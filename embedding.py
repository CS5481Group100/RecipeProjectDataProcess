import argparse
import json
import math
import os
import sys
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from concurrent.futures import ProcessPoolExecutor

# This file provides a streaming + parallel embedding pipeline for very large
# JSONL datasets (one JSON recipe per line). It avoids langchain and pydantic
# dependencies so it works on Python 3.14.

# ---------------------- 1. JSON -> text converter ----------------------
def json_to_text(recipe_json: Dict) -> str:
    name = recipe_json.get('name', '未知名称')
    dish = recipe_json.get('dish', '家常菜')
    description = recipe_json.get('description', '无描述')

    ingredients = recipe_json.get('recipeIngredient', []) or []
    ingredients_str = '、'.join(ingredients) if ingredients else '无食材信息'

    instructions = recipe_json.get('recipeInstructions', []) or []
    if instructions:
        instructions_str = '；'.join([f'{i+1}. {step.strip()}' for i, step in enumerate(instructions)])
    else:
        instructions_str = '无步骤信息'

    author = recipe_json.get('author', '未知作者')
    keywords = recipe_json.get('keywords', []) or []
    keywords_str = '、'.join(keywords) if keywords else '无关键词'

    text = (
        f"菜品：{name}（{dish}）。{description} "
        f"所需食材：{ingredients_str}。 "
        f"烹饪步骤：{instructions_str}。 "
        f"相关关键词：{keywords_str}。作者：{author}"
    )
    text = text.replace('乱泉水', '矿泉水').strip().replace('  ', ' ')
    return text


# ---------------------- 2. Simple Chinese-aware splitter ----------------------
def split_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    # Split keeping punctuation (。；，) attached to sentences
    parts = re.split('(。|；|，)', text)
    sentences = [''.join(i) for i in zip(parts[::2], parts[1::2] + [''])]
    sentences = [s for s in sentences if s.strip()]

    chunks: List[str] = []
    current: List[str] = []
    cur_len = 0

    for s in sentences:
        slen = len(s)
        if cur_len + slen <= chunk_size:
            current.append(s)
            cur_len += slen
        else:
            if current:
                chunks.append(''.join(current))

            # prepare new chunk: optionally include tail overlap from previous chunk
            if chunks and overlap > 0:
                tail = chunks[-1][-overlap:]
                current = [tail, s]
                cur_len = len(tail) + slen
            else:
                current = [s]
                cur_len = slen

    if current:
        chunks.append(''.join(current))

    return [c for c in chunks if len(c) > 50]


# ---------------------- 3. Embedding worker helpers ----------------------
# We use a process pool. Each worker will initialize its own SentenceTransformer
# to avoid model pickling issues and to take advantage of multiple CPUs/GPUs.
_WORKER_MODEL: Optional[SentenceTransformer] = None

def _worker_init(model_name: str):
    global _WORKER_MODEL
    _WORKER_MODEL = SentenceTransformer(model_name)

def _worker_embed(batch_texts: List[str]) -> np.ndarray:
    # Called inside worker process
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        raise RuntimeError("Worker model not initialized")
    arr = _WORKER_MODEL.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
    return arr.astype('float32')


# ---------------------- 4. Streaming pipeline ----------------------
def build_index_streaming(
    input_path: str,
    model_name: str = 'all-MiniLM-L6-v2',
    chunk_size: int = 400,
    overlap: int = 50,
    chunks_per_batch: int = 2048,
    workers: int = None,
    save_prefix: str = 'recipe_rag',
    max_lines: Optional[int] = None,
):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    # Files
    chunks_out_path = f'{save_prefix}_chunks.jsonl'
    index_path = f'{save_prefix}_index.faiss'

    # Buffers
    chunk_buffer: List[str] = []

    # We'll create FAISS index after we know embedding dimension
    index: Optional[faiss.IndexFlatL2] = None

    # Start process pool
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(model_name,)) as exe:
        # open input and chunks output
        with open(input_path, 'r', encoding='utf-8') as fin, open(chunks_out_path, 'w', encoding='utf-8') as fout:
            line_count = 0
            for line in fin:
                line_count += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # skip malformed line but continue
                    continue

                text = json_to_text(obj)
                cks = split_text(text, chunk_size=chunk_size, overlap=overlap)
                for c in cks:
                    chunk_buffer.append(c)
                # when buffer large enough, dispatch to workers
                if len(chunk_buffer) >= chunks_per_batch:
                    _dispatch_and_add(exe, chunk_buffer, fout, index, index_path)
                    # after dispatch, index may be created and returned
                    # read index from disk to update reference
                    if os.path.exists(index_path):
                        index = faiss.read_index(index_path)
                    chunk_buffer = []

                # if a maximum number of lines was requested, stop after processing
                if max_lines is not None and line_count >= max_lines:
                    # break out of the for-loop; final flush will happen below
                    break

            # final flush
            if chunk_buffer:
                _dispatch_and_add(exe, chunk_buffer, fout, index, index_path)
                if os.path.exists(index_path):
                    index = faiss.read_index(index_path)

    print(f"Processed ~{line_count} lines. Index saved to {index_path}, chunks to {chunks_out_path}")
    return index, chunks_out_path


def _dispatch_and_add(exe: ProcessPoolExecutor, chunk_buffer: List[str], fout, index: Optional[faiss.IndexFlatL2], index_path: str):
    # Split chunk_buffer into roughly equal sub-batches for workers
    workers = exe._max_workers
    total = len(chunk_buffer)
    batch_size = max(1, math.ceil(total / workers))
    batches = [chunk_buffer[i:i+batch_size] for i in range(0, total, batch_size)]

    # Submit map to workers and collect embeddings
    results = list(exe.map(_worker_embed, batches))

    # write chunks to file in order (one chunk per line)
    for c in chunk_buffer:
        fout.write(json.dumps({'chunk': c}, ensure_ascii=False) + '\n')

    # stack embeddings in same order
    emb_list = [r for r in results if r is not None]
    embeddings = np.vstack(emb_list).astype('float32')

    # create index if needed
    if index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
    # add to index
    index.add(embeddings)
    # persist index to disk after each batch to keep progress
    faiss.write_index(index, index_path)


# ---------------------- 5. Command line / example ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', type=str, default='recipe_corpus_full.json', help='Input JSONL file (one JSON per line)')
    p.add_argument('--model', type=str, default='all-MiniLM-L6-v2')
    p.add_argument('--chunk_size', type=int, default=400)
    p.add_argument('--overlap', type=int, default=50)
    p.add_argument('--chunks_per_batch', type=int, default=2048)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--save_prefix', type=str, default='recipe_rag')
    p.add_argument('--max-lines', type=int, default=None, help='Maximum number of input lines to process (for testing)')
    return p.parse_args()


if __name__ == '__main__':
    # If user runs without the big file present, fall back to a small sample pipeline
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"Input {args.input} not found, running a small built-in demo to validate execution.")
        # small demo dataset (keeps prior sample)
        sample = [
            {
                'name': '红烧滩羊肉',
                'dish': 'Unknown',
                'description': '每到桂花香满城后，就可以吃羊肉温补身体了',
                'recipeIngredient': ['1kg羊肉', '5片姜', '3瓣蒜', '适量花椒'],
                'recipeInstructions': ['焯水', '炒香', '炖煮40分钟'],
                'author': 'demo',
                'keywords': ['红烧', '羊肉']
            }
        ]

        # run the original simple pipeline for smoke test
        texts = [json_to_text(d) for d in sample]
        chunks = []
        for t in texts:
            chunks.extend(split_text(t, chunk_size=args.chunk_size, overlap=args.overlap))
        print('Chunks:', chunks)
        model = SentenceTransformer(args.model)
        emb = model.encode(chunks, convert_to_numpy=True)
        emb = emb.astype('float32')
        idx = faiss.IndexFlatL2(emb.shape[1])
        idx.add(emb)
        print('Demo index size:', idx.ntotal)
        sys.exit(0)

    # run streaming build on provided large file
    build_index_streaming(
        input_path=args.input,
        model_name=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        chunks_per_batch=args.chunks_per_batch,
        workers=args.workers,
        save_prefix=args.save_prefix,
        max_lines=args.max_lines,
    )
