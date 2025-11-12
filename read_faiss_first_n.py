#!/usr/bin/env python3
"""Read and print/save the first N vectors from a FAISS index file.

Usage:
    python read_faiss_first_n.py --index recipe_rag_index.faiss --n 10 --out first10.json

This script attempts to reconstruct the first N vectors using
`index.reconstruct(i)`. If the index type doesn't support reconstruction
an informative message will be printed.
"""
import argparse
import json
import os
import sys
from typing import List

import faiss
import numpy as np


def read_first_n(index_path: str, n: int) -> List[List[float]]:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    index = faiss.read_index(index_path)
    total = index.ntotal
    if total == 0:
        return []

    n = min(n, total)
    vectors = []

    # Many FAISS indexes support reconstruct(i). Use it if available.
    if hasattr(index, 'reconstruct'):
        for i in range(n):
            try:
                vec = faiss.vector_to_array(index.reconstruct(i))
            except Exception:
                # Some composite indexes may require using the inner index
                # or otherwise may not support reconstruct. Re-raise with
                # helpful context.
                raise RuntimeError(f"Failed to reconstruct vector {i} from index of type {type(index)}")
            vectors.append([float(x) for x in vec])
    else:
        # If reconstruct isn't available, try a fallback: search for the
        # vector closest to the i-th id by running a dummy search using
        # the stored vectors — but without reconstruct this is not possible.
        raise RuntimeError("Index does not support reconstruct(i); cannot read raw vectors")

    return vectors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', '-i', type=str, default='recipe_rag_index.faiss', help='Path to FAISS index file')
    p.add_argument('--n', type=int, default=10, help='Number of vectors to read')
    p.add_argument('--out', '-o', type=str, default=None, help='Optional output JSON file to save vectors')
    args = p.parse_args()

    try:
        vectors = read_first_n(args.index, args.n)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Read {len(vectors)} vectors from {args.index} (requested {args.n})")
    for i, v in enumerate(vectors):
        print(f"[{i}] len={len(v)} first5={v[:5]}")

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'index': os.path.basename(args.index), 'vectors': vectors}, f, ensure_ascii=False, indent=2)
        print(f"Saved vectors to {args.out}")


if __name__ == '__main__':
    main()
