import argparse
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


def load_json(file_path: str) -> Any:
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def save_json(data: Any, file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


DEFAULT_INPUT_PATH = "./recipe_corpus_sample.json"
DEFAULT_OUTPUT_PATH = "./recipe_chunks.json"
DEFAULT_META_PATH = "./recipe_meta.json"
DEFAULT_MAX_CHARS = 800
MIN_BODY_CHARS = 200
INVALID_DISH_VALUES = {"", "unknown", "null", "none", "undefined", "nan"}
LEV_WEIGHT = 0.6
JACCARD_WEIGHT = 0.4
KEYWORD_DUP_THRESHOLD = 0.4


def normalize_text(value: Any) -> str:
    """Normalize scalar values to stripped text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_dish(dish: Any, fallback: str) -> str:
    """Replace invalid dish values with the recipe name."""
    candidate = normalize_text(dish)
    if not candidate or candidate.lower() in INVALID_DISH_VALUES:
        safe_fallback = normalize_text(fallback) or "Unknown Recipe"
        return safe_fallback
    return candidate


def ensure_list(value: Any) -> List[str]:
    """Force value into a list of non-empty strings."""
    if isinstance(value, list):
        items = value
    elif value is None:
        items = []
    elif isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    result: List[str] = []
    for item in items:
        text = normalize_text(item)
        if text:
            result.append(text)
    return result


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance via dynamic programming."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = current_row
    return prev_row[-1]


def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity over whitespace tokens."""
    tokens_a = set(normalize_text(a).lower().split())
    tokens_b = set(normalize_text(b).lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def keyword_similarity(a: str, b: str) -> float:
    """Blend Levenshtein similarity and Jaccard score."""
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    lev_sim = 1.0 - (levenshtein_distance(a, b) / max_len)
    jac_sim = jaccard_similarity(a, b)
    return (LEV_WEIGHT * lev_sim) + (JACCARD_WEIGHT * jac_sim)


def deduplicate_keywords(keywords: Sequence[Any], threshold: float = KEYWORD_DUP_THRESHOLD) -> List[str]:
    """Remove near-duplicate keywords using weighted similarity."""
    unique: List[str] = []
    for raw_kw in keywords or []:
        keyword = normalize_text(raw_kw)
        if not keyword:
            continue
        duplicate = False
        for existing in unique:
            if keyword_similarity(keyword, existing) >= threshold:
                duplicate = True
                break
        if not duplicate:
            unique.append(keyword)
    return unique


def format_description(value: Any) -> str:
    return normalize_text(value)


def format_ingredients(value: Any) -> str:
    items = ensure_list(value)
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def format_instructions(value: Any) -> str:
    steps = ensure_list(value)
    if not steps:
        return ""
    return "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))


def build_keywords_text(keywords: Sequence[str]) -> str:
    safe_keywords = list(keywords)
    if not safe_keywords:
        return "关键词：暂无"
    return f"关键词：{'，'.join(safe_keywords)}"


def split_long_text(text: str, max_length: int) -> List[str]:
    """Split long text into chunks that stay under the limit."""
    clean_text = text.strip()
    if not clean_text:
        return []
    if len(clean_text) <= max_length:
        return [clean_text]
    segments = [seg for seg in re.split(r'(?<=[。！？!?])\s+|\n+', clean_text) if seg]
    if not segments:
        return [clean_text]
    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0
    for segment in segments:
        seg = segment.strip()
        if not seg:
            continue
        seg_len = len(seg) + (1 if buffer else 0)
        if seg_len > max_length:
            # Hard wrap segments that are longer than the budget.
            for i in range(0, len(seg), max_length):
                part = seg[i:i + max_length]
                if buffer:
                    chunks.append("\n".join(buffer))
                    buffer = []
                    current_len = 0
                chunks.append(part)
            continue
        if current_len + seg_len > max_length and buffer:
            chunks.append("\n".join(buffer))
            buffer = [seg]
            current_len = len(seg)
        else:
            buffer.append(seg)
            current_len += seg_len
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


def build_section_chunks(label: str, content: str, body_limit: int) -> List[Dict[str, str]]:
    if not content:
        return []
    bodies = split_long_text(content, max(body_limit, MIN_BODY_CHARS))
    section_chunks: List[Dict[str, str]] = []
    for idx, body in enumerate(bodies):
        suffix = "" if idx == 0 else f"（续{idx}）"
        section_chunks.append(
            {
                "type": f"{label}{idx}",
                "content": f"{label}{suffix}：\n{body}",
            }
        )
    return section_chunks


def build_sample_chunks(sample: Dict[str, Any], origin_id: int, max_chunk_chars: int) -> List[str]:
    name = normalize_text(sample.get("name")) or f"Recipe-{origin_id}"
    dish = normalize_dish(sample.get("dish"), name)
    header = f"{name}-{dish}" if name != dish else name
    keywords = deduplicate_keywords(sample.get("keywords", []))
    keywords_text = build_keywords_text(keywords)
    body_budget = max(max_chunk_chars - len(header) - len(keywords_text) - 16, MIN_BODY_CHARS)
    sections: List[Dict[str, str]] = []

    description = format_description(sample.get("description"))
    sections.extend(build_section_chunks("描述", description, body_budget))

    ingredients = format_ingredients(sample.get("recipeIngredient"))
    sections.extend(build_section_chunks("原料", ingredients, body_budget))

    instructions = format_instructions(sample.get("recipeInstructions"))
    sections.extend(build_section_chunks("做法步骤", instructions, body_budget))

    chunks: List[Dict[str, str]] = []
    for section in sections:
        chunk_text = f"{header}\n{section['content']}\n{keywords_text}"
        chunks.append(
            {
                "type": section["type"],
                "text": chunk_text,
            }
        )
    return chunks


def iter_samples(raw_data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw_data, list):
        for item in raw_data:
            if isinstance(item, dict):
                yield item
    elif isinstance(raw_data, dict):
        for value in raw_data.values():
            if isinstance(value, dict):
                yield value


def process_dataset(
    input_path: str,
    output_path: str,
    max_chunk_chars: int,
    meta_output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    data = load_json(input_path)
    processed: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    chunk_id = 0
    for origin_id, sample in enumerate(iter_samples(data)):
        meta[str(origin_id)] = sample
        sample_chunks = build_sample_chunks(sample, origin_id, max_chunk_chars)
        for chunk in sample_chunks:
            processed.append(
                {
                    "origin_id": origin_id,
                    "chunk_id": chunk_id,
                    "type": chunk["type"],
                    "text": chunk["text"],
                }
            )
            chunk_id += 1
    if output_path:
        save_json(processed, output_path)
    if meta_output_path:
        save_json(meta, meta_output_path)
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess recipe data into RAG-ready chunks.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_PATH, help="Path to the source JSON file.")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_PATH,
        help="Where to store the processed chunk JSON output.",
    )
    parser.add_argument(
        "--meta",
        default=DEFAULT_META_PATH,
        help="Where to store the origin_id to raw item mapping JSON.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters per chunk including header and keywords.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = process_dataset(args.input, args.output, args.max_chars, args.meta)
    print(f"Generated {len(processed)} chunks -> {args.output}\nSaved meta -> {args.meta}")


if __name__ == "__main__":
    main()