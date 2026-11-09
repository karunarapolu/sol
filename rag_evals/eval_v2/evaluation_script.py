#!/usr/bin/env python3
"""
Deterministic retrieval evaluation.

Run:
    python rag_evals/eval_v2/evaluation_script.py

Reads only eval*.json files inside eval_v2.
Writes eval_metrics.json inside eval_v2.

Uses Python standard library only.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "eval_metrics.json"
K = 5


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).casefold().split())


def parse_id_set(value: Any) -> set[str]:
    if value is None:
        return set()

    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}

    if isinstance(value, dict):
        return {str(item).strip() for item in value.keys()}

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            return {
                item.strip()
                for item in re.findall(r"""['"]([^'"]+)['"]""", value)
            }

        if isinstance(parsed, (list, tuple, set)):
            return {str(item).strip() for item in parsed if str(item).strip()}

        if isinstance(parsed, dict):
            return {str(item).strip() for item in parsed.keys()}

    return set()


def find_value_case_insensitive(data: Any, wanted: str) -> Any:
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).casefold() == wanted.casefold():
                return value

        for value in data.values():
            found = find_value_case_insensitive(value, wanted)
            if found is not None:
                return found

    elif isinstance(data, list):
        for value in data:
            found = find_value_case_insensitive(value, wanted)
            if found is not None:
                return found

    return None


def extract_retrieved_ids(data: dict[str, Any]) -> tuple[set[str], set[str]]:
    output = find_value_case_insensitive(data, "output")

    dense_ids = set()
    bm25_ids = set()

    if isinstance(output, dict):
        dense_ids = parse_id_set(
            find_value_case_insensitive(output, "Dense IDs")
        )
        bm25_ids = parse_id_set(
            find_value_case_insensitive(output, "BM25 IDs")
        )

    elif isinstance(output, str):
        dense_match = re.search(
            r"Dense\s+IDs\s*:\s*(\{.*?\}|\[.*?\])",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )

        bm25_match = re.search(
            r"BM25\s+IDs\s*:\s*(\{.*?\}|\[.*?\])",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if dense_match:
            dense_ids = parse_id_set(dense_match.group(1))

        if bm25_match:
            bm25_ids = parse_id_set(bm25_match.group(1))

    return dense_ids, bm25_ids


def extract_question(data: dict[str, Any]) -> str:
    question = find_value_case_insensitive(data, "question")
    return str(question).strip() if question is not None else ""


def extract_chunk_records(value: Any) -> tuple[set[str], set[str]]:
    ids = set()
    texts = set()

    def walk(node: Any):
        if isinstance(node, dict):
            chunk_id = node.get("chunk_id", node.get("id"))

            if chunk_id is not None and not isinstance(chunk_id, (dict, list)):
                ids.add(str(chunk_id).strip())

            payload = node.get("payload")

            if isinstance(payload, dict):
                payload_id = payload.get("chunk_id", payload.get("id"))
                if payload_id is not None:
                    ids.add(str(payload_id).strip())

                for key in ("text", "page_content", "content"):
                    if payload.get(key) is not None:
                        text = normalize_text(payload[key])
                        if text:
                            texts.add(text)

            for key in ("text", "page_content", "content"):
                if node.get(key) is not None and not isinstance(node[key], (dict, list)):
                    text = normalize_text(node[key])
                    if text:
                        texts.add(text)

            for child in node.values():
                if isinstance(child, (dict, list)):
                    walk(child)

        elif isinstance(node, list):
            for child in node:
                walk(child)

        elif isinstance(node, str):
            text = normalize_text(node)
            if text:
                texts.add(text)

    walk(value)
    return ids, texts


def load_correct_chunks(data: dict[str, Any]) -> tuple[set[str], set[str]]:
    correct = find_value_case_insensitive(data, "correct chunk")

    if correct is None:
        correct = find_value_case_insensitive(data, "correct_chunk")

    if correct is None:
        return set(), set()

    # Special eval_3 format:
    # "correct chunk": {
    #     "text": "Any 3 chunks from eval_v1/eval3_crt_chunks.json"
    # }
    if isinstance(correct, dict):
        reference = correct.get("text")

        if isinstance(reference, str):
            match = re.search(
                r"Any\s+3\s+chunks\s+from\s+(.+\.json)\s*$",
                reference.strip(),
                flags=re.IGNORECASE,
            )

            if match:
                relative_path = Path(match.group(1).strip())

                # Resolve relative to rag_evals/.
                referenced_file = SCRIPT_DIR.parent / relative_path

                if not referenced_file.is_file():
                    raise FileNotFoundError(
                        f"Cannot find correct chunks file: {referenced_file}"
                    )

                with referenced_file.open("r", encoding="utf-8") as file:
                    referenced_data = json.load(file)

                points = find_value_case_insensitive(referenced_data, "points")

                if isinstance(points, list):
                    selected_chunks = points[:3]
                elif isinstance(referenced_data, list):
                    selected_chunks = referenced_data[:3]
                else:
                    selected_chunks = [referenced_data]

                return extract_chunk_records(selected_chunks)

    return extract_chunk_records(correct)


def evaluate_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    dense_ids, bm25_ids = extract_retrieved_ids(data)

    # All retrieved chunks are the union of Dense and BM25.
    retrieved_ids = dense_ids | bm25_ids

    correct_ids, correct_texts = load_correct_chunks(data)

    if retrieved_ids and correct_ids:
        true_positives = len(retrieved_ids & correct_ids)
        correct_count = len(correct_ids)

    else:
        output = find_value_case_insensitive(data, "output")
        retrieved_texts = extract_chunk_records(output)[1]

        if correct_texts and retrieved_texts:
            true_positives = len(correct_texts & retrieved_texts)
            correct_count = len(correct_texts)
        else:
            true_positives = len(retrieved_ids & correct_ids)
            correct_count = len(correct_ids)

    precision = (
        true_positives / len(retrieved_ids)
        if retrieved_ids else 0.0
    )

    recall = (
        true_positives / correct_count
        if correct_count else 0.0
    )

    return {
        "question": extract_question(data),
        "source_file": path.name,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "counts": {
            "correct_chunks": correct_count,
            "dense_chunks": len(dense_ids),
            "bm25_chunks": len(bm25_ids),
        },
    }


def main():
    # Only JSON files inside eval_v2 are considered.
    files = sorted(
        path for path in SCRIPT_DIR.glob("eval*.json")
        if path.is_file()
        and path.name != OUTPUT_FILE.name
    )

    if not files:
        print("No evaluation files found.")
        return 1

    print(f"Found {len(files)} evaluation files.")

    per_query = []

    for path in files:
        print(f"Evaluating {path.name}...")

        try:
            result = evaluate_file(path)
            per_query.append(result)

        except Exception as error:
            print(f"Error in {path.name}: {error}", file=sys.stderr)
            return 1

    result = {
        "evaluation_info": {
            "folder": SCRIPT_DIR.name,
            "evaluated_files": [path.name for path in files],
            "total_queries": len(per_query),
            "k": K,
            "llm_used": False,
            "dependencies": "Python standard library only",
            "important_note": (
                "End-to-end scores are deterministic lexical/structural "
                "proxies, not semantic judgments. Retrieved chunks are "
                "calculated as the union of Dense and BM25 chunk IDs. "
                "Precision and recall use this union. Dense IDs and BM25 IDs "
                "are unordered, so ranked @k metrics cannot be calculated."
            ),
        },
        "per_query": per_query,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"\nEvaluation complete: {OUTPUT_FILE}")
    print(f"Total queries: {len(per_query)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())