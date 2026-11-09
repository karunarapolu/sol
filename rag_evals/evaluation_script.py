#!/usr/bin/env python3
"""
Deterministic evaluation for logged RAG retrieval and answers.

Usage (from the project root):
    python rag_evals/evaluation_script.py eval_v1

The argument may be a folder name relative to this script's directory or a
path to a folder. Results are written to <folder>/eval_metrics.json.

No LLM, API key, RAGAS, or third-party package is required.

Important:
- Retrieval metrics use binary chunk relevance (chunk ID first, then exact text
  matching). If the final log omits chunk IDs, exact normalized chunk text is
  used to identify relevant chunks.
- End-to-end metrics are deterministic lexical/structural proxies. They are
  NOT equivalent to human or LLM semantic judgments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


# Small stopword list: removes common function words from lexical comparisons.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "me", "more", "most", "my", "of", "on", "or", "our",
    "ours", "she", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "then", "there", "these", "they", "this", "those",
    "to", "too", "us", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your", "yours",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokens(value: Any, *, remove_stopwords: bool = True) -> list[str]:
    if value is None:
        return []
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_@.+'-]*", str(value).lower())
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return words


def token_set(value: Any) -> set[str]:
    return set(tokens(value))


def normalize_text(value: Any) -> str:
    return " ".join(tokens(value, remove_stopwords=False))


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def rounded(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def recursively_find_points(obj: Any) -> list[dict]:
    """Find Qdrant-style points anywhere inside a nested object."""
    found: list[dict] = []
    if isinstance(obj, dict):
        points = obj.get("points")
        if isinstance(points, list):
            found.extend(p for p in points if isinstance(p, dict))
        for value in obj.values():
            if isinstance(value, (dict, list)):
                found.extend(recursively_find_points(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(recursively_find_points(item))
    return found


def extract_correct_chunks(data: Any) -> list[dict]:
    """
    Accepts:
      - {"correct chunk": {"result": {"points": [...]}}}
      - {"correct_chunk": [...]}
      - lists of {"id": ..., "payload": {"text": ...}}
      - ordinary strings / text-bearing dictionaries
    """
    chunks: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(chunk_id: Any = None, text: Any = None) -> None:
        if text is None and chunk_id is None:
            return
        cid = str(chunk_id) if chunk_id is not None else None
        txt = str(text) if text is not None else ""
        key = (cid or "", normalize_text(txt))
        if key not in seen:
            seen.add(key)
            chunks.append({"id": cid, "text": txt})

    points = recursively_find_points(data)
    for point in points:
        payload = point.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        add(
            point.get("id", payload.get("chunk_id")),
            payload.get("text", point.get("text")),
        )

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            # Avoid treating a Qdrant point twice; add() deduplicates it.
            payload = obj.get("payload")
            if isinstance(payload, dict) and payload.get("text") is not None:
                add(obj.get("id", payload.get("chunk_id")), payload.get("text"))
            elif obj.get("text") is not None:
                # "text" may itself be a nested object, as in the user's
                # {"text": "Any 3 from ..."} convention.
                if isinstance(obj["text"], (str, int, float)):
                    placeholder = str(obj["text"]).strip().lower()
                    if not placeholder.startswith("any ") or "from " not in placeholder:
                        add(obj.get("id", obj.get("chunk_id")), obj["text"])
            for key, value in obj.items():
                if key not in {"payload"} and isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            add(None, obj)

    walk(data)
    return chunks


def find_correct_chunks(record: dict, eval_path: Path) -> list[dict]:
    correct_value = None
    for key in ("correct chunk", "correct_chunk", "correct_chunks",
                "relevant_chunks", "ground_truth_chunks"):
        if key in record:
            correct_value = record[key]
            break

    chunks = extract_correct_chunks(correct_value) if correct_value is not None else []

    # Optional companion file, e.g. eval3_crt_chunks.json. Merge it with the
    # inline ground truth if it contains additional chunk records.
    companion_candidates = [
        eval_path.with_name(eval_path.stem + "_crt_chunks.json"),
        eval_path.with_name(eval_path.stem.replace("_crt_chunks", "") + "_crt_chunks.json"),
    ]
    for companion in companion_candidates:
        if companion.exists() and companion != eval_path:
            try:
                chunks.extend(extract_correct_chunks(load_json(companion)))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"WARNING: Could not read {companion.name}: {exc}", file=sys.stderr)
            break

    # Deduplicate after combining.
    unique: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        key = (chunk.get("id") or "", normalize_text(chunk.get("text", "")))
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def extract_id(text: str) -> str | None:
    patterns = (
        r"""Metadata:\s*\{[^}]*['_"]_id['_"]\s*:\s*['"]?(\d+)""",
        r"""(?:chunk_id|point_id)\s*[:=]\s*['"]?(\d+)""",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def parse_logged_chunks(output: str) -> dict[str, list[dict]]:
    """Parse the three retrieval stages from the user's text log format."""
    stages: dict[str, list[dict]] = {"dense": [], "bm25": [], "final": []}

    # Dense retrieval: each chunk extends until the next dense chunk or section.
    dense_section = re.search(
        r"========== DENSE RETRIEVAL ==========(.*?)(?========== BM25 RESULTS ==========)",
        output, flags=re.DOTALL
    )
    if dense_section:
        body = dense_section.group(1)
        matches = list(re.finditer(r"--- DENSE CHUNK \d+ ---\n", body))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            raw = body[match.end():end].strip()
            raw = re.sub(r"\n(?=Metadata:)", "\n", raw)
            stages["dense"].append({"id": extract_id(raw), "text": raw})

    # BM25 results: split on Rank lines, preserving each result's text.
    bm25_section = re.search(
        r"========== BM25 RESULTS ==========(.*?)(?=FINAL RETRIEVED CHUNKS|$)",
        output, flags=re.DOTALL
    )
    if bm25_section:
        body = bm25_section.group(1)
        matches = list(re.finditer(r"(?m)^Rank:\s*\d+.*$", body))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            raw = body[match.start():end].strip()
            raw = re.sub(r"(?m)^Rank:\s*\d+.*?(?=\nText:)", "", raw, count=1, flags=re.DOTALL)
            raw = re.sub(r"(?m)^Text:\s*", "", raw, count=1)
            stages["bm25"].append({"id": extract_id(raw), "text": raw})

    # Final retrieval: this is the primary stage used for retrieval_level.
    final_match = re.search(
        r"FINAL RETRIEVED CHUNKS\s*=+\s*(.*?)(?:\n={10,}|$)",
        output, flags=re.DOTALL
    )
    if final_match:
        body = final_match.group(1)
        matches = list(re.finditer(r"--- CHUNK \d+ ---\n", body))
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            raw = body[match.end():end].strip()
            raw = re.sub(r"\n-+\s*$", "", raw).strip()
            stages["final"].append({"id": extract_id(raw), "text": raw})

    return stages


def is_relevant(retrieved: dict, relevant_chunks: list[dict]) -> bool:
    rid = retrieved.get("id")
    rtext = normalize_text(retrieved.get("text", ""))

    for expected in relevant_chunks:
        eid = expected.get("id")
        etext = normalize_text(expected.get("text", ""))

        # Exact ID match is the strongest signal.
        if rid is not None and eid is not None and str(rid) == str(eid):
            return True

        # Logs sometimes omit IDs from the final section. In that case use
        # exact normalized text, not fuzzy semantic similarity.
        if rtext and etext and (rtext == etext or rtext in etext or etext in rtext):
            return True
    return False


def retrieval_metrics(retrieved: list[dict], relevant: list[dict], k: int = 5) -> dict:
    if not relevant:
        return {
            "precision": None, "recall": None, "f1_score": None,
            "hit_rate": None, "mean_reciprocal_rank_mrr": None,
            "mean_average_precision_map": None, "ndcg_at_k": None,
            "precision_at_k": None, "recall_at_k": None,
            "k": k, "retrieved_count": len(retrieved), "relevant_count": 0,
        }

    relevance = [is_relevant(item, relevant) for item in retrieved]
    true_positives = sum(relevance)
    precision = safe_div(true_positives, len(retrieved))
    recall = safe_div(true_positives, len(relevant))
    f1 = safe_div(2 * precision * recall, precision + recall)

    first_rank = next((i + 1 for i, hit in enumerate(relevance) if hit), None)
    mrr = 1 / first_rank if first_rank else 0.0

    precisions_at_relevant_ranks = []
    seen_relevant = 0
    for rank, hit in enumerate(relevance, start=1):
        if hit:
            seen_relevant += 1
            precisions_at_relevant_ranks.append(seen_relevant / rank)
    ap = safe_div(sum(precisions_at_relevant_ranks), len(relevant))

    top = relevance[:k]
    p_at_k = safe_div(sum(top), k)
    r_at_k = safe_div(sum(top), len(relevant))

    dcg = sum((1 / math.log2(rank + 1)) for rank, hit in enumerate(top, start=1) if hit)
    ideal_relevant = min(len(relevant), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    ndcg = safe_div(dcg, idcg)

    return {
        "precision": rounded(precision),
        "recall": rounded(recall),
        "f1_score": rounded(f1),
        "hit_rate": 1.0 if true_positives else 0.0,
        "mean_reciprocal_rank_mrr": rounded(mrr),
        "mean_average_precision_map": rounded(ap),
        "ndcg_at_k": rounded(ndcg),
        "precision_at_k": rounded(p_at_k),
        "recall_at_k": rounded(r_at_k),
        "k": k,
        "retrieved_count": len(retrieved),
        "relevant_count": len(relevant),
        "matched_relevant_count": true_positives,
    }


def cosine_similarity(a: Any, b: Any) -> float:
    """Cosine similarity of binary token vectors; dependency-free."""
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / math.sqrt(len(sa) * len(sb))


def token_f1(prediction: Any, reference: Any) -> float:
    pred, ref = token_set(prediction), token_set(reference)
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    p = overlap / len(pred)
    r = overlap / len(ref)
    return safe_div(2 * p * r, p + r)


def split_sentences(text: str) -> list[str]:
    # Keep bullet/list lines as separate units as well.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip(" \t-*•") for p in parts if p.strip(" \t-*•")]


def end_to_end_metrics(answer: str, reference_answer: str, context: str) -> dict:
    answer_tokens = token_set(answer)
    reference_tokens = token_set(reference_answer)
    context_tokens = token_set(context)

    # Answer relevance: lexical F1 against the supplied reference answer.
    answer_relevance = token_f1(answer, reference_answer)

    # Context utilization: share of meaningful answer terms present in context.
    context_utilization = safe_div(len(answer_tokens & context_tokens), len(answer_tokens))

    # Groundedness proxy: fraction of answer sentences with at least 30% of
    # their meaningful terms present in the retrieved context.
    sentences = split_sentences(answer)
    sentence_support = []
    for sentence in sentences:
        st = token_set(sentence)
        if not st:
            continue
        sentence_support.append(safe_div(len(st & context_tokens), len(st)))
    groundedness = (
        sum(score >= 0.30 for score in sentence_support) / len(sentence_support)
        if sentence_support else 0.0
    )

    # Hallucination-rate proxy: complement of the sentence-level groundedness
    # proxy. It is not factual verification.
    hallucination_rate = 1.0 - groundedness

    # Coherence proxy: proportion of answer units that are not fragments.
    # A unit is considered structurally complete if it has >= 3 meaningful
    # words, or is a short list/name item (>= 2 words).
    coherence_units = split_sentences(answer)
    complete_units = 0
    for unit in coherence_units:
        raw_words = tokens(unit)
        if len(raw_words) >= 3 or (len(raw_words) >= 2 and not re.search(r"[.!?]$", unit)):
            complete_units += 1
    response_coherence = safe_div(complete_units, len(coherence_units))

    # Overall relevancy score: cosine similarity against the reference.
    relevancy_score = cosine_similarity(answer, reference_answer)

    return {
        "answer_relevance": rounded(answer_relevance),
        "context_utilization": rounded(context_utilization),
        "groundedness": rounded(groundedness),
        "hallucination_rate": rounded(hallucination_rate),
        "response_coherence": rounded(response_coherence),
        "relevancy_score": rounded(relevancy_score),
        "method": {
            "answer_relevance": "Token-level precision/recall F1 against accurate_answer.",
            "context_utilization": "Fraction of meaningful answer tokens also found in retrieved context.",
            "groundedness": "Fraction of answer sentences with at least 30% meaningful-token overlap with retrieved context.",
            "hallucination_rate": "1 - groundedness proxy; not factual hallucination detection.",
            "response_coherence": "Structural completeness heuristic; not semantic coherence.",
            "relevancy_score": "Cosine similarity between binary token sets of answer and accurate_answer.",
        },
    }


def average_dicts(rows: list[dict], keys: Iterable[str]) -> dict:
    result = {}
    for key in keys:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        result[key] = rounded(statistics.mean(values)) if values else None
    return result


def evaluate_record(record: dict, eval_path: Path, k: int) -> dict:
    question = str(record.get("question", ""))
    answer = str(record.get("answer", ""))
    reference_answer = str(
        record.get("accurate_answer")
        or record.get("expected_answer")
        or record.get("ground_truth")
        or ""
    )
    output = str(record.get("output", ""))

    correct = find_correct_chunks(record, eval_path)
    stages = parse_logged_chunks(output)

    # The primary retrieval metrics use the final reranked/combined retrieval.
    # Fall back to dense retrieval if the log does not contain final chunks.
    primary_stage = "final" if stages["final"] else "dense"
    primary_retrieved = stages[primary_stage]

    # Build context from final retrieved chunks, falling back to dense then
    # BM25. This context is what the answer-generation stage should have seen.
    context_chunks = stages["final"] or stages["dense"] or stages["bm25"]
    context = "\n".join(chunk["text"] for chunk in context_chunks)

    return {
        "question": question,
        "source_file": eval_path.name,
        "retrieval_stage_used": primary_stage,
        "retrieval_level": retrieval_metrics(primary_retrieved, correct, k),
        "retrieval_stages": {
            stage: retrieval_metrics(chunks, correct, k)
            for stage, chunks in stages.items()
        },
        "end_to_end": end_to_end_metrics(answer, reference_answer, context),
        "counts": {
            "correct_chunks": len(correct),
            "dense_chunks": len(stages["dense"]),
            "bm25_chunks": len(stages["bm25"]),
            "final_chunks": len(stages["final"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval logs and answer quality without an LLM."
    )
    parser.add_argument(
        "eval_folder",
        help="Folder containing eval*.json, e.g. eval_v1 (relative to rag_evals/).",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Cutoff for P@k, R@k and NDCG@k (default: 5).",
    )
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be >= 1")

    supplied = Path(args.eval_folder)
    eval_dir = supplied if supplied.is_absolute() else (
        supplied if supplied.exists() else Path(__file__).resolve().parent / supplied
    )
    if not eval_dir.is_dir():
        print(f"ERROR: Evaluation folder not found: {eval_dir}", file=sys.stderr)
        return 1

    eval_files = sorted(
        p for p in eval_dir.glob("eval*.json")
        if p.name != "eval_metrics.json" and not p.stem.endswith("_crt_chunks")
    )
    if not eval_files:
        print(f"ERROR: No eval*.json files found in {eval_dir}", file=sys.stderr)
        return 1

    per_query = []
    for path in eval_files:
        try:
            data = load_json(path)
            records = data if isinstance(data, list) else [data]
            for index, record in enumerate(records, start=1):
                if not isinstance(record, dict):
                    print(f"WARNING: Skipping non-object record in {path.name}", file=sys.stderr)
                    continue
                result = evaluate_record(record, path, args.k)
                result["record_index"] = index
                per_query.append(result)
                rel = result["retrieval_level"]
                print(
                    f"[{path.name} #{index}] "
                    f"Precision={rel['precision']} Recall={rel['recall']} "
                    f"F1={rel['f1_score']} MRR={rel['mean_reciprocal_rank_mrr']}"
                )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: Skipping {path.name}: {exc}", file=sys.stderr)

    if not per_query:
        print("ERROR: No valid evaluation records were processed.", file=sys.stderr)
        return 1

    retrieval_keys = [
        "precision", "recall", "f1_score", "hit_rate",
        "mean_reciprocal_rank_mrr", "mean_average_precision_map",
        "ndcg_at_k", "precision_at_k", "recall_at_k",
    ]
    end_to_end_keys = [
        "answer_relevance", "context_utilization", "groundedness",
        "hallucination_rate", "response_coherence", "relevancy_score",
    ]

    retrieval_rows = [row["retrieval_level"] for row in per_query]
    e2e_rows = [row["end_to_end"] for row in per_query]

    stage_averages = {}
    for stage in ("dense", "bm25", "final"):
        stage_averages[stage] = average_dicts(
            [row["retrieval_stages"][stage] for row in per_query],
            retrieval_keys,
        )

    result_document = {
        "evaluation_info": {
            "folder": str(eval_dir),
            "evaluated_files": [p.name for p in eval_files],
            "total_queries": len(per_query),
            "k": args.k,
            "llm_used": False,
            "dependencies": "Python standard library only",
            "important_note": (
                "End-to-end scores are deterministic lexical/structural proxies, "
                "not semantic judgments. Retrieval matching uses chunk IDs when "
                "available and exact normalized chunk text otherwise."
            ),
        },
        "retrieval_level": average_dicts(retrieval_rows, retrieval_keys),
        "end_to_end": average_dicts(e2e_rows, end_to_end_keys),
        "retrieval_stage_averages": stage_averages,
        "per_query": per_query,
    }

    output_path = eval_dir / "eval_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_document, f, indent=2, ensure_ascii=False)

    print("\nEvaluation complete.")
    print(f"Queries evaluated: {len(per_query)}")
    print(f"Metrics saved to: {output_path}")
    print("\nAggregate retrieval metrics:")
    for key, value in result_document["retrieval_level"].items():
        print(f"  {key}: {value}")
    print("\nAggregate end-to-end metrics:")
    for key, value in result_document["end_to_end"].items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
