"""Run multilingual retrieval latency/recall experiments.

Experiments:
  A. SentenceTransformer CPU thread count sweep
  B. Qdrant retrieval depth sweep with recall
  C. Production reranker outlier investigation
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.pipeline import (
    PHRASE_WEIGHT,
    TOP_K_FINAL,
    VECTOR_WEIGHT,
    LEXICAL_WEIGHT,
    lexical_overlap,
    phrase_score,
    rerank,
)


COLLECTION_NAME = "hh_goa_rag_multilingual"
QDRANT_PATH = "data/qdrant"
PASSAGES_FILE = Path("data/processed/multilingual/passages.jsonl")
REPORT_FILE = Path("data/processed/multilingual/multilingual_experiments_report.json")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384
EXPECTED_VECTORS = 6046

SUPPORTED_LANGUAGES = (
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "ta",
    "ur",
)


@dataclass
class StageStats:
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p100_ms: float


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = (len(ordered) - 1) * (pct / 100)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]

    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    stats = StageStats(
        avg_ms=round(statistics.mean(values), 2) if values else 0.0,
        p50_ms=round(percentile(values, 50), 2),
        p95_ms=round(percentile(values, 95), 2),
        p100_ms=round(max(values), 2) if values else 0.0,
    )
    return asdict(stats)


def load_indexed_queries(max_queries: int | None) -> list[dict[str, Any]]:
    by_language: dict[str, list[dict[str, Any]]] = {
        language: []
        for language in SUPPORTED_LANGUAGES
    }
    seen: set[tuple[str, Any]] = set()

    with PASSAGES_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            item = json.loads(line)
            language = item["language"]
            key = (language, item["query_id"])
            if key in seen:
                continue

            seen.add(key)
            by_language[language].append(
                {
                    "language": language,
                    "query_id": item["query_id"],
                    "query": item["query"],
                }
            )

    queries = []
    max_language_size = max(len(items) for items in by_language.values())
    for index in range(max_language_size):
        for language in SUPPORTED_LANGUAGES:
            items = by_language[language]
            if index >= len(items):
                continue

            queries.append(items[index])
            if max_queries is not None and len(queries) >= max_queries:
                return queries

    return queries


def validate_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        raise ValueError(f"missing collection: {COLLECTION_NAME}")

    info = client.get_collection(COLLECTION_NAME)
    vectors = info.config.params.vectors
    if info.points_count != EXPECTED_VECTORS:
        raise ValueError(f"expected {EXPECTED_VECTORS} points, found {info.points_count}")
    if getattr(vectors, "size", None) != VECTOR_SIZE:
        raise ValueError(f"expected {VECTOR_SIZE} dimensions, found {vectors}")
    if getattr(vectors, "distance", None) != models.Distance.COSINE:
        raise ValueError(f"expected COSINE distance, found {vectors}")


def set_threads(thread_count: int | None) -> int | None:
    try:
        import torch
    except ImportError:
        return None

    if thread_count is not None:
        torch.set_num_threads(thread_count)

    return int(torch.get_num_threads())


def load_model() -> tuple[SentenceTransformer, float]:
    start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    return model, (time.perf_counter() - start) * 1000


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    vector = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"query vector dimension mismatch: {len(vector)}")
    return vector


def qdrant_search(
    client: QdrantClient,
    vector: np.ndarray,
    depth: int,
) -> list[Any]:
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=depth,
        with_payload=[
            "chunk_id",
            "passage_id",
            "query_id",
            "text",
            "language",
            "is_selected",
            "chunk_strategy",
            "word_count",
        ],
        with_vectors=False,
    ).points


def target_found(item: dict[str, Any], hits: list[Any], rank: int) -> bool:
    target = (item["language"], item["query_id"])
    for hit in hits[:rank]:
        payload = hit.payload or {}
        if (payload.get("language"), payload.get("query_id")) == target:
            return True
    return False


def recall_summary(queries: list[dict[str, Any]], retrieved: list[list[Any]], depth: int) -> dict[str, float]:
    total = len(queries)
    summary = {}
    for rank in (1, 5, 10):
        hits = sum(
            1
            for item, result_hits in zip(queries, retrieved)
            if target_found(item, result_hits, rank)
        )
        summary[f"recall_at_{rank}"] = round((hits / total) * 100, 2) if total else 0.0
    summary["note"] = (
        f"Recall@10 at Top-{depth} depth is recall within the {depth} retrieved candidates."
        if depth < 10
        else ""
    )
    return summary


def run_experiment_a(queries: list[dict[str, Any]], warmup: int) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT A - EMBEDDING CPU THREADS")
    print("=" * 70)

    configs: list[tuple[str, int | None]] = [
        ("current", None),
        ("threads_1", 1),
        ("threads_2", 2),
        ("threads_4", 4),
        ("threads_6", 6),
        ("threads_8", 8),
    ]
    results = []

    for label, thread_count in configs:
        actual_threads = set_threads(thread_count)
        model, model_load_ms = load_model()

        for item in queries[:warmup]:
            encode_query(model, item["query"])

        timings = []
        for item in queries:
            start = time.perf_counter()
            encode_query(model, item["query"])
            timings.append((time.perf_counter() - start) * 1000)

        result = {
            "label": label,
            "requested_threads": thread_count,
            "actual_torch_threads": actual_threads,
            "model_load_ms": round(model_load_ms, 2),
            "embedding": summarize(timings),
        }
        results.append(result)

        stats = result["embedding"]
        print(
            f"{label:<10} threads={actual_threads} "
            f"P50={stats['p50_ms']:.2f}ms "
            f"P95={stats['p95_ms']:.2f}ms "
            f"P100={stats['p100_ms']:.2f}ms"
        )

        del model
        gc.collect()

    best = min(results, key=lambda item: item["embedding"]["p95_ms"])
    return {
        "queries": len(queries),
        "warmup": warmup,
        "results": results,
        "best_by_p95": best,
    }


def run_experiment_b(
    queries: list[dict[str, Any]],
    warmup: int,
    thread_count: int | None,
) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT B - RETRIEVAL DEPTH")
    print("=" * 70)

    actual_threads = set_threads(thread_count)
    model, model_load_ms = load_model()
    client = QdrantClient(path=QDRANT_PATH)

    try:
        validate_collection(client)

        for item in queries[:warmup]:
            vector = encode_query(model, item["query"])
            hits = qdrant_search(client, vector, 20)
            rerank(item["query"], hits)

        results = []
        for depth in (20, 10, 8, 5):
            embedding_ms = []
            qdrant_ms = []
            rerank_ms = []
            total_ms = []
            retrieved_by_query = []

            for item in queries:
                total_start = time.perf_counter()

                start = time.perf_counter()
                vector = encode_query(model, item["query"])
                embedding_ms.append((time.perf_counter() - start) * 1000)

                start = time.perf_counter()
                hits = qdrant_search(client, vector, depth)
                qdrant_ms.append((time.perf_counter() - start) * 1000)

                start = time.perf_counter()
                rerank(item["query"], hits)
                rerank_ms.append((time.perf_counter() - start) * 1000)

                total_ms.append((time.perf_counter() - total_start) * 1000)
                retrieved_by_query.append(hits)

            result = {
                "top_k": depth,
                "embedding": summarize(embedding_ms),
                "qdrant": summarize(qdrant_ms),
                "rerank": summarize(rerank_ms),
                "total": summarize(total_ms),
                "recall": recall_summary(queries, retrieved_by_query, depth),
            }
            results.append(result)

            total = result["total"]
            recall = result["recall"]
            print(
                f"Top-{depth:<2} total P50={total['p50_ms']:.2f}ms "
                f"P95={total['p95_ms']:.2f}ms P100={total['p100_ms']:.2f}ms | "
                f"R@1={recall['recall_at_1']:.2f}% "
                f"R@5={recall['recall_at_5']:.2f}% "
                f"R@10={recall['recall_at_10']:.2f}%"
            )

        return {
            "queries": len(queries),
            "warmup": warmup,
            "actual_torch_threads": actual_threads,
            "model_load_ms": round(model_load_ms, 2),
            "results": results,
        }
    finally:
        client.close()
        del model
        gc.collect()


def legacy_rerank(query: str, hits: list[Any]) -> list[Any]:
    if not hits:
        return []

    scores = []
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text", "")
        vector_score = float(hit.score)
        lexical_score = lexical_overlap(query, text)
        exact_phrase_score = phrase_score(query, text)
        final_score = (
            VECTOR_WEIGHT * vector_score
            + LEXICAL_WEIGHT * lexical_score
            + PHRASE_WEIGHT * exact_phrase_score
        )
        scores.append((final_score, vector_score, lexical_score, exact_phrase_score, hit))

    scores.sort(key=lambda item: item[0], reverse=True)
    return scores[:TOP_K_FINAL]


def describe_case(item: dict[str, Any], hits: list[Any], elapsed_ms: float) -> dict[str, Any]:
    texts = [str((hit.payload or {}).get("text", "")) for hit in hits]
    lengths = [len(text) for text in texts]
    words = [len(text.split()) for text in texts]
    languages = Counter((hit.payload or {}).get("language") for hit in hits)
    return {
        "elapsed_ms": round(elapsed_ms, 4),
        "query_language": item["language"],
        "query_id": item["query_id"],
        "query_chars": len(item["query"]),
        "query_words": len(item["query"].split()),
        "hits": len(hits),
        "candidate_languages": dict(languages),
        "total_candidate_chars": sum(lengths),
        "total_candidate_words": sum(words),
        "max_candidate_chars": max(lengths) if lengths else 0,
        "max_candidate_words": max(words) if words else 0,
        "chunk_ids": [
            (hit.payload or {}).get("chunk_id")
            for hit in hits[:5]
        ],
    }


def time_reranker(
    reranker: Callable[[str, list[Any]], list[Any]],
    item: dict[str, Any],
    hits: list[Any],
) -> float:
    start = time.perf_counter()
    reranker(item["query"], hits)
    return (time.perf_counter() - start) * 1000


def run_experiment_c(
    queries: list[dict[str, Any]],
    warmup: int,
    thread_count: int | None,
) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("EXPERIMENT C - RERANKER OUTLIER")
    print("=" * 70)

    set_threads(thread_count)
    model, _ = load_model()
    client = QdrantClient(path=QDRANT_PATH)

    try:
        validate_collection(client)

        query_hits = []
        for item in queries[:warmup]:
            vector = encode_query(model, item["query"])
            hits = qdrant_search(client, vector, 20)
            rerank(item["query"], hits)

        for item in queries:
            vector = encode_query(model, item["query"])
            hits = qdrant_search(client, vector, 20)
            query_hits.append((item, hits))

        legacy_timings = []
        optimized_timings = []
        optimized_cases = []
        legacy_cases = []

        for item, hits in query_hits:
            legacy_elapsed = time_reranker(legacy_rerank, item, hits)
            optimized_elapsed = time_reranker(rerank, item, hits)

            legacy_timings.append(legacy_elapsed)
            optimized_timings.append(optimized_elapsed)
            legacy_cases.append(describe_case(item, hits, legacy_elapsed))
            optimized_cases.append(describe_case(item, hits, optimized_elapsed))

        worst_optimized_index = max(
            range(len(optimized_timings)),
            key=lambda index: optimized_timings[index],
        )
        worst_item, worst_hits = query_hits[worst_optimized_index]
        repeat_timings = [
            time_reranker(rerank, worst_item, worst_hits)
            for _ in range(50)
        ]

        legacy_stats = summarize(legacy_timings)
        optimized_stats = summarize(optimized_timings)
        repeat_stats = summarize(repeat_timings)

        print(
            f"Legacy    P50={legacy_stats['p50_ms']:.2f}ms "
            f"P95={legacy_stats['p95_ms']:.2f}ms "
            f"P100={legacy_stats['p100_ms']:.2f}ms"
        )
        print(
            f"Optimized P50={optimized_stats['p50_ms']:.2f}ms "
            f"P95={optimized_stats['p95_ms']:.2f}ms "
            f"P100={optimized_stats['p100_ms']:.2f}ms"
        )
        print(
            f"Worst-case repeat x50 P50={repeat_stats['p50_ms']:.2f}ms "
            f"P95={repeat_stats['p95_ms']:.2f}ms "
            f"P100={repeat_stats['p100_ms']:.2f}ms"
        )

        return {
            "queries": len(queries),
            "warmup": warmup,
            "legacy_rerank": {
                "stats": legacy_stats,
                "top_outliers": sorted(
                    legacy_cases,
                    key=lambda case: case["elapsed_ms"],
                    reverse=True,
                )[:10],
            },
            "optimized_production_rerank": {
                "stats": optimized_stats,
                "top_outliers": sorted(
                    optimized_cases,
                    key=lambda case: case["elapsed_ms"],
                    reverse=True,
                )[:10],
            },
            "worst_optimized_repeat_x50": {
                "case": describe_case(
                    worst_item,
                    worst_hits,
                    max(repeat_timings),
                ),
                "stats": repeat_stats,
            },
            "optimization": (
                "Production rerank now precomputes query normalization/tokens/phrases "
                "once and normalizes each candidate document once."
            ),
        }
    finally:
        client.close()
        del model
        gc.collect()


def print_final_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("MULTILINGUAL EXPERIMENTS COMPLETE")
    print("=" * 70)

    print("\nExperiment A - Embedding threads")
    for result in report["experiment_a"]["results"]:
        stats = result["embedding"]
        print(
            f"{result['label']:<10} "
            f"threads={result['actual_torch_threads']} "
            f"P50={stats['p50_ms']:.2f} "
            f"P95={stats['p95_ms']:.2f} "
            f"P100={stats['p100_ms']:.2f}"
        )

    print("\nExperiment B - Retrieval depth")
    print("Depth | Total P50 | Total P95 | Total P100 | R@1 | R@5 | R@10")
    print("-" * 68)
    for result in report["experiment_b"]["results"]:
        total = result["total"]
        recall = result["recall"]
        print(
            f"{result['top_k']:>5} | "
            f"{total['p50_ms']:>9.2f} | "
            f"{total['p95_ms']:>9.2f} | "
            f"{total['p100_ms']:>10.2f} | "
            f"{recall['recall_at_1']:>4.2f} | "
            f"{recall['recall_at_5']:>4.2f} | "
            f"{recall['recall_at_10']:>5.2f}"
        )

    print("\nExperiment C - Reranker")
    legacy = report["experiment_c"]["legacy_rerank"]["stats"]
    optimized = report["experiment_c"]["optimized_production_rerank"]["stats"]
    repeat = report["experiment_c"]["worst_optimized_repeat_x50"]["stats"]
    print(
        f"Legacy    P50={legacy['p50_ms']:.2f} "
        f"P95={legacy['p95_ms']:.2f} "
        f"P100={legacy['p100_ms']:.2f}"
    )
    print(
        f"Optimized P50={optimized['p50_ms']:.2f} "
        f"P95={optimized['p95_ms']:.2f} "
        f"P100={optimized['p100_ms']:.2f}"
    )
    print(
        f"Worst repeat x50 P50={repeat['p50_ms']:.2f} "
        f"P95={repeat['p95_ms']:.2f} "
        f"P100={repeat['p100_ms']:.2f}"
    )
    print(f"\nReport: {REPORT_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multilingual RAG latency/recall experiments.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=500,
        help="Balanced indexed query count. Use 0 for all indexed queries.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup queries before each measured experiment.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Thread count for experiments B/C. Omit to use current torch default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_queries = None if args.max_queries == 0 else args.max_queries
    queries = load_indexed_queries(max_queries)
    query_counts = Counter(item["language"] for item in queries)

    client = QdrantClient(path=QDRANT_PATH)
    try:
        validate_collection(client)
    finally:
        client.close()

    report = {
        "collection": COLLECTION_NAME,
        "model": MODEL_NAME,
        "queries": len(queries),
        "query_counts_by_language": dict(query_counts),
        "experiment_a": run_experiment_a(queries, args.warmup),
        "experiment_b": run_experiment_b(queries, args.warmup, args.threads),
        "experiment_c": run_experiment_c(queries, args.warmup, args.threads),
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_final_report(report)


if __name__ == "__main__":
    main()
