"""Benchmark the multilingual post-STT text pipeline.

Measured stages:
  embedding -> Qdrant -> rerank -> context compression -> answer generation -> total

STT and TTS are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import QDRANT_COLLECTION
from app.pipeline import RAGEngine


PASSAGES_FILE = Path("data/processed/multilingual/passages.jsonl")
REPORT_FILE = Path("data/processed/multilingual/post_stt_benchmark_report.json")
LATENCY_BUDGET_MS = 200

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
    return {
        "avg_ms": round(statistics.mean(values), 2) if values else 0.0,
        "p50_ms": round(percentile(values, 50), 2),
        "p95_ms": round(percentile(values, 95), 2),
        "p99_ms": round(percentile(values, 99), 2),
        "p100_ms": round(max(values), 2) if values else 0.0,
    }


def load_queries(max_queries: int | None) -> list[dict[str, Any]]:
    by_language = {language: [] for language in SUPPORTED_LANGUAGES}
    seen = set()

    with PASSAGES_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            item = json.loads(line)
            key = (item["language"], item["query_id"])
            if key in seen:
                continue

            seen.add(key)
            by_language[item["language"]].append(
                {
                    "language": item["language"],
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


def target_found(item: dict[str, Any], sources: list[dict[str, Any]], rank: int) -> bool:
    target = (item["language"], item["query_id"])
    for source in sources[:rank]:
        if (source.get("language"), source.get("query_id")) == target:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark multilingual post-STT text pipeline.",
    )
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    max_queries = None if args.max_queries == 0 else args.max_queries
    queries = load_queries(max_queries)
    if not queries:
        raise SystemExit("No benchmark queries found.")

    print("=" * 70)
    print("HH GOA RAG - MULTILINGUAL POST-STT BENCHMARK")
    print("=" * 70)
    print(f"Collection : {QDRANT_COLLECTION}")
    print(f"Queries    : {len(queries)}")
    print("Loading engine once...")

    engine = RAGEngine()

    warmups = queries[: min(args.warmup, len(queries))]
    print(f"Warmup     : {len(warmups)}")
    for item in warmups:
        engine.process(
            item["query"],
            language=item["language"],
        )

    timings = {
        "embedding": [],
        "qdrant": [],
        "rerank": [],
        "compression": [],
        "llm": [],
        "answer": [],
        "total": [],
    }
    context_before = []
    context_after = []
    recall = {1: 0, 5: 0, 10: 0}
    query_counts = Counter()
    blocked = 0

    print("Running benchmark...")
    for index, item in enumerate(queries, start=1):
        start = time.perf_counter()
        result = engine.process(
            item["query"],
            language=item["language"],
        )
        outer_total = (time.perf_counter() - start) * 1000

        stage = result.get("timings", {})
        timings["embedding"].append(float(stage.get("embedding_ms", 0)))
        timings["qdrant"].append(float(stage.get("qdrant_ms", 0)))
        timings["rerank"].append(float(stage.get("rerank_ms", 0)))
        timings["compression"].append(float(stage.get("compression_ms", 0)))
        timings["llm"].append(float(stage.get("llm_ms", 0)))
        timings["answer"].append(float(stage.get("answer_ms", 0)))
        timings["total"].append(outer_total)

        compressed = result.get("compressed_context", {})
        context_before.append(int(compressed.get("chars_before", 0)))
        context_after.append(int(compressed.get("chars_after", 0)))

        sources = result.get("sources", [])
        for rank in (1, 5, 10):
            if target_found(item, sources, rank):
                recall[rank] += 1

        query_counts[item["language"]] += 1
        if result.get("blocked"):
            blocked += 1

        if index % 100 == 0:
            print(f"[{index}/{len(queries)}] total={outer_total:.2f}ms")

    stage_stats = {
        name: summarize(values)
        for name, values in timings.items()
    }
    recall_percent = {
        f"recall_at_{rank}": round((count / len(queries)) * 100, 2)
        for rank, count in recall.items()
    }
    report = {
        "collection": QDRANT_COLLECTION,
        "queries": len(queries),
        "query_counts_by_language": dict(query_counts),
        "stage_stats": stage_stats,
        "recall": recall_percent,
        "blocked_percent": round((blocked / len(queries)) * 100, 2),
        "context": {
            "before_avg_chars": round(statistics.mean(context_before), 2),
            "before_p95_chars": round(percentile(context_before, 95), 2),
            "after_avg_chars": round(statistics.mean(context_after), 2),
            "after_p95_chars": round(percentile(context_after, 95), 2),
        },
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "budget_pass": stage_stats["total"]["p95_ms"] < LATENCY_BUDGET_MS,
    }
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nStage       | P50    | P95    | P99    | P100")
    print("-" * 54)
    for name in ("embedding", "qdrant", "rerank", "compression", "llm", "total"):
        stats = stage_stats[name]
        print(
            f"{name:<11} | "
            f"{stats['p50_ms']:>6.2f} | "
            f"{stats['p95_ms']:>6.2f} | "
            f"{stats['p99_ms']:>6.2f} | "
            f"{stats['p100_ms']:>7.2f}"
        )

    print("\nRecall:")
    print(f"Recall@1  : {recall_percent['recall_at_1']:.2f}%")
    print(f"Recall@5  : {recall_percent['recall_at_5']:.2f}%")
    print(f"Recall@10 : {recall_percent['recall_at_10']:.2f}%")
    print("\nContext:")
    print(f"Before avg/p95 chars: {report['context']['before_avg_chars']} / {report['context']['before_p95_chars']}")
    print(f"After  avg/p95 chars: {report['context']['after_avg_chars']} / {report['context']['after_p95_chars']}")
    print(f"\nP95 total < {LATENCY_BUDGET_MS} ms: {report['budget_pass']}")
    print(f"Report: {REPORT_FILE}")
    engine.close()


if __name__ == "__main__":
    main()
