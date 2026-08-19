"""Manual live benchmark for the multilingual Qwen API RAG path.

This script intentionally makes no outbound API calls unless --allow-live-api is
passed. It loads the RAG engine once, warms up outside measured latency, then
records real per-request stage timings for:

  embedding -> Qdrant -> rerank -> compression -> Qwen API -> total
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

from app.config import (  # noqa: E402
    ANSWER_BACKEND,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    MAX_NEW_TOKENS,
    QDRANT_COLLECTION,
    QWEN_MODEL,
)
from app.pipeline import RAGEngine  # noqa: E402


PASSAGES_FILE = Path("data/processed/multilingual/passages.jsonl")
REPORT_FILE = Path("data/processed/multilingual/qwen_api_benchmark_report.json")
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
            language = item.get("language")
            if language not in by_language:
                continue
            key = (language, item.get("query_id"))
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
            if index >= len(by_language[language]):
                continue
            queries.append(by_language[language][index])
            if max_queries is not None and len(queries) >= max_queries:
                return queries
    return queries


def target_found(item: dict[str, Any], sources: list[dict[str, Any]], rank: int) -> bool:
    target = (item["language"], item["query_id"])
    return any(
        (source.get("language"), source.get("query_id")) == target
        for source in sources[:rank]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the live multilingual Qwen API retrieval benchmark.",
    )
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument("--max-queries", type=int, default=13)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--report-file", type=Path, default=REPORT_FILE)
    args = parser.parse_args()

    if not args.allow_live_api:
        raise SystemExit(
            "Refusing to call the live LLM API. Re-run with --allow-live-api "
            "after setting LLM_API_KEY/HF_API_KEY in .env."
        )

    if ANSWER_BACKEND not in {"qwen", "qwen_api"}:
        raise SystemExit("Set ANSWER_BACKEND=qwen_api before running the live LLM benchmark.")

    max_queries = None if args.max_queries == 0 else args.max_queries
    queries = load_queries(max_queries)
    if not queries:
        raise SystemExit("No multilingual benchmark queries found.")

    print("=" * 70)
    print("HH GOA RAG - LIVE QWEN API MULTILINGUAL BENCHMARK")
    print("=" * 70)
    print(f"Collection : {QDRANT_COLLECTION}")
    print(f"Provider   : {LLM_PROVIDER}")
    print(f"Model      : {QWEN_MODEL}")
    print(f"Tokens     : {MAX_NEW_TOKENS}")
    print(f"Timeout    : {LLM_TIMEOUT_SECONDS}s")
    print(f"Queries    : {len(queries)}")
    print(f"Warmup     : {min(args.warmup, len(queries))}")
    print("Loading engine once...")

    engine = RAGEngine()
    if not engine.answer_generator.available:
        engine.close()
        raise SystemExit("Qwen API client is unavailable. Check LLM_API_KEY/HF_API_KEY in .env.")

    try:
        for item in queries[: min(args.warmup, len(queries))]:
            warm = engine.process(item["query"], language=item["language"])
            if warm.get("reason") in {"qwen_api_key_missing", "qwen_api_timeout", "qwen_api_error"}:
                raise SystemExit(f"Warmup failed: {warm.get('reason')}")

        timings = {
            "embedding": [],
            "qdrant": [],
            "rerank": [],
            "compression": [],
            "llm": [],
            "total": [],
        }
        recall = {1: 0, 5: 0, 10: 0}
        query_counts = Counter()
        blocked_reasons = Counter()

        print("Running live benchmark...")
        for index, item in enumerate(queries, start=1):
            start = time.perf_counter()
            result = engine.process(item["query"], language=item["language"])
            outer_total = (time.perf_counter() - start) * 1000

            stage = result.get("timings", {})
            timings["embedding"].append(float(stage.get("embedding_ms", 0.0)))
            timings["qdrant"].append(float(stage.get("qdrant_ms", 0.0)))
            timings["rerank"].append(float(stage.get("rerank_ms", 0.0)))
            timings["compression"].append(float(stage.get("compression_ms", 0.0)))
            timings["llm"].append(float(stage.get("llm_ms", 0.0)))
            timings["total"].append(outer_total)

            top20 = result.get("retrieval", {}).get("top20", [])
            for rank in (1, 5, 10):
                if target_found(item, top20, rank):
                    recall[rank] += 1

            query_counts[item["language"]] += 1
            if result.get("blocked"):
                blocked_reasons[result.get("reason", "blocked")] += 1

            if index % 25 == 0 or index == len(queries):
                print(
                    f"[{index}/{len(queries)}] "
                    f"llm={stage.get('llm_ms', 0.0):.2f}ms "
                    f"total={outer_total:.2f}ms "
                    f"reason={result.get('reason')}"
                )

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
            "answer_backend": ANSWER_BACKEND,
            "llm_provider": LLM_PROVIDER,
            "llm_model": QWEN_MODEL,
            "max_new_tokens": MAX_NEW_TOKENS,
            "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
            "queries": len(queries),
            "warmup": min(args.warmup, len(queries)),
            "query_counts_by_language": dict(query_counts),
            "stage_stats": stage_stats,
            "recall": recall_percent,
            "blocked_reasons": dict(blocked_reasons),
            "latency_budget_ms": LATENCY_BUDGET_MS,
            "budget_pass": stage_stats["total"]["p95_ms"] < LATENCY_BUDGET_MS,
            "live_llm_measured": True,
        }
        args.report_file.write_text(
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
        print(f"\nP95 total < {LATENCY_BUDGET_MS} ms: {report['budget_pass']}")
        print(f"Report: {args.report_file}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
