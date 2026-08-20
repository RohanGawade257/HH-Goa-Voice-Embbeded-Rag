"""Auditable live benchmark for the multilingual Qwen API RAG path.

Phase 1 model: Qwen/Qwen3-0.6B (set via QWEN_MODEL env var or config default).

The benchmark measures the current production pipeline:

  embedding -> Qdrant -> rerank -> compression -> remote Qwen API -> answer

REQUEST CLASSIFICATION
----------------------
Every request receives exactly one status:
  SUCCESS    -- valid HTTP response, non-empty answer, timing valid
  HTTP_ERROR -- non-2xx HTTP response (401, 429, 500, etc.)
  TIMEOUT    -- httpx.TimeoutException
  EXCEPTION  -- any other exception

Failed LLM requests are recorded in the raw report but excluded from latency
percentiles and PASS/FAIL decisions.  Warmup requests are never included in
measured statistics.  The benchmark continues after individual request failures
and never crashes the entire run.

Token-limit experiment (--token-experiment): runs the same pipeline for each
of 16/32/48/64 max-tokens and records per-config reports plus a comparison
table.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
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
from app.generation.llm import EXCEPTION, HTTP_ERROR, SUCCESS, TIMEOUT  # noqa: E402
from app.pipeline import RAGEngine  # noqa: E402


PASSAGES_FILE = Path("data/processed/multilingual/passages.jsonl")
REPORT_FILE = Path("data/processed/multilingual/qwen_api_benchmark_report.json")
REPORT_DIR = Path("data/processed/multilingual")
LATENCY_BUDGET_MS = 200
DEFAULT_WARMUP = 10
DEFAULT_REQUESTS = 50
DEFAULT_QUERY_POOL_SIZE = 13
DEFAULT_MIN_SUCCESSFUL_REQUESTS = 20
TOKEN_EXPERIMENT_CONFIGS = (16, 32, 48, 64)
STATUSES = (SUCCESS, HTTP_ERROR, TIMEOUT, EXCEPTION)

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
        "samples": len(values),
    }


def load_query_pool(limit: int = DEFAULT_QUERY_POOL_SIZE) -> list[dict[str, Any]]:
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
            if len(queries) >= limit:
                return queries
    return queries


def repeated_query(pool: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if not pool:
        raise ValueError("query pool is empty")
    return pool[index % len(pool)]


def target_found(item: dict[str, Any], sources: list[dict[str, Any]], rank: int) -> bool:
    target = (item["language"], item["query_id"])
    return any(
        (source.get("language"), source.get("query_id")) == target
        for source in sources[:rank]
    )


def stage_total(stage: dict[str, Any]) -> float:
    return round(
        float(stage.get("embedding_ms", 0.0))
        + float(stage.get("qdrant_ms", 0.0))
        + float(stage.get("rerank_ms", 0.0))
        + float(stage.get("compression_ms", 0.0))
        + float(stage.get("llm_ms", 0.0)),
        2,
    )


def classify_result(result: dict[str, Any]) -> dict[str, Any]:
    generation = result.get("answer_generation", {}) or {}
    status = generation.get("status")
    if status not in STATUSES:
        reason = result.get("reason", "")
        if reason == "qwen_api_timeout":
            status = TIMEOUT
        elif reason == "qwen_api_http_error":
            status = HTTP_ERROR
        elif not result.get("blocked") and result.get("answer"):
            status = SUCCESS
        else:
            status = EXCEPTION

    return {
        "status": status,
        "reason": result.get("reason", generation.get("reason", "")),
        "http_status": generation.get("http_status"),
        "timeout_seconds": generation.get("timeout_seconds"),
        "exception_type": generation.get("exception_type"),
        "error": generation.get("error"),
        "answer_chars": len(str(result.get("answer", ""))),
        "prompt_chars": generation.get("prompt_chars"),
        "context_chars": generation.get("context_chars"),
        "grounded": bool(result.get("grounded", False)),
        "blocked": bool(result.get("blocked", False)),
        "provider": generation.get("provider"),
        "model": generation.get("model"),
        "retry_count": 0,
        "succeeded_after_retry": False,
    }


def request_record(
    request_number: int,
    item: dict[str, Any],
    result: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    stage = result.get("timings", {}) or {}
    classification = classify_result(result)
    top20 = result.get("retrieval", {}).get("top20", [])
    record = {
        "phase": phase,
        "request_number": request_number,
        "query_id": item.get("query_id"),
        "language": item.get("language"),
        "status": classification["status"],
        "reason": classification["reason"],
        "timing_ms": {
            "embedding": float(stage.get("embedding_ms", 0.0)),
            "qdrant": float(stage.get("qdrant_ms", 0.0)),
            "rerank": float(stage.get("rerank_ms", 0.0)),
            "compression": float(stage.get("compression_ms", 0.0)),
            "llm": float(stage.get("llm_ms", 0.0)),
            "total": stage_total(stage),
        },
        "http_status": classification["http_status"],
        "timeout_seconds": classification["timeout_seconds"],
        "exception_type": classification["exception_type"],
        "error": classification["error"],
        "answer_chars": classification["answer_chars"],
        "prompt_chars": classification["prompt_chars"],
        "context_chars": classification["context_chars"],
        "grounded": classification["grounded"],
        "blocked": classification["blocked"],
        "provider": classification["provider"],
        "model": classification["model"],
        "retry_count": classification["retry_count"],
        "succeeded_after_retry": classification["succeeded_after_retry"],
        "target_found_at_1": target_found(item, top20, 1),
        "target_found_at_5": target_found(item, top20, 5),
        "target_found_at_10": target_found(item, top20, 10),
    }
    return record


def exception_record(
    request_number: int,
    item: dict[str, Any],
    exc: Exception,
    phase: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "request_number": request_number,
        "query_id": item.get("query_id"),
        "language": item.get("language"),
        "status": EXCEPTION,
        "reason": "benchmark_exception",
        "timing_ms": {
            "embedding": 0.0,
            "qdrant": 0.0,
            "rerank": 0.0,
            "compression": 0.0,
            "llm": 0.0,
            "total": 0.0,
        },
        "http_status": None,
        "timeout_seconds": None,
        "exception_type": type(exc).__name__,
        "error": str(exc)[:500],
        "answer_chars": 0,
        "prompt_chars": None,
        "context_chars": None,
        "grounded": False,
        "blocked": True,
        "provider": None,
        "model": None,
        "retry_count": 0,
        "succeeded_after_retry": False,
        "target_found_at_1": False,
        "target_found_at_5": False,
        "target_found_at_10": False,
    }


def run_requests(
    engine: Any,
    query_pool: list[dict[str, Any]],
    count: int,
    phase: str,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    records = []
    for index in range(count):
        item = repeated_query(query_pool, index)
        request_number = index + 1
        try:
            result = engine.process(
                item["query"],
                language=item["language"],
                max_tokens=max_tokens,
            )
            records.append(request_record(request_number, item, result, phase))
        except Exception as exc:
            records.append(exception_record(request_number, item, exc, phase))
    return records


def build_report(
    measured_records: list[dict[str, Any]],
    warmup_records: list[dict[str, Any]],
    query_pool: list[dict[str, Any]],
    requested_requests: int,
    min_successful_requests: int,
) -> dict[str, Any]:
    status_counts = Counter(record["status"] for record in measured_records)
    warmup_status_counts = Counter(record["status"] for record in warmup_records)
    successful_records = [
        record
        for record in measured_records
        if record["status"] == SUCCESS
    ]

    stage_latency = {
        "embedding": [],
        "qdrant": [],
        "rerank": [],
        "compression": [],
        "llm": [],
        "total": [],
    }
    for record in successful_records:
        timing = record["timing_ms"]
        for stage in stage_latency:
            stage_latency[stage].append(float(timing.get(stage, 0.0)))

    stage_stats = {
        stage: summarize(values)
        for stage, values in stage_latency.items()
    }
    failure_examples = [
        {
            "request_number": record["request_number"],
            "status": record["status"],
            "reason": record["reason"],
            "http_status": record["http_status"],
            "exception_type": record["exception_type"],
            "error": record["error"],
        }
        for record in measured_records
        if record["status"] != SUCCESS
    ][:5]
    total_requests = len(measured_records)
    successful_requests = len(successful_records)
    failure_count = total_requests - successful_requests
    total_p95 = stage_stats["total"]["p95_ms"]
    if successful_requests < min_successful_requests:
        target_status = "INCONCLUSIVE"
    elif total_p95 <= LATENCY_BUDGET_MS:
        target_status = "PASS"
    else:
        target_status = "FAIL"

    recall_denominator = total_requests or 1
    recall = {
        "recall_at_1": round(
            sum(1 for record in measured_records if record["target_found_at_1"])
            / recall_denominator
            * 100,
            2,
        ),
        "recall_at_5": round(
            sum(1 for record in measured_records if record["target_found_at_5"])
            / recall_denominator
            * 100,
            2,
        ),
        "recall_at_10": round(
            sum(1 for record in measured_records if record["target_found_at_10"])
            / recall_denominator
            * 100,
            2,
        ),
    }

    return {
        "collection": QDRANT_COLLECTION,
        "answer_backend": ANSWER_BACKEND,
        "provider": LLM_PROVIDER,
        "model": QWEN_MODEL,
        "behavior": "concise-instruct-no-thinking",
        "max_new_tokens": MAX_NEW_TOKENS,
        "timeout_seconds": LLM_TIMEOUT_SECONDS,
        "retry_policy": {"retries": 0},
        "warmup": len(warmup_records),
        "warmup_status_counts": dict(warmup_status_counts),
        "requested_requests": requested_requests,
        "total_requests": total_requests,
        "unique_queries": len(query_pool),
        "successful_requests": successful_requests,
        "http_errors": status_counts.get(HTTP_ERROR, 0),
        "timeouts": status_counts.get(TIMEOUT, 0),
        "exceptions": status_counts.get(EXCEPTION, 0),
        "success_rate": round((successful_requests / total_requests) * 100, 2)
        if total_requests
        else 0.0,
        "failure_rate": round((failure_count / total_requests) * 100, 2)
        if total_requests
        else 0.0,
        "latency_ms": {
            "llm": stage_stats["llm"],
            "total": stage_stats["total"],
        },
        "stage_latency_ms": stage_stats,
        "recall": recall,
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "min_successful_requests": min_successful_requests,
        "target_status": target_status,
        "benchmark_infrastructure_api_failure": successful_requests == 0,
        "api_configuration_blocker": (
            failure_examples[0]
            if successful_requests == 0 and failure_examples
            else None
        ),
        "failure_examples": failure_examples,
        "requests": measured_records,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_report(report: dict[str, Any], path: Path) -> None:
    print("=" * 60)
    print("HH GOA RAG - LIVE LLM BENCHMARK")
    print("=" * 60)
    print(f"Provider: {report['provider']}")
    print(f"Model: {report['model']}")
    print(f"Warmup: {report['warmup']}")
    print(f"Measured requests: {report['total_requests']}")
    print(f"Unique queries: {report['unique_queries']}")

    print("\n" + "-" * 60)
    print("REQUEST RESULTS")
    print("-" * 60)
    print(f"SUCCESS: {report['successful_requests']}")
    print(f"HTTP_ERROR: {report['http_errors']}")
    print(f"TIMEOUT: {report['timeouts']}")
    print(f"EXCEPTION: {report['exceptions']}")
    print(f"SUCCESS RATE: {report['success_rate']:.2f}%")

    llm = report["latency_ms"]["llm"]
    total = report["latency_ms"]["total"]
    print("\n" + "-" * 60)
    print("SUCCESSFUL LLM LATENCY")
    print("-" * 60)
    print(f"P50: {llm['p50_ms']:.2f} ms")
    print(f"P95: {llm['p95_ms']:.2f} ms")
    print(f"P99: {llm['p99_ms']:.2f} ms")
    print(f"P100: {llm['p100_ms']:.2f} ms")

    print("\n" + "-" * 60)
    print("SUCCESSFUL TOTAL POST-STT LATENCY")
    print("-" * 60)
    print(f"P50: {total['p50_ms']:.2f} ms")
    print(f"P95: {total['p95_ms']:.2f} ms")
    print(f"P99: {total['p99_ms']:.2f} ms")
    print(f"P100: {total['p100_ms']:.2f} ms")

    print("\n" + "-" * 60)
    print("STAGE LATENCY")
    print("-" * 60)
    for name in ("embedding", "qdrant", "rerank", "compression", "llm"):
        stats = report["stage_latency_ms"][name]
        print(
            f"{name.capitalize()}: "
            f"P50={stats['p50_ms']:.2f} "
            f"P95={stats['p95_ms']:.2f} "
            f"P99={stats['p99_ms']:.2f} "
            f"P100={stats['p100_ms']:.2f} ms"
        )

    print("\n" + "-" * 60)
    print("TARGET")
    print("-" * 60)
    print(f"POST-STT P95 TARGET: <= {report['latency_budget_ms']} ms")
    print(f"ACTUAL: {total['p95_ms']:.2f} ms")
    print(f"STATUS: {report['target_status']}")
    print(f"Report: {path}")
    print("=" * 60)


def _report_path_for_tokens(base_dir: Path, max_tokens: int) -> Path:
    return base_dir / f"qwen25_benchmark_tokens{max_tokens}.json"


def run_token_experiment(
    query_pool: list[dict[str, Any]],
    token_configs: tuple[int, ...],
    warmup: int,
    requests: int,
    min_successes: int,
    report_dir: Path,
) -> dict[int, dict[str, Any]]:
    """Run one full benchmark pass per token config.

    The RAG engine (embedding model + Qdrant) is loaded ONCE and reused across
    all token configs so retrieval latency is measured under identical warm
    conditions.  Only max_tokens changes between configs.
    """
    results: dict[int, dict[str, Any]] = {}

    print("Loading RAG engine once for token experiment...")
    engine = RAGEngine()
    try:
        # Shared warmup across all token configs — warms embedder and Qdrant.
        print(f"Running {warmup} shared warmup requests (max_tokens={token_configs[0]})...")
        run_requests(engine, query_pool, warmup, "warmup", max_tokens=token_configs[0])

        for max_tokens in token_configs:
            print(f"\n--- Token config: max_tokens={max_tokens} ---")
            print(f"  Running {requests} measured requests...")
            measured = run_requests(
                engine, query_pool, requests, "measured", max_tokens=max_tokens
            )
            report = build_report(
                measured_records=measured,
                warmup_records=[],  # warmup already done once above
                query_pool=query_pool,
                requested_requests=requests,
                min_successful_requests=min_successes,
            )
            # Override the max_new_tokens field to reflect the actual config used.
            report["max_new_tokens"] = max_tokens
            path = _report_path_for_tokens(report_dir, max_tokens)
            write_report(report, path)
            print_report(report, path)
            results[max_tokens] = report
    finally:
        engine.close()

    return results


def print_comparison_table(results: dict[int, dict[str, Any]]) -> None:
    """Print the Phase 2 comparison table across all token configs."""
    print("\n" + "=" * 100)
    print("PHASE 2 — TOKEN EXPERIMENT COMPARISON TABLE")
    print("=" * 100)
    header = (
        f"{'Model':<32} {'MaxTok':>7} {'SuccRate':>9} "
        f"{'LLM P50':>8} {'LLM P95':>8} {'LLM P99':>8} "
        f"{'Tot P50':>8} {'Tot P95':>8} {'Tot P99':>8} {'Target':>12}"
    )
    print(header)
    print("-" * 100)
    for max_tokens, report in sorted(results.items()):
        llm = report["latency_ms"]["llm"]
        total = report["latency_ms"]["total"]
        row = (
            f"{report['model']:<32} {max_tokens:>7} "
            f"{report['success_rate']:>8.1f}% "
            f"{llm['p50_ms']:>8.0f} {llm['p95_ms']:>8.0f} {llm['p99_ms']:>8.0f} "
            f"{total['p50_ms']:>8.0f} {total['p95_ms']:>8.0f} {total['p99_ms']:>8.0f} "
            f"{report['target_status']:>12}"
        )
        print(row)
    print("=" * 100)

    # Engineering conclusion
    successful_runs = {k: v for k, v in results.items() if v["successful_requests"] > 0}
    if successful_runs:
        best_tokens = min(successful_runs, key=lambda k: successful_runs[k]["latency_ms"]["total"]["p95_ms"])
        best = successful_runs[best_tokens]
        best_p95 = best["latency_ms"]["total"]["p95_ms"]
        best_llm_p95 = best["latency_ms"]["llm"]["p95_ms"]
        print(f"\nBest configuration : max_tokens={best_tokens}")
        print(f"Best total P95     : {best_p95:.0f} ms")
        print(f"Best LLM P95       : {best_llm_p95:.0f} ms")
        print(f"Official target    : <= {LATENCY_BUDGET_MS} ms")
        if best["target_status"] == "PASS":
            print("STATUS             : PASS — target met")
        elif best["target_status"] == "FAIL":
            print("STATUS             : FAIL — target not met even at smallest token budget")
        else:
            print("STATUS             : INCONCLUSIVE — insufficient successful requests")

        # Token-reduction analysis
        p95_values = {k: v["latency_ms"]["total"]["p95_ms"] for k, v in successful_runs.items()}
        if len(p95_values) >= 2:
            sorted_configs = sorted(p95_values.items())
            delta = sorted_configs[-1][1] - sorted_configs[0][1]
            print(f"\nTotal P95 delta (16 vs 64 tokens): {delta:.0f} ms")
            if delta > 100:
                print("Token reduction IS materially affecting latency (>100 ms delta).")
            elif delta > 30:
                print("Token reduction has MODERATE effect on latency (30–100 ms delta).")
            else:
                print("Token reduction has MINIMAL effect on latency (<30 ms delta).")
                print("Remote inference/network overhead is the dominant bottleneck.")
    else:
        print("\nNo successful requests — cannot determine best configuration.")
        print("STATUS: INCONCLUSIVE — API/infrastructure failure")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the live multilingual Qwen API retrieval benchmark.",
    )
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--query-pool-size", type=int, default=DEFAULT_QUERY_POOL_SIZE)
    parser.add_argument("--min-successes", type=int, default=DEFAULT_MIN_SUCCESSFUL_REQUESTS)
    parser.add_argument("--report-file", type=Path, default=REPORT_FILE)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Override max output tokens for a single run. "
            "If omitted and --token-experiment is not set, uses MAX_NEW_TOKENS from config."
        ),
    )
    parser.add_argument(
        "--token-experiment",
        action="store_true",
        help=(
            "Run the Phase 2 token-limit experiment: benchmark all of "
            f"{TOKEN_EXPERIMENT_CONFIGS} max-token configs and print a comparison table."
        ),
    )
    args = parser.parse_args()

    if not args.allow_live_api:
        raise SystemExit(
            "Refusing to call the live LLM API. Re-run with --allow-live-api "
            "after setting LLM_API_KEY/HF_API_KEY in .env."
        )
    if ANSWER_BACKEND not in {"qwen", "qwen_api"}:
        raise SystemExit("Set ANSWER_BACKEND=qwen_api before running the live LLM benchmark.")
    if args.requests < 1:
        raise SystemExit("--requests must be >= 1.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0.")

    query_pool = load_query_pool(max(1, args.query_pool_size))
    if not query_pool:
        raise SystemExit("No multilingual benchmark queries found.")

    if args.token_experiment:
        # Phase 2 multi-config experiment
        results = run_token_experiment(
            query_pool=query_pool,
            token_configs=TOKEN_EXPERIMENT_CONFIGS,
            warmup=args.warmup,
            requests=args.requests,
            min_successes=args.min_successes,
            report_dir=args.report_file.parent if args.report_file != REPORT_FILE else REPORT_DIR,
        )
        print_comparison_table(results)
        # Also write the canonical report file as the best-result config
        best_tokens = min(
            results,
            key=lambda k: results[k]["latency_ms"]["total"]["p95_ms"],
            default=next(iter(results), None),
        )
        if best_tokens is not None:
            write_report(results[best_tokens], args.report_file)
    else:
        # Single-run mode (Phase 1 compatible)
        print("Loading RAG engine once...")
        engine = RAGEngine()
        try:
            print(f"Running {args.warmup} warmup requests...")
            warmup_records = run_requests(
                engine, query_pool, args.warmup, "warmup", max_tokens=args.max_tokens
            )
            print(f"Running {args.requests} measured requests...")
            measured_records = run_requests(
                engine, query_pool, args.requests, "measured", max_tokens=args.max_tokens
            )
        finally:
            engine.close()

        report = build_report(
            measured_records=measured_records,
            warmup_records=warmup_records,
            query_pool=query_pool,
            requested_requests=args.requests,
            min_successful_requests=args.min_successes,
        )
        if args.max_tokens is not None:
            report["max_new_tokens"] = args.max_tokens
        write_report(report, args.report_file)
        print_report(report, args.report_file)


if __name__ == "__main__":
    main()
