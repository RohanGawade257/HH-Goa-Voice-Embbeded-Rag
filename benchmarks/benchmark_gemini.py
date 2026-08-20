"""Standalone Gemini LLM latency benchmark.

Measures TTFT (time-to-first-token) and total generation time for
gemini-2.5-flash-lite in non-thinking mode, independently of the
retrieval pipeline.  This isolates the LLM leg so you can assess whether the
model alone fits within the latency budget before running the full RAG bench.

Pipeline context
----------------
The full post-STT pipeline is:
    embedding + Qdrant + rerank + compression + LLM generation

This benchmark measures ONLY the LLM generation step with a fixed synthetic
prompt that approximates a realistic compressed-context answer request.

Usage
-----
    python benchmarks/benchmark_gemini.py --allow-live-api
    python benchmarks/benchmark_gemini.py --allow-live-api --warmup 5 --requests 30
    python benchmarks/benchmark_gemini.py --allow-live-api --max-output-tokens 60

Report
------
Written to: data/processed/multilingual/gemini_llm_benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import (  # noqa: E402
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_THINKING_BUDGET,
    GEMINI_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
)
from app.generation.gemini import GeminiAnswerGenerator  # noqa: E402
from app.generation.llm import EXCEPTION, HTTP_ERROR, SUCCESS, TIMEOUT  # noqa: E402

REPORT_FILE = Path("data/processed/multilingual/gemini_llm_benchmark_report.json")
LATENCY_BUDGET_MS = 200
DEFAULT_WARMUP = 5
DEFAULT_REQUESTS = 30
DEFAULT_MIN_SUCCESSFUL = 15

# Synthetic probe prompts — representative of what the RAG pipeline sends.
# Language coverage: Hindi, Bengali, Tamil, Marathi, Gujarati, Kannada,
#                    Malayalam, Punjabi, Odia, Urdu, Nepali, Sanskrit, Assamese
PROBE_QUERIES = [
    {
        "language": "hi",
        "query": "भारत की राजधानी कहाँ है?",
        "context": "नई दिल्ली भारत की राजधानी है। यह उत्तर भारत में स्थित है।",
    },
    {
        "language": "bn",
        "query": "বাংলাদেশের রাজধানী কোথায়?",
        "context": "ঢাকা বাংলাদেশের রাজধানী এবং বৃহত্তম শহর।",
    },
    {
        "language": "ta",
        "query": "தமிழ்நாட்டின் தலைநகரம் எது?",
        "context": "சென்னை தமிழ்நாட்டின் தலைநகரம் மற்றும் மிகப்பெரிய நகரம்.",
    },
    {
        "language": "mr",
        "query": "महाराष्ट्राची राजधानी कोणती आहे?",
        "context": "मुंबई महाराष्ट्राची राजधानी आहे आणि भारतातील सर्वात मोठे शहर आहे.",
    },
    {
        "language": "gu",
        "query": "ગુજરાતની રાજધાની ક્યાં છે?",
        "context": "ગાંધીનગર ગુજરાતની રાજધાની છે.",
    },
    {
        "language": "kn",
        "query": "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?",
        "context": "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ.",
    },
    {
        "language": "ml",
        "query": "കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?",
        "context": "തിരുവനന്തപുരം കേരളത്തിന്റെ തലസ്ഥാനമാണ്.",
    },
    {
        "language": "pa",
        "query": "ਪੰਜਾਬ ਦੀ ਰਾਜਧਾਨੀ ਕਿੱਥੇ ਹੈ?",
        "context": "ਚੰਡੀਗੜ੍ਹ ਪੰਜਾਬ ਦੀ ਰਾਜਧਾਨੀ ਹੈ।",
    },
    {
        "language": "or",
        "query": "ଓଡ଼ିଶାର ରାଜଧାନୀ କେଉଁଠି?",
        "context": "ଭୁବନେଶ୍ୱର ଓଡ଼ିଶାର ରାଜଧାନୀ।",
    },
    {
        "language": "ur",
        "query": "پاکستان کا دارالحکومت کہاں ہے؟",
        "context": "اسلام آباد پاکستان کا دارالحکومت ہے۔",
    },
    {
        "language": "ne",
        "query": "नेपालको राजधानी कहाँ छ?",
        "context": "काठमाडौं नेपालको राजधानी हो।",
    },
    {
        "language": "sa",
        "query": "भारतस्य राजधानी का अस्ति?",
        "context": "नवदिल्ली भारतस्य राजधानी अस्ति।",
    },
    {
        "language": "as",
        "query": "অসমৰ ৰাজধানী ক'ত?",
        "context": "দিছপুৰ অসমৰ ৰাজধানী।",
    },
]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (1 - (idx - lo)) + ordered[hi] * (idx - lo)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "p100_ms": 0.0, "samples": 0}
    return {
        "avg_ms": round(statistics.mean(values), 2),
        "p50_ms": round(percentile(values, 50), 2),
        "p95_ms": round(percentile(values, 95), 2),
        "p99_ms": round(percentile(values, 99), 2),
        "p100_ms": round(max(values), 2),
        "samples": len(values),
    }


# ---------------------------------------------------------------------------
# Single-request runner
# ---------------------------------------------------------------------------

def run_probe(
    generator: GeminiAnswerGenerator,
    probe: dict[str, str],
    request_number: int,
    phase: str,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    result = generator.generate(
        query=probe["query"],
        language=probe["language"],
        context=probe["context"],
        max_tokens=max_output_tokens,
    )
    return {
        "phase": phase,
        "request_number": request_number,
        "language": probe["language"],
        "status": result["status"],
        "reason": result.get("reason", ""),
        "ttft_ms": result.get("ttft_ms", 0.0),
        "llm_ms": result.get("llm_ms", 0.0),
        "answer_chars": len(result.get("answer", "")),
        "exception_type": result.get("exception_type"),
        "error": result.get("error"),
        "model": result.get("model", generator.model_name),
        "thinking_budget": result.get("thinking_budget", generator.thinking_budget),
        "max_output_tokens_used": result.get("max_output_tokens", generator.max_output_tokens),
    }


def run_requests(
    generator: GeminiAnswerGenerator,
    probes: list[dict[str, str]],
    count: int,
    phase: str,
    max_output_tokens: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i in range(count):
        probe = probes[i % len(probes)]
        try:
            rec = run_probe(generator, probe, i + 1, phase, max_output_tokens)
        except Exception as exc:
            rec = {
                "phase": phase,
                "request_number": i + 1,
                "language": probe.get("language", "?"),
                "status": EXCEPTION,
                "reason": "benchmark_exception",
                "ttft_ms": 0.0,
                "llm_ms": 0.0,
                "answer_chars": 0,
                "exception_type": type(exc).__name__,
                "error": str(exc)[:400],
                "model": generator.model_name,
                "thinking_budget": generator.thinking_budget,
                "max_output_tokens_used": generator.max_output_tokens,
            }
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    measured: list[dict[str, Any]],
    warmup: list[dict[str, Any]],
    max_output_tokens: int,
    min_successful: int,
) -> dict[str, Any]:
    from collections import Counter
    status_counts = Counter(r["status"] for r in measured)
    warmup_counts = Counter(r["status"] for r in warmup)

    ok = [r for r in measured if r["status"] == SUCCESS]
    ttft_vals = [r["ttft_ms"] for r in ok]
    llm_vals  = [r["llm_ms"]  for r in ok]

    n_total = len(measured)
    n_ok = len(ok)
    n_fail = n_total - n_ok

    total_p95 = summarize(llm_vals)["p95_ms"]
    if n_ok < min_successful:
        target_status = "INCONCLUSIVE"
    elif total_p95 <= LATENCY_BUDGET_MS:
        target_status = "PASS"
    else:
        target_status = "FAIL"

    failure_examples = [
        {
            "request_number": r["request_number"],
            "status": r["status"],
            "reason": r["reason"],
            "exception_type": r["exception_type"],
            "error": r["error"],
        }
        for r in measured
        if r["status"] != SUCCESS
    ][:5]

    return {
        "provider": "gemini",
        "model": GEMINI_MODEL,
        "thinking_budget": GEMINI_THINKING_BUDGET,
        "max_output_tokens": max_output_tokens,
        "temperature": LLM_TEMPERATURE,
        "timeout_seconds": GEMINI_TIMEOUT_SECONDS,
        "warmup": len(warmup),
        "warmup_status_counts": dict(warmup_counts),
        "total_requests": n_total,
        "successful_requests": n_ok,
        "http_errors": status_counts.get(HTTP_ERROR, 0),
        "timeouts": status_counts.get(TIMEOUT, 0),
        "exceptions": status_counts.get(EXCEPTION, 0),
        "success_rate": round(n_ok / n_total * 100, 2) if n_total else 0.0,
        "failure_rate": round(n_fail / n_total * 100, 2) if n_total else 0.0,
        "latency_ms": {
            "ttft": summarize(ttft_vals),
            "llm_total": summarize(llm_vals),
        },
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "min_successful_requests": min_successful,
        "target_status": target_status,
        "benchmark_scope": "llm_only",
        "benchmark_infrastructure_api_failure": n_ok == 0,
        "api_configuration_blocker": failure_examples[0] if n_ok == 0 and failure_examples else None,
        "failure_examples": failure_examples,
        "requests": measured,
    }


# ---------------------------------------------------------------------------
# Human-readable printer
# ---------------------------------------------------------------------------

def print_report(report: dict[str, Any], path: Path) -> None:
    t = report["latency_ms"]["ttft"]
    l = report["latency_ms"]["llm_total"]
    print("\n" + "=" * 60)
    print("HH GOA RAG — GEMINI LLM LATENCY BENCHMARK")
    print("=" * 60)
    print(f"Provider         : {report['provider']}")
    print(f"Model            : {report['model']}")
    print(f"Thinking budget  : {report['thinking_budget']}  (0 = non-thinking)")
    print(f"Max output tokens: {report['max_output_tokens']}")
    print(f"Temperature      : {report['temperature']}")
    print(f"Warmup           : {report['warmup']}")
    print(f"Measured requests: {report['total_requests']}")

    print("\n" + "-" * 60)
    print("REQUEST RESULTS")
    print("-" * 60)
    print(f"SUCCESS   : {report['successful_requests']}")
    print(f"HTTP_ERROR: {report['http_errors']}")
    print(f"TIMEOUT   : {report['timeouts']}")
    print(f"EXCEPTION : {report['exceptions']}")
    print(f"SUCCESS RATE: {report['success_rate']:.1f}%")

    print("\n" + "-" * 60)
    print("TTFT  (time-to-first-token, successful requests only)")
    print("-" * 60)
    print(f"P50 : {t['p50_ms']:.0f} ms")
    print(f"P95 : {t['p95_ms']:.0f} ms")
    print(f"P99 : {t['p99_ms']:.0f} ms")
    print(f"P100: {t['p100_ms']:.0f} ms")
    print(f"Avg : {t['avg_ms']:.0f} ms")

    print("\n" + "-" * 60)
    print("TOTAL LLM GENERATION TIME  (successful requests only)")
    print("-" * 60)
    print(f"P50 : {l['p50_ms']:.0f} ms")
    print(f"P95 : {l['p95_ms']:.0f} ms")
    print(f"P99 : {l['p99_ms']:.0f} ms")
    print(f"P100: {l['p100_ms']:.0f} ms")
    print(f"Avg : {l['avg_ms']:.0f} ms")

    print("\n" + "-" * 60)
    print("TARGET  (LLM generation P95 contribution to post-STT budget)")
    print("-" * 60)
    print(f"POST-STT P95 BUDGET: <= {report['latency_budget_ms']} ms  (full pipeline)")
    print(f"LLM TOTAL P95      : {l['p95_ms']:.0f} ms")
    print(f"STATUS             : {report['target_status']}")
    print(f"Report             : {path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Gemini LLM latency benchmark (TTFT + total generation).",
    )
    parser.add_argument("--allow-live-api", action="store_true",
                        help="Required to actually call the Gemini API.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Warmup requests excluded from statistics (default: {DEFAULT_WARMUP}).")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS,
                        help=f"Measured requests (default: {DEFAULT_REQUESTS}).")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Override GEMINI_MAX_OUTPUT_TOKENS for this run.")
    parser.add_argument("--min-successes", type=int, default=DEFAULT_MIN_SUCCESSFUL,
                        help=f"Min successes for PASS/FAIL decision (default: {DEFAULT_MIN_SUCCESSFUL}).")
    parser.add_argument("--report-file", type=Path, default=REPORT_FILE)
    args = parser.parse_args()

    if not args.allow_live_api:
        raise SystemExit(
            "Refusing to call the live Gemini API.\n"
            "Re-run with --allow-live-api after setting GEMINI_API_KEY in .env."
        )
    if not GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is not set.\n"
            "Get a key at https://aistudio.google.com/app/apikey and add it to .env."
        )
    if args.requests < 1:
        raise SystemExit("--requests must be >= 1.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0.")

    effective_tokens = args.max_output_tokens or GEMINI_MAX_OUTPUT_TOKENS
    print(f"Initialising Gemini generator (model={GEMINI_MODEL}, "
          f"thinking_budget={GEMINI_THINKING_BUDGET}, "
          f"max_output_tokens={effective_tokens})...")
    generator = GeminiAnswerGenerator()
    if not generator.available:
        raise SystemExit(f"Gemini generator unavailable: {generator.load_error}")

    print(f"Running {args.warmup} warmup requests...")
    warmup_records = run_requests(generator, PROBE_QUERIES, args.warmup, "warmup", effective_tokens)

    print(f"Running {args.requests} measured requests...")
    measured_records = run_requests(generator, PROBE_QUERIES, args.requests, "measured", effective_tokens)

    report = build_report(measured_records, warmup_records, effective_tokens, args.min_successes)

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report, args.report_file)


if __name__ == "__main__":
    main()
