"""
HH Goa 2026 — Voice RAG
End-to-End Production Pipeline Benchmark

Measures the COMPLETE pipeline in two clearly separated phases:

  PHASE 1 — Post-STT fast path  (the <200 ms requirement)
  ─────────────────────────────────────────────────────────
  This is what is measured against the 200 ms latency budget.
  STT is done externally (by Sarvam); this phase starts the moment
  a transcript arrives.

    embed → Qdrant → rerank → compress → extractive answer

  Stages timed individually:
    embedding_ms   — MiniLM-L12-v2 query encoding
    qdrant_ms      — vector search (local SQLite)
    rerank_ms      — lexical hybrid reranker
    compression_ms — context compressor
    answer_ms      — extractive sentence selection + guardrails
    direct_ms      — total of the above (= time_to_direct_answer)

  Budget gate: direct p95 < 200 ms

  PHASE 2 — LLM enhancement  (async, best-effort)
  ─────────────────────────────────────────────────
  Gemini (or Qwen) runs AFTER the direct answer is delivered.
  This phase is NOT on the critical path and NOT subject to the
  200 ms budget.  It is skipped gracefully if the LLM is unavailable.

    direct_ms + llm_ms = time_to_llm_answer

  Stages:
    llm_ms         — Gemini/Qwen generation wall-clock time
    ttft_ms        — Gemini time-to-first-token
    total_ms       — direct + llm

Usage
─────
  python benchmarks/benchmark.py                    # 200 queries, default pool
  python benchmarks/benchmark.py 500               # 500 queries
  python benchmarks/benchmark.py 200 --warmup 20   # custom warmup
  python benchmarks/benchmark.py 200 --llm         # force LLM even if backend=extractive
  python benchmarks/benchmark.py 200 --no-llm      # skip LLM entirely

Output
──────
  Terminal: two-section table (direct path + LLM path)
  JSON:     data/benchmark_report.json
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

# ── offline model cache (no HF download during benchmark) ────────────────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline import RAGEngine                           # noqa: E402
from app.config import (                                     # noqa: E402
    ANSWER_BACKEND,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    GEMINI_THINKING_BUDGET,
    QDRANT_COLLECTION,
    QWEN_MODEL,
    LLM_PROVIDER,
)
from app.generation.llm import SUCCESS                       # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

LATENCY_BUDGET_MS   = 200   # post-STT direct-answer P95 gate
ENGINEERING_TARGET  = 150   # comfortable safety margin

QUERY_FILE = Path("data/hindi_sample_1000.jsonl")

REPORT_FILE = Path("data/benchmark_report.json")

# Multilingual query pool used when the JSONL dataset is unavailable.
# 2 representative queries per language (13 languages = 26 queries).
FALLBACK_QUERY_POOL: list[tuple[str, str]] = [
    ("hi", "मैनहट्टन परियोजना क्या थी?"),
    ("hi", "डीएनए की संरचना कैसी है?"),
    ("bn", "ম্যানহাটন প্রকল্পের তাৎক্ষণিক প্রভাব কী?"),
    ("bn", "ডিএনএ কী এবং এটি কীভাবে কাজ করে?"),
    ("gu", "મેનહટન પ્રોજેક્ટની સફળતાની તાત્કાલિક અસર શું હતી?"),
    ("gu", "સૌર ઊર્જા કેવી રીતે કાર્ય કરે છે?"),
    ("kn", "ಮ್ಯಾನ್‌ಹ್ಯಾಟನ್ ಯೋಜನೆಯ ಯಶಸ್ಸಿನ ತಕ್ಷಣದ ಪರಿಣಾಮ ಏನು?"),
    ("kn", "ಸೌರ ಶಕ್ತಿ ಹೇಗೆ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ?"),
    ("ml", "മാൻഹാട്ടൻ പദ്ധതിയുടെ വിജയത്തിന്റെ ഉടനടി ആഘാതം എന്തായിരുന്നു?"),
    ("ml", "ഡിഎൻഎ എന്താണ്?"),
    ("mr", "मॅनहॅटन प्रकल्पाच्या यशाचा तात्काळ काय परिणाम झाला?"),
    ("mr", "सूर्य ऊर्जा कशी कार्य करते?"),
    ("ne", "म्यानहट्टन परियोजनाको सफलताको तत्काल प्रभाव के थियो?"),
    ("ne", "डिएनए कसरी काम गर्छ?"),
    ("or", "ମ୍ୟାନହାଟନ ପ୍ରକଳ୍ପର ସଫଳତାର ତତ୍‌କ୍ଷଣାତ ପ୍ରଭାବ କ'ଣ ଥିଲା?"),
    ("or", "ଡିଏନଏ କ'ଣ?"),
    ("pa", "ਮੈਨਹੈਟਨ ਪ੍ਰੋਜੈਕਟ ਦੀ ਸਫਲਤਾ ਦਾ ਤੁਰੰਤ ਪ੍ਰਭਾਵ ਕੀ ਸੀ?"),
    ("pa", "ਸੂਰਜੀ ਊਰਜਾ ਕਿਵੇਂ ਕੰਮ ਕਰਦੀ ਹੈ?"),
    ("sa", "म्यान्ह्याटन्-परियोजनायाः सफलतायाः तत्क्षणात् कः प्रभावः आसीत्?"),
    ("sa", "सूर्यशक्तिः कथं कार्यं करोति?"),
    ("ta", "மன்ஹாட்டன் திட்டத்தின் வெற்றியின் உடனடி விளைவு என்ன?"),
    ("ta", "டிஎன்ஏ என்றால் என்ன?"),
    ("ur", "مین ہیٹن پروجیکٹ کی کامیابی کا فوری اثر کیا تھا؟"),
    ("ur", "شمسی توانائی کیسے کام کرتی ہے؟"),
    ("as", "মেনহাটন প্ৰকল্পৰ সফলতাৰ তাৎক্ষণিক প্ৰভাৱ কি আছিল?"),
    ("as", "ডিএনএ কি আৰু ই কেনেকৈ কাম কৰে?"),
]


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100)
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    return ordered[lo] + (idx - lo) * (ordered[hi] - ordered[lo])


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p70": 0.0, "p95": 0.0, "p99": 0.0, "p100": 0.0, "n": 0}
    return {
        "avg":  round(statistics.mean(values), 2),
        "p50":  round(percentile(values, 50),  2),
        "p70":  round(percentile(values, 70),  2),
        "p95":  round(percentile(values, 95),  2),
        "p99":  round(percentile(values, 99),  2),
        "p100": round(max(values),             2),
        "n":    len(values),
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUERY LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_queries(max_q: int) -> list[tuple[str, str]]:
    """Load (language, query) pairs from the dataset JSONL.

    Falls back to FALLBACK_QUERY_POOL if the file is missing.
    Language is taken from the record's 'language' field (short code, e.g. 'hi').
    If that field is missing, defaults to 'hi'.
    """
    if not QUERY_FILE.exists():
        print(f"  [warn] {QUERY_FILE} not found — using built-in fallback pool ({len(FALLBACK_QUERY_POOL)} queries)")
        return FALLBACK_QUERY_POOL[:max_q]

    queries: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(QUERY_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            q   = item.get("query", "").strip()
            qid = item.get("query_id", "")
            if not q or qid in seen:
                continue
            seen.add(qid)
            lang = item.get("language", "hi")
            queries.append((lang, q))
            if len(queries) >= max_q:
                break

    if not queries:
        print(f"  [warn] {QUERY_FILE} yielded no queries — using fallback pool")
        return FALLBACK_QUERY_POOL[:max_q]

    return queries


# ─────────────────────────────────────────────────────────────────────────────
# SECTION PRINTER
# ─────────────────────────────────────────────────────────────────────────────

_W_STAGE = 14   # column width for stage name
_W_NUM   = 8    # column width for each numeric value


def _row(name: str, stats: dict[str, float]) -> str:
    return (
        f"  {name:<{_W_STAGE}}"
        f"{stats['avg']:>{_W_NUM}.2f}"
        f"{stats['p50']:>{_W_NUM}.2f}"
        f"{stats['p70']:>{_W_NUM}.2f}"
        f"{stats['p95']:>{_W_NUM}.2f}"
        f"{stats['p99']:>{_W_NUM}.2f}"
        f"{stats['p100']:>{_W_NUM}.2f}"
        f"   (n={stats['n']})"
    )


def _header() -> str:
    cols = ["avg", "p50", "p70", "p95", "p99", "p100"]
    hdr  = "  " + " " * _W_STAGE
    hdr += "".join(f"{c:>{_W_NUM}}" for c in cols)
    hdr += "   (ms)"
    return hdr


def _sep() -> str:
    return "  " + "─" * (_W_STAGE + _W_NUM * 6 + 9)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── CLI ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end HH-Goa-RAG benchmark.\n"
            "Phase 1: post-STT direct-answer latency (≤200 ms budget).\n"
            "Phase 2: LLM enhancement latency (async, best-effort)."
        ),
    )
    parser.add_argument(
        "n", nargs="?", type=int, default=200,
        help="Number of measured queries (default: 200).",
    )
    parser.add_argument(
        "--warmup", type=int, default=15,
        help="Warmup queries discarded before measurement (default: 15).",
    )
    parser.add_argument(
        "--llm", dest="llm_mode", action="store_true", default=None,
        help="Force LLM phase even if backend is 'extractive'.",
    )
    parser.add_argument(
        "--no-llm", dest="llm_mode", action="store_false",
        help="Skip LLM phase entirely.",
    )
    parser.add_argument(
        "--report", type=Path, default=REPORT_FILE,
        help=f"Where to write the JSON report (default: {REPORT_FILE}).",
    )
    args = parser.parse_args()

    n_queries = max(1, args.n)
    warmup    = max(0, args.warmup)

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  HH GOA 2026 — Voice RAG  |  End-to-End Benchmark")
    print("=" * 70)
    print(f"  Collection  : {QDRANT_COLLECTION}")
    print(f"  Embedding   : {EMBEDDING_MODEL.split('/')[-1]}")
    print(f"  Backend     : {ANSWER_BACKEND}")
    print(f"  Warmup      : {warmup} queries (discarded)")
    print(f"  Measured    : {n_queries} queries")
    print(f"  Budget      : direct-answer P95 < {LATENCY_BUDGET_MS} ms  (post-STT)")
    print()

    # ── Engine ────────────────────────────────────────────────────────────────
    print("  Loading RAG engine …")
    engine = RAGEngine()

    llm_available = engine.answer_generator.available
    llm_name      = (
        GEMINI_MODEL if ANSWER_BACKEND == "gemini"
        else QWEN_MODEL if ANSWER_BACKEND in {"qwen", "qwen_api"}
        else "—"
    )

    # Determine whether to run LLM phase
    if args.llm_mode is True:
        run_llm = True
    elif args.llm_mode is False:
        run_llm = False
    else:
        # auto: run if the LLM generator is actually available
        run_llm = llm_available and ANSWER_BACKEND in {"gemini", "qwen", "qwen_api"}

    print(f"  LLM         : {llm_name}  ({'will measure' if run_llm else 'skipped — not available or extractive backend'})")
    print()

    # ── Query pool ────────────────────────────────────────────────────────────
    pool = load_queries(max(n_queries + warmup, 500))
    print(f"  Query pool  : {len(pool)} unique queries")
    if len(pool) < warmup + n_queries:
        print(f"  [warn] pool smaller than warmup+n — queries will repeat (round-robin)")
    print()

    # ── Warmup ────────────────────────────────────────────────────────────────
    if warmup > 0:
        print(f"  Warming up ({warmup} queries) …")
        for i in range(warmup):
            lang, q = pool[i % len(pool)]
            try:
                engine.process_dual(q, lang)
            except Exception:
                pass
        print("  Warmup complete.\n")

    # ── Measured loop ─────────────────────────────────────────────────────────
    print(f"  Running {n_queries} measured queries …\n")

    # Per-query records
    records: list[dict[str, Any]] = []

    # Accumulated timing buckets
    t_embed   : list[float] = []
    t_qdrant  : list[float] = []
    t_rerank  : list[float] = []
    t_compress: list[float] = []
    t_answer  : list[float] = []
    t_direct  : list[float] = []   # total for Phase 1 = direct answer
    t_llm     : list[float] = []   # LLM wall-clock
    t_ttft    : list[float] = []   # Gemini time-to-first-token
    t_total   : list[float] = []   # direct + llm

    n_grounded  = 0
    n_blocked   = 0
    n_llm_ok    = 0
    n_llm_skip  = 0
    n_llm_fail  = 0

    for i in range(n_queries):
        lang, q = pool[(warmup + i) % len(pool)]
        rec: dict[str, Any] = {"n": i + 1, "lang": lang, "query": q[:80]}

        # ── Phase 1: retrieve + compress + extractive answer ─────────────────
        wall_direct_start = time.perf_counter()
        try:
            direct_result, state = engine.process_dual(q, lang)
        except Exception as exc:
            rec.update({
                "phase1_status": "ERROR",
                "phase1_error": str(exc),
                "direct_ms": None,
            })
            records.append(rec)
            continue
        direct_ms = (time.perf_counter() - wall_direct_start) * 1000

        timings = direct_result.get("timings", {})
        emb_ms  = float(timings.get("embedding_ms",   0))
        qdr_ms  = float(timings.get("qdrant_ms",      0))
        rnk_ms  = float(timings.get("rerank_ms",      0))
        cmp_ms  = float(timings.get("compression_ms", 0))
        ans_ms  = float(timings.get("answer_ms",      0))

        t_embed.append(emb_ms)
        t_qdrant.append(qdr_ms)
        t_rerank.append(rnk_ms)
        t_compress.append(cmp_ms)
        t_answer.append(ans_ms)
        t_direct.append(direct_ms)

        grounded = direct_result.get("grounded", False)
        blocked  = direct_result.get("blocked",  False)
        if grounded:
            n_grounded += 1
        if blocked:
            n_blocked += 1

        rec.update({
            "phase1_status": "OK",
            "grounded": grounded,
            "blocked":  blocked,
            "retrieved_chunks": direct_result.get("retrieved_chunks", 0),
            "direct_ms":  round(direct_ms, 2),
            "embedding_ms":   round(emb_ms,  2),
            "qdrant_ms":      round(qdr_ms,  2),
            "rerank_ms":      round(rnk_ms,  2),
            "compression_ms": round(cmp_ms,  2),
            "answer_ms":      round(ans_ms,  2),
        })

        # ── Phase 2: LLM enhancement ──────────────────────────────────────────
        if run_llm and state is not None:
            try:
                llm_result = engine.answer_generator.generate(
                    state["query"],
                    state["language_code"] or "",
                    state["compression_result"]["context"],
                )
                llm_ms_val  = float(llm_result.get("llm_ms", 0) or llm_result.get("latency_ms", 0))
                ttft_ms_val = float(llm_result.get("ttft_ms", 0))
                llm_status  = llm_result.get("status", "EXCEPTION")

                if llm_status == SUCCESS:
                    n_llm_ok += 1
                    t_llm.append(llm_ms_val)
                    t_ttft.append(ttft_ms_val)
                    t_total.append(direct_ms + llm_ms_val)
                    rec.update({
                        "phase2_status": "OK",
                        "llm_ms":       round(llm_ms_val, 2),
                        "ttft_ms":      round(ttft_ms_val, 2),
                        "total_ms":     round(direct_ms + llm_ms_val, 2),
                        "llm_reason":   llm_result.get("reason", ""),
                        "llm_attempts": llm_result.get("attempt_count", 1),
                    })
                else:
                    n_llm_fail += 1
                    rec.update({
                        "phase2_status": "FAIL",
                        "llm_reason":    llm_result.get("reason", ""),
                        "llm_error":     llm_result.get("error", ""),
                        "llm_attempts":  llm_result.get("attempt_count", 1),
                    })
            except Exception as exc:
                n_llm_fail += 1
                rec["phase2_status"] = "ERROR"
                rec["llm_error"]     = str(exc)
        else:
            n_llm_skip += 1
            rec["phase2_status"] = "SKIPPED"

        records.append(rec)

        # Progress ticker every 50 queries
        if (i + 1) % 50 == 0:
            p95_so_far = percentile(t_direct, 95) if t_direct else 0
            print(f"  [{i+1:>4}/{n_queries}]  direct P95 so far: {p95_so_far:.1f} ms")

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_ok       = len(t_direct)
    n_error    = n_queries - n_ok
    direct_p95 = percentile(t_direct, 95) if t_direct else 0

    pass_official    = direct_p95 <= LATENCY_BUDGET_MS   if n_ok > 0 else False
    pass_engineering = direct_p95 <= ENGINEERING_TARGET  if n_ok > 0 else False

    # ── Phase 1 table ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  PHASE 1 — POST-STT DIRECT ANSWER  (the <200 ms critical path)")
    print("=" * 70)
    print(f"  Successful  : {n_ok} / {n_queries}")
    if n_error:
        print(f"  Errors      : {n_error}")
    print(f"  Grounded    : {n_grounded} ({n_grounded/n_ok*100:.1f}%)" if n_ok else "")
    print(f"  Blocked     : {n_blocked}")
    print()
    print(_header())
    print(_sep())
    for stage_name, bucket in [
        ("embedding",   t_embed),
        ("qdrant",      t_qdrant),
        ("rerank",      t_rerank),
        ("compression", t_compress),
        ("answer",      t_answer),
        ("─── DIRECT ─", t_direct),
    ]:
        if bucket:
            print(_row(stage_name, summarize(bucket)))

    print()
    print(f"  Latency budget  : {LATENCY_BUDGET_MS} ms (post-STT, P95)")
    print(f"  Engineering tgt : {ENGINEERING_TARGET} ms (P95, safety margin)")
    p95_str = f"{direct_p95:.2f} ms" if n_ok else "N/A"
    print(f"  Direct P95      : {p95_str}")
    p100_str = f"{max(t_direct):.2f} ms" if t_direct else "N/A"
    print(f"  Direct P100     : {p100_str}")
    print()

    if n_ok == 0:
        print("  STATUS: INCONCLUSIVE (all queries errored)")
    elif pass_official:
        status_line = "  ✓ PASS  — direct answer within budget"
        if pass_engineering:
            status_line += f" (engineering target {ENGINEERING_TARGET} ms also met)"
        print(status_line)
    else:
        print(f"  ✗ FAIL  — direct P95 {direct_p95:.2f} ms exceeds {LATENCY_BUDGET_MS} ms budget")
        # Identify the bottleneck
        for stage_name, bucket in [("embedding", t_embed), ("qdrant", t_qdrant)]:
            if bucket:
                print(f"    Bottleneck hint — {stage_name} P95: {percentile(bucket, 95):.2f} ms")

    # ── Phase 2 table ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  PHASE 2 — LLM ENHANCEMENT  (async, NOT on the <200 ms path)")
    print("=" * 70)

    if not run_llm:
        print(f"  LLM phase skipped")
        if not llm_available:
            print(f"  Reason: LLM generator unavailable (backend={ANSWER_BACKEND}, no API key or SDK)")
        else:
            print(f"  Reason: --no-llm flag set")
        print()
        print("  To include LLM timings, ensure GEMINI_API_KEY is set and re-run.")
        print("  Use --llm to force this phase even when the backend auto-detects unavailable.")
    elif n_llm_ok == 0 and n_llm_fail > 0:
        print(f"  LLM attempted : {n_llm_ok + n_llm_fail}")
        print(f"  LLM succeeded : 0")
        print(f"  LLM failed    : {n_llm_fail}  (API key missing, rate limit, or network issue)")
        print()
        print("  No LLM timing data available for this run.")
    else:
        print(f"  LLM model     : {llm_name}")
        print(f"  LLM attempted : {n_llm_ok + n_llm_fail}")
        print(f"  LLM succeeded : {n_llm_ok}")
        if n_llm_fail:
            print(f"  LLM failed    : {n_llm_fail}  (retries counted — see records)")
        if n_llm_skip:
            print(f"  LLM skipped   : {n_llm_skip}  (no state returned from pipeline)")
        print()

        if t_llm:
            print(_header())
            print(_sep())
            if t_ttft and any(v > 0 for v in t_ttft):
                print(_row("ttft (gemini)", summarize(t_ttft)))
            print(_row("llm total",    summarize(t_llm)))
            print(_row("─── TOTAL ─",  summarize(t_total)))
            print()
            print(f"  NOTE: total = direct_ms + llm_ms")
            print(f"  The direct answer is delivered after direct_ms regardless.")
            print(f"  LLM is an async enhancement — it never blocks the user response.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    p50_d  = percentile(t_direct, 50) if t_direct else 0
    p70_d  = percentile(t_direct, 70) if t_direct else 0
    p100_d = max(t_direct) if t_direct else 0
    print(f"  Direct answer  P50  : {p50_d:.1f} ms")
    print(f"  Direct answer  P70  : {p70_d:.1f} ms")
    print(f"  Direct answer  P95  : {direct_p95:.1f} ms")
    print(f"  Direct answer  P100 : {p100_d:.1f} ms")
    if t_llm:
        p50_l  = percentile(t_llm, 50)
        p95_l  = percentile(t_llm, 95)
        p50_tot = percentile(t_total, 50)
        p95_tot = percentile(t_total, 95)
        print(f"  LLM generation P50  : {p50_l:.1f} ms   (async, not in 200ms budget)")
        print(f"  LLM generation P95  : {p95_l:.1f} ms")
        print(f"  Total pipeline P50  : {p50_tot:.1f} ms  (direct + llm)")
        print(f"  Total pipeline P95  : {p95_tot:.1f} ms")
    budget_result = (
        "PASS" if pass_official
        else "FAIL" if n_ok > 0
        else "INCONCLUSIVE"
    )
    print(f"  Budget gate (<{LATENCY_BUDGET_MS}ms P95) : {budget_result}")
    print("=" * 70)
    print()

    # ── JSON report ───────────────────────────────────────────────────────────
    report: dict[str, Any] = {
        "benchmark": "end_to_end_pipeline",
        "config": {
            "collection":    QDRANT_COLLECTION,
            "embedding":     EMBEDDING_MODEL,
            "answer_backend": ANSWER_BACKEND,
            "llm_model":     llm_name,
            "llm_provider":  LLM_PROVIDER,
            "warmup":        warmup,
            "n_queries":     n_queries,
        },
        "phase1_direct_answer": {
            "note": (
                "Post-STT fast path. This is the performance-critical window. "
                "The 200ms latency budget applies here."
            ),
            "n_ok":       n_ok,
            "n_error":    n_error,
            "n_grounded": n_grounded,
            "n_blocked":  n_blocked,
            "stages": {
                "embedding":   summarize(t_embed),
                "qdrant":      summarize(t_qdrant),
                "rerank":      summarize(t_rerank),
                "compression": summarize(t_compress),
                "answer":      summarize(t_answer),
                "direct_total": summarize(t_direct),
            },
            "budget": {
                "target_p95_ms":      LATENCY_BUDGET_MS,
                "engineering_p95_ms": ENGINEERING_TARGET,
                "actual_p95_ms":      round(direct_p95, 2),
                "pass_official":      pass_official,
                "pass_engineering":   pass_engineering,
                "status":             budget_result,
            },
        },
        "phase2_llm_enhancement": {
            "note": (
                "LLM runs AFTER direct answer is delivered. "
                "Not subject to the 200ms budget. Skipped gracefully if unavailable."
            ),
            "run":         run_llm,
            "n_ok":        n_llm_ok,
            "n_fail":      n_llm_fail,
            "n_skip":      n_llm_skip,
            "model":       llm_name,
            "stages": {
                "ttft_gemini": summarize(t_ttft) if t_ttft else None,
                "llm_total":   summarize(t_llm)  if t_llm  else None,
                "pipeline_total_direct_plus_llm": summarize(t_total) if t_total else None,
            },
        },
        "records": records,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Report saved → {args.report}")
    print()

    engine.close()

    # ── Exit code ─────────────────────────────────────────────────────────────
    if not pass_official and n_ok > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
