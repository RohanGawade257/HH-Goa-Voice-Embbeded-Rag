"""Direct-answer latency benchmark for HH-Goa-Rag multilingual pipeline.

Phase 3E — Measure time_to_direct_answer independently of Gemini.

The fast path measured here is:
    STT complete → START
      embedding → Qdrant → rerank → compress → extractive answer
    STOP → time_to_direct_answer

This is the performance-critical path. The 200 ms target applies here.
The engineering target (with deployment safety margin) is ≤150 ms P95.

Gemini is NOT called in this benchmark. That is measured separately.

Historical reference only (do NOT cite as current):
    Hindi extractive P50 ≈ 38 ms  (old hindi collection, different pipeline)

Usage:
    python benchmarks/benchmark_direct_answer.py
    python benchmarks/benchmark_direct_answer.py --warmup 10 --requests 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["ANSWER_BACKEND"] = "extractive"  # no LLM needed

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline import RAGEngine  # noqa: E402

REPORT_DIR = REPO_ROOT / "data" / "processed" / "multilingual"
REPORT_FILE = REPORT_DIR / "direct_answer_benchmark.json"

LATENCY_TARGET_MS = 200        # official requirement
ENGINEERING_TARGET_MS = 150    # recommended safety margin

# 2 representative queries per language = 26 total
QUERY_POOL = [
    # Assamese
    ("as-IN", "মেনহাটন প্ৰকল্পৰ সফলতাৰ তাৎক্ষণিক প্ৰভাৱ কি আছিল?"),
    ("as-IN", "অপৰাধীৰ অপৰাধমূলক কাৰ্যৰ ফলত হোৱা ক্ষতি মোচন কৰা কি?"),
    # Bengali
    ("bn-IN", "ম্যানহাটন প্রকল্পের সাফল্যের তাৎক্ষণিক প্রভাব কী ছিল?"),
    ("bn-IN", "ডিএনএ কী এবং এটি কীভাবে কাজ করে?"),
    # Gujarati
    ("gu-IN", "મેનહટન પ્રોજેક્ટની સફળતાની તાત્કાલિક અસર શું હતી?"),
    ("gu-IN", "સૌર ઊર્જા કેવી રીતે કાર્ય કરે છે?"),
    # Hindi
    ("hi-IN", "मैनहट्टन परियोजना क्या थी?"),
    ("hi-IN", "डीएनए की संरचना कैसी है?"),
    # Kannada
    ("kn-IN", "ಮ್ಯಾನ್‌ಹ್ಯಾಟನ್ ಯೋಜನೆಯ ಯಶಸ್ಸಿನ ತಕ್ಷಣದ ಪರಿಣಾಮ ಏನು ಆಗಿತ್ತು?"),
    ("kn-IN", "ಸೌರ ಶಕ್ತಿ ಹೇಗೆ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ?"),
    # Malayalam
    ("ml-IN", "മാൻഹാട്ടൻ പദ്ധതിയുടെ വിജയത്തിന്റെ ഉടനടി ആഘാതം എന്തായിരുന്നു?"),
    ("ml-IN", "ഡിഎൻഎ എന്താണ്?"),
    # Marathi
    ("mr-IN", "मॅनहॅटन प्रकल्पाच्या यशाचा तात्काळ काय परिणाम झाला?"),
    ("mr-IN", "सूर्य ऊर्जा कशी कार्य करते?"),
    # Nepali
    ("ne-IN", "म्यानहट्टन परियोजनाको सफलताको तत्काल प्रभाव के थियो?"),
    ("ne-IN", "डिएनए कसरी काम गर्छ?"),
    # Odia
    ("or-IN", "ମ୍ୟାନହାଟନ ପ୍ରକଳ୍ପର ସଫଳତାର ତତ୍‌କ୍ଷଣାତ ପ୍ରଭାବ କ'ଣ ଥିଲା?"),
    ("or-IN", "ଡିଏନଏ କ'ଣ?"),
    # Punjabi
    ("pa-IN", "ਮੈਨਹੈਟਨ ਪ੍ਰੋਜੈਕਟ ਦੀ ਸਫਲਤਾ ਦਾ ਤੁਰੰਤ ਪ੍ਰਭਾਵ ਕੀ ਸੀ?"),
    ("pa-IN", "ਸੂਰਜੀ ਊਰਜਾ ਕਿਵੇਂ ਕੰਮ ਕਰਦੀ ਹੈ?"),
    # Sanskrit
    ("sa-IN", "म्यान्ह्याटन्-परियोजनायाः सफलतायाः तत्क्षणात् कः प्रभावः आसीत्?"),
    ("sa-IN", "सूर्यशक्तिः कथं कार्यं करोति?"),
    # Tamil
    ("ta-IN", "மன்ஹாட்டன் திட்டத்தின் வெற்றியின் உடனடி விளைவு என்ன?"),
    ("ta-IN", "டிஎன்ஏ என்றால் என்ன?"),
    # Urdu
    ("ur-IN", "مین ہیٹن پروجیکٹ کی کامیابی کا فوری اثر کیا تھا؟"),
    ("ur-IN", "شمسی توانائی کیسے کام کرتی ہے؟"),
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (idx - lo) * (ordered[hi] - ordered[lo])


def run_benchmark(warmup: int, n_requests: int) -> dict:
    print("=" * 65)
    print("HH-GOA-RAG  --  DIRECT-ANSWER LATENCY BENCHMARK")
    print("Phase 3E: time_to_direct_answer (extractive, no LLM)")
    print("=" * 65)
    print()
    print(f"Warmup    : {warmup} requests (discarded)")
    print(f"Measured  : {n_requests} requests")
    print(f"Query pool: {len(QUERY_POOL)} unique queries (2 per language)")
    print()

    print("Loading RAG engine...")
    engine = RAGEngine()
    print("Engine ready.\n")

    # ---- warmup ----
    if warmup > 0:
        print(f"Warming up ({warmup} requests)...")
        for i in range(warmup):
            lang, q = QUERY_POOL[i % len(QUERY_POOL)]
            try:
                engine.process_dual(q, lang)
            except Exception:
                pass
        print("Warmup done.\n")

    # ---- measured loop ----
    print("Measuring...")
    records = []
    for i in range(n_requests):
        lang, q = QUERY_POOL[i % len(QUERY_POOL)]
        try:
            t_start = time.perf_counter()
            result, _ = engine.process_dual(q, lang)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            timings = result.get("timings", {})
            records.append({
                "n": i + 1,
                "lang": lang,
                "query": q[:60],
                "status": "SUCCESS",
                "retrieved_chunks": result.get("retrieved_chunks", 0),
                "blocked": result.get("blocked", False),
                "time_to_direct_ms": round(elapsed_ms, 2),
                "embedding_ms": timings.get("embedding_ms", 0),
                "qdrant_ms": timings.get("qdrant_ms", 0),
                "rerank_ms": timings.get("rerank_ms", 0),
                "compression_ms": timings.get("compression_ms", 0),
                "answer_ms": timings.get("answer_ms", 0),
            })
        except Exception as exc:
            records.append({
                "n": i + 1,
                "lang": lang,
                "query": q[:60],
                "status": "EXCEPTION",
                "error": str(exc),
                "time_to_direct_ms": None,
            })

    # ---- aggregate ----
    successes = [r for r in records if r["status"] == "SUCCESS"]
    exceptions = [r for r in records if r["status"] == "EXCEPTION"]
    n_ok = len(successes)

    times = [r["time_to_direct_ms"] for r in successes]
    emb   = [r["embedding_ms"] for r in successes]
    qdr   = [r["qdrant_ms"] for r in successes]
    rnk   = [r["rerank_ms"] for r in successes]
    cmp   = [r["compression_ms"] for r in successes]
    ans   = [r["answer_ms"] for r in successes]

    p50  = percentile(times, 50)
    p95  = percentile(times, 95)
    p99  = percentile(times, 99)
    p100 = percentile(times, 100)

    target_ok_p95  = p95 <= LATENCY_TARGET_MS if n_ok > 0 else False
    engineering_ok = p95 <= ENGINEERING_TARGET_MS if n_ok > 0 else False

    status = "INCONCLUSIVE"
    if n_ok >= max(5, n_requests // 5):
        status = "PASS" if target_ok_p95 else "FAIL"

    # ---- print report ----
    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Total measured      : {n_requests}")
    print(f"  Successful          : {n_ok}")
    print(f"  Exceptions          : {len(exceptions)}")
    print(f"  Success rate        : {n_ok/n_requests:.1%}")
    print()
    print("  TIME-TO-DIRECT-ANSWER (successful requests only)")
    print(f"    P50  : {p50:.1f} ms")
    print(f"    P95  : {p95:.1f} ms")
    print(f"    P99  : {p99:.1f} ms")
    print(f"    P100 : {p100:.1f} ms")
    print()
    print("  STAGE BREAKDOWN (P50 of successful requests)")
    print(f"    Embedding   : {percentile(emb, 50):.1f} ms")
    print(f"    Qdrant      : {percentile(qdr, 50):.1f} ms")
    print(f"    Reranking   : {percentile(rnk, 50):.1f} ms")
    print(f"    Compression : {percentile(cmp, 50):.1f} ms")
    print(f"    Answer      : {percentile(ans, 50):.1f} ms")
    print()
    print(f"  Target  (<{LATENCY_TARGET_MS} ms P95)       : {'PASS' if target_ok_p95 else 'FAIL'}")
    print(f"  Target  (<{ENGINEERING_TARGET_MS} ms P95 eng): {'PASS' if engineering_ok else 'FAIL'}")
    print()
    print(f"  STATUS: {status}")
    print()
    print("  NOTE: This is a localhost measurement. Deployed latency will differ")
    print("        due to network overhead and production hardware differences.")
    print()
    print("  HISTORICAL REFERENCE (do NOT cite as current):")
    print("    Hindi extractive P50 ~38ms — old hindi-only collection, different pipeline.")
    print("=" * 65)

    report = {
        "benchmark": "direct_answer_latency",
        "phase": "3E",
        "note": (
            "Localhost measurement. Deployed latency includes network overhead. "
            "Time_to_direct_answer is the performance-critical path (<200ms target). "
            "Gemini (time_to_llm_answer) is benchmarked separately."
        ),
        "historical_note": (
            "Hindi extractive P50~38ms was measured on the old hindi-only Qdrant "
            "collection. Do NOT cite as current performance."
        ),
        "warmup": warmup,
        "requested_requests": n_requests,
        "unique_queries": len(QUERY_POOL),
        "successful_requests": n_ok,
        "exceptions": len(exceptions),
        "success_rate": round(n_ok / n_requests, 4) if n_requests else 0,
        "latency_ms": {
            "time_to_direct": {
                "p50":  round(p50, 2),
                "p95":  round(p95, 2),
                "p99":  round(p99, 2),
                "p100": round(p100, 2),
            },
            "embedding_p50":    round(percentile(emb, 50), 2),
            "qdrant_p50":       round(percentile(qdr, 50), 2),
            "reranking_p50":    round(percentile(rnk, 50), 2),
            "compression_p50":  round(percentile(cmp, 50), 2),
            "answer_p50":       round(percentile(ans, 50), 2),
        },
        "target": {
            "post_stt_direct_p95_ms": LATENCY_TARGET_MS,
            "engineering_p95_ms": ENGINEERING_TARGET_MS,
            "pass_official": target_ok_p95,
            "pass_engineering": engineering_ok,
            "status": status,
        },
        "requests": records,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved -> {REPORT_FILE}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark time_to_direct_answer for the multilingual RAG pipeline"
    )
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup requests (default: 10)")
    parser.add_argument("--requests", type=int, default=50,
                        help="Number of measured requests (default: 50)")
    args = parser.parse_args()

    run_benchmark(warmup=args.warmup, n_requests=args.requests)
