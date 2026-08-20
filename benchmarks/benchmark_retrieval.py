# -*- coding: utf-8 -*-
"""Retrieval quality benchmark for the multilingual Qdrant pipeline.

Phase 3A — Measure before changing.

Tests:
  - Recall@3 and Recall@K for K in (10, 20, 30)
  - Stage latency (embedding, Qdrant, rerank, compression, time_to_direct_answer)
  - Language-affinity A/B: does adding a 0.05 language-match bonus improve Recall@3?

Recall definition used here:
  A query PASSES if at least one chunk in the returned top-3 has the same
  query_id as the query.  This tests whether the pipeline retrieves a
  passage that belongs to the same topic group as the question.

Cross-language retrieval is intentional — the corpus contains the same
query_id in 13 languages.  The embedding model is multilingual so it can
retrieve the correct answer in any language.

Usage:
    python benchmarks/benchmark_retrieval.py
    python benchmarks/benchmark_retrieval.py --warmup 5 --queries-per-lang 2

The script saves results to:
    data/processed/multilingual/retrieval_benchmark.json
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
os.environ["ANSWER_BACKEND"] = "extractive"  # no LLM needed

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipeline import (  # noqa: E402
    RAGEngine,
    COLLECTION_NAME,
    TOP_K_FINAL,
    VECTOR_WEIGHT,
    LEXICAL_WEIGHT,
    PHRASE_WEIGHT,
    build_query_features,
    score_document,
)
from app.context_compressor import compress_context  # noqa: E402
from app.answer_generator import generate_extractive_answer  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "multilingual"
REPORT_DIR = REPO_ROOT / "data" / "processed" / "multilingual"
REPORT_FILE = REPORT_DIR / "retrieval_benchmark.json"

SUPPORTED_LANGUAGES = [
    "as", "bn", "gu", "hi", "kn", "ml",
    "mr", "ne", "or", "pa", "sa", "ta", "ur",
]

# Candidate pool sizes to test
K_VALUES = [10, 20, 30]

# Language-affinity bonus weight (applied temporarily for A/B test only)
LANG_AFFINITY_WEIGHT = 0.05


# ============================================================
# QUERY LOADING
# ============================================================

def load_queries(queries_per_lang: int = 2) -> list[dict]:
    """Load N queries per language from the raw JSONL files.

    Returns list of:
        {lang, query_id, query}

    Uses the first N DESCRIPTION-type queries from each language file.
    """
    queries: list[dict] = []
    for lang in SUPPORTED_LANGUAGES:
        path = DATA_DIR / f"{lang}_sample_1000.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {lang}")
            continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        descs = [r for r in rows if r.get("query_type") == "DESCRIPTION"]
        for r in descs[:queries_per_lang]:
            queries.append({
                "lang": lang,
                "query_id": r["query_id"],
                "query": r["query"],
            })
    return queries


# ============================================================
# PERCENTILE HELPER
# ============================================================

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


# ============================================================
# LANGUAGE-AFFINITY RERANKER (A/B variant — inline, no pipeline change)
# ============================================================

def rerank_with_language_affinity(query: str, hits, language_code: str | None):
    """Rerank hits with an optional language-affinity bonus.

    Same as the production rerank() but adds LANG_AFFINITY_WEIGHT * match.
    Weights sum to 1.0 only when language_code is provided.
    When language_code is None the result is identical to baseline.
    """
    if not hits:
        return []

    # Adjust base weights to make room for the affinity bonus
    v_weight = VECTOR_WEIGHT - LANG_AFFINITY_WEIGHT / 2
    l_weight = LEXICAL_WEIGHT
    p_weight = PHRASE_WEIGHT - LANG_AFFINITY_WEIGHT / 2
    lang_w   = LANG_AFFINITY_WEIGHT if language_code else 0.0

    normalized_query, query_tokens, query_phrases = build_query_features(query)

    scores = []
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text", "")
        vector_score = float(hit.score)

        lexical_score, exact_phrase_score = score_document(
            normalized_query, query_tokens, query_phrases, text
        )

        # Language affinity: 1.0 if the stored language matches
        hit_lang = payload.get("language", "")
        lang_match = 1.0 if (language_code and hit_lang == language_code) else 0.0

        final_score = (
            v_weight * vector_score
            + l_weight * lexical_score
            + p_weight * exact_phrase_score
            + lang_w * lang_match
        )

        scores.append((final_score, vector_score, lexical_score, exact_phrase_score, hit))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:TOP_K_FINAL]


# ============================================================
# SINGLE QUERY BENCHMARK
# ============================================================

def run_query(
    engine: RAGEngine,
    query: str,
    query_id: int,
    language: str,
    top_k: int,
    use_lang_affinity: bool = False,
) -> dict:
    """Run one query and return detailed timing + recall result."""

    t_pipeline_start = time.perf_counter()

    # 1. Embed
    t = time.perf_counter()
    qvec = engine.embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    embedding_ms = (time.perf_counter() - t) * 1000

    # 2. Qdrant
    t = time.perf_counter()
    response = engine.client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=top_k,
        with_payload=["chunk_id", "passage_id", "query_id", "text", "language"],
        with_vectors=False,
    )
    hits = response.points
    qdrant_ms = (time.perf_counter() - t) * 1000

    # 3. Rerank (baseline or with affinity)
    t = time.perf_counter()
    if use_lang_affinity:
        reranked = rerank_with_language_affinity(query, hits, language)
    else:
        from app.pipeline import rerank as baseline_rerank
        reranked = baseline_rerank(query, hits)
    rerank_ms = (time.perf_counter() - t) * 1000

    # 4. Build top-3 result list
    top3 = []
    for rank, item in enumerate(reranked, start=1):
        _, vs, ls, ps, hit = item
        payload = hit.payload or {}
        top3.append({
            "rank": rank,
            "chunk_id": payload.get("chunk_id"),
            "query_id": payload.get("query_id"),
            "language": payload.get("language"),
            "text": payload.get("text", ""),
            "vector_score": vs,
        })

    # 5. Compress
    t = time.perf_counter()
    compression_result = compress_context(query, top3)
    compression_ms = (time.perf_counter() - t) * 1000

    # 6. Extractive answer
    t = time.perf_counter()
    snippets = [{"text": s["text"], "score": s.get("score", 0.0)}
                for s in compression_result["snippets"]]
    generate_extractive_answer(query, snippets)
    answer_ms = (time.perf_counter() - t) * 1000

    time_to_direct_ms = (time.perf_counter() - t_pipeline_start) * 1000

    # Recall: does any top-3 chunk belong to this query's topic group?
    recall_at_3 = any(c.get("query_id") == query_id for c in top3)

    # Recall in the raw Qdrant pool (before reranking)
    pool_query_ids = {hit.payload.get("query_id") for hit in hits}
    recall_at_k = query_id in pool_query_ids

    # Language of top-1 result
    top1_lang = top3[0]["language"] if top3 else None

    return {
        "recall_at_3": recall_at_3,
        "recall_at_k": recall_at_k,
        "top1_correct_lang": top1_lang == language,
        "timings": {
            "embedding_ms": round(embedding_ms, 2),
            "qdrant_ms": round(qdrant_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "compression_ms": round(compression_ms, 2),
            "answer_ms": round(answer_ms, 2),
            "time_to_direct_ms": round(time_to_direct_ms, 2),
        },
    }


# ============================================================
# MAIN BENCHMARK LOOP
# ============================================================

def run_benchmark(warmup: int, queries_per_lang: int) -> dict:
    print("=" * 65)
    print("HH-GOA-RAG  —  RETRIEVAL QUALITY BENCHMARK")
    print("Phase 3A: Measure before changing")
    print("=" * 65)
    print()

    print("Loading RAG engine…")
    engine = RAGEngine()
    print("Engine ready.")
    print()

    queries = load_queries(queries_per_lang)
    print(f"Query pool: {len(queries)} queries ({queries_per_lang} per language × {len(SUPPORTED_LANGUAGES)} languages)")
    print()

    # ---------- warmup ----------
    if warmup > 0:
        print(f"Warming up ({warmup} requests)…")
        for i in range(warmup):
            q = queries[i % len(queries)]
            try:
                run_query(engine, q["query"], q["query_id"], q["lang"], top_k=10)
            except Exception:
                pass
        print("Warmup done.\n")

    # ---------- K sweep ----------
    k_results: dict[int, dict] = {}
    for k in K_VALUES:
        print(f"-- TOP_K = {k} {'-'*45}")
        per_query: list[dict] = []
        for q in queries:
            try:
                r = run_query(engine, q["query"], q["query_id"], q["lang"],
                              top_k=k, use_lang_affinity=False)
                r["query"] = q["query"][:60]
                r["lang"] = q["lang"]
                per_query.append(r)
            except Exception as exc:
                per_query.append({
                    "query": q["query"][:60], "lang": q["lang"],
                    "recall_at_3": False, "recall_at_k": False,
                    "error": str(exc),
                })

        n = len(per_query)
        recall3 = sum(1 for r in per_query if r.get("recall_at_3")) / n if n else 0
        recallK = sum(1 for r in per_query if r.get("recall_at_k")) / n if n else 0
        times_direct = [r["timings"]["time_to_direct_ms"]
                        for r in per_query if "timings" in r]
        times_qdrant = [r["timings"]["qdrant_ms"]
                        for r in per_query if "timings" in r]
        times_embed  = [r["timings"]["embedding_ms"]
                        for r in per_query if "timings" in r]

        p50 = percentile(times_direct, 50)
        p95 = percentile(times_direct, 95)
        print(f"  Recall@3 = {recall3:.1%}   Recall@{k} = {recallK:.1%}")
        print(f"  time_to_direct P50={p50:.1f}ms  P95={p95:.1f}ms")
        print(f"  embed P50={percentile(times_embed,50):.1f}ms  "
              f"qdrant P50={percentile(times_qdrant,50):.1f}ms")
        print()

        k_results[k] = {
            "recall_at_3": round(recall3, 4),
            "recall_at_k": round(recallK, 4),
            "time_to_direct_p50": round(p50, 2),
            "time_to_direct_p95": round(p95, 2),
            "embedding_p50": round(percentile(times_embed, 50), 2),
            "qdrant_p50":    round(percentile(times_qdrant, 50), 2),
            "per_query": per_query,
        }

    # ---------- Language-affinity A/B  (always at K=10 for comparability) ----------
    print("-- Language-affinity A/B test (K=10) " + "-" * 28)
    ab_per_query: list[dict] = []
    for q in queries:
        try:
            r = run_query(engine, q["query"], q["query_id"], q["lang"],
                          top_k=10, use_lang_affinity=True)
            r["query"] = q["query"][:60]
            r["lang"] = q["lang"]
            ab_per_query.append(r)
        except Exception as exc:
            ab_per_query.append({
                "query": q["query"][:60], "lang": q["lang"],
                "recall_at_3": False, "recall_at_k": False,
                "error": str(exc),
            })

    n = len(ab_per_query)
    ab_recall3 = sum(1 for r in ab_per_query if r.get("recall_at_3")) / n if n else 0
    baseline_recall3 = k_results[10]["recall_at_3"]
    delta = ab_recall3 - baseline_recall3
    ab_times = [r["timings"]["time_to_direct_ms"]
                for r in ab_per_query if "timings" in r]
    ab_p50 = percentile(ab_times, 50)
    ab_p95 = percentile(ab_times, 95)

    print(f"  Baseline  Recall@3 = {baseline_recall3:.1%}")
    print(f"  +LangAff  Recall@3 = {ab_recall3:.1%}  (delta={delta:+.1%})")
    print(f"  +LangAff  time_to_direct P50={ab_p50:.1f}ms  P95={ab_p95:.1f}ms")
    print()

    lang_affinity_result = {
        "recall_at_3": round(ab_recall3, 4),
        "recall_at_3_delta_vs_k10_baseline": round(delta, 4),
        "time_to_direct_p50": round(ab_p50, 2),
        "time_to_direct_p95": round(ab_p95, 2),
        "per_query": ab_per_query,
    }

    # ---------- Decision ----------
    print("=" * 65)
    print("DECISION TABLE")
    print("=" * 65)

    header = f"{'K':>5}  {'Recall@3':>10}  {'Recall@K':>10}  {'Direct P50':>12}  {'Direct P95':>12}"
    print(header)
    print("-" * len(header))
    for k, r in k_results.items():
        print(f"{k:>5}  {r['recall_at_3']:>10.1%}  {r['recall_at_k']:>10.1%}"
              f"  {r['time_to_direct_p50']:>10.1f}ms  {r['time_to_direct_p95']:>10.1f}ms")
    print()

    # Automatic recommendation: adopt K if Recall@3 improves by ≥2pp AND
    # time_to_direct P95 increase is ≤ 10ms vs K=10
    best_k = 10
    best_recall = k_results[10]["recall_at_3"]
    base_p95 = k_results[10]["time_to_direct_p95"]
    for k in [20, 30]:
        r = k_results[k]
        recall_gain = r["recall_at_3"] - best_recall
        latency_cost = r["time_to_direct_p95"] - base_p95
        if recall_gain >= 0.02 and latency_cost <= 10.0:
            best_k = k
            best_recall = r["recall_at_3"]

    adopt_lang_affinity = (
        lang_affinity_result["recall_at_3_delta_vs_k10_baseline"] >= 0.02
        and (lang_affinity_result["time_to_direct_p95"] - base_p95) <= 5.0
    )

    print(f"Recommended TOP_K : {best_k} "
          f"({'same as current' if best_k == 10 else 'CHANGE JUSTIFIED'})")
    print(f"Adopt lang-affinity: {'YES — JUSTIFIED' if adopt_lang_affinity else 'NO — not justified'}")
    print()

    report: dict[str, Any] = {
        "benchmark": "retrieval_quality",
        "phase": "3A",
        "warmup": warmup,
        "queries_per_lang": queries_per_lang,
        "total_queries": len(queries),
        "languages": SUPPORTED_LANGUAGES,
        "k_values_tested": K_VALUES,
        "results_by_k": {str(k): {k2: v for k2, v in v.items() if k2 != "per_query"}
                         for k, v in k_results.items()},
        "language_affinity_ab": {k2: v for k2, v in lang_affinity_result.items()
                                  if k2 != "per_query"},
        "recommendation": {
            "top_k": best_k,
            "top_k_changed": best_k != 10,
            "adopt_language_affinity": adopt_lang_affinity,
            "rationale": (
                f"TOP_K={best_k} chosen: recall gain ≥2pp and latency cost ≤10ms. "
                f"Language affinity {'adopted' if adopt_lang_affinity else 'not adopted'}: "
                f"delta={lang_affinity_result['recall_at_3_delta_vs_k10_baseline']:+.1%}."
            ),
        },
        "historical_note": (
            "Hindi extractive baseline P50≈38ms was measured on the old hindi-only "
            "Qdrant collection with a different pipeline. Do NOT cite as current performance."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved → {REPORT_FILE}")
    print()
    return report


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieval quality benchmark for HH-Goa-RAG multilingual pipeline"
    )
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup requests (default: 5)")
    parser.add_argument("--queries-per-lang", type=int, default=2,
                        help="Number of queries per language (default: 2, total=26)")
    args = parser.parse_args()

    run_benchmark(warmup=args.warmup, queries_per_lang=args.queries_per_lang)
