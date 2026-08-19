"""
Official-format benchmark using the production RAGEngine singleton.
This is the authoritative measurement — it uses the same code path
as the production API.

Format matches the official benchmark specification:
  avg / p50 / p95 / p99 per stage
  p95 gate against LATENCY_BUDGET_MS
  sys.exit(1) on failure
"""
import json
import statistics
import sys
from pathlib import Path

from app.pipeline import RAGEngine

# ============================================================
# BUDGET (matches official format)
# ============================================================

LATENCY_BUDGET_MS = 200  # post-STT p95 gate

QUERY_FILE = Path("data/hindi_sample_1000.jsonl")


# ============================================================
# QUERIES
# ============================================================

def load_queries(max_q: int = 500) -> list:
    if not QUERY_FILE.exists():
        return []
    queries, seen = [], set()
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item.get("query", "").strip()
            qid = item.get("query_id")
            if q and qid and qid not in seen:
                seen.add(qid)
                queries.append(q)
            if len(queries) >= max_q:
                break
    return queries


# ============================================================
# PERCENTILE (official formula)
# ============================================================

def percentile(values: list, pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    print("=" * 65)
    print("HH Goa 2026 — Voice RAG | Production Pipeline Benchmark")
    print(f"Budget: {LATENCY_BUDGET_MS} ms (post-STT, p95 gate)")
    print("=" * 65)

    # --------------------------------------------------------
    # INIT (single singleton — same as production API)
    # --------------------------------------------------------

    print("\nLoading RAG engine...")
    engine = RAGEngine()

    queries = load_queries(500)
    print(f"Query pool : {len(queries)} unique queries")
    print(f"Benchmark  : {n} iterations")

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    print("\nWarming up (model load + first inference)...")
    for q in queries[:10]:
        engine.process(q)
    print("Warmup complete.")

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    embed_ms, search_ms, rerank_ms, answer_ms, total_ms = [], [], [], [], []

    print(f"\nRunning {n} queries...")

    for i in range(n):
        result = engine.process(queries[i % len(queries)])
        t = result["timings"]
        embed_ms.append(t["embedding_ms"])
        search_ms.append(t["qdrant_ms"])
        rerank_ms.append(t["rerank_ms"])
        answer_ms.append(t["answer_ms"])
        total_ms.append(t["total_ms"])

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(f"\nRan {n} queries\n")

    print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'p100':>8}   (ms)")
    print("-" * 60)

    stages = [
        ("embed",   embed_ms),
        ("search",  search_ms),
        ("rerank",  rerank_ms),
        ("answer",  answer_ms),
        ("total",   total_ms),
    ]

    for name, vals in stages:
        print(
            f"{name:<12}"
            f"{statistics.mean(vals):>8.2f}"
            f"{percentile(vals, 50):>8.2f}"
            f"{percentile(vals, 95):>8.2f}"
            f"{percentile(vals, 99):>8.2f}"
            f"{max(vals):>8.2f}"
        )

    p95_total = percentile(total_ms, 95)
    print(f"\nLatency budget: {LATENCY_BUDGET_MS} ms  |  p95 total: {p95_total:.2f} ms  |  p100 (max): {max(total_ms):.2f} ms")

    if p95_total <= LATENCY_BUDGET_MS:
        print("PASS: within budget")
    else:
        print("FAIL: over budget -- optimize embedding or search stage")
        # Identify bottleneck
        for name, vals in stages[:2]:
            p = percentile(vals, 95)
            print(f"  {name} p95: {p:.2f} ms")
        sys.exit(1)


if __name__ == "__main__":
    main()
