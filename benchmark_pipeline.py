import json
import time
import statistics
from pathlib import Path

from pipeline import RAGEngine


# ============================================================
# CONFIG
# ============================================================

QUERY_FILE = Path(
    "data/hindi_sample_1000.jsonl"
)

WARMUP_QUERIES = 10


# ============================================================
# PERCENTILE
# ============================================================

def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    index = int(
        round(
            (p / 100) * (len(values) - 1)
        )
    )

    return values[index]


# ============================================================
# LOAD QUERIES
# ============================================================

def load_queries():

    queries = []

    seen = set()

    with open(
        QUERY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            item = json.loads(line)

            query_id = item.get(
                "query_id"
            )

            query = item.get(
                "query"
            )

            if query_id is None:
                continue

            if not query:
                continue

            if query_id in seen:
                continue

            seen.add(query_id)

            queries.append(
                {
                    "query_id": query_id,
                    "query": query
                }
            )

    return queries


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG - STEP 12")
    print("COMPLETE POST-STT PIPELINE BENCHMARK")
    print("=" * 70)

    print()
    print("Loading RAG engine...")

    engine = RAGEngine()

    queries = load_queries()

    print()
    print(
        f"Benchmark queries: {len(queries)}"
    )

    if not queries:

        print(
            "ERROR: No benchmark queries found."
        )

        return

    # ========================================================
    # WARMUP
    # ========================================================

    print()
    print(
        f"Running {WARMUP_QUERIES} warmup queries..."
    )

    for item in queries[
        :WARMUP_QUERIES
    ]:

        engine.process(
            item["query"]
        )

    print("Warmup complete.")

    # ========================================================
    # STORAGE
    # ========================================================

    total_times = []

    embedding_times = []
    qdrant_times = []
    rerank_times = []
    answer_times = []

    recall_1 = 0
    recall_3 = 0
    recall_5 = 0

    grounded_count = 0
    blocked_count = 0

    processed = 0

    # ========================================================
    # BENCHMARK
    # ========================================================

    print()
    print("Running benchmark...")
    print()

    benchmark_start = (
        time.perf_counter()
    )

    for item in queries:

        query_id = item[
            "query_id"
        ]

        query = item[
            "query"
        ]

        # ----------------------------------------------------
        # COMPLETE POST-STT PIPELINE
        # ----------------------------------------------------

        start = time.perf_counter()

        result = engine.process(
            query
        )

        measured_total_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # ----------------------------------------------------
        # PIPELINE TIMINGS
        # ----------------------------------------------------

        timings = result.get(
            "timings",
            {}
        )

        embedding_ms = float(
            timings.get(
                "embedding_ms",
                0
            )
        )

        qdrant_ms = float(
            timings.get(
                "qdrant_ms",
                0
            )
        )

        rerank_ms = float(
            timings.get(
                "rerank_ms",
                0
            )
        )

        answer_ms = float(
            timings.get(
                "answer_ms",
                0
            )
        )

        # Use our outer measurement
        # as the authoritative end-to-end
        # post-STT measurement.
        total_ms = measured_total_ms

        # ----------------------------------------------------
        # STORE LATENCIES
        # ----------------------------------------------------

        total_times.append(
            total_ms
        )

        embedding_times.append(
            embedding_ms
        )

        qdrant_times.append(
            qdrant_ms
        )

        rerank_times.append(
            rerank_ms
        )

        answer_times.append(
            answer_ms
        )

        # ----------------------------------------------------
        # GROUNDING / GUARDRAIL
        # ----------------------------------------------------

        if result.get(
            "grounded",
            False
        ):

            grounded_count += 1

        if result.get(
            "blocked",
            False
        ):

            blocked_count += 1

        # ----------------------------------------------------
        # RETRIEVAL RECALL
        # ----------------------------------------------------

        sources = result.get(
            "sources",
            []
        )

        retrieved_ids = []

        for source in sources:

            if not isinstance(
                source,
                dict
            ):
                continue

            retrieved_query_id = source.get(
                "query_id"
            )

            if retrieved_query_id is not None:

                retrieved_ids.append(
                    retrieved_query_id
                )

        # Recall@1
        if query_id in retrieved_ids[:1]:

            recall_1 += 1

        # Recall@3
        if query_id in retrieved_ids[:3]:

            recall_3 += 1

        # Recall@5
        if query_id in retrieved_ids[:5]:

            recall_5 += 1

        processed += 1

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if processed % 50 == 0:

            print(
                f"[{processed}/{len(queries)}] "
                f"Total={total_ms:.2f} ms"
            )

    benchmark_runtime = (
        time.perf_counter()
        - benchmark_start
    ) * 1000

    # ========================================================
    # STATISTICS
    # ========================================================

    p50 = percentile(
        total_times,
        50
    )

    p70 = percentile(
        total_times,
        70
    )

    p100 = max(
        total_times
    )

    mean = statistics.mean(
        total_times
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 12 COMPLETE")
    print("=" * 70)

    print()
    print("QUERY COUNT")
    print("-" * 70)

    print(
        f"Queries processed : {processed}"
    )

    print()
    print("POST-STT PIPELINE LATENCY")
    print("-" * 70)

    print(
        f"P50  : {p50:.2f} ms"
    )

    print(
        f"P70  : {p70:.2f} ms"
    )

    print(
        f"P100 : {p100:.2f} ms"
    )

    print(
        f"Mean : {mean:.2f} ms"
    )

    print()
    print("PIPELINE COMPONENTS")
    print("-" * 70)

    print(
        f"Embedding P50 : "
        f"{percentile(embedding_times, 50):.2f} ms"
    )

    print(
        f"Qdrant P50    : "
        f"{percentile(qdrant_times, 50):.2f} ms"
    )

    print(
        f"Rerank P50    : "
        f"{percentile(rerank_times, 50):.2f} ms"
    )

    print(
        f"Answer P50    : "
        f"{percentile(answer_times, 50):.2f} ms"
    )

    print()
    print("RETRIEVAL QUALITY")
    print("-" * 70)

    print(
        f"Recall@1 : "
        f"{recall_1 / processed * 100:.2f}%"
    )

    print(
        f"Recall@3 : "
        f"{recall_3 / processed * 100:.2f}%"
    )

    print(
        f"Recall@5 : "
        f"{recall_5 / processed * 100:.2f}%"
    )

    print()
    print("GROUNDING / GUARDRAILS")
    print("-" * 70)

    print(
        f"Grounded : "
        f"{grounded_count / processed * 100:.2f}%"
    )

    print(
        f"Blocked  : "
        f"{blocked_count / processed * 100:.2f}%"
    )

    print()
    print("BENCHMARK RUNTIME")
    print("-" * 70)

    print(
        f"Total benchmark time : "
        f"{benchmark_runtime:.2f} ms"
    )

    print()
    print("LATENCY TARGET")
    print("-" * 70)

    print(
        "Post-STT target : < 200 ms"
    )

    print(
        f"P50             : {p50:.2f} ms"
    )

    print(
        f"P70             : {p70:.2f} ms"
    )

    print(
        f"P100            : {p100:.2f} ms"
    )

    print()

    if p100 < 200:

        print(
            "STATUS: PASS"
        )

        print(
            "Every measured query completed "
            "under the 200 ms post-STT target."
        )

    elif p70 < 200:

        print(
            "STATUS: PARTIAL"
        )

        print(
            "P70 is under 200 ms, but P100 "
            "needs optimization."
        )

    else:

        print(
            "STATUS: FAIL"
        )

        print(
            "The post-STT pipeline exceeds "
            "the 200 ms target."
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()