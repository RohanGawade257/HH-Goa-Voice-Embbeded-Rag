import json
import time
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.pipeline import RAGEngine


# ============================================================
# HH GOA RAG - STEP 19
# TOP-20 vs TOP-3 RETRIEVAL DIAGNOSTICS
#
# Purpose:
#
# Benchmark:
#
# Query
#   ↓
# Embedding
#   ↓
# Qdrant Top-20
#   ↓
# Fast Reranker
#   ↓
# Top-3
#
# We measure:
#
# 1. Original Qdrant Top-20 Recall
# 2. Reranked Top-3 Recall
# 3. Reranking losses
# 4. Correct result position
# 5. Retrieval latency
#
# This step DOES NOT modify pipeline.py
# ============================================================


# ============================================================
# CONFIG
# ============================================================

BENCHMARK_DIR = Path("data")

POSSIBLE_FILES = [
    Path("data/benchmark_queries.json"),
    Path("data/queries.json"),
    Path("benchmark_queries.json"),
    Path("queries.json"),
]

TOP20 = 20
TOP3 = 3


# ============================================================
# LOAD BENCHMARK
# ============================================================

def find_benchmark_file():

    for path in POSSIBLE_FILES:

        if path.exists():
            return path

    # Fallback: search recursively
    candidates = []

    for pattern in [
        "*benchmark*.json",
        "*queries*.json",
    ]:

        candidates.extend(
            Path(".").rglob(pattern)
        )

    candidates = [
        p for p in candidates
        if "qdrant" not in str(p).lower()
    ]

    if candidates:
        return candidates[0]

    return None


def load_queries():
    """
    Load the existing HH Goa benchmark.

    The project benchmark is stored as JSONL:
        data/hindi_sample_1000.jsonl

    Each line contains one JSON record.
    """

    candidates = [
        Path("data/hindi_sample_1000.jsonl"),
        Path("data/benchmark_queries.json"),
        Path("data/queries.json"),
        Path("benchmark_queries.json"),
        Path("queries.json"),
    ]

    benchmark_path = None

    for path in candidates:
        if path.exists():
            benchmark_path = path
            break

    if benchmark_path is None:
        raise FileNotFoundError(
            "\nCould not find benchmark file.\n\n"
            "Expected the existing project benchmark:\n"
            "  data/hindi_sample_1000.jsonl\n"
        )

    print(f"Benchmark file: {benchmark_path}")

    # --------------------------------------------------------
    # JSONL benchmark
    # --------------------------------------------------------

    if benchmark_path.suffix.lower() == ".jsonl":

        queries = []

        with open(
            benchmark_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)

                except json.JSONDecodeError as e:

                    raise ValueError(
                        f"Invalid JSON on line "
                        f"{line_number} of "
                        f"{benchmark_path}: {e}"
                    )

                queries.append(item)

        return queries

    # --------------------------------------------------------
    # Normal JSON
    # --------------------------------------------------------

    with open(
        benchmark_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # Handle either:
    #
    # [
    #   {...},
    #   {...}
    # ]
    #
    # or
    #
    # {
    #   "queries": [...]
    # }

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "queries" in data:
            return data["queries"]

        if "data" in data:
            return data["data"]

    raise ValueError(
        f"Unsupported benchmark format in "
        f"{benchmark_path}"
    )

# ============================================================
# EXTRACT GROUND TRUTH
# ============================================================

def get_query_text(item):

    for key in [
        "query",
        "question",
        "text",
        "query_text"
    ]:

        value = item.get(key)

        if isinstance(value, str):
            return value.strip()

    return ""


def get_ground_truth_id(item):

    for key in [
        "query_id",
        "expected_query_id",
        "ground_truth_query_id",
        "passage_id"
    ]:

        value = item.get(key)

        if value is not None:
            return str(value)

    return None


# ============================================================
# FIND ID IN RESULTS
# ============================================================

def result_ids(results):

    ids = []

    for item in results:

        query_id = item.get(
            "query_id"
        )

        if query_id is not None:
            ids.append(
                str(query_id)
            )

    return ids


def find_rank(results, expected_id):

    if expected_id is None:
        return None

    for index, item in enumerate(
        results,
        start=1
    ):

        query_id = item.get(
            "query_id"
        )

        if query_id is not None:

            if str(query_id) == str(
                expected_id
            ):

                return index

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG - STEP 19")
    print("TOP-20 vs TOP-3 RETRIEVAL DIAGNOSTICS")
    print("=" * 70)

    print()
    print("Loading RAG engine...")

    engine = RAGEngine()

    queries = load_queries()

    print(
        f"Queries: {len(queries)}"
    )

    print()

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print("Running 10 warmup queries...")

    for item in queries[:10]:

        query = get_query_text(item)

        if query:
            engine.process(query)

    print("Warmup complete.")
    print()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = 0

    top20_hits = 0
    top3_hits = 0

    top20_miss = 0
    rerank_loss = 0

    top1_hits = 0

    ranks = []

    retrieval_times = []
    rerank_times = []
    total_times = []

    top20_only_examples = []
    complete_failures = []

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print("Running diagnostics...")
    print()

    benchmark_start = time.perf_counter()

    for index, item in enumerate(
        queries,
        start=1
    ):

        query = get_query_text(item)
        expected_id = get_ground_truth_id(item)

        if not query:
            continue

        result = engine.process(query)

        total += 1

        retrieval = result.get(
            "retrieval",
            {}
        )

        top20 = retrieval.get(
            "top20",
            []
        )

        top3 = retrieval.get(
            "top3",
            []
        )

        # ----------------------------------------------------
        # IDs
        # ----------------------------------------------------

        top20_ids = result_ids(top20)
        top3_ids = result_ids(top3)

        rank20 = find_rank(
            top20,
            expected_id
        )

        rank3 = find_rank(
            top3,
            expected_id
        )

        # ----------------------------------------------------
        # Top-20 recall
        # ----------------------------------------------------

        if rank20 is not None:

            top20_hits += 1

            ranks.append(rank20)

            if rank20 == 1:
                top1_hits += 1

        else:

            top20_miss += 1

        # ----------------------------------------------------
        # Top-3 recall
        # ----------------------------------------------------

        if rank3 is not None:

            top3_hits += 1

        # ----------------------------------------------------
        # Reranking loss
        #
        # Correct result existed in Top-20
        # but disappeared from Top-3.
        # ----------------------------------------------------

        if (
            rank20 is not None
            and rank3 is None
        ):

            rerank_loss += 1

            if len(
                top20_only_examples
            ) < 30:

                top20_only_examples.append(
                    {
                        "query": query,
                        "expected_id": expected_id,
                        "top20_rank": rank20,
                        "top3_ids": top3_ids,
                        "top20_ids": top20_ids,
                    }
                )

        # ----------------------------------------------------
        # Complete retrieval failure
        # ----------------------------------------------------

        if (
            rank20 is None
            and len(complete_failures) < 30
        ):

            complete_failures.append(
                {
                    "query": query,
                    "expected_id": expected_id,
                    "top20_ids": top20_ids,
                }
            )

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        timings = result.get(
            "timings",
            {}
        )

        if "qdrant_ms" in timings:
            retrieval_times.append(
                timings["qdrant_ms"]
            )

        if "rerank_ms" in timings:
            rerank_times.append(
                timings["rerank_ms"]
            )

        if "total_ms" in timings:
            total_times.append(
                timings["total_ms"]
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if index % 100 == 0:

            print(
                f"[{index}/{len(queries)}]"
            )

    benchmark_time = (
        time.perf_counter()
        - benchmark_start
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    def pct(value):

        if total == 0:
            return 0.0

        return (
            value / total
        ) * 100

    def percentile(values, p):

        if not values:
            return 0.0

        values = sorted(values)

        position = (
            len(values) - 1
        ) * p

        lower = int(position)

        upper = min(
            lower + 1,
            len(values) - 1
        )

        fraction = (
            position - lower
        )

        return (
            values[lower]
            + (
                values[upper]
                - values[lower]
            )
            * fraction
        )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 19 COMPLETE")
    print("=" * 70)

    print()
    print("QUERY COUNTS")
    print("-" * 70)

    print(
        f"Queries processed : {total}"
    )

    print()
    print("RETRIEVAL QUALITY")
    print("-" * 70)

    print(
        f"Qdrant Top-1 Recall : "
        f"{pct(top1_hits):.2f}%"
    )

    print(
        f"Qdrant Top-20 Recall: "
        f"{pct(top20_hits):.2f}%"
    )

    print(
        f"Top-20 misses       : "
        f"{top20_miss} "
        f"({pct(top20_miss):.2f}%)"
    )

    print(
        f"Reranked Top-3 Recall: "
        f"{pct(top3_hits):.2f}%"
    )

    print()
    print("RERANKING ANALYSIS")
    print("-" * 70)

    print(
        f"Correct in Top-20    : "
        f"{top20_hits}"
    )

    print(
        f"Correct in Top-3     : "
        f"{top3_hits}"
    )

    print(
        f"Reranking losses     : "
        f"{rerank_loss}"
    )

    if top20_hits > 0:

        rerank_loss_pct = (
            rerank_loss
            / top20_hits
        ) * 100

    else:

        rerank_loss_pct = 0.0

    print(
        f"Rerank loss rate     : "
        f"{rerank_loss_pct:.2f}%"
    )

    # ========================================================
    # RANK DISTRIBUTION
    # ========================================================

    print()
    print("CORRECT RESULT POSITION IN TOP-20")
    print("-" * 70)

    if ranks:

        for start, end in [
            (1, 1),
            (2, 3),
            (4, 5),
            (6, 10),
            (11, 20)
        ]:

            count = sum(
                1
                for rank in ranks
                if start <= rank <= end
            )

            print(
                f"Rank {start}-{end:<2} : "
                f"{count:4d} "
                f"({pct(count):.2f}%)"
            )

    # ========================================================
    # LATENCY
    # ========================================================

    print()
    print("LATENCY")
    print("-" * 70)

    if retrieval_times:

        print(
            f"Qdrant P50  : "
            f"{percentile(retrieval_times, 0.50):.2f} ms"
        )

        print(
            f"Qdrant P95  : "
            f"{percentile(retrieval_times, 0.95):.2f} ms"
        )

    if rerank_times:

        print(
            f"Rerank P50  : "
            f"{percentile(rerank_times, 0.50):.2f} ms"
        )

        print(
            f"Rerank P95  : "
            f"{percentile(rerank_times, 0.95):.2f} ms"
        )

    if total_times:

        print(
            f"Total P50   : "
            f"{percentile(total_times, 0.50):.2f} ms"
        )

        print(
            f"Total P95   : "
            f"{percentile(total_times, 0.95):.2f} ms"
        )

        print(
            f"Total P100  : "
            f"{max(total_times):.2f} ms"
        )

    print()
    print(
        f"Benchmark runtime : "
        f"{benchmark_time:.2f} seconds"
    )

    # ========================================================
    # RERANK LOSSES
    # ========================================================

    print()
    print("CORRECT IN TOP-20 BUT LOST DURING RERANKING")
    print("-" * 70)

    if not top20_only_examples:

        print("None.")

    else:

        for item in top20_only_examples:

            print()
            print(
                f"Query: {item['query']}"
            )

            print(
                f"Expected query_id: "
                f"{item['expected_id']}"
            )

            print(
                f"Correct Top-20 rank: "
                f"{item['top20_rank']}"
            )

            print(
                f"Top-3 IDs: "
                f"{item['top3_ids']}"
            )

    # ========================================================
    # COMPLETE FAILURES
    # ========================================================

    print()
    print("NOT FOUND IN TOP-20")
    print("-" * 70)

    if not complete_failures:

        print("None.")

    else:

        for item in complete_failures[:20]:

            print()
            print(
                f"Query: {item['query']}"
            )

            print(
                f"Expected query_id: "
                f"{item['expected_id']}"
            )

            print(
                f"Retrieved Top-20 IDs: "
                f"{item['top20_ids']}"
            )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    print()

    if top20_hits > 0:

        print(
            f"Top-20 successfully retrieves "
            f"the ground-truth result for "
            f"{pct(top20_hits):.2f}% of queries."
        )

    print(
        f"Only {pct(top3_hits):.2f}% survive "
        f"into the final Top-3."
    )

    print()

    if rerank_loss_pct > 10:

        print(
            "WARNING: Reranker is losing a "
            "significant number of correct results."
        )

        print(
            "Priority: improve reranking "
            "before changing the evidence gate."
        )

    elif pct(top20_miss) > 20:

        print(
            "WARNING: Qdrant retrieval itself "
            "is the main bottleneck."
        )

        print(
            "Priority: improve embeddings/chunking "
            "or retrieval strategy."
        )

    else:

        print(
            "Retrieval pipeline appears reasonably "
            "stable. Continue to evidence-gate analysis."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT change thresholds automatically "
        "from this script."
    )

    print(
        "This step is diagnostic only."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()