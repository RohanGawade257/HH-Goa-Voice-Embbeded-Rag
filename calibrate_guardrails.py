import json
import statistics
from pathlib import Path

from pipeline import RAGEngine


# ============================================================
# CONFIG
# ============================================================

QUERY_FILE = Path("data/hindi_sample_1000.jsonl")

SCORE_BUCKETS = [
    (0.00, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01),
]


# ============================================================
# LOAD QUERIES
# ============================================================

def load_queries():

    queries = []

    with open(
        QUERY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        seen = set()

        for line in f:

            item = json.loads(line)

            query_id = item.get("query_id")
            query = item.get("query")

            if query_id is None or not query:
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
# SAFE FLOAT
# ============================================================

def safe_float(value):

    try:
        return float(value)
    except Exception:
        return 0.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG - STEP 13")
    print("GUARDRAIL CALIBRATION")
    print("=" * 70)

    print()
    print("Loading RAG engine...")

    engine = RAGEngine()

    queries = load_queries()

    print()
    print(f"Queries: {len(queries)}")

    if not queries:

        print("No queries found.")

        return

    # ========================================================
    # STORAGE
    # ========================================================

    top1_scores = []
    top3_scores = []
    top5_scores = []

    lexical_scores = []

    grounded_count = 0
    blocked_count = 0

    retrieved_gt_at_1 = 0
    retrieved_gt_at_3 = 0
    retrieved_gt_at_5 = 0

    # score -> counts
    bucket_stats = {}

    for low, high in SCORE_BUCKETS:

        bucket_stats[
            f"{low:.2f}-{high:.2f}"
        ] = {
            "queries": 0,
            "grounded": 0,
            "blocked": 0,
            "gt_retrieved": 0
        }

    examples = []

    # ========================================================
    # PROCESS
    # ========================================================

    for index, item in enumerate(queries, start=1):

        query = item["query"]
        query_id = item["query_id"]

        result = engine.process(query)

        sources = result.get(
            "sources",
            []
        )

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        vector_scores = [
            safe_float(
                source.get("vector_score", 0)
            )
            for source in sources
        ]

        vector_scores.sort(
            reverse=True
        )

        lexical = [
            safe_float(
                source.get("lexical_score", 0)
            )
            for source in sources
        ]

        lexical.sort(
            reverse=True
        )

        top1 = (
            vector_scores[0]
            if len(vector_scores) >= 1
            else 0.0
        )

        top3 = (
            max(vector_scores[:3])
            if vector_scores
            else 0.0
        )

        top5 = (
            max(vector_scores[:5])
            if vector_scores
            else 0.0
        )

        max_lexical = (
            max(lexical)
            if lexical
            else 0.0
        )

        top1_scores.append(top1)
        top3_scores.append(top3)
        top5_scores.append(top5)
        lexical_scores.append(max_lexical)

        # ----------------------------------------------------
        # Ground truth retrieval
        # ----------------------------------------------------

        source_ids = [
            source.get("query_id")
            for source in sources
        ]

        gt1 = query_id in source_ids[:1]
        gt3 = query_id in source_ids[:3]
        gt5 = query_id in source_ids[:5]

        if gt1:
            retrieved_gt_at_1 += 1

        if gt3:
            retrieved_gt_at_3 += 1

        if gt5:
            retrieved_gt_at_5 += 1

        # ----------------------------------------------------
        # Guardrail result
        # ----------------------------------------------------

        grounded = bool(
            result.get("grounded", False)
        )

        blocked = bool(
            result.get("blocked", False)
        )

        if grounded:
            grounded_count += 1

        if blocked:
            blocked_count += 1

        # ----------------------------------------------------
        # Score bucket
        # ----------------------------------------------------

        for low, high in SCORE_BUCKETS:

            if low <= top1 < high:

                key = (
                    f"{low:.2f}-{high:.2f}"
                )

                bucket = bucket_stats[key]

                bucket["queries"] += 1

                if grounded:
                    bucket["grounded"] += 1

                if blocked:
                    bucket["blocked"] += 1

                if gt5:
                    bucket["gt_retrieved"] += 1

                break

        # ----------------------------------------------------
        # Interesting examples
        # ----------------------------------------------------

        if (
            len(examples) < 40
            and gt5
            and blocked
        ):

            examples.append(
                {
                    "query": query,
                    "query_id": query_id,
                    "top1": top1,
                    "lexical": max_lexical,
                    "grounded": grounded,
                    "blocked": blocked,
                    "reason": result.get(
                        "reason",
                        ""
                    ),
                    "gt_at_5": gt5
                }
            )

        if index % 100 == 0:

            print(
                f"[{index}/{len(queries)}]"
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    def percentile(values, p):

        if not values:
            return 0.0

        values = sorted(values)

        position = (
            (len(values) - 1)
            * p
            / 100
        )

        lower = int(position)
        upper = min(
            lower + 1,
            len(values) - 1
        )

        fraction = position - lower

        return (
            values[lower]
            + (
                values[upper]
                - values[lower]
            )
            * fraction
        )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("GUARDRAIL CALIBRATION RESULTS")
    print("=" * 70)

    print()
    print("QUERY COUNTS")
    print("-" * 70)

    print(
        f"Total queries       : {len(queries)}"
    )

    print(
        f"Grounded            : "
        f"{grounded_count} "
        f"({grounded_count / len(queries) * 100:.2f}%)"
    )

    print(
        f"Blocked             : "
        f"{blocked_count} "
        f"({blocked_count / len(queries) * 100:.2f}%)"
    )

    print()
    print("GROUND-TRUTH RETRIEVAL")
    print("-" * 70)

    print(
        f"Ground truth @1     : "
        f"{retrieved_gt_at_1 / len(queries) * 100:.2f}%"
    )

    print(
        f"Ground truth @3     : "
        f"{retrieved_gt_at_3 / len(queries) * 100:.2f}%"
    )

    print(
        f"Ground truth @5     : "
        f"{retrieved_gt_at_5 / len(queries) * 100:.2f}%"
    )

    print()
    print("VECTOR SCORE DISTRIBUTION")
    print("-" * 70)

    print(
        f"Top-1 P50           : "
        f"{percentile(top1_scores, 50):.4f}"
    )

    print(
        f"Top-1 P70           : "
        f"{percentile(top1_scores, 70):.4f}"
    )

    print(
        f"Top-1 P90           : "
        f"{percentile(top1_scores, 90):.4f}"
    )

    print(
        f"Top-1 P95           : "
        f"{percentile(top1_scores, 95):.4f}"
    )

    print(
        f"Top-1 P99           : "
        f"{percentile(top1_scores, 99):.4f}"
    )

    print(
        f"Top-1 Min           : "
        f"{min(top1_scores):.4f}"
    )

    print(
        f"Top-1 Max           : "
        f"{max(top1_scores):.4f}"
    )

    print()
    print("LEXICAL OVERLAP")
    print("-" * 70)

    print(
        f"Lexical P50         : "
        f"{percentile(lexical_scores, 50):.4f}"
    )

    print(
        f"Lexical P70         : "
        f"{percentile(lexical_scores, 70):.4f}"
    )

    print(
        f"Lexical P90         : "
        f"{percentile(lexical_scores, 90):.4f}"
    )

    print(
        f"Lexical P95         : "
        f"{percentile(lexical_scores, 95):.4f}"
    )

    # ========================================================
    # BUCKET ANALYSIS
    # ========================================================

    print()
    print("SCORE BUCKET ANALYSIS")
    print("-" * 70)

    print(
        f"{'Score':<12}"
        f"{'Queries':<10}"
        f"{'Grounded':<12}"
        f"{'Blocked':<10}"
        f"{'GT@5':<10}"
        f"{'GT@5 %':<10}"
    )

    for key, stats in bucket_stats.items():

        count = stats["queries"]

        if count == 0:
            continue

        gt_percent = (
            stats["gt_retrieved"]
            / count
            * 100
        )

        print(
            f"{key:<12}"
            f"{count:<10}"
            f"{stats['grounded']:<12}"
            f"{stats['blocked']:<10}"
            f"{stats['gt_retrieved']:<10}"
            f"{gt_percent:<10.2f}"
        )

    # ========================================================
    # FALSE REFUSAL ANALYSIS
    # ========================================================

    print()
    print("POTENTIAL FALSE REFUSALS")
    print("-" * 70)

    false_refusals = 0

    for example in examples:

        false_refusals += 1

        print()
        print(
            f"Query: {example['query']}"
        )

        print(
            f"Top-1 score: "
            f"{example['top1']:.4f}"
        )

        print(
            f"Lexical: "
            f"{example['lexical']:.4f}"
        )

        print(
            f"GT@5: "
            f"{example['gt_at_5']}"
        )

        print(
            f"Reason: "
            f"{example['reason']}"
        )

    print()
    print(
        f"Potential false refusals shown: "
        f"{false_refusals}"
    )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    print()
    print("=" * 70)
    print("CALIBRATION INTERPRETATION")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT change the guardrail threshold yet."
    )

    print(
        "Use the score buckets above to determine "
        "where legitimate retrieved answers are "
        "currently being blocked."
    )

    print()
    print(
        "The next step will combine:"
    )

    print(
        "  1. Vector similarity"
    )

    print(
        "  2. Lexical overlap"
    )

    print(
        "  3. Ground-truth retrieval"
    )

    print(
        "  4. Answer evidence"
    )

    print(
        "  5. Off-topic / unsafe detection"
    )

    print(
        "into a calibrated evidence gate."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()