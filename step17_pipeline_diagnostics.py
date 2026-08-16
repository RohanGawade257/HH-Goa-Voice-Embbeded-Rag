import json
import re
import time
from pathlib import Path

from pipeline import RAGEngine


# ============================================================
# CONFIG
# ============================================================

QUERY_FILE = Path(
    "data/hindi_sample_1000.jsonl"
)

MAX_QUERIES = 1000


# ============================================================
# TEXT
# ============================================================

WORD_RE = re.compile(
    r"\w+",
    re.UNICODE
)


def normalize(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^\w\u0900-\u097F\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text):

    return set(
        WORD_RE.findall(
            normalize(text)
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_queries():

    if not QUERY_FILE.exists():

        raise FileNotFoundError(
            f"Missing: {QUERY_FILE}"
        )

    queries = []

    seen = set()

    with open(
        QUERY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

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

            if len(queries) >= MAX_QUERIES:
                break

    return queries


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG - STEP 17")
    print("PIPELINE DIAGNOSTICS")
    print("=" * 70)

    print()
    print("Loading RAG engine...")

    engine = RAGEngine()

    queries = load_queries()

    print()
    print(
        f"Queries: {len(queries)}"
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    total = 0

    retrieval_gt1 = 0
    retrieval_gt5 = 0
    retrieval_gt10 = 0

    rerank_gt1 = 0
    rerank_gt3 = 0

    allowed = 0
    blocked = 0

    # Correct retrieval but blocked
    retrieval_correct_blocked = 0

    # Correct reranked result but blocked
    rerank_correct_blocked = 0

    # Correct retrieval and allowed
    correct_and_allowed = 0

    # Retrieval failure
    retrieval_failure = 0

    # ========================================================
    # CASE COLLECTION
    # ========================================================

    retrieval_failures = []
    rerank_failures = []
    blocked_correct_cases = []

    # ========================================================
    # RUN
    # ========================================================

    print()
    print("Running diagnostics...")
    print()

    for index, item in enumerate(
        queries,
        start=1
    ):

        query = item["query"]

        query_id = item[
            "query_id"
        ]

        result = engine.process(
            query
        )

        sources = result.get(
            "sources",
            []
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # pipeline.py currently returns only the final
        # reranked Top-3.
        #
        # Therefore we can measure rerank recall here,
        # but NOT original Qdrant Top-20 recall.
        # ----------------------------------------------------

        retrieved_ids = [

            source.get(
                "query_id"
            )

            for source in sources
        ]

        # ----------------------------------------------------
        # RERANK RECALL
        # ----------------------------------------------------

        if query_id in retrieved_ids[:1]:

            rerank_gt1 += 1

        if query_id in retrieved_ids[:3]:

            rerank_gt3 += 1

        # ----------------------------------------------------
        # CURRENT PIPELINE DECISION
        # ----------------------------------------------------

        is_blocked = bool(
            result.get(
                "blocked",
                False
            )
        )

        if is_blocked:

            blocked += 1

        else:

            allowed += 1

        # ----------------------------------------------------
        # BLOCKED BUT CORRECT RESULT EXISTS
        # ----------------------------------------------------

        if (
            is_blocked
            and query_id in retrieved_ids
        ):

            rerank_correct_blocked += 1

            if len(
                blocked_correct_cases
            ) < 50:

                blocked_correct_cases.append(
                    {
                        "query": query,
                        "query_id": query_id,
                        "reason": result.get(
                            "reason",
                            ""
                        ),
                        "sources": len(
                            sources
                        )
                    }
                )

        # ----------------------------------------------------
        # CORRECT + ALLOWED
        # ----------------------------------------------------

        if (
            not is_blocked
            and query_id in retrieved_ids
        ):

            correct_and_allowed += 1

        # ----------------------------------------------------
        # RERANK FAILURE
        # ----------------------------------------------------

        if query_id not in retrieved_ids:

            rerank_failures.append(
                {
                    "query": query,
                    "query_id": query_id,
                    "top_ids": retrieved_ids
                }
            )

        total += 1

        if index % 100 == 0:

            print(
                f"[{index}/{len(queries)}]"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 17 COMPLETE")
    print("=" * 70)

    print()
    print("QUERY COUNTS")
    print("-" * 70)

    print(
        f"Queries processed : {total}"
    )

    print()
    print("CURRENT FINAL RESULTS")
    print("-" * 70)

    print(
        f"Allowed           : "
        f"{allowed} "
        f"({allowed / total * 100:.2f}%)"
    )

    print(
        f"Blocked           : "
        f"{blocked} "
        f"({blocked / total * 100:.2f}%)"
    )

    print()
    print("RERANK RECALL")
    print("-" * 70)

    print(
        f"Rerank Recall@1   : "
        f"{rerank_gt1 / total * 100:.2f}%"
    )

    print(
        f"Rerank Recall@3   : "
        f"{rerank_gt3 / total * 100:.2f}%"
    )

    print()
    print("GATE DIAGNOSTICS")
    print("-" * 70)

    print(
        f"Correct result + allowed : "
        f"{correct_and_allowed}"
    )

    print(
        f"Correct result + blocked : "
        f"{rerank_correct_blocked}"
    )

    print(
        f"Rerank result missing    : "
        f"{len(rerank_failures)}"
    )

    print()
    print(
        "Correct result blocked % : "
        f"{rerank_correct_blocked / total * 100:.2f}%"
    )

    print()
    print("IMPORTANT")
    print("-" * 70)

    print(
        "pipeline.py currently exposes only the final "
        "reranked Top-3 sources."
    )

    print(
        "Therefore this script cannot independently "
        "recalculate original Qdrant Recall@5."
    )

    print(
        "The next architecture change should expose "
        "both Top-20 retrieval and Top-3 reranking."
    )

    # ========================================================
    # BLOCKED CORRECT CASES
    # ========================================================

    print()
    print(
        "CORRECT RERANKED RESULT BUT BLOCKED"
    )

    print("-" * 70)

    if not blocked_correct_cases:

        print(
            "None found."
        )

    else:

        for case in blocked_correct_cases[:20]:

            print()

            print(
                f"Query: {case['query']}"
            )

            print(
                f"Query ID: "
                f"{case['query_id']}"
            )

            print(
                f"Reason: "
                f"{case['reason']}"
            )

            print(
                f"Sources: "
                f"{case['sources']}"
            )

    # ========================================================
    # RERANK FAILURES
    # ========================================================

    print()
    print(
        "RERANK FAILURES"
    )

    print("-" * 70)

    for case in rerank_failures[:20]:

        print()

        print(
            f"Query: {case['query']}"
        )

        print(
            f"Expected query_id: "
            f"{case['query_id']}"
        )

        print(
            f"Retrieved IDs: "
            f"{case['top_ids']}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":

    main()