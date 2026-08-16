import json
from collections import Counter
from pathlib import Path

from pipeline import RAGEngine


QUERY_FILE = Path(
    "data/hindi_sample_1000.jsonl"
)

OUTPUT_FILE = Path(
    "data/guardrail_diagnostics.jsonl"
)


def load_queries():

    queries = []

    with open(
        QUERY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            item = json.loads(line)

            if not item.get("query"):
                continue

            if item.get("query_id") is None:
                continue

            queries.append(item)

    return queries


def main():

    print("=" * 70)
    print("HH GOA RAG - STEP 13")
    print("GUARDRAIL DIAGNOSTICS")
    print("=" * 70)

    engine = RAGEngine()

    queries = load_queries()

    print()
    print(
        f"Queries: {len(queries)}"
    )

    reasons = Counter()

    blocked_examples = []

    grounded_examples = []

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for index, item in enumerate(
            queries,
            start=1
        ):

            result = engine.process(
                item["query"]
            )

            reason = result.get(
                "reason",
                "unknown"
            )

            blocked = result.get(
                "blocked",
                False
            )

            grounded = result.get(
                "grounded",
                False
            )

            reasons[reason] += 1

            record = {
                "query_id": item[
                    "query_id"
                ],

                "query": item[
                    "query"
                ],

                "answer": result.get(
                    "answer"
                ),

                "grounded": grounded,

                "blocked": blocked,

                "reason": reason,

                "sources": result.get(
                    "sources",
                    []
                )
            }

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

            if blocked and len(
                blocked_examples
            ) < 20:

                blocked_examples.append(
                    record
                )

            if grounded and len(
                grounded_examples
            ) < 10:

                grounded_examples.append(
                    record
                )

            if index % 100 == 0:

                print(
                    f"[{index}/{len(queries)}]"
                )

    print()
    print("=" * 70)
    print("BLOCK / GROUNDING REASONS")
    print("=" * 70)

    for reason, count in reasons.most_common():

        print(
            f"{reason:<35} {count}"
        )

    print()
    print("=" * 70)
    print("BLOCKED EXAMPLES")
    print("=" * 70)

    for item in blocked_examples:

        print()
        print(
            "Query:",
            item["query"]
        )

        print(
            "Reason:",
            item["reason"]
        )

        print(
            "Sources:",
            len(item["sources"])
        )

    print()
    print("=" * 70)
    print("GROUNDED EXAMPLES")
    print("=" * 70)

    for item in grounded_examples:

        print()
        print(
            "Query:",
            item["query"]
        )

        print(
            "Answer:",
            item["answer"]
        )

        print(
            "Reason:",
            item["reason"]
        )

    print()
    print(
        f"Diagnostics saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()