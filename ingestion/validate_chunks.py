import json
import statistics
from collections import defaultdict


PASSAGES_FILE = "data/processed/passages_1000.jsonl"
CHUNKS_FILE = "data/processed/chunks_1000.jsonl"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():

    print("=" * 65)
    print("HH GOA RAG - STEP 6.5")
    print("Chunk Quality Validation")
    print("=" * 65)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    passages = load_jsonl(PASSAGES_FILE)
    chunks = load_jsonl(CHUNKS_FILE)

    print(f"\nClean passages : {len(passages)}")
    print(f"Chunks         : {len(chunks)}")

    # ---------------------------------------------------------
    # GROUP CHUNKS BY PASSAGE
    # ---------------------------------------------------------

    chunks_by_passage = defaultdict(list)

    for chunk in chunks:
        chunks_by_passage[
            chunk["passage_id"]
        ].append(chunk)

    # ---------------------------------------------------------
    # 1. COVERAGE
    # ---------------------------------------------------------

    passage_ids = {
        passage["passage_id"]
        for passage in passages
    }

    chunked_passage_ids = set(
        chunks_by_passage.keys()
    )

    missing_passages = (
        passage_ids - chunked_passage_ids
    )

    print("\n" + "-" * 65)
    print("1. PASSAGE COVERAGE")
    print("-" * 65)

    print(
        f"Passages with chunks : "
        f"{len(chunked_passage_ids)} / {len(passage_ids)}"
    )

    print(
        f"Missing passages     : "
        f"{len(missing_passages)}"
    )

    # ---------------------------------------------------------
    # 2. SELECTED PASSAGE COVERAGE
    # ---------------------------------------------------------

    selected_passages = [
        p for p in passages
        if p["is_selected"]
    ]

    selected_with_chunks = [
        p for p in selected_passages
        if p["passage_id"] in chunks_by_passage
    ]

    print("\n" + "-" * 65)
    print("2. GROUND-TRUTH COVERAGE")
    print("-" * 65)

    print(
        f"Selected passages : "
        f"{len(selected_passages)}"
    )

    print(
        f"Selected preserved: "
        f"{len(selected_with_chunks)}"
    )

    if selected_passages:

        coverage = (
            len(selected_with_chunks)
            / len(selected_passages)
        ) * 100

        print(
            f"Coverage          : "
            f"{coverage:.2f}%"
        )

    # ---------------------------------------------------------
    # 3. CHUNK SIZE
    # ---------------------------------------------------------

    chunk_lengths = [
        chunk["word_count"]
        for chunk in chunks
    ]

    print("\n" + "-" * 65)
    print("3. CHUNK SIZE DISTRIBUTION")
    print("-" * 65)

    print(
        f"Minimum : {min(chunk_lengths)} words"
    )

    print(
        f"Maximum : {max(chunk_lengths)} words"
    )

    print(
        f"Average : "
        f"{statistics.mean(chunk_lengths):.2f} words"
    )

    print(
        f"Median  : "
        f"{statistics.median(chunk_lengths):.2f} words"
    )

    # ---------------------------------------------------------
    # 4. CHUNK PERCENTILES
    # ---------------------------------------------------------

    sorted_lengths = sorted(chunk_lengths)

    def percentile(data, percentage):

        index = int(
            len(data) * percentage
        )

        index = min(
            index,
            len(data) - 1
        )

        return data[index]

    print("\nPercentiles:")

    for p in [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99
    ]:

        print(
            f"P{int(p * 100):2d}: "
            f"{percentile(sorted_lengths, p)} words"
        )

    # ---------------------------------------------------------
    # 5. STRATEGY DISTRIBUTION
    # ---------------------------------------------------------

    strategy_counts = defaultdict(int)

    for chunk in chunks:

        strategy_counts[
            chunk["chunk_strategy"]
        ] += 1

    print("\n" + "-" * 65)
    print("4. STRATEGY DISTRIBUTION")
    print("-" * 65)

    for strategy, count in sorted(
        strategy_counts.items()
    ):

        percentage = (
            count / len(chunks)
        ) * 100

        print(
            f"{strategy:<20} "
            f"{count:>6} "
            f"({percentage:.2f}%)"
        )

    # ---------------------------------------------------------
    # 6. CHUNKS PER PASSAGE
    # ---------------------------------------------------------

    chunks_per_passage = [
        len(chunk_list)
        for chunk_list in chunks_by_passage.values()
    ]

    print("\n" + "-" * 65)
    print("5. CHUNKS PER PASSAGE")
    print("-" * 65)

    print(
        f"Minimum : "
        f"{min(chunks_per_passage)}"
    )

    print(
        f"Maximum : "
        f"{max(chunks_per_passage)}"
    )

    print(
        f"Average : "
        f"{statistics.mean(chunks_per_passage):.2f}"
    )

    print(
        f"Median  : "
        f"{statistics.median(chunks_per_passage):.2f}"
    )

    # ---------------------------------------------------------
    # 7. OVERLAP
    # ---------------------------------------------------------

    overlap_chunks = [
        chunk
        for chunk in chunks
        if chunk["has_overlap"]
    ]

    print("\n" + "-" * 65)
    print("6. OVERLAP")
    print("-" * 65)

    print(
        f"Chunks with overlap : "
        f"{len(overlap_chunks)}"
    )

    if chunks:

        overlap_percentage = (
            len(overlap_chunks)
            / len(chunks)
        ) * 100

        print(
            f"Percentage          : "
            f"{overlap_percentage:.2f}%"
        )

    # ---------------------------------------------------------
    # 8. POTENTIAL PROBLEMS
    # ---------------------------------------------------------

    oversized = [
        chunk
        for chunk in chunks
        if chunk["word_count"] > 140
    ]

    tiny_chunks = [
        chunk
        for chunk in chunks
        if chunk["word_count"] < 5
    ]

    print("\n" + "-" * 65)
    print("7. POTENTIAL PROBLEMS")
    print("-" * 65)

    print(
        f"Oversized chunks (>140 words): "
        f"{len(oversized)}"
    )

    print(
        f"Tiny chunks (<5 words)       : "
        f"{len(tiny_chunks)}"
    )

    # ---------------------------------------------------------
    # 9. FINAL VERDICT
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    problems = []

    if missing_passages:
        problems.append(
            f"{len(missing_passages)} passages missing chunks"
        )

    if selected_passages:
        coverage = (
            len(selected_with_chunks)
            / len(selected_passages)
        ) * 100

        if coverage < 100:
            problems.append(
                "Some ground-truth passages were lost"
            )

    if tiny_chunks:
        problems.append(
            f"{len(tiny_chunks)} tiny chunks found"
        )

    if problems:

        print("\n⚠ Validation found issues:")

        for problem in problems:
            print(f"  - {problem}")

    else:

        print(
            "\n✓ No structural problems detected."
        )

    print("\nValidation finished.")


if __name__ == "__main__":
    main()