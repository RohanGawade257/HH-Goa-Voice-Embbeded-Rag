import json
import re
import time
import statistics
from pathlib import Path

from pipeline import RAGEngine


# ============================================================
# CONFIG
# ============================================================

QUERY_FILE = Path("data/hindi_sample_1000.jsonl")

MAX_QUERIES = 1000

PRINT_EVERY = 100

# Minimum proportion of meaningful answer tokens
# that should appear in retrieved evidence.
MIN_EVIDENCE_OVERLAP = 0.30

# We don't want tiny/common words to dominate the score.
MIN_TOKEN_LENGTH = 2


# ============================================================
# TEXT UTILITIES
# ============================================================

WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:

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


def tokenize(text: str):

    text = normalize(text)

    tokens = WORD_RE.findall(text)

    return {
        token
        for token in tokens
        if len(token) >= MIN_TOKEN_LENGTH
    }


def evidence_overlap(answer: str, evidence: str) -> float:

    answer_tokens = tokenize(answer)
    evidence_tokens = tokenize(evidence)

    if not answer_tokens:
        return 0.0

    return (
        len(answer_tokens & evidence_tokens)
        / len(answer_tokens)
    )


def best_evidence_overlap(answer, sources):

    if not sources:
        return 0.0, None

    best_score = 0.0
    best_source = None

    for source in sources:

        text = source.get("text", "")

        score = evidence_overlap(
            answer,
            text
        )

        if score > best_score:

            best_score = score
            best_source = source

    return best_score, best_source


# ============================================================
# LOAD QUERIES
# ============================================================

def load_queries():

    if not QUERY_FILE.exists():

        raise FileNotFoundError(
            f"Missing query file: {QUERY_FILE}"
        )

    queries = []

    seen = set()

    with open(
        QUERY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            query_id = item.get("query_id")
            query = item.get("query")

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
# PERCENTILE
# ============================================================

def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    index = int(
        round(
            (p / 100)
            * (len(values) - 1)
        )
    )

    return values[index]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG - STEP 16")
    print("ANSWER QUALITY + GROUNDING BENCHMARK")
    print("=" * 70)

    print()
    print("Loading RAG engine...")

    engine = RAGEngine()

    print()
    print("Loading benchmark queries...")

    queries = load_queries()

    print(
        f"Queries: {len(queries)}"
    )

    if not queries:

        print(
            "No benchmark queries found."
        )

        return

    # ========================================================
    # METRICS
    # ========================================================

    total = 0

    retrieval_success = 0

    grounded = 0
    blocked = 0

    grounded_with_evidence = 0
    grounded_without_evidence = 0

    false_refusal = 0

    potential_unsupported = 0

    overlap_scores = []

    total_times = []
    embedding_times = []
    qdrant_times = []
    rerank_times = []
    answer_times = []

    # ========================================================
    # DEBUG CASES
    # ========================================================

    false_refusal_cases = []
    unsupported_cases = []
    blocked_cases = []

    # ========================================================
    # BENCHMARK
    # ========================================================

    print()
    print("Running benchmark...")
    print()

    benchmark_start = time.perf_counter()

    for index, item in enumerate(
        queries,
        start=1
    ):

        query = item["query"]
        query_id = item["query_id"]

        start = time.perf_counter()

        result = engine.process(
            query
        )

        total_ms = (
            time.perf_counter()
            - start
        ) * 1000

        total += 1

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        answer = result.get(
            "answer",
            ""
        )

        is_grounded = bool(
            result.get(
                "grounded",
                False
            )
        )

        is_blocked = bool(
            result.get(
                "blocked",
                False
            )
        )

        sources = result.get(
            "sources",
            []
        )

        # ----------------------------------------------------
        # TIMINGS
        # ----------------------------------------------------

        timings = result.get(
            "timings",
            {}
        )

        total_times.append(
            total_ms
        )

        embedding_times.append(
            timings.get(
                "embedding_ms",
                0.0
            )
        )

        qdrant_times.append(
            timings.get(
                "qdrant_ms",
                0.0
            )
        )

        rerank_times.append(
            timings.get(
                "rerank_ms",
                0.0
            )
        )

        answer_times.append(
            timings.get(
                "answer_ms",
                0.0
            )
        )

        # ----------------------------------------------------
        # GROUND TRUTH RETRIEVAL
        # ----------------------------------------------------

        retrieved_query_ids = [
            source.get("query_id")
            for source in sources
        ]

        gt5 = (
            query_id
            in retrieved_query_ids[:5]
        )

        if gt5:

            retrieval_success += 1

        # ----------------------------------------------------
        # GROUNDING
        # ----------------------------------------------------

        if is_grounded:

            grounded += 1

        if is_blocked:

            blocked += 1

            blocked_cases.append(
                {
                    "query": query,
                    "query_id": query_id,
                    "reason": result.get(
                        "reason",
                        ""
                    )
                }
            )

        # ----------------------------------------------------
        # FALSE REFUSAL
        # ----------------------------------------------------

        if (
            is_blocked
            and gt5
        ):

            false_refusal += 1

            if len(false_refusal_cases) < 50:

                false_refusal_cases.append(
                    {
                        "query": query,
                        "query_id": query_id,
                        "reason": result.get(
                            "reason",
                            ""
                        )
                    }
                )

        # ----------------------------------------------------
        # ANSWER EVIDENCE
        # ----------------------------------------------------

        if is_grounded and answer:

            overlap, best_source = (
                best_evidence_overlap(
                    answer,
                    sources
                )
            )

            overlap_scores.append(
                overlap
            )

            if overlap >= MIN_EVIDENCE_OVERLAP:

                grounded_with_evidence += 1

            else:

                grounded_without_evidence += 1

                if len(unsupported_cases) < 50:

                    unsupported_cases.append(
                        {
                            "query": query,
                            "answer": answer,
                            "overlap": overlap,
                            "source": (
                                best_source.get(
                                    "text",
                                    ""
                                )
                                if best_source
                                else ""
                            )
                        }
                    )

        # ----------------------------------------------------
        # POTENTIAL UNSUPPORTED ANSWER
        # ----------------------------------------------------

        if (
            is_grounded
            and answer
            and sources
        ):

            overlap, _ = (
                best_evidence_overlap(
                    answer,
                    sources
                )
            )

            if overlap < MIN_EVIDENCE_OVERLAP:

                potential_unsupported += 1

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if index % PRINT_EVERY == 0:

            print(
                f"[{index}/{len(queries)}]"
            )

    benchmark_time = (
        time.perf_counter()
        - benchmark_start
    )

    # ========================================================
    # RESULTS
    # ========================================================

    retrieval_percent = (
        retrieval_success
        / total
        * 100
    )

    grounded_percent = (
        grounded
        / total
        * 100
    )

    blocked_percent = (
        blocked
        / total
        * 100
    )

    false_refusal_percent = (
        false_refusal
        / total
        * 100
    )

    if grounded:

        grounded_evidence_percent = (
            grounded_with_evidence
            / grounded
            * 100
        )

    else:

        grounded_evidence_percent = 0.0

    unsupported_percent = (
        potential_unsupported
        / total
        * 100
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 16 COMPLETE")
    print("=" * 70)

    print()
    print("QUERY COUNTS")
    print("-" * 70)

    print(
        f"Queries processed       : {total}"
    )

    print(
        f"Grounded answers        : "
        f"{grounded} "
        f"({grounded_percent:.2f}%)"
    )

    print(
        f"Blocked answers         : "
        f"{blocked} "
        f"({blocked_percent:.2f}%)"
    )

    print()
    print("RETRIEVAL")
    print("-" * 70)

    print(
        f"Ground truth @5         : "
        f"{retrieval_success} "
        f"({retrieval_percent:.2f}%)"
    )

    print(
        f"False refusals          : "
        f"{false_refusal} "
        f"({false_refusal_percent:.2f}%)"
    )

    print()
    print("ANSWER EVIDENCE")
    print("-" * 70)

    print(
        f"Grounded with evidence  : "
        f"{grounded_with_evidence} "
        f"({grounded_evidence_percent:.2f}% "
        f"of grounded)"
    )

    print(
        f"Grounded without strong "
        f"evidence                : "
        f"{grounded_without_evidence}"
    )

    print(
        f"Potential unsupported   : "
        f"{potential_unsupported} "
        f"({unsupported_percent:.2f}%)"
    )

    print(
        f"Evidence overlap P50    : "
        f"{percentile(overlap_scores, 50):.3f}"
    )

    print(
        f"Evidence overlap P70    : "
        f"{percentile(overlap_scores, 70):.3f}"
    )

    print(
        f"Evidence overlap P90    : "
        f"{percentile(overlap_scores, 90):.3f}"
    )

    print()
    print("LATENCY")
    print("-" * 70)

    print(
        f"Total P50              : "
        f"{percentile(total_times, 50):.2f} ms"
    )

    print(
        f"Total P70              : "
        f"{percentile(total_times, 70):.2f} ms"
    )

    print(
        f"Total P100             : "
        f"{max(total_times):.2f} ms"
    )

    print(
        f"Embedding P50          : "
        f"{percentile(embedding_times, 50):.2f} ms"
    )

    print(
        f"Qdrant P50             : "
        f"{percentile(qdrant_times, 50):.2f} ms"
    )

    print(
        f"Rerank P50             : "
        f"{percentile(rerank_times, 50):.2f} ms"
    )

    print(
        f"Answer P50             : "
        f"{percentile(answer_times, 50):.2f} ms"
    )

    print()
    print("BENCHMARK RUNTIME")
    print("-" * 70)

    print(
        f"Total benchmark time   : "
        f"{benchmark_time:.2f} seconds"
    )

    # ========================================================
    # FALSE REFUSALS
    # ========================================================

    print()
    print("FALSE REFUSAL EXAMPLES")
    print("-" * 70)

    if not false_refusal_cases:

        print(
            "None found."
        )

    else:

        for case in false_refusal_cases[:20]:

            print()
            print(
                f"Query: {case['query']}"
            )

            print(
                f"Reason: {case['reason']}"
            )

    # ========================================================
    # UNSUPPORTED ANSWERS
    # ========================================================

    print()
    print("POTENTIAL UNSUPPORTED ANSWERS")
    print("-" * 70)

    if not unsupported_cases:

        print(
            "None found."
        )

    else:

        for case in unsupported_cases[:20]:

            print()
            print(
                f"Query: {case['query']}"
            )

            print(
                f"Answer: {case['answer']}"
            )

            print(
                f"Evidence overlap: "
                f"{case['overlap']:.3f}"
            )

    # ========================================================
    # STATUS
    # ========================================================

    print()
    print("ASSESSMENT")
    print("-" * 70)

    if (
        false_refusal_percent <= 5
        and unsupported_percent <= 5
    ):

        print(
            "STATUS: GOOD"
        )

        print(
            "The current evidence gate provides "
            "reasonable retrieval and answer grounding."
        )

    elif false_refusal_percent <= 5:

        print(
            "STATUS: NEEDS ANSWER-GROUNDING REVIEW"
        )

        print(
            "Retrieval is acceptable, but some "
            "grounded answers need stronger evidence."
        )

    else:

        print(
            "STATUS: NEEDS RETRIEVAL/GATE REVIEW"
        )

        print(
            "Too many legitimate queries are "
            "being refused."
        )

    print()
    print(
        "NOTE:"
    )

    print(
        "This benchmark does not use an LLM."
    )

    print(
        "Evidence overlap is a deterministic "
        "diagnostic signal, not a substitute for "
        "human answer-quality evaluation."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":

    main()