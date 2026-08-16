import json
import re
import time
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient


# ============================================================
# CONFIG
# ============================================================

COLLECTION_NAME = "hh_goa_rag_hindi"
QDRANT_PATH = "data/qdrant"

QUERY_FILE = "data/hindi_sample_1000.jsonl"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 3

# ------------------------------------------------------------
# CALIBRATED EVIDENCE GATE
# ------------------------------------------------------------

VECTOR_THRESHOLD = 0.60
LEXICAL_THRESHOLD = 0.55
RERANK_THRESHOLD = 0.55


# ============================================================
# TEXT UTILITIES
# ============================================================

WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:

    text = text.lower().strip()

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

    return text


def tokenize(text: str):

    return set(
        WORD_RE.findall(
            normalize(text)
        )
    )


def lexical_overlap(
    query: str,
    document: str
) -> float:

    q_tokens = tokenize(query)
    d_tokens = tokenize(document)

    if not q_tokens or not d_tokens:
        return 0.0

    return (
        len(q_tokens.intersection(d_tokens))
        / len(q_tokens)
    )


def phrase_score(
    query: str,
    document: str
) -> float:

    q = normalize(query)
    d = normalize(document)

    if not q:
        return 0.0

    # Exact query
    if q in d:
        return 1.0

    q_words = q.split()

    if len(q_words) >= 3:

        for size in (3, 4):

            for i in range(
                len(q_words) - size + 1
            ):

                phrase = " ".join(
                    q_words[
                        i:i + size
                    ]
                )

                if phrase in d:
                    return 0.7

    return 0.0


# ============================================================
# FAST RERANK
# ============================================================

def rerank(query, hits):

    if not hits:
        return []

    results = []

    for hit in hits:

        payload = hit.payload or {}

        text = payload.get(
            "text",
            ""
        )

        vector_score = float(
            hit.score
        )

        lexical_score = lexical_overlap(
            query,
            text
        )

        phrase_score_value = phrase_score(
            query,
            text
        )

        final_score = (
            0.70 * vector_score
            + 0.20 * lexical_score
            + 0.10 * phrase_score_value
        )

        results.append(
            (
                final_score,
                vector_score,
                lexical_score,
                phrase_score_value,
                hit
            )
        )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results[:TOP_K_FINAL]


# ============================================================
# LOAD QUERIES
# ============================================================

def load_queries():

    path = Path(QUERY_FILE)

    if not path.exists():

        raise FileNotFoundError(
            f"Missing query file: {path}"
        )

    queries = []

    seen = set()

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            query_id = item.get(
                "query_id"
            )

            query = item.get(
                "query"
            )

            if (
                query_id is None
                or not query
            ):
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
# EVIDENCE GATE
# ============================================================

def evidence_gate(
    vector_score,
    lexical_score,
    rerank_score
):
    """
    Calibrated evidence gate.

    We intentionally allow multiple routes:

    1. Strong vector + lexical evidence
    2. Strong rerank + lexical evidence

    This avoids blocking legitimate queries simply
    because one score is slightly weak.
    """

    # Route 1:
    # Strong semantic retrieval + lexical evidence
    if (
        vector_score >= VECTOR_THRESHOLD
        and lexical_score >= LEXICAL_THRESHOLD
    ):

        return True, "strong_vector_lexical"

    # Route 2:
    # Strong reranked result + lexical evidence
    if (
        rerank_score >= RERANK_THRESHOLD
        and lexical_score >= LEXICAL_THRESHOLD
    ):

        return True, "strong_rerank_lexical"

    return False, "insufficient_evidence"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG - STEP 15")
    print("PRODUCTION EVIDENCE GATE")
    print("=" * 70)

    print()
    print("Loading embedding model...")

    embedder = TextEmbedding(
        model_name=MODEL_NAME
    )

    print("Embedding model loaded.")

    print()
    print("Opening Qdrant...")

    client = QdrantClient(
        path=QDRANT_PATH
    )

    print("Qdrant opened.")

    print()
    print("Loading benchmark queries...")

    queries = load_queries()

    print(
        f"Queries: {len(queries)}"
    )

    if not queries:
        print("No queries found.")
        return

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    allowed = 0
    blocked = 0

    grounded_retrieval = 0

    false_refusals = 0
    potential_false_accepts = 0

    gate_decisions = {
        "strong_vector_lexical": 0,
        "strong_rerank_lexical": 0,
        "insufficient_evidence": 0
    }

    total_times = []
    embedding_times = []
    qdrant_times = []
    rerank_times = []

    false_refusal_examples = []
    false_accept_examples = []

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    print()
    print("Running evidence gate benchmark...")
    print()

    for index, item in enumerate(
        queries,
        start=1
    ):

        query = item["query"]
        query_id = item["query_id"]

        pipeline_start = (
            time.perf_counter()
        )

        # ====================================================
        # EMBEDDING
        # ====================================================

        start = time.perf_counter()

        query_vector = list(
            embedder.embed(
                [query]
            )
        )[0]

        embedding_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # ====================================================
        # QDRANT
        # ====================================================

        start = time.perf_counter()

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K_RETRIEVAL,
            with_payload=[
                "chunk_id",
                "passage_id",
                "query_id",
                "text",
                "is_selected",
                "chunk_strategy",
                "word_count"
            ],
            with_vectors=False
        )

        hits = response.points

        qdrant_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # ====================================================
        # GROUND TRUTH RETRIEVAL
        # ====================================================

        retrieved_ids = [
            (
                hit.payload or {}
            ).get("query_id")
            for hit in hits
        ]

        gt5 = (
            query_id
            in retrieved_ids[:5]
        )

        if gt5:
            grounded_retrieval += 1

        # ====================================================
        # RERANK
        # ====================================================

        start = time.perf_counter()

        reranked = rerank(
            query,
            hits
        )

        rerank_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # ====================================================
        # NO RESULTS
        # ====================================================

        if not reranked:

            allowed_flag = False

            decision = (
                "insufficient_evidence"
            )

            vector_score = 0.0
            lexical_score = 0.0
            rerank_score = 0.0

        else:

            top = reranked[0]

            (
                rerank_score,
                vector_score,
                lexical_score,
                phrase_value,
                hit
            ) = top

            allowed_flag, decision = (
                evidence_gate(
                    vector_score,
                    lexical_score,
                    rerank_score
                )
            )

        # ====================================================
        # DECISION COUNTS
        # ====================================================

        gate_decisions[
            decision
        ] += 1

        if allowed_flag:

            allowed += 1

        else:

            blocked += 1

        # ====================================================
        # FALSE REFUSAL
        # ====================================================

        if (
            not allowed_flag
            and gt5
        ):

            false_refusals += 1

            if len(
                false_refusal_examples
            ) < 30:

                false_refusal_examples.append(
                    {
                        "query": query,
                        "vector": vector_score,
                        "lexical": lexical_score,
                        "rerank": rerank_score,
                        "decision": decision
                    }
                )

        # ====================================================
        # POTENTIAL FALSE ACCEPT
        # ====================================================

        if (
            allowed_flag
            and not gt5
        ):

            potential_false_accepts += 1

            if len(
                false_accept_examples
            ) < 20:

                false_accept_examples.append(
                    {
                        "query": query,
                        "vector": vector_score,
                        "lexical": lexical_score,
                        "rerank": rerank_score,
                        "decision": decision
                    }
                )

        # ====================================================
        # LATENCY
        # ====================================================

        total_ms = (
            time.perf_counter()
            - pipeline_start
        ) * 1000

        embedding_times.append(
            embedding_ms
        )

        qdrant_times.append(
            qdrant_ms
        )

        rerank_times.append(
            rerank_ms
        )

        total_times.append(
            total_ms
        )

        if index % 100 == 0:

            print(
                f"[{index}/{len(queries)}]"
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    def percentile(
        values,
        p
    ):

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

    total_queries = len(
        queries
    )

    false_refusal_rate = (
        false_refusals
        / total_queries
        * 100
    )

    false_accept_rate = (
        potential_false_accepts
        / total_queries
        * 100
    )

    allowed_precision = 0.0

    if allowed > 0:

        allowed_precision = (
            (allowed - potential_false_accepts)
            / allowed
            * 100
        )

    gt5_rate = (
        grounded_retrieval
        / total_queries
        * 100
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 15 COMPLETE")
    print("=" * 70)

    print()
    print("EVIDENCE GATE CONFIGURATION")
    print("-" * 70)

    print(
        f"Vector threshold  : "
        f"{VECTOR_THRESHOLD:.2f}"
    )

    print(
        f"Lexical threshold : "
        f"{LEXICAL_THRESHOLD:.2f}"
    )

    print(
        f"Rerank threshold  : "
        f"{RERANK_THRESHOLD:.2f}"
    )

    print()
    print("GATE RESULTS")
    print("-" * 70)

    print(
        f"Total queries          : "
        f"{total_queries}"
    )

    print(
        f"Allowed               : "
        f"{allowed} "
        f"({allowed / total_queries * 100:.2f}%)"
    )

    print(
        f"Blocked               : "
        f"{blocked} "
        f"({blocked / total_queries * 100:.2f}%)"
    )

    print()
    print(
        f"Ground truth @5       : "
        f"{grounded_retrieval} "
        f"({gt5_rate:.2f}%)"
    )

    print(
        f"False refusals        : "
        f"{false_refusals} "
        f"({false_refusal_rate:.2f}%)"
    )

    print(
        f"Potential false accepts: "
        f"{potential_false_accepts} "
        f"({false_accept_rate:.2f}%)"
    )

    print(
        f"Allowed precision     : "
        f"{allowed_precision:.2f}%"
    )

    # ========================================================
    # GATE DECISIONS
    # ========================================================

    print()
    print("GATE DECISIONS")
    print("-" * 70)

    for name, count in (
        gate_decisions.items()
    ):

        print(
            f"{name:<35}"
            f"{count}"
        )

    # ========================================================
    # FALSE REFUSALS
    # ========================================================

    print()
    print("REMAINING FALSE REFUSALS")
    print("-" * 70)

    if not false_refusal_examples:

        print(
            "None found."
        )

    else:

        for example in (
            false_refusal_examples
        ):

            print()
            print(
                f"Query: "
                f"{example['query']}"
            )

            print(
                f"Vector: "
                f"{example['vector']:.4f}"
            )

            print(
                f"Lexical: "
                f"{example['lexical']:.4f}"
            )

            print(
                f"Rerank: "
                f"{example['rerank']:.4f}"
            )

            print(
                f"Decision: "
                f"{example['decision']}"
            )

    # ========================================================
    # FALSE ACCEPTS
    # ========================================================

    print()
    print("POTENTIAL FALSE ACCEPTS")
    print("-" * 70)

    if not false_accept_examples:

        print(
            "None found."
        )

    else:

        for example in (
            false_accept_examples
        ):

            print()
            print(
                f"Query: "
                f"{example['query']}"
            )

            print(
                f"Vector: "
                f"{example['vector']:.4f}"
            )

            print(
                f"Lexical: "
                f"{example['lexical']:.4f}"
            )

            print(
                f"Rerank: "
                f"{example['rerank']:.4f}"
            )

            print(
                f"Decision: "
                f"{example['decision']}"
            )

    # ========================================================
    # LATENCY
    # ========================================================

    print()
    print("LATENCY")
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
        f"Total P50     : "
        f"{percentile(total_times, 50):.2f} ms"
    )

    print(
        f"Total P70     : "
        f"{percentile(total_times, 70):.2f} ms"
    )

    print(
        f"Total P100    : "
        f"{percentile(total_times, 100):.2f} ms"
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("ASSESSMENT")
    print("-" * 70)

    if (
        false_refusal_rate <= 5.0
        and allowed_precision >= 90.0
    ):

        print(
            "STATUS: GOOD"
        )

        print(
            "The evidence gate has low false refusals "
            "and high allowed precision."
        )

    elif false_refusal_rate <= 10.0:

        print(
            "STATUS: NEEDS REVIEW"
        )

        print(
            "The gate has acceptable recall but "
            "should be reviewed before production."
        )

    else:

        print(
            "STATUS: NEEDS CALIBRATION"
        )

        print(
            "Too many relevant queries are still "
            "being blocked."
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()