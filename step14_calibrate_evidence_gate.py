import json
import time
from pathlib import Path
from statistics import mean

from fastembed import TextEmbedding
from qdrant_client import QdrantClient


# ============================================================
# CONFIG
# ============================================================

COLLECTION_NAME = "hh_goa_rag_hindi"
QDRANT_PATH = "data/qdrant"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

QUERY_FILE = Path("data/hindi_sample_1000.jsonl")

TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 3


# ============================================================
# IMPORTANT
# ============================================================
#
# These are calibration configurations.
#
# We are NOT changing the production guardrail yet.
#
# The script tests several combinations of:
#
#   vector similarity
#   lexical overlap
#   reranked score
#
# against ground-truth retrieval.
#
# ============================================================


CONFIGS = [

    {
        "name": "STRICT",
        "vector": 0.80,
        "lexical": 0.75,
        "rerank": 0.75,
    },

    {
        "name": "BALANCED",
        "vector": 0.75,
        "lexical": 0.70,
        "rerank": 0.70,
    },

    {
        "name": "BALANCED_LOW",
        "vector": 0.70,
        "lexical": 0.65,
        "rerank": 0.65,
    },

    {
        "name": "RECALL_FOCUSED",
        "vector": 0.65,
        "lexical": 0.60,
        "rerank": 0.60,
    },

    {
        "name": "RECALL_FOCUSED_LOW",
        "vector": 0.60,
        "lexical": 0.55,
        "rerank": 0.55,
    },

]


# ============================================================
# TEXT UTILITIES
# ============================================================

import re

WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text):

    text = str(text).lower().strip()

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


def tokenize(text):

    return set(
        WORD_RE.findall(
            normalize(text)
        )
    )


def lexical_overlap(query, document):

    q_tokens = tokenize(query)
    d_tokens = tokenize(document)

    if not q_tokens or not d_tokens:
        return 0.0

    intersection = (
        q_tokens.intersection(d_tokens)
    )

    return (
        len(intersection)
        / len(q_tokens)
    )


# ============================================================
# PHRASE SCORE
# ============================================================

def phrase_score(query, document):

    q = normalize(query)
    d = normalize(document)

    if not q:
        return 0.0

    if q in d:
        return 1.0

    words = q.split()

    if len(words) >= 3:

        for size in (3, 4):

            for i in range(
                len(words) - size + 1
            ):

                phrase = " ".join(
                    words[i:i + size]
                )

                if phrase in d:
                    return 0.7

    return 0.0


# ============================================================
# RERANK
# ============================================================

VECTOR_WEIGHT = 0.70
LEXICAL_WEIGHT = 0.20
PHRASE_WEIGHT = 0.10


def rerank(query, hits):

    results = []

    for hit in hits:

        payload = (
            hit.payload or {}
        )

        text = payload.get(
            "text",
            ""
        )

        vector_score = float(
            hit.score
        )

        lexical = lexical_overlap(
            query,
            text
        )

        phrase = phrase_score(
            query,
            text
        )

        final_score = (
            VECTOR_WEIGHT * vector_score
            + LEXICAL_WEIGHT * lexical
            + PHRASE_WEIGHT * phrase
        )

        results.append(
            {
                "hit": hit,
                "vector": vector_score,
                "lexical": lexical,
                "phrase": phrase,
                "rerank": final_score,
                "text": text,
            }
        )

    results.sort(
        key=lambda x: x["rerank"],
        reverse=True
    )

    return results[:TOP_K_FINAL]


# ============================================================
# LOAD QUERIES
# ============================================================

def load_queries():

    queries = []

    if not QUERY_FILE.exists():

        raise FileNotFoundError(
            f"Missing query file: {QUERY_FILE}"
        )

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
                    "query": query,
                }
            )

    return queries


# ============================================================
# EVIDENCE GATE
# ============================================================

def evidence_gate(
    top_vector,
    top_lexical,
    top_rerank,
    top3_vectors,
    top3_lexicals,
    config
):

    vector_threshold = config[
        "vector"
    ]

    lexical_threshold = config[
        "lexical"
    ]

    rerank_threshold = config[
        "rerank"
    ]

    # --------------------------------------------------------
    # RULE 1
    #
    # Strong semantic + lexical evidence
    # --------------------------------------------------------

    if (
        top_vector >= vector_threshold
        and
        top_lexical >= lexical_threshold
    ):

        return True, "strong_vector_lexical"


    # --------------------------------------------------------
    # RULE 2
    #
    # Strong reranked evidence
    # --------------------------------------------------------

    if (
        top_rerank >= rerank_threshold
        and
        top_lexical >= lexical_threshold
    ):

        return True, "strong_rerank_lexical"


    # --------------------------------------------------------
    # RULE 3
    #
    # Evidence distributed across top-3.
    #
    # This is important because the correct chunk
    # doesn't necessarily have to be rank #1.
    # --------------------------------------------------------

    evidence_count = 0

    for vector, lexical in zip(
        top3_vectors,
        top3_lexicals
    ):

        if (
            vector >= vector_threshold
            and
            lexical >= lexical_threshold
        ):

            evidence_count += 1

    if evidence_count >= 2:

        return True, "multi_chunk_evidence"


    return False, "insufficient_evidence"


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
    print("HH GOA RAG - STEP 14")
    print("CALIBRATED EVIDENCE GATE")
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

        print(
            "No benchmark queries found."
        )

        return

    # ========================================================
    # RESULT STORAGE
    # ========================================================

    results = []

    embedding_times = []
    qdrant_times = []
    rerank_times = []

    # ========================================================
    # RUN RETRIEVAL ONCE
    #
    # This is important.
    #
    # We don't rerun embedding/Qdrant for every threshold.
    # Retrieval results are cached in memory.
    #
    # ========================================================

    print()
    print(
        "Running retrieval calibration..."
    )

    for index, item in enumerate(
        queries,
        start=1
    ):

        query = item["query"]
        query_id = item["query_id"]

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # QDRANT
        # ----------------------------------------------------

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
                "word_count",
            ],

            with_vectors=False,
        )

        hits = response.points

        qdrant_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # ----------------------------------------------------
        # RERANK
        # ----------------------------------------------------

        start = time.perf_counter()

        ranked = rerank(
            query,
            hits
        )

        rerank_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # ----------------------------------------------------
        # TOP-1 INFORMATION
        # ----------------------------------------------------

        if ranked:

            top = ranked[0]

            top_vector = top[
                "vector"
            ]

            top_lexical = top[
                "lexical"
            ]

            top_rerank = top[
                "rerank"
            ]

        else:

            top_vector = 0.0
            top_lexical = 0.0
            top_rerank = 0.0

        # ----------------------------------------------------
        # TOP-3 INFORMATION
        # ----------------------------------------------------

        top3_vectors = [
            x["vector"]
            for x in ranked[:3]
        ]

        top3_lexicals = [
            x["lexical"]
            for x in ranked[:3]
        ]

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        retrieved_ids = [

            hit.payload.get(
                "query_id"
            )

            for hit in hits
        ]

        gt_at_5 = (
            query_id
            in retrieved_ids[:5]
        )

        gt_at_3 = (
            query_id
            in retrieved_ids[:3]
        )

        gt_at_1 = (
            query_id
            in retrieved_ids[:1]
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        results.append(
            {
                "query_id": query_id,
                "query": query,

                "top_vector":
                    top_vector,

                "top_lexical":
                    top_lexical,

                "top_rerank":
                    top_rerank,

                "top3_vectors":
                    top3_vectors,

                "top3_lexicals":
                    top3_lexicals,

                "gt_at_1":
                    gt_at_1,

                "gt_at_3":
                    gt_at_3,

                "gt_at_5":
                    gt_at_5,
            }
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

        if index % 100 == 0:

            print(
                f"[{index}/{len(queries)}]"
            )

    # ========================================================
    # TEST CONFIGURATIONS
    # ========================================================

    print()
    print("=" * 70)
    print("EVIDENCE GATE CALIBRATION")
    print("=" * 70)

    calibration_results = []

    for config in CONFIGS:

        allowed = 0
        blocked = 0

        allowed_gt5 = 0
        blocked_gt5 = 0

        false_refusals = 0
        potential_false_accepts = 0

        reasons = {}

        for result in results:

            allow, reason = evidence_gate(

                result["top_vector"],

                result["top_lexical"],

                result["top_rerank"],

                result["top3_vectors"],

                result["top3_lexicals"],

                config
            )

            reasons[reason] = (
                reasons.get(
                    reason,
                    0
                ) + 1
            )

            if allow:

                allowed += 1

                if result["gt_at_5"]:

                    allowed_gt5 += 1

                else:

                    potential_false_accepts += 1

            else:

                blocked += 1

                if result["gt_at_5"]:

                    blocked_gt5 += 1

                    false_refusals += 1

        allowed_pct = (
            allowed
            / len(results)
            * 100
        )

        blocked_pct = (
            blocked
            / len(results)
            * 100
        )

        false_refusal_pct = (
            false_refusals
            / len(results)
            * 100
        )

        false_accept_pct = (
            potential_false_accepts
            / len(results)
            * 100
        )

        # Precision of allowed answers
        if allowed:

            allowed_precision = (
                allowed_gt5
                / allowed
                * 100
            )

        else:

            allowed_precision = 0.0

        # ====================================================
        # SCORE
        #
        # We want:
        #
        # high allowed GT5
        # low false refusals
        # low false accepts
        #
        # This is only a calibration score.
        # ====================================================

        calibration_score = (
            allowed_gt5
            - potential_false_accepts * 0.75
        )

        calibration_results.append(
    {
        "config": config,

        "allowed": allowed,
        "blocked": blocked,

        "allowed_gt5": allowed_gt5,

        "allowed_pct":
            allowed_pct,

        "blocked_pct":
            blocked_pct,

        "false_refusals":
            false_refusals,

        "false_refusal_pct":
            false_refusal_pct,

        "false_accepts":
            potential_false_accepts,

        "false_accept_pct":
            false_accept_pct,

        "allowed_precision":
            allowed_precision,

        "score":
            calibration_score,

        "reasons":
            reasons,
    }
)
    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print(
        f"{'CONFIG':<22}"
        f"{'ALLOW':>8}"
        f"{'BLOCK':>8}"
        f"{'GT5 ALLOW':>11}"
        f"{'FALSE REF':>11}"
        f"{'FALSE ACC':>11}"
        f"{'PRECISION':>11}"
    )

    print("-" * 82)

    for item in calibration_results:

        print(
            f"{item['config']['name']:<22}"
            f"{item['allowed_pct']:>7.2f}%"
            f"{item['blocked_pct']:>7.2f}%"
            f"{item['allowed_gt5']:>11}"
            f"{item['false_refusals']:>11}"
            f"{item['false_accepts']:>11}"
            f"{item['allowed_precision']:>10.2f}%"
        )

    # ========================================================
    # BEST CONFIG
    # ========================================================

    best = max(
        calibration_results,
        key=lambda x: x["score"]
    )

    print()
    print("=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)

    config = best["config"]

    print()
    print(
        f"Name       : {config['name']}"
    )

    print(
        f"Vector     : {config['vector']:.2f}"
    )

    print(
        f"Lexical    : {config['lexical']:.2f}"
    )

    print(
        f"Rerank     : {config['rerank']:.2f}"
    )

    print()

    print(
        f"Allowed    : "
        f"{best['allowed']} "
        f"({best['allowed_pct']:.2f}%)"
    )

    print(
        f"Blocked    : "
        f"{best['blocked']} "
        f"({best['blocked_pct']:.2f}%)"
    )

    print(
        f"False refusals : "
        f"{best['false_refusals']} "
        f"({best['false_refusal_pct']:.2f}%)"
    )

    print(
        f"Potential false accepts : "
        f"{best['false_accepts']} "
        f"({best['false_accept_pct']:.2f}%)"
    )

    print(
        f"Allowed precision : "
        f"{best['allowed_precision']:.2f}%"
    )

    # ========================================================
    # REASONS
    # ========================================================

    print()
    print("GATE DECISIONS")
    print("-" * 70)

    for reason, count in sorted(
        best["reasons"].items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{reason:<30}"
            f"{count:>8}"
        )

    # ========================================================
    # HIGH-CONFIDENCE FALSE REFUSALS
    # ========================================================

    print()
    print("=" * 70)
    print("REMAINING HIGH-CONFIDENCE FALSE REFUSALS")
    print("=" * 70)

    count = 0

    for result in results:

        allow, reason = evidence_gate(

            result["top_vector"],
            result["top_lexical"],
            result["top_rerank"],
            result["top3_vectors"],
            result["top3_lexicals"],
            config
        )

        if (
            not allow
            and result["gt_at_5"]
        ):

            print()

            print(
                f"Query: {result['query']}"
            )

            print(
                f"Vector: "
                f"{result['top_vector']:.4f}"
            )

            print(
                f"Lexical: "
                f"{result['top_lexical']:.4f}"
            )

            print(
                f"Rerank: "
                f"{result['top_rerank']:.4f}"
            )

            print(
                f"Reason: {reason}"
            )

            count += 1

            if count >= 20:
                break

    # ========================================================
    # LATENCY
    # ========================================================

    print()
    print("=" * 70)
    print("CALIBRATION RETRIEVAL LATENCY")
    print("=" * 70)

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

    print()
    print(
        "NOTE:"
    )

    print(
        "This script only calibrates the evidence gate."
    )

    print(
        "It does NOT modify pipeline.py."
    )

    print(
        "Do not copy the recommended thresholds into "
        "production until reviewing the results."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()