import time
import re

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from app.answer_generator import generate_extractive_answer


# ============================================================
# HH GOA RAG - STEP 18
# COMPLETE POST-STT PIPELINE
#
# Pipeline:
#
# STT text
#    ↓
# Query embedding
#    ↓
# Qdrant Top-20 retrieval
#    ↓
# Fast reranking
#    ↓
# Top-3
#    ↓
# Guardrails / Answer generation
#
# IMPORTANT:
# Both Top-20 and Top-3 are preserved for diagnostics.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

COLLECTION_NAME = "hh_goa_rag_hindi"

QDRANT_PATH = "data/qdrant"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 3

VECTOR_WEIGHT = 0.70
LEXICAL_WEIGHT = 0.20
PHRASE_WEIGHT = 0.10


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
        len(
            q_tokens.intersection(d_tokens)
        )
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

    # Exact complete query.
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

    scores = []

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

        exact_phrase_score = phrase_score(
            query,
            text
        )

        final_score = (
            VECTOR_WEIGHT * vector_score
            + LEXICAL_WEIGHT * lexical_score
            + PHRASE_WEIGHT * exact_phrase_score
        )

        scores.append(
            (
                final_score,
                vector_score,
                lexical_score,
                exact_phrase_score,
                hit
            )
        )

    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scores[:TOP_K_FINAL]


# ============================================================
# RAG ENGINE
# ============================================================

class RAGEngine:

    def __init__(self):

        print(
            "Loading embedding model..."
        )

        self.embedder = TextEmbedding(
            model_name=MODEL_NAME
        )

        print(
            "Opening Qdrant..."
        )

        self.client = QdrantClient(
            path=QDRANT_PATH
        )

        print(
            "RAG engine ready."
        )

    # ========================================================
    # PROCESS QUERY
    # ========================================================

    def process(self, query: str):

        pipeline_start = time.perf_counter()

        query = query.strip()

        if not query:

            return {
                "answer": "Please provide a question.",
                "grounded": False,
                "blocked": True,
                "reason": "empty_query",

                "retrieval": {
                    "top20": [],
                    "top3": []
                },

                "retrieved_chunks": 0,

                "sources": [],

                "timings": {
                    "embedding_ms": 0.0,
                    "qdrant_ms": 0.0,
                    "rerank_ms": 0.0,
                    "answer_ms": 0.0,
                    "total_ms": 0.0
                }
            }

        # ====================================================
        # 1. QUERY EMBEDDING
        # ====================================================

        start = time.perf_counter()

        query_vector = list(
            self.embedder.embed(
                [query]
            )
        )[0]

        embedding_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # ====================================================
        # 2. QDRANT TOP-20 RETRIEVAL
        # ====================================================

        start = time.perf_counter()

        response = self.client.query_points(

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
        # 3. PRESERVE ORIGINAL TOP-20
        # ====================================================

        top20_results = []

        for rank, hit in enumerate(
            hits,
            start=1
        ):

            payload = hit.payload or {}

            top20_results.append(
                {
                    "rank": rank,

                    "chunk_id": payload.get(
                        "chunk_id"
                    ),

                    "passage_id": payload.get(
                        "passage_id"
                    ),

                    "query_id": payload.get(
                        "query_id"
                    ),

                    "text": payload.get(
                        "text",
                        ""
                    ),

                    "vector_score": float(
                        hit.score
                    ),

                    "word_count": payload.get(
                        "word_count"
                    ),

                    "chunk_strategy": payload.get(
                        "chunk_strategy"
                    )
                }
            )

        # ====================================================
        # 4. FAST RERANK
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
        # 5. CONVERT TOP-3
        # ====================================================

        top3_results = []

        for rank, item in enumerate(
            reranked,
            start=1
        ):

            (
                final_score,
                vector_score,
                lexical_score,
                phrase_score_value,
                hit
            ) = item

            payload = hit.payload or {}

            top3_results.append(
                {
                    "rank": rank,

                    "chunk_id": payload.get(
                        "chunk_id"
                    ),

                    "passage_id": payload.get(
                        "passage_id"
                    ),

                    "query_id": payload.get(
                        "query_id"
                    ),

                    "text": payload.get(
                        "text",
                        ""
                    ),

                    "score": final_score,

                    "vector_score": vector_score,

                    "lexical_score": lexical_score,

                    "phrase_score": phrase_score_value,

                    "word_count": payload.get(
                        "word_count"
                    ),

                    "chunk_strategy": payload.get(
                        "chunk_strategy"
                    )
                }
            )

        # ====================================================
        # 6. ANSWER GENERATION
        # ====================================================

        answer_start = time.perf_counter()

        answer_result = generate_extractive_answer(
            query,
            top3_results
        )

        answer_ms = (
            time.perf_counter()
            - answer_start
        ) * 1000

        # ====================================================
        # 7. TOTAL LATENCY
        # ====================================================

        total_ms = (
            time.perf_counter()
            - pipeline_start
        ) * 1000

        # ====================================================
        # 8. FINAL RESULT
        # ====================================================

        return {

            "query": query,

            "answer": answer_result[
                "answer"
            ],

            "grounded": answer_result[
                "grounded"
            ],

            "blocked": answer_result[
                "blocked"
            ],

            "reason": answer_result[
                "reason"
            ],

            # IMPORTANT:
            # Preserve both retrieval stages.

            "retrieval": {

                "top20": top20_results,

                "top3": top3_results
            },

            "retrieved_chunks": len(
                top3_results
            ),

            # Backward compatibility.

            "sources": top3_results,

            "timings": {

                "embedding_ms": round(
                    embedding_ms,
                    2
                ),

                "qdrant_ms": round(
                    qdrant_ms,
                    2
                ),

                "rerank_ms": round(
                    rerank_ms,
                    2
                ),

                "answer_ms": round(
                    answer_ms,
                    2
                ),

                "total_ms": round(
                    total_ms,
                    2
                )
            }
        }


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG - STEP 18")
    print("TOP-20 + TOP-3 RETRIEVAL TEST")
    print("=" * 70)

    engine = RAGEngine()

    query = "मैनहट्टन परियोजना क्या थी?"

    print()
    print(
        f"Question: {query}"
    )

    result = engine.process(
        query
    )

    print()

    print(
        "Answer:"
    )

    print(
        result["answer"]
    )

    print()

    print(
        "Grounded:",
        result["grounded"]
    )

    print(
        "Blocked:",
        result["blocked"]
    )

    print(
        "Reason:",
        result["reason"]
    )

    print()

    print("=" * 70)
    print("RETRIEVAL")
    print("=" * 70)

    print(
        "Top-20 retrieved:",
        len(
            result["retrieval"]["top20"]
        )
    )

    print(
        "Top-3 reranked:",
        len(
            result["retrieval"]["top3"]
        )
    )

    print()

    print("TOP-3 RESULTS")

    for item in result[
        "retrieval"
    ]["top3"]:

        print(
            f"Rank {item['rank']} | "
            f"query_id={item['query_id']} | "
            f"score={item['score']:.4f} | "
            f"vector={item['vector_score']:.4f} | "
            f"lexical={item['lexical_score']:.4f}"
        )

    print()

    print("=" * 70)
    print("LATENCY")
    print("=" * 70)

    timings = result["timings"]

    print(
        f"Embedding : {timings['embedding_ms']:.2f} ms"
    )

    print(
        f"Qdrant    : {timings['qdrant_ms']:.2f} ms"
    )

    print(
        f"Rerank    : {timings['rerank_ms']:.2f} ms"
    )

    print(
        f"Answer    : {timings['answer_ms']:.2f} ms"
    )

    print(
        f"TOTAL     : {timings['total_ms']:.2f} ms"
    )

    print()

    if timings["total_ms"] < 200:

        print(
            "STATUS: PASS"
        )

    else:

        print(
            "STATUS: FAIL"
        )


if __name__ == "__main__":
    main()