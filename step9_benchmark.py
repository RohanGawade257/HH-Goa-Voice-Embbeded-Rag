import json
import time
import statistics
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient


# ============================================================
# CONFIG
# ============================================================

QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "hh_goa_rag_hindi"

SOURCE_FILE = Path("data/hindi_sample_1000.jsonl")

# THIS MUST MATCH THE MODEL USED TO CREATE YOUR QDRANT VECTORS
EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

RERANKER_MODEL = (
    "jinaai/jina-reranker-v2-base-multilingual"
)

INITIAL_TOP_K = 20


# ============================================================
# PERCENTILE
# ============================================================

def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    fraction = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

def load_queries():

    queries = []

    with open(
        SOURCE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            record = json.loads(line)

            query_id = record.get("query_id")

            query = record.get("query")

            passages = record.get(
                "passages",
                {}
            )

            english = passages.get(
                "English_passages",
                []
            )

            selected = passages.get(
                "is_selected",
                []
            )

            if not query:
                continue

            selected_indexes = []

            for i, flag in enumerate(selected):

                if flag == 1:
                    selected_indexes.append(i)

            queries.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "selected_indexes": selected_indexes,
                    "english_passages": english
                }
            )

    return queries


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG - STEP 9")
    print("TOP-20 RETRIEVAL + MULTILINGUAL RERANKING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Qdrant
    # --------------------------------------------------------

    print("\nLoading Qdrant...")

    client = QdrantClient(
        path=QDRANT_PATH
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    embedder = TextEmbedding(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Load reranker
    # --------------------------------------------------------

    print("\nLoading reranker...")

    print(
        f"Model: {RERANKER_MODEL}"
    )

    reranker = TextCrossEncoder(
        model_name=RERANKER_MODEL
    )

    # --------------------------------------------------------
    # Load queries
    # --------------------------------------------------------

    queries = load_queries()

    print(
        f"\nBenchmark queries: {len(queries)}"
    )

    if not queries:

        raise RuntimeError(
            "ZERO QUERIES LOADED. "
            "Check hindi_sample_1000.jsonl"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    embedding_times = []
    qdrant_times = []
    retrieval_times = []
    rerank_times = []
    total_times = []

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print("\nWarming up models...")

    warmup_query = "मैनहट्टन परियोजना क्या थी?"

    list(
        embedder.query_embed(
            warmup_query
        )
    )

    list(
        reranker.rerank(
            warmup_query,
            [
                "मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान बनाई गई थी।",
                "भारत की राजधानी नई दिल्ली है।"
            ]
        )
    )

    print("Warmup complete.")

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print("\nRunning benchmark...\n")

    for number, item in enumerate(
        queries,
        start=1
    ):

        query = item["query"]

        # ====================================================
        # EMBEDDING
        # ====================================================

        start = time.perf_counter()

        query_vector = list(
            embedder.query_embed(
                query
            )
        )[0]

        embedding_ms = (
            time.perf_counter() - start
        ) * 1000

        embedding_times.append(
            embedding_ms
        )

        # ====================================================
        # QDRANT
        # ====================================================

        start = time.perf_counter()

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=INITIAL_TOP_K,
            with_payload=True
        )

        qdrant_ms = (
            time.perf_counter() - start
        ) * 1000

        qdrant_times.append(
            qdrant_ms
        )

        # ====================================================
        # COLLECT DOCUMENTS
        # ====================================================

        documents = []

        for point in response.points:

            payload = point.payload or {}

            text = payload.get(
                "text",
                ""
            )

            if text:
                documents.append(text)

        # ====================================================
        # RETRIEVAL TIME
        # ====================================================

        retrieval_ms = (
            embedding_ms
            + qdrant_ms
        )

        retrieval_times.append(
            retrieval_ms
        )

        # ====================================================
        # RERANK
        # ====================================================

        start = time.perf_counter()

        scores = list(
            reranker.rerank(
                query,
                documents
            )
        )

        rerank_ms = (
            time.perf_counter() - start
        ) * 1000

        rerank_times.append(
            rerank_ms
        )

        # ====================================================
        # TOTAL
        # ====================================================

        total_ms = (
            embedding_ms
            + qdrant_ms
            + rerank_ms
        )

        total_times.append(
            total_ms
        )

        # ====================================================
        # PROGRESS
        # ====================================================

        if number % 25 == 0:

            print(
                f"[{number}/{len(queries)}] "
                f"Embedding={embedding_ms:.2f} ms | "
                f"Qdrant={qdrant_ms:.2f} ms | "
                f"Rerank={rerank_ms:.2f} ms | "
                f"Total={total_ms:.2f} ms"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 9 COMPLETE")
    print("=" * 70)

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    print("\nLATENCY")
    print("-" * 70)

    print(
        f"Embedding P50 : "
        f"{percentile(embedding_times, 0.50):.2f} ms"
    )

    print(
        f"Embedding P70 : "
        f"{percentile(embedding_times, 0.70):.2f} ms"
    )

    print(
        f"Embedding P100: "
        f"{max(embedding_times):.2f} ms"
    )

    print()

    print(
        f"Qdrant P50    : "
        f"{percentile(qdrant_times, 0.50):.2f} ms"
    )

    print(
        f"Qdrant P70    : "
        f"{percentile(qdrant_times, 0.70):.2f} ms"
    )

    print(
        f"Qdrant P100   : "
        f"{max(qdrant_times):.2f} ms"
    )

    print()

    print(
        f"Retrieval P50 : "
        f"{percentile(retrieval_times, 0.50):.2f} ms"
    )

    print(
        f"Retrieval P70 : "
        f"{percentile(retrieval_times, 0.70):.2f} ms"
    )

    print(
        f"Retrieval P100: "
        f"{max(retrieval_times):.2f} ms"
    )

    print()

    print(
        f"Rerank P50    : "
        f"{percentile(rerank_times, 0.50):.2f} ms"
    )

    print(
        f"Rerank P70    : "
        f"{percentile(rerank_times, 0.70):.2f} ms"
    )

    print(
        f"Rerank P100   : "
        f"{max(rerank_times):.2f} ms"
    )

    print()

    print(
        f"TOTAL P50     : "
        f"{percentile(total_times, 0.50):.2f} ms"
    )

    print(
        f"TOTAL P70     : "
        f"{percentile(total_times, 0.70):.2f} ms"
    )

    print(
        f"TOTAL P100    : "
        f"{max(total_times):.2f} ms"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    print("\nTARGET")
    print("-" * 70)

    print(
        "Post-STT requirement : < 200 ms"
    )

    print(
        "Current measurement  : "
        "Embedding + Qdrant + Reranking"
    )

    print()

    p50 = percentile(
        total_times,
        0.50
    )

    p70 = percentile(
        total_times,
        0.70
    )

    p100 = max(total_times)

    print(
        f"P50  : {p50:.2f} ms"
    )

    print(
        f"P70  : {p70:.2f} ms"
    )

    print(
        f"P100 : {p100:.2f} ms"
    )

    if p100 < 200:

        print(
            "\n✓ Retrieval + reranking "
            "fits under 200 ms."
        )

    else:

        print(
            "\n⚠ Retrieval + reranking "
            "does NOT fit under 200 ms."
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "LLM generation is NOT included yet."
    )

    print(
        "The final pipeline still needs:"
    )

    print(
        "LLM + guardrails + API overhead."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
    