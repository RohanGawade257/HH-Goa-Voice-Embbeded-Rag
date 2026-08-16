import json
import time
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient


# ============================================================
# CONFIG
# ============================================================

COLLECTION_NAME = "hh_goa_rag_hindi"
QDRANT_PATH = "data/qdrant"

SOURCE_FILE = Path(
    "data/hindi_sample_1000.jsonl"
)

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 10
MAX_QUERIES = 656


# ============================================================
# PERCENTILE
# ============================================================

def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * p / 100

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1
    )

    weight = index - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    return " ".join(
        text.strip().lower().split()
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth():

    ground_truth = {}

    with open(
        SOURCE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            query_id = record["query_id"]

            query = record["query"]

            passages = record["passages"]

            hindi_passages = passages[
                "Translated_passages"
            ]

            selected_flags = passages[
                "is_selected"
            ]

            selected_passages = []

            for passage, flag in zip(
                hindi_passages,
                selected_flags
            ):

                if flag == 1:

                    selected_passages.append(
                        passage
                    )

            if selected_passages:

                ground_truth[query_id] = {
                    "query": query,
                    "selected": selected_passages
                }

    return ground_truth


# ============================================================
# GROUND TRUTH MATCHING
# ============================================================

def passage_matches(
    retrieved,
    ground_truth
):

    retrieved = normalize_text(
        retrieved
    )

    ground_truth = normalize_text(
        ground_truth
    )

    # Exact containment
    if ground_truth in retrieved:
        return True

    if retrieved in ground_truth:
        return True

    # Chunking can split a passage.
    # Calculate word overlap.

    gt_words = ground_truth.split()

    retrieved_words = set(
        retrieved.split()
    )

    if len(gt_words) < 5:
        return False

    matched = sum(
        1
        for word in gt_words
        if word in retrieved_words
    )

    overlap = matched / len(gt_words)

    return overlap >= 0.80


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("HH GOA RAG - STEP 8")
    print("Multilingual Retrieval Benchmark")
    print("=" * 65)

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    model_start = time.perf_counter()

    embedder = TextEmbedding(
        model_name=MODEL_NAME
    )

    model_load_ms = (
        time.perf_counter()
        - model_start
    ) * 1000

    print(
        f"Model loaded : "
        f"{model_load_ms:.2f} ms"
    )

    # --------------------------------------------------------
    # QDRANT
    # --------------------------------------------------------

    print("\nConnecting to Qdrant...")

    client = QdrantClient(
        path=QDRANT_PATH
    )

    collection = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Collection   : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Vectors      : "
        f"{collection.points_count}"
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    print("\nLoading Hindi ground truth...")

    ground_truth = load_ground_truth()

    print(
        f"Ground-truth queries : "
        f"{len(ground_truth)}"
    )

    queries = list(
        ground_truth.items()
    )[:MAX_QUERIES]

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    print("\nWarming up...")

    list(
        embedder.embed(
            ["मैनहट्टन परियोजना क्या थी?"]
        )
    )

    print("Warm-up complete.")

    # --------------------------------------------------------
    # LATENCY ARRAYS
    # --------------------------------------------------------

    embedding_times = []
    qdrant_times = []
    total_times = []

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    print("\nRunning benchmark...")
    print(
        f"Queries : {len(queries)}"
    )

    for index, (query_id, data) in enumerate(
        queries,
        start=1
    ):

        query = data["query"]

        selected = data["selected"]

        # ====================================================
        # EMBEDDING
        # ====================================================

        start = time.perf_counter()

        vector = list(
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

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector.tolist(),
            limit=TOP_K,
            with_payload=True,
            with_vectors=False,
        ).points

        qdrant_ms = (
            time.perf_counter()
            - start
        ) * 1000

        total_ms = (
            embedding_ms
            + qdrant_ms
        )

        embedding_times.append(
            embedding_ms
        )

        qdrant_times.append(
            qdrant_ms
        )

        total_times.append(
            total_ms
        )

        # ====================================================
        # RETRIEVED TEXT
        # ====================================================

        retrieved = []

        for result in results:

            payload = (
                result.payload
                or {}
            )

            text = payload.get(
                "text",
                ""
            )

            if text:
                retrieved.append(
                    text
                )

        # ====================================================
        # FIND BEST RANK
        # ====================================================

        best_rank = None

        for gt in selected:

            for rank, text in enumerate(
                retrieved,
                start=1
            ):

                if passage_matches(
                    text,
                    gt
                ):

                    if (
                        best_rank is None
                        or rank < best_rank
                    ):
                        best_rank = rank

                    break

        # ====================================================
        # RECALL
        # ====================================================

        if best_rank is not None:

            if best_rank <= 1:
                recall_1 += 1

            if best_rank <= 5:
                recall_5 += 1

            if best_rank <= 10:
                recall_10 += 1

        # ====================================================
        # PROGRESS
        # ====================================================

        if index % 50 == 0:

            print(
                f"[{index}/{len(queries)}] "
                f"{total_ms:.2f} ms"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    total = len(queries)

    print("\n")
    print("=" * 65)
    print("STEP 8 COMPLETE")
    print("=" * 65)

    print("\nRETRIEVAL QUALITY")
    print("-" * 65)

    print(
        f"Recall@1  : "
        f"{recall_1 / total * 100:.2f}%"
    )

    print(
        f"Recall@5  : "
        f"{recall_5 / total * 100:.2f}%"
    )

    print(
        f"Recall@10 : "
        f"{recall_10 / total * 100:.2f}%"
    )

    print("\nLATENCY")
    print("-" * 65)

    print(
        f"Embedding P50 : "
        f"{percentile(embedding_times, 50):.2f} ms"
    )

    print(
        f"Embedding P70 : "
        f"{percentile(embedding_times, 70):.2f} ms"
    )

    print(
        f"Embedding P100: "
        f"{percentile(embedding_times, 100):.2f} ms"
    )

    print()

    print(
        f"Qdrant P50    : "
        f"{percentile(qdrant_times, 50):.2f} ms"
    )

    print(
        f"Qdrant P70    : "
        f"{percentile(qdrant_times, 70):.2f} ms"
    )

    print(
        f"Qdrant P100   : "
        f"{percentile(qdrant_times, 100):.2f} ms"
    )

    print()

    print(
        f"Retrieval P50 : "
        f"{percentile(total_times, 50):.2f} ms"
    )

    print(
        f"Retrieval P70 : "
        f"{percentile(total_times, 70):.2f} ms"
    )

    print(
        f"Retrieval P100: "
        f"{percentile(total_times, 100):.2f} ms"
    )

    print("\nTARGET")
    print("-" * 65)

    print(
        "Post-STT complete pipeline: < 200 ms"
    )

    print(
        "Current stage: "
        "Query Embedding + Qdrant"
    )

    print("=" * 65)

    client.close()


if __name__ == "__main__":
    main()