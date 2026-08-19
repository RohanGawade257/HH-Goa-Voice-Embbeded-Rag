import json
import os
import time
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/processed/chunks_1000.jsonl"

COLLECTION_NAME = "hh_goa_rag_hindi"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BATCH_SIZE = 128

QDRANT_PATH = "data/qdrant"

# paraphrase-multilingual-MiniLM-L12-v2 = 384 dimensions
VECTOR_SIZE = 384


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    chunks = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                chunks.append(
                    json.loads(line)
                )

    return chunks


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("HH GOA RAG - STEP 7")
    print("Embedding Generation")
    print("=" * 65)

    chunks = load_chunks()

    print(
        f"\nLoaded chunks: {len(chunks)}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    model_start = time.perf_counter()

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    model_load_time = (
        time.perf_counter()
        - model_start
    ) * 1000

    print(
        f"Model loaded in "
        f"{model_load_time:.2f} ms"
    )

    # --------------------------------------------------------
    # QDRANT
    # --------------------------------------------------------

    os.makedirs(
        QDRANT_PATH,
        exist_ok=True
    )

    client = QdrantClient(
        path=QDRANT_PATH
    )

    # --------------------------------------------------------
    # CREATE COLLECTION
    # --------------------------------------------------------

    if client.collection_exists(
        COLLECTION_NAME
    ):

        print(
            f"\nDeleting existing collection: "
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(
            COLLECTION_NAME
        )

    print(
        f"\nCreating collection: "
        f"{COLLECTION_NAME}"
    )

    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=models.VectorParams(

            size=VECTOR_SIZE,

            distance=models.Distance.COSINE
        )
    )

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "\nGenerating embeddings..."
    )

    embedding_start = time.perf_counter()

    embeddings = embedding_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    ) * 1000

    print(
        f"Embedding time: "
        f"{embedding_time:.2f} ms"
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    print(
        "\nUploading vectors to Qdrant..."
    )

    upload_start = time.perf_counter()

    points = []

    for i, chunk in enumerate(chunks):

        payload = {
            "chunk_id": chunk["chunk_id"],
            "passage_id": chunk["passage_id"],
            "query_id": chunk["query_id"],
            "passage_index": chunk["passage_index"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "language": chunk["language"],
            "word_count": chunk["word_count"],
            "chunk_strategy": chunk["chunk_strategy"],
            "has_overlap": chunk["has_overlap"],
            "is_selected": chunk["is_selected"],
            "query_type": chunk["query_type"],
        }

        points.append(
            models.PointStruct(

                id=i,

                vector=embeddings[i].tolist(),

                payload=payload
            )
        )

        # Upload in batches
        if len(points) >= BATCH_SIZE:

            client.upsert(

                collection_name=COLLECTION_NAME,

                points=points
            )

            points = []

    # Remaining points

    if points:

        client.upsert(

            collection_name=COLLECTION_NAME,

            points=points
        )

    upload_time = (
        time.perf_counter()
        - upload_start
    ) * 1000

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"\nUpload time: "
        f"{upload_time:.2f} ms"
    )

    print(
        f"Collection points: "
        f"{collection_info.points_count}"
    )

    print("\n" + "=" * 65)
    print("STEP 7 COMPLETE")
    print("=" * 65)

    print(
        f"\nVectors generated : {len(embeddings)}"
    )

    print(
        f"Vector dimension   : {VECTOR_SIZE}"
    )

    print(
        f"Embedding time     : "
        f"{embedding_time:.2f} ms"
    )

    print(
        f"Upload time        : "
        f"{upload_time:.2f} ms"
    )

    print(
        f"\nQdrant collection  : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Qdrant storage     : "
        f"{QDRANT_PATH}"
    )


if __name__ == "__main__":
    main()
