import json
import re
import time
import statistics
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ============================================================
# CONFIG
# ============================================================

COLLECTION_NAME = "hh_goa_rag_hindi"

QDRANT_PATH = "data/qdrant"

CHUNKS_FILE = "data/processed/chunks_1000.jsonl"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 3

# Fast reranking weights
VECTOR_WEIGHT = 0.70
LEXICAL_WEIGHT = 0.20
PHRASE_WEIGHT = 0.10


# ============================================================
# TEXT UTILITIES
# ============================================================

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    text = text.lower().strip()

    # Keep Devanagari + normal Unicode words
    text = re.sub(r"[^\w\u0900-\u097F\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text


def tokenize(text: str):
    return set(WORD_RE.findall(normalize(text)))


def lexical_overlap(query: str, document: str) -> float:
    q_tokens = tokenize(query)
    d_tokens = tokenize(document)

    if not q_tokens or not d_tokens:
        return 0.0

    intersection = q_tokens.intersection(d_tokens)

    return len(intersection) / len(q_tokens)


def phrase_score(query: str, document: str) -> float:
    q = normalize(query)
    d = normalize(document)

    if not q:
        return 0.0

    # Exact query occurrence
    if q in d:
        return 1.0

    # Check meaningful phrases
    q_words = q.split()

    if len(q_words) >= 3:

        for size in [3, 4]:
            for i in range(len(q_words) - size + 1):

                phrase = " ".join(q_words[i:i + size])

                if phrase in d:
                    return 0.7

    return 0.0


# ============================================================
# FAST RERANKER
# ============================================================

def rerank(query, hits):

    if not hits:
        return []

    scores = []

    for hit in hits:

        payload = hit.payload or {}

        text = payload.get("text", "")

        vector_score = float(hit.score)

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
# LOAD QUERIES
# ============================================================

def load_queries():

    queries = []

    # Use the original sampled dataset because it contains
    # the actual Hindi query and query_id.
    path = Path("data/hindi_sample_1000.jsonl")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing query dataset: {path}"
        )

    seen = set()

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            item = json.loads(line)

            query_id = item.get("query_id")
            query = item.get("query")

            if query_id is None or not query:
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG - STEP 9")
    print("FAST LATENCY-AWARE RERANKING")
    print("=" * 70)

    print()
    print("Embedding model:")
    print(MODEL_NAME)

    print()
    print("Loading embedding model...")

    embedder = SentenceTransformer(
        MODEL_NAME
    )

    print("Embedding model loaded.")

    print()
    print("Opening Qdrant...")

    client = QdrantClient(
        path=QDRANT_PATH
    )

    print("Qdrant opened.")

    queries = load_queries()

    print()
    print(f"Benchmark queries: {len(queries)}")

    if not queries:
        print("No benchmark queries found.")
        return

    embedding_times = []
    qdrant_times = []
    rerank_times = []
    total_times = []

    recall_before = {
        1: 0,
        5: 0,
        10: 0
    }

    recall_after = {
        1: 0,
        3: 0
    }

    processed = 0

    print()
    print("Running benchmark...")
    print()

    for item in queries:

        query = item["query"]
        query_id = item["query_id"]

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        start = time.perf_counter()

        query_vector = embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embedding_ms = (
            time.perf_counter() - start
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
                "word_count"
            ],
            with_vectors=False
        )

        hits = response.points

        qdrant_ms = (
            time.perf_counter() - start
        ) * 1000

        # ----------------------------------------------------
        # BEFORE RERANKING
        # ----------------------------------------------------

        retrieved_ids = [
            hit.payload.get("query_id")
            for hit in hits
        ]

        # Ground truth:
        # The source query_id identifies the original query.
        target = query_id

        for k in [1, 5, 10]:

            if target in retrieved_ids[:k]:

                recall_before[k] += 1

        # ----------------------------------------------------
        # FAST RERANK
        # ----------------------------------------------------

        start = time.perf_counter()

        reranked = rerank(
            query,
            hits
        )

        rerank_ms = (
            time.perf_counter() - start
        ) * 1000

        reranked_ids = [
            result[4].payload.get("query_id")
            for result in reranked
        ]

        for k in [1, 3]:

            if target in reranked_ids[:k]:

                recall_after[k] += 1

        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        total_ms = (
            embedding_ms
            + qdrant_ms
            + rerank_ms
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

        total_times.append(
            total_ms
        )

        processed += 1

        if processed % 25 == 0:

            print(
                f"[{processed}/{len(queries)}] "
                f"Embedding={embedding_ms:.2f} ms | "
                f"Qdrant={qdrant_ms:.2f} ms | "
                f"Rerank={rerank_ms:.2f} ms | "
                f"Total={total_ms:.2f} ms"
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    def percentile(values, p):

        values = sorted(values)

        if not values:
            return 0.0

        index = int(
            round(
                (p / 100)
                * (len(values) - 1)
            )
        )

        return values[index]

    def print_stats(name, values):

        print(
            f"{name:<15}"
            f"P50={percentile(values, 50):>8.2f} ms | "
            f"P70={percentile(values, 70):>8.2f} ms | "
            f"P100={max(values):>8.2f} ms"
        )

    print()
    print("=" * 70)
    print("STEP 9 COMPLETE")
    print("=" * 70)

    print()
    print("RETRIEVAL BEFORE FAST RERANK")
    print("-" * 70)

    print(
        f"Recall@1  : "
        f"{recall_before[1] / processed * 100:.2f}%"
    )

    print(
        f"Recall@5  : "
        f"{recall_before[5] / processed * 100:.2f}%"
    )

    print(
        f"Recall@10 : "
        f"{recall_before[10] / processed * 100:.2f}%"
    )

    print()
    print("RETRIEVAL AFTER FAST RERANK")
    print("-" * 70)

    print(
        f"Recall@1  : "
        f"{recall_after[1] / processed * 100:.2f}%"
    )

    print(
        f"Recall@3  : "
        f"{recall_after[3] / processed * 100:.2f}%"
    )

    print()
    print("LATENCY")
    print("-" * 70)

    print_stats(
        "Embedding",
        embedding_times
    )

    print_stats(
        "Qdrant",
        qdrant_times
    )

    print_stats(
        "Fast rerank",
        rerank_times
    )

    print_stats(
        "TOTAL",
        total_times
    )

    print()
    print("TARGET")
    print("-" * 70)

    p50 = percentile(total_times, 50)
    p70 = percentile(total_times, 70)
    p100 = max(total_times)

    print(
        f"Post-STT target : < 200 ms"
    )

    print(
        f"P50             : {p50:.2f} ms"
    )

    print(
        f"P70             : {p70:.2f} ms"
    )

    print(
        f"P100            : {p100:.2f} ms"
    )

    print()

    if p100 < 200:

        print(
            "STATUS: PASS - every measured query "
            "was under 200 ms."
        )

    elif p70 < 200:

        print(
            "STATUS: PARTIAL - P70 is under 200 ms, "
            "but tail latency needs optimization."
        )

    else:

        print(
            "STATUS: FAIL - retrieval path needs "
            "further optimization."
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()