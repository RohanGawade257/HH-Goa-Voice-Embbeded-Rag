import time

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


def embed_query(query: str):

    start = time.perf_counter()

    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return embedding, latency_ms


if __name__ == "__main__":

    query = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"

    vector, latency = embed_query(query)

    print(
        f"\nVector dimensions: "
        f"{len(vector)}"
    )

    print(
        f"Query embedding latency: "
        f"{latency:.2f} ms"
    )