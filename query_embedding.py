import time

from fastembed import TextEmbedding


MODEL_NAME = "intfloat/multilingual-e5-small"


print("Loading embedding model...")

model = TextEmbedding(
    model_name=MODEL_NAME
)


def embed_query(query: str):

    start = time.perf_counter()

    # E5 retrieval convention
    query_text = f"query: {query}"

    embedding = next(
        model.embed([query_text])
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    return embedding, latency_ms


if __name__ == "__main__":

    query = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"

    vector, latency = embed_query(
        query
    )

    print(
        f"\nVector dimensions: "
        f"{len(vector)}"
    )

    print(
        f"Query embedding latency: "
        f"{latency:.2f} ms"
    )
    