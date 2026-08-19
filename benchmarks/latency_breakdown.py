import sys, time, statistics
sys.stdout.reconfigure(encoding="utf-8")
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION = "hh_goa_rag_hindi"
QDRANT_PATH = "data/qdrant"

emb = SentenceTransformer(MODEL)
client = QdrantClient(path=QDRANT_PATH)

queries = [
    "मैनहट्टन परियोजना क्या थी?",
    "सौर ऊर्जा कैसे काम करती है?",
    "डीएनए की संरचना क्या है?",
    "प्रकाश की गति कितनी है?",
    "भारत की राजधानी क्या है?",
]

# Warmup x10
for _ in range(10):
    v = emb.encode("warmup", convert_to_numpy=True, normalize_embeddings=True)
    client.query_points(collection_name=COLLECTION, query=v, limit=20, with_payload=False, with_vectors=False)

# Detailed breakdown
embed_times = []
search_times = []
total_times = []

N = 100
for i in range(N):
    q = queries[i % len(queries)]

    t0 = time.perf_counter()
    # Exactly how pipeline.py does it
    query_vector = emb.encode(q, convert_to_numpy=True, normalize_embeddings=True)
    t1 = time.perf_counter()

    client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=20,
        with_payload=["chunk_id","passage_id","query_id","text","is_selected","chunk_strategy","word_count"],
        with_vectors=False
    )
    t2 = time.perf_counter()

    embed_times.append((t1-t0)*1000)
    search_times.append((t2-t1)*1000)
    total_times.append((t2-t0)*1000)

def pct(vals, p):
    vals = sorted(vals)
    k = (len(vals)-1)*p/100
    f = int(k)
    return vals[f] + (k-f)*(vals[min(f+1,len(vals)-1)]-vals[f])

print(f"n={N} queries | sentence-transformers | paraphrase-multilingual-MiniLM-L12-v2")
print()
print(f"{'stage':<12}{'avg':>8}{'p50':>8}{'p95':>8}{'p99':>8}   (ms)")
print("-" * 52)
for name, v in [("embed", embed_times), ("search", search_times), ("total", total_times)]:
    print(f"{name:<12}{statistics.mean(v):>8.2f}{pct(v,50):>8.2f}{pct(v,95):>8.2f}{pct(v,99):>8.2f}")

print()
p95 = pct(total_times, 95)
print(f"Budget: 200 ms | p95 total: {p95:.2f} ms")
print("PASS" if p95 <= 200 else "FAIL")

# Check encode overhead without numpy conversion
import numpy as np
t0 = time.perf_counter()
for _ in range(100):
    v = emb.encode("test", convert_to_numpy=True, normalize_embeddings=True)
t1 = time.perf_counter()

print()
print(f"encode() avg per call: {(t1-t0)*10:.2f} ms avg")
client.close()
