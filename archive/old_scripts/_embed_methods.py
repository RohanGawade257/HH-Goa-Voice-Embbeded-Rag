import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from fastembed import TextEmbedding
import numpy as np

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
emb = TextEmbedding(model_name=MODEL)

# Warmup
for _ in range(5):
    list(emb.embed(["warmup"]))

# Method 1: list(emb.embed([q]))[0]  -- what pipeline.py does currently
times1 = []
for _ in range(50):
    t = time.perf_counter()
    v = list(emb.embed(["मैनहट्टन परियोजना"]))[0]
    times1.append((time.perf_counter()-t)*1000)

# Method 2: next(iter(emb.embed([q])))
times2 = []
for _ in range(50):
    t = time.perf_counter()
    v = next(iter(emb.embed(["मैनहट्टन परियोजना"])))
    times2.append((time.perf_counter()-t)*1000)

# Method 3: emb.embed([q], batch_size=1) -- explicit batch size
times3 = []
for _ in range(50):
    t = time.perf_counter()
    v = list(emb.embed(["मैनहट्टन परियोजना"], batch_size=1))[0]
    times3.append((time.perf_counter()-t)*1000)

# Method 4: np.array(list(...)) vs direct
times4 = []
arr = None
for _ in range(50):
    t = time.perf_counter()
    gen = emb.embed(["मैनहट्टन परियोजना"])
    v = next(gen)
    times4.append((time.perf_counter()-t)*1000)

def p50(ts): return sorted(ts)[len(ts)//2]

print("Embed call method comparison (P50, ms):")
print(f"  list(embed([q]))[0]           : {p50(times1):.2f}")
print(f"  next(iter(embed([q])))         : {p50(times2):.2f}")
print(f"  list(embed([q], batch_size=1)): {p50(times3):.2f}")
print(f"  next(embed([q]))               : {p50(times4):.2f}")
print()
print(f"Vector dim: {len(v)}, type: {type(v)}")
