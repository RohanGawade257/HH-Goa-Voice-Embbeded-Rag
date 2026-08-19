import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.pipeline import RAGEngine

print("Loading engine...")
engine = RAGEngine()

# Warmup
for _ in range(5):
    engine.process("मैनहट्टन परियोजना क्या थी?")

# Test queries
queries = [
    "मैनहट्टन परियोजना क्या थी?",
    "सौर ऊर्जा कैसे काम करती है?",
    "भारत की राजधानी क्या है?",
    "पृथ्वी का वजन कितना है?",
    "डीएनए क्या है?",
    "प्रकाश की गति कितनी है?",
    "ताजमहल कहाँ है?",
    "पानी का रासायनिक सूत्र क्या है?",
]

print()
print("Single-query latency measurements (post-warmup):")
print("-" * 70)

totals = []
for q in queries:
    result = engine.process(q)
    t = result["timings"]
    totals.append(t["total_ms"])
    print(
        f"  embed={t['embedding_ms']:.1f}ms "
        f"qdrant={t['qdrant_ms']:.1f}ms "
        f"rerank={t['rerank_ms']:.1f}ms "
        f"answer={t['answer_ms']:.1f}ms "
        f"TOTAL={t['total_ms']:.1f}ms "
        f"grounded={result['grounded']}"
    )

print()
print(f"Min total:  {min(totals):.1f} ms")
print(f"Max total:  {max(totals):.1f} ms")
print(f"Mean total: {sum(totals)/len(totals):.1f} ms")
print()

# Show a sample answer
r = engine.process("मैनहट्टन परियोजना क्या थी?")
print("Sample answer:")
print(r["answer"])
print("Grounded:", r["grounded"])
print("Reason:", r["reason"])
