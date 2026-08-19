"""
Diagnose the 65% blocked rate.
Samples 100 queries and shows block reasons.
"""
import json
import sys
from collections import Counter
from pathlib import Path

# Use UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.pipeline import RAGEngine

engine = RAGEngine()

queries = []
with open("data/hindi_sample_1000.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            qid = item.get("query_id")
            q = item.get("query", "")
            if qid and q:
                queries.append({"query_id": qid, "query": q})

# Sample 200 queries
sample = queries[:200]

reasons = Counter()
blocked_examples = []
allowed_examples = []

for item in sample:
    result = engine.process(item["query"])

    if result.get("blocked"):
        reasons[result.get("reason", "unknown")] += 1
        if len(blocked_examples) < 10:
            blocked_examples.append({
                "query": item["query"],
                "reason": result.get("reason"),
                "retrieved": len(result.get("sources", [])),
                "top_score": result["retrieval"]["top3"][0]["score"]
                if result["retrieval"]["top3"] else 0.0,
                "overlap": result.get("context_overlap"),
            })
    else:
        allowed_examples.append({
            "query": item["query"],
            "answer": result.get("answer", "")[:100],
        })

print("=" * 70)
print("BLOCK REASON ANALYSIS (n=200)")
print("=" * 70)
print()
print("Block reasons:")
for reason, count in reasons.most_common():
    print(f"  {reason}: {count}")

print()
print(f"Total blocked:  {sum(reasons.values())}")
print(f"Total allowed:  {200 - sum(reasons.values())}")
print(f"Block rate:     {sum(reasons.values()) / 200 * 100:.1f}%")

print()
print("=" * 70)
print("BLOCKED EXAMPLES (first 10)")
print("=" * 70)
for ex in blocked_examples:
    print(f"  Query:    {ex['query'][:80]}")
    print(f"  Reason:   {ex['reason']}")
    print(f"  Sources:  {ex['retrieved']}")
    print(f"  Score:    {ex['top_score']:.4f}")
    print()

print()
print("=" * 70)
print("ALLOWED EXAMPLES (first 5)")
print("=" * 70)
for ex in allowed_examples[:5]:
    print(f"  Query:  {ex['query'][:80]}")
    print(f"  Answer: {ex['answer']}")
    print()
