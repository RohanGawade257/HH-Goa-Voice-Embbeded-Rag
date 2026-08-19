"""
Deep-dive on context_overlap threshold issue.
"""
import json
import re
import sys
sys.stdout.reconfigure(encoding="utf-8")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.pipeline import RAGEngine

engine = RAGEngine()

# Test the overlap calculation on specific cases
def keyword_overlap(query, context):
    query_words = set(
        w.lower()
        for w in re.findall(r"\w+", query)
        if len(w) > 2
    )
    context_words = set(
        w.lower()
        for w in re.findall(r"\w+", context)
        if len(w) > 2
    )
    if not query_words:
        return 0.0
    return len(query_words & context_words) / len(query_words)

# Sample queries that are blocked
test_queries = [
    "विभिन्न प्रकार की सामाजिक सुरक्षा विकलांगता",
    "कारों पर अमेरिकी ध्वज के स्टिकर का क्या अर्थ है?",
    "फ्लूम किस दिशा में बहता है",
    "क्रेविस को परिभाषित करें",
    "समुद्र तल से ऊंचाई पर",
]

for query in test_queries:
    result = engine.process(query)
    top3 = result["retrieval"]["top3"]
    if top3:
        combined = " ".join(s["text"] for s in top3)
        overlap = keyword_overlap(query, combined)
        q_words = set(w.lower() for w in re.findall(r"\w+", query) if len(w) > 2)
        c_words = set(w.lower() for w in re.findall(r"\w+", combined) if len(w) > 2)
        common = q_words & c_words
        print(f"Query: {query}")
        print(f"  Overlap: {overlap:.4f}")
        print(f"  Query words ({len(q_words)}): {list(q_words)[:8]}")
        print(f"  Common words ({len(common)}): {list(common)[:5]}")
        print(f"  Top score: {top3[0]['score']:.4f}")
        print(f"  Top text: {top3[0]['text'][:120]}")
        print()
