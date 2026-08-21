"""
Smoke-test the FastAPI app endpoints using TestClient.
"""

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ["ANSWER_BACKEND"] = "extractive"
os.environ["SARVAM_API_KEY"] = ""
os.environ["SARVAM_STT_MOCK"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)

print("=" * 60)
print("FastAPI Endpoint Tests")
print("=" * 60)
print()

print("[1] GET /health")
resp = client.get("/health")
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
data = resp.json()
assert data["status"] == "ok"
assert data["rag_engine"]["status"] == "loaded"
assert data["qdrant"]["status"] == "ok"
print(f"  Status: {data['status']}")
print(f"  Qdrant points: {data['qdrant']['points']}")
print(f"  P95 target: {data['performance_target']['post_stt_p95_target_ms']} ms")
print("  PASS")
print()

print("[2] POST /query")
resp = client.post(
    "/query",
    json={
        "query": "\u092e\u0948\u0928\u0939\u091f\u094d\u091f\u0928 \u092a\u0930\u093f\u092f\u094b\u091c\u0928\u093e \u0915\u094d\u092f\u093e \u0925\u0940?",
        "language": "hi-IN",
    },
)
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
data = resp.json()
print(f"  Total ms: {data['timings']['total_ms']:.1f}")
print(f"  Grounded: {data['grounded']}")
print(f"  Retrieved chunks: {data['retrieved_chunks']}")
print(f"  Answer (first 80): {data['answer'][:80]}")
assert data["retrieved_chunks"] == 3
print("  PASS")
print()

print("[3] POST /query (validation - empty)")
resp = client.post("/query", json={"query": ""})
assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
print(f"  Status: {resp.status_code} (validation error - expected)")
print("  PASS")
print()

print("[4] POST /query (off-topic)")
resp = client.post("/query", json={"query": "write python code"})
assert resp.status_code == 200
data = resp.json()
print(f"  Blocked: {data['blocked']}")
print(f"  Reason: {data['reason']}")
assert data["blocked"] is True
assert data["reason"] == "off_topic"
print("  PASS")
print()

print("[5] POST /voice (mock STT)")
wav_header = (
    b"RIFF" + (44).to_bytes(4, "little") + b"WAVE" +
    b"fmt " + (16).to_bytes(4, "little") +
    (1).to_bytes(2, "little") +
    (1).to_bytes(2, "little") +
    (16000).to_bytes(4, "little") +
    (32000).to_bytes(4, "little") +
    (2).to_bytes(2, "little") +
    (16).to_bytes(2, "little") +
    b"data" + (0).to_bytes(4, "little")
)
resp = client.post(
    "/voice",
    files={"file": ("test.wav", wav_header, "audio/wav")},
    data={"language": "hi-IN"},
)
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
data = resp.json()
print(f"  STT provider: {data['stt_provider']}")
print(f"  Transcript: {data['transcript']}")
print(f"  RAG total: {data['rag_timings']['total_ms']:.1f} ms")
assert data["stt_provider"] == "mock"
print("  PASS")
print()

print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
