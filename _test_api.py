"""
Test the FastAPI app endpoints using TestClient (no server needed).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

print("=" * 60)
print("FastAPI Endpoint Tests")
print("=" * 60)
print()

# Test 1: Health
print("[1] GET /health")
resp = client.get("/health")
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
data = resp.json()
assert data["status"] == "ok"
assert data["rag_engine"]["status"] == "loaded"
assert data["qdrant"]["status"] == "ok"
qdrant_points = data["qdrant"]["points"]
p50 = data["performance_target"]["last_benchmark_p50_ms"]
print(f"  Status: {data['status']}")
print(f"  Qdrant points: {qdrant_points}")
print(f"  Benchmark P50: {p50} ms")
print(f"  PASS")
print()

# Test 2: Query
print("[2] POST /query")
resp = client.post("/query", json={"query": "मैनहट्टन परियोजना क्या थी?"})
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
data = resp.json()
total_ms = data["timings"]["total_ms"]
grounded = data["grounded"]
retrieved = data["retrieved_chunks"]
print(f"  Total ms: {total_ms:.1f}")
print(f"  Grounded: {grounded}")
print(f"  Retrieved chunks: {retrieved}")
print(f"  Answer (first 80): {data['answer'][:80]}")
assert grounded is True, "Expected grounded answer"
assert retrieved == 3
assert total_ms < 200, f"Latency too high: {total_ms}"
print(f"  PASS")
print()

# Test 3: Empty query
print("[3] POST /query (validation - empty)")
resp = client.post("/query", json={"query": ""})
assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
print(f"  Status: {resp.status_code} (validation error - expected)")
print(f"  PASS")
print()

# Test 4: Off-topic query
print("[4] POST /query (off-topic)")
resp = client.post("/query", json={"query": "write python code"})
assert resp.status_code == 200
data = resp.json()
blocked = data["blocked"]
reason = data["reason"]
print(f"  Blocked: {blocked}")
print(f"  Reason: {reason}")
assert blocked is True
assert reason == "off_topic"
print(f"  PASS")
print()

# Test 5: Voice endpoint (mock STT)
print("[5] POST /voice (mock STT)")
# Create a minimal valid WAV header
wav_header = (
    b"RIFF" + (44).to_bytes(4, "little") + b"WAVE" +
    b"fmt " + (16).to_bytes(4, "little") +
    (1).to_bytes(2, "little") +  # PCM
    (1).to_bytes(2, "little") +  # mono
    (16000).to_bytes(4, "little") +  # sample rate
    (32000).to_bytes(4, "little") +  # byte rate
    (2).to_bytes(2, "little") +  # block align
    (16).to_bytes(2, "little") +  # bits per sample
    b"data" + (0).to_bytes(4, "little")
)
resp = client.post(
    "/voice",
    files={"file": ("test.wav", wav_header, "audio/wav")},
    data={"language": "hi-IN"},
)
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
data = resp.json()
stt_provider = data["stt_provider"]
transcript = data["transcript"]
rag_total = data["rag_timings"]["total_ms"]
print(f"  STT provider: {stt_provider}")
print(f"  Transcript: {transcript}")
print(f"  RAG total: {rag_total:.1f} ms")
assert stt_provider == "mock"
assert rag_total < 200
print(f"  PASS")
print()

print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
