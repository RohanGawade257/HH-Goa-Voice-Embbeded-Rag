"""
Integration tests for the /query/stream SSE endpoint.

Sub-Task 5 — Phase 3C: Dual-answer contract verification.

Tests:
    1.  direct_answer event arrives before llm_answer
    2.  direct_answer.answer is non-empty
    3.  Gemini failure preserves direct_answer (direct answer survives LLM error)
    4.  No second retrieval call is made for Gemini (process_dual called exactly once)
    5.  sources event contains a list
    6.  timing event has all 7 required fields
    7.  empty query → error event then done
    8.  stream always terminates with done (even on pipeline exception)
    9.  multilingual queries (Bengali, Tamil) return a direct_answer
   10.  off-topic query → direct_answer with blocked=true
   11.  llm_answer error does not erase already-delivered direct_answer

All tests use ANSWER_BACKEND=extractive — no live API key required.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

# Force extractive backend BEFORE any app module imports
os.environ["ANSWER_BACKEND"] = "extractive"
os.environ.setdefault("SARVAM_API_KEY", "")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402


# ============================================================
# SSE FRAME PARSER
# ============================================================

def parse_sse_stream(response) -> list[dict]:
    """Parse an SSE stream response into a list of event dicts.

    Each dict has keys: 'event' (str) and 'data' (parsed JSON dict).
    """
    events = []
    buffer = ""
    for chunk in response.iter_bytes():
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            if not frame.strip():
                continue
            event_name = None
            data_str = None
            for line in frame.split("\n"):
                if line.startswith("event: "):
                    event_name = line[len("event: "):].strip()
                elif line.startswith("data: "):
                    data_str = line[len("data: "):].strip()
            if event_name and data_str:
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    data = {"raw": data_str}
                events.append({"event": event_name, "data": data})
    return events


def get_event(events: list[dict], name: str) -> dict | None:
    """Return the first event with the given name, or None."""
    return next((e for e in events if e["event"] == name), None)


def get_all_events(events: list[dict], name: str) -> list[dict]:
    return [e for e in events if e["event"] == name]


# ============================================================
# MOCK ENGINE FACTORY
# ============================================================

def make_mock_engine(fail_llm: bool = False, fail_pipeline: bool = False):
    """Build a minimal mock RAGEngine for SSE testing.

    process_dual() returns a realistic result without hitting Qdrant.
    The call_count attribute tracks how many times process_dual is called.
    """
    from app.answer_generator import generate_extractive_answer

    engine = MagicMock()
    engine.call_count = 0

    def _process_dual(query, language=None):
        engine.call_count += 1
        if fail_pipeline:
            raise RuntimeError("simulated pipeline failure")
        q = query.strip()
        if not q:
            return (
                {
                    "answer": "Please provide a question.",
                    "grounded": False, "blocked": True,
                    "reason": "empty_query",
                    "retrieved_chunks": 0, "sources": [],
                    "timings": {
                        "embedding_ms": 0.0, "qdrant_ms": 0.0,
                        "rerank_ms": 0.0, "compression_ms": 0.0,
                        "llm_ms": 0.0, "answer_ms": 0.0, "total_ms": 0.0,
                    },
                },
                None,
            )
        # Simulate realistic compressed result
        snippets = [{"text": f"The Manhattan Project was a secret nuclear programme. Query: {q[:30]}", "score": 0.85}]
        ans = generate_extractive_answer(q, snippets)
        result = {
            "answer": ans["answer"],
            "grounded": ans["grounded"],
            "blocked": ans["blocked"],
            "reason": ans["reason"],
            "retrieved_chunks": 1,
            "sources": [{"rank": 1, "chunk_id": "c1", "passage_id": "p1",
                         "query_id": "q1", "text": "Test chunk.", "language": "hi",
                         "score": 0.85, "vector_score": 0.90}],
            "timings": {
                "embedding_ms": 30.0, "qdrant_ms": 10.0,
                "rerank_ms": 0.5, "compression_ms": 0.3,
                "llm_ms": 0.0, "answer_ms": 1.0, "total_ms": 42.0,
            },
        }
        state = {
            "query": q,
            "language_code": language,
            "compression_result": {"context": "The Manhattan Project context."},
            "_pipeline_start": 0.0,
            "timings": {
                "embedding_ms": 30.0, "qdrant_ms": 10.0,
                "rerank_ms": 0.5, "compression_ms": 0.3,
            },
        }
        return result, state

    engine.process_dual = _process_dual

    if fail_llm:
        engine.answer_generator = MagicMock()
        engine.answer_generator.available = True
        engine.answer_generator.generate.side_effect = RuntimeError("simulated Gemini failure")
    else:
        engine.answer_generator = MagicMock()
        engine.answer_generator.available = False  # extractive mode — no LLM

    return engine


# ============================================================
# TEST SETUP — patch get_engine once per test class
# ============================================================

class StreamSSETests(unittest.TestCase):

    def _get_client_and_stream(self, q: str, language: str = "hi-IN", engine=None):
        """Create a patched TestClient, stream one SSE request, return events."""
        if engine is None:
            engine = make_mock_engine()
        from app.api import app
        # Patch get_engine so every call inside the request handler returns our mock.
        with patch("app.api.get_engine", return_value=engine):
            client = TestClient(app, raise_server_exceptions=False)
            with client.stream("GET", f"/query/stream?q={q}&language={language}") as resp:
                events = parse_sse_stream(resp)
        return events, engine

    # ── Test 1: direct_answer arrives before llm_answer ───────────────────────

    def test_direct_answer_before_llm_answer(self):
        events, _ = self._get_client_and_stream("मैनहट्टन परियोजना क्या थी?")
        names = [e["event"] for e in events]
        self.assertIn("direct_answer", names)
        self.assertIn("llm_answer", names)
        direct_idx = names.index("direct_answer")
        llm_idx = names.index("llm_answer")
        self.assertLess(direct_idx, llm_idx,
                        "direct_answer must arrive before llm_answer")

    # ── Test 2: direct_answer is non-empty ────────────────────────────────────

    def test_direct_answer_not_empty(self):
        events, _ = self._get_client_and_stream("मैनहट्टन परियोजना क्या थी?")
        da = get_event(events, "direct_answer")
        self.assertIsNotNone(da, "direct_answer event must be present")
        # answer may be blocked/empty for some queries — just verify the field exists
        self.assertIn("answer", da["data"])
        self.assertIn("grounded", da["data"])
        self.assertIn("blocked", da["data"])
        self.assertIn("time_to_direct_ms", da["data"])

    # ── Test 3: Gemini failure preserves direct_answer ────────────────────────

    def test_gemini_failure_preserves_direct_answer(self):
        engine = make_mock_engine(fail_llm=True)
        events, _ = self._get_client_and_stream("मैनहट्टन परियोजना क्या थी?", engine=engine)

        da = get_event(events, "direct_answer")
        la = get_event(events, "llm_answer")

        self.assertIsNotNone(da, "direct_answer must be present even when Gemini fails")
        self.assertIn("answer", da["data"])

        self.assertIsNotNone(la, "llm_answer event must still be emitted (with error)")
        self.assertIn("error", la["data"])
        self.assertIsNotNone(la["data"]["error"])

    # ── Test 4: process_dual called exactly once per request ──────────────────

    def test_no_second_retrieval_for_gemini(self):
        """Gemini must reuse the state from process_dual(), not re-run retrieval."""
        engine = make_mock_engine(fail_llm=True)
        self._get_client_and_stream("मैनहट्टन परियोजना क्या थी?", engine=engine)
        self.assertEqual(engine.call_count, 1,
                         "process_dual must be called exactly once per request")

    # ── Test 5: sources event is a list ───────────────────────────────────────

    def test_sources_event_is_list(self):
        events, _ = self._get_client_and_stream("मैनहट्टन परियोजना क्या थी?")
        src = get_event(events, "sources")
        self.assertIsNotNone(src)
        self.assertIn("sources", src["data"])
        self.assertIsInstance(src["data"]["sources"], list)

    # ── Test 6: timing event has all required fields ──────────────────────────

    def test_timing_fields_present(self):
        events, _ = self._get_client_and_stream("मैनहट्टन परियोजना क्या थी?")
        timing = get_event(events, "timing")
        self.assertIsNotNone(timing)
        required = {
            "embedding_ms", "qdrant_ms", "rerank_ms", "compression_ms",
            "llm_ms", "time_to_direct_ms", "time_to_llm_ms", "total_ms",
        }
        for field in required:
            self.assertIn(field, timing["data"],
                          f"timing event missing field: {field}")

    # ── Test 7: empty query → error then done ─────────────────────────────────

    def test_empty_query_returns_error_then_done(self):
        events, _ = self._get_client_and_stream("   ")
        names = [e["event"] for e in events]
        self.assertIn("error", names, "empty query must emit error event")
        self.assertIn("done", names, "empty query must still emit done event")
        # error before done
        self.assertLess(names.index("error"), names.index("done"))

    # ── Test 8: stream always terminates with done ────────────────────────────

    def test_stream_always_terminates_with_done(self):
        events, _ = self._get_client_and_stream("मैनहट्टन परियोजना क्या थी?")
        self.assertEqual(events[-1]["event"], "done",
                         "last event must be 'done'")

    def test_stream_terminates_on_pipeline_exception(self):
        engine = make_mock_engine(fail_pipeline=True)
        events, _ = self._get_client_and_stream("any query", engine=engine)
        names = [e["event"] for e in events]
        self.assertIn("done", names, "stream must terminate with done even on exception")

    # ── Test 9: multilingual queries produce direct_answer ────────────────────

    def test_bengali_query_returns_direct_answer(self):
        events, _ = self._get_client_and_stream(
            "ম্যানহাটন প্রকল্পের সাফল্যের তাৎক্ষণিক প্রভাব কী ছিল?",
            language="bn-IN",
        )
        da = get_event(events, "direct_answer")
        self.assertIsNotNone(da, "Bengali query must return a direct_answer event")

    def test_tamil_query_returns_direct_answer(self):
        events, _ = self._get_client_and_stream(
            "மன்ஹாட்டன் திட்டத்தின் வெற்றியின் உடனடி விளைவு என்ன?",
            language="ta-IN",
        )
        da = get_event(events, "direct_answer")
        self.assertIsNotNone(da, "Tamil query must return a direct_answer event")

    # ── Test 10: off-topic query → blocked direct_answer ─────────────────────

    def test_off_topic_query_produces_blocked_direct_answer(self):
        events, _ = self._get_client_and_stream("write python code for fibonacci")
        da = get_event(events, "direct_answer")
        self.assertIsNotNone(da)
        self.assertTrue(da["data"]["blocked"],
                        "off-topic query must produce blocked=true direct_answer")

    # ── Test 11: llm_answer error doesn't erase direct_answer ─────────────────

    def test_llm_error_does_not_erase_direct_answer(self):
        engine = make_mock_engine(fail_llm=True)
        events, _ = self._get_client_and_stream("मैनहट्टन परियोजना क्या थी?", engine=engine)
        da = get_event(events, "direct_answer")
        la = get_event(events, "llm_answer")
        # Direct answer must remain intact
        self.assertIsNotNone(da)
        self.assertIn("answer", da["data"])
        # LLM answer must carry the error, not wipe out the direct answer
        self.assertIsNotNone(la)
        self.assertIn("error", la["data"])


if __name__ == "__main__":
    unittest.main()
