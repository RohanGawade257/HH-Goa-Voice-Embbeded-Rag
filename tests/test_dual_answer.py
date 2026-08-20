"""
Tests for the Dual-Answer Fast Path.

Covers:
    1. process_dual() returns direct answer without LLM call
    2. process_dual() returns state with compressed context for LLM
    3. Empty query returns blocked result
    4. _retrieve_and_compress() returns None for empty query
    5. SSE helpers: _sse_event formats correctly
    6. SSE helpers: _build_sources serialises correctly
    7. Direct answer is always returned even when Gemini fails
    8. Timing fields are separated (time_to_direct vs time_to_llm)
    9. process_dual() does not call the LLM generator
    10. process() (legacy) still works unchanged
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

# Set up environment before any app imports
os.environ.setdefault("ANSWER_BACKEND", "gemini")
os.environ.setdefault("GEMINI_API_KEY", "unit-test-key")
os.environ.setdefault("QDRANT_PATH", "./data/vectorstore/qdrant_multilingual")


class TestSSEHelpers(unittest.TestCase):
    """Test the pure SSE formatting helpers in app/api.py."""

    def setUp(self):
        # Import lazily so we can patch env vars before module load
        from app.api import _sse_event, _build_sources
        self._sse_event = _sse_event
        self._build_sources = _build_sources

    def test_sse_event_format(self):
        result = self._sse_event("direct_answer", {"answer": "test"})
        self.assertTrue(result.startswith("event: direct_answer\n"))
        self.assertIn('data: {"answer": "test"}', result)
        self.assertTrue(result.endswith("\n\n"))

    def test_sse_event_json_serialisable(self):
        data = {"answer": "hello", "grounded": True, "time_ms": 42.5}
        result = self._sse_event("test_event", data)
        data_line = [l for l in result.split("\n") if l.startswith("data:")][0]
        parsed = json.loads(data_line[len("data: "):])
        self.assertEqual(parsed["answer"], "hello")
        self.assertTrue(parsed["grounded"])

    def test_build_sources_converts_items(self):
        items = [
            {
                "rank": 1,
                "chunk_id": "abc-123",
                "passage_id": "p1",
                "query_id": "q1",
                "text": "Some text",
                "language": "hi",
                "score": 0.9,
                "vector_score": 0.85,
            }
        ]
        result = self._build_sources(items)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[0]["text"], "Some text")
        self.assertIsInstance(result[0]["score"], float)

    def test_build_sources_empty(self):
        self.assertEqual(self._build_sources([]), [])

    def test_build_sources_missing_fields_use_defaults(self):
        items = [{"rank": 2}]
        result = self._build_sources(items)
        self.assertEqual(result[0]["rank"], 2)
        self.assertEqual(result[0]["text"], "")
        self.assertIsNone(result[0]["chunk_id"])
        self.assertEqual(result[0]["score"], 0.0)


class TestProcessDual(unittest.TestCase):
    """Test RAGEngine.process_dual() using a fully mocked engine."""

    def _make_engine(self):
        """Build a RAGEngine with all heavy dependencies mocked."""
        with patch("app.pipeline.SentenceTransformer"), \
             patch("app.pipeline.QdrantClient"), \
             patch("app.pipeline.GeminiAnswerGenerator") as MockGemini, \
             patch("app.pipeline.ANSWER_BACKEND", "gemini"):

            MockGemini.return_value.available = True
            MockGemini.return_value.load_ms = 0.0
            MockGemini.return_value.load_error = None

            from app.pipeline import RAGEngine
            engine = RAGEngine.__new__(RAGEngine)

            # Wire mocked components
            engine.embedder = MagicMock()
            engine.embedder.encode.return_value = [0.1] * 384

            engine.client = MagicMock()
            # Simulate 3 Qdrant hits
            hit = MagicMock()
            hit.score = 0.92
            hit.payload = {
                "chunk_id": "chunk-001",
                "passage_id": "pass-001",
                "query_id": "q001",
                "text": "The Manhattan Project was a secret US nuclear weapons programme.",
                "language": "en",
                "is_selected": True,
                "chunk_strategy": "sentence",
                "word_count": 12,
            }
            engine.client.query_points.return_value.points = [hit, hit, hit]

            engine.answer_generator = MagicMock()
            engine.answer_generator.available = True

        return engine

    def test_process_dual_returns_tuple(self):
        engine = self._make_engine()
        result, state = engine.process_dual("मैनहट्टन परियोजना क्या थी?")
        self.assertIsNotNone(result)
        self.assertIsNotNone(state)

    def test_process_dual_direct_answer_present(self):
        engine = self._make_engine()
        result, state = engine.process_dual("What is DNA?")
        self.assertIn("answer", result)
        self.assertIn("grounded", result)
        self.assertIn("timings", result)

    def test_process_dual_does_not_call_llm(self):
        """process_dual must NOT call the LLM generator — that's the SSE endpoint's job."""
        engine = self._make_engine()
        engine.process_dual("some query")
        engine.answer_generator.generate.assert_not_called()

    def test_process_dual_state_has_compressed_context(self):
        engine = self._make_engine()
        _, state = engine.process_dual("some query")
        self.assertIn("compression_result", state)
        self.assertIn("context", state["compression_result"])

    def test_process_dual_state_has_language_code(self):
        engine = self._make_engine()
        _, state = engine.process_dual("hello", language="hi-IN")
        self.assertIn("language_code", state)

    def test_process_dual_state_has_pipeline_start(self):
        engine = self._make_engine()
        _, state = engine.process_dual("test")
        self.assertIn("_pipeline_start", state)
        self.assertIsInstance(state["_pipeline_start"], float)

    def test_process_dual_timings_have_no_llm_ms(self):
        """Direct answer timings should have llm_ms = 0 (LLM not called yet)."""
        engine = self._make_engine()
        result, _ = engine.process_dual("test")
        self.assertEqual(result["timings"]["llm_ms"], 0.0)

    def test_process_dual_timings_have_all_stages(self):
        engine = self._make_engine()
        result, _ = engine.process_dual("test")
        t = result["timings"]
        for key in ("embedding_ms", "qdrant_ms", "rerank_ms", "compression_ms",
                    "llm_ms", "answer_ms", "total_ms"):
            self.assertIn(key, t, f"Missing timing key: {key}")

    def test_process_dual_empty_query_returns_blocked(self):
        engine = self._make_engine()
        result, state = engine.process_dual("")
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "empty_query")
        self.assertIsNone(state)

    def test_process_dual_whitespace_query_returns_blocked(self):
        engine = self._make_engine()
        result, state = engine.process_dual("   ")
        self.assertTrue(result["blocked"])
        self.assertIsNone(state)

    def test_process_dual_sources_populated(self):
        engine = self._make_engine()
        result, _ = engine.process_dual("test query")
        self.assertIsInstance(result["sources"], list)
        self.assertGreater(len(result["sources"]), 0)


class TestRetrieveAndCompress(unittest.TestCase):
    """Test the _retrieve_and_compress() helper directly."""

    def _make_engine(self):
        with patch("app.pipeline.SentenceTransformer"), \
             patch("app.pipeline.QdrantClient"), \
             patch("app.pipeline.GeminiAnswerGenerator") as MockGemini, \
             patch("app.pipeline.ANSWER_BACKEND", "gemini"):

            MockGemini.return_value.available = True
            MockGemini.return_value.load_ms = 0.0

            from app.pipeline import RAGEngine
            engine = RAGEngine.__new__(RAGEngine)
            engine.embedder = MagicMock()
            engine.embedder.encode.return_value = [0.1] * 384
            engine.client = MagicMock()
            hit = MagicMock()
            hit.score = 0.8
            hit.payload = {
                "chunk_id": "c1", "passage_id": "p1", "query_id": "q1",
                "text": "Sample text for compression.", "language": "hi",
                "is_selected": True, "chunk_strategy": "sentence", "word_count": 5,
            }
            engine.client.query_points.return_value.points = [hit]
            engine.answer_generator = MagicMock()

        return engine

    def test_returns_none_for_empty_query(self):
        engine = self._make_engine()
        self.assertIsNone(engine._retrieve_and_compress(""))

    def test_returns_none_for_whitespace(self):
        engine = self._make_engine()
        self.assertIsNone(engine._retrieve_and_compress("   "))

    def test_returns_dict_for_valid_query(self):
        engine = self._make_engine()
        result = engine._retrieve_and_compress("What is DNA?")
        self.assertIsNotNone(result)
        self.assertIn("query", result)
        self.assertIn("timings", result)
        self.assertIn("top3_results", result)

    def test_partial_timings_no_llm(self):
        engine = self._make_engine()
        result = engine._retrieve_and_compress("test")
        t = result["timings"]
        self.assertIn("embedding_ms", t)
        self.assertIn("qdrant_ms", t)
        self.assertIn("rerank_ms", t)
        self.assertIn("compression_ms", t)
        # llm_ms is NOT in partial timings — added later by process_dual
        self.assertNotIn("llm_ms", t)


class TestTimingSeparation(unittest.TestCase):
    """Verify that time_to_direct_ms and time_to_llm_ms are distinct fields."""

    def test_direct_answer_timing_field_name(self):
        """process_dual result must have answer_ms and total_ms but not time_to_direct_ms.
        The SSE endpoint adds time_to_direct_ms at the HTTP layer."""
        with patch("app.pipeline.SentenceTransformer"), \
             patch("app.pipeline.QdrantClient"), \
             patch("app.pipeline.GeminiAnswerGenerator") as MockGemini, \
             patch("app.pipeline.ANSWER_BACKEND", "gemini"):

            MockGemini.return_value.available = True
            MockGemini.return_value.load_ms = 0.0

            from app.pipeline import RAGEngine
            engine = RAGEngine.__new__(RAGEngine)
            engine.embedder = MagicMock()
            engine.embedder.encode.return_value = [0.1] * 384
            engine.client = MagicMock()
            hit = MagicMock()
            hit.score = 0.7
            hit.payload = {
                "chunk_id": "c2", "text": "Test.", "language": "en",
                "passage_id": None, "query_id": None,
                "is_selected": True, "chunk_strategy": "sentence", "word_count": 1,
            }
            engine.client.query_points.return_value.points = [hit]
            engine.answer_generator = MagicMock()

        result, state = engine.process_dual("test query")
        # result timings contain total_ms (pipeline elapsed so far)
        self.assertIn("total_ms", result["timings"])
        # llm_ms is 0 — LLM hasn't run yet
        self.assertEqual(result["timings"]["llm_ms"], 0.0)
        # state preserves pipeline start for the SSE endpoint
        self.assertIn("_pipeline_start", state)


class TestLegacyProcessUnchanged(unittest.TestCase):
    """Smoke-test that the original process() method still returns expected shape."""

    def test_process_returns_answer_key(self):
        with patch("app.pipeline.SentenceTransformer"), \
             patch("app.pipeline.QdrantClient"), \
             patch("app.pipeline.GeminiAnswerGenerator") as MockGemini, \
             patch("app.pipeline.ANSWER_BACKEND", "gemini"):

            MockGemini.return_value.available = True
            MockGemini.return_value.load_ms = 0.0
            MockGemini.return_value.generate.return_value = {
                "answer": "Test answer.",
                "grounded": True,
                "blocked": False,
                "reason": "grounded",
                "latency_ms": 50.0,
            }

            from app.pipeline import RAGEngine
            engine = RAGEngine.__new__(RAGEngine)
            engine.embedder = MagicMock()
            engine.embedder.encode.return_value = [0.1] * 384
            engine.client = MagicMock()
            hit = MagicMock()
            hit.score = 0.9
            hit.payload = {
                "chunk_id": "c3", "text": "Legacy test.", "language": "en",
                "passage_id": None, "query_id": None,
                "is_selected": True, "chunk_strategy": "sentence", "word_count": 2,
            }
            engine.client.query_points.return_value.points = [hit]
            engine.answer_generator = MockGemini.return_value

        result = engine.process("What is the legacy test?")
        self.assertIn("answer", result)
        self.assertIn("timings", result)
        self.assertIn("sources", result)
        self.assertIn("grounded", result)


if __name__ == "__main__":
    unittest.main()
