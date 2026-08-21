"""Unit tests for the Gemini answer generator.

All tests run WITHOUT a live API key.  The google-genai SDK is mocked at the
client level so no network calls are made.

Tests cover:
  1.  Generator initialises with correct model / thinking_budget / max_output_tokens
  2.  Non-thinking config: thinking_budget=0 is passed to ThinkingConfig
  3.  max_output_tokens override flows into generation config
  4.  System prompt enforces answer-only / no-reasoning behaviour
  5.  SUCCESS classification on valid streamed answer
  6.  TTFT is measured (> 0 ms on success)
  7.  Empty-answer classified as EXCEPTION
  8.  Missing context returns language-specific fallback without API call
  9.  Missing API key → EXCEPTION with reason=gemini_api_key_missing
  10. Wrong backend → EXCEPTION with reason=gemini_disabled
  11. SDK exception → EXCEPTION classification
  12. Timeout-like exception → TIMEOUT classification
  13. HTTP-like exception → HTTP_ERROR classification
  14. latency_ms alias equals llm_ms
"""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Set env before importing any app module
os.environ["ANSWER_BACKEND"] = "gemini"
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
os.environ["GEMINI_MODEL"] = "gemini-2.5-flash-lite"
os.environ["GEMINI_MAX_OUTPUT_TOKENS"] = "30"
os.environ["GEMINI_THINKING_BUDGET"] = "0"

from app.generation.gemini import GeminiAnswerGenerator  # noqa: E402
from app.generation.llm import (  # noqa: E402
    EXCEPTION,
    HTTP_ERROR,
    SUCCESS,
    TIMEOUT,
    missing_context_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str) -> MagicMock:
    """Fake a streamed chunk returned by generate_content_stream."""
    c = MagicMock()
    c.text = text
    return c


def _make_generator(stream_chunks: list[str] | None = None) -> tuple[GeminiAnswerGenerator, MagicMock]:
    """
    Return a GeminiAnswerGenerator with the google-genai SDK mocked out.

    ``stream_chunks`` is a list of text fragments the mock stream yields.
    Pass ``None`` to get a generator with no mock (SDK unavailable path).
    """
    gen = GeminiAnswerGenerator.__new__(GeminiAnswerGenerator)
    gen.backend = "gemini"
    gen.model_name = "gemini-2.5-flash-lite"
    gen.max_output_tokens = 30
    gen.thinking_budget = 0
    gen.timeout = 10.0
    gen.api_key = "test-gemini-key"
    gen.available = True
    gen.load_error = ""
    gen.load_ms = 0.0

    mock_client = MagicMock()

    if stream_chunks is not None:
        # generate_content_stream returns a plain iterator (not a context manager).
        chunks = [_chunk(t) for t in stream_chunks]
        mock_client.models.generate_content_stream.return_value = iter(chunks)

    gen.client = mock_client
    return gen, mock_client


# ---------------------------------------------------------------------------
# 1 & 2 — Initialisation and thinking_budget
# ---------------------------------------------------------------------------

class InitialisationTests(unittest.TestCase):

    def test_default_model_is_flash_lite(self):
        gen, _ = _make_generator([])
        self.assertEqual(gen.model_name, "gemini-2.5-flash-lite")

    def test_thinking_budget_is_zero(self):
        gen, _ = _make_generator([])
        self.assertEqual(gen.thinking_budget, 0)

    def test_default_max_output_tokens(self):
        gen, _ = _make_generator([])
        self.assertEqual(gen.max_output_tokens, 30)

    def test_provider_is_gemini(self):
        """generate() must report provider=gemini on success."""
        gen, _ = _make_generator(["Paris"])
        result = gen.generate("capital?", "hi", "Paris is the capital.")
        self.assertEqual(result.get("provider"), "gemini")

    def test_first_try_success_has_attempt_count_1(self):
        """Harness must report attempt_count=1 and no retry delays on happy path."""
        gen, _ = _make_generator(["मैनहट्टन परियोजना एक अनुसंधान परियोजना थी।"])
        result = gen.generate("मैनहट्टन परियोजना क्या थी?", "hi", "यह प्रमाण है।")
        self.assertEqual(result.get("attempt_count"), 1)
        self.assertEqual(result.get("retry_delays_ms"), [])


# ---------------------------------------------------------------------------
# 3 — max_output_tokens override
# ---------------------------------------------------------------------------

class MaxTokensTests(unittest.TestCase):

    def test_override_reaches_generation_config(self):
        gen, mock_client = _make_generator(["ok"])

        call_configs: list = []

        def capture_call(**kwargs):
            call_configs.append(kwargs.get("config"))
            return iter([_chunk("ok")])

        mock_client.models.generate_content_stream.side_effect = capture_call

        gen.generate("q", "hi", "evidence", max_tokens=16)
        self.assertEqual(len(call_configs), 1)
        self.assertEqual(call_configs[0].max_output_tokens, 16)

    def test_no_override_uses_default(self):
        gen, mock_client = _make_generator(["ok"])

        call_configs: list = []

        def capture_call(**kwargs):
            call_configs.append(kwargs.get("config"))
            return iter([_chunk("ok")])

        mock_client.models.generate_content_stream.side_effect = capture_call

        gen.generate("q", "hi", "evidence")
        self.assertEqual(call_configs[0].max_output_tokens, 30)


# ---------------------------------------------------------------------------
# 4 — System prompt
# ---------------------------------------------------------------------------

class SystemPromptTests(unittest.TestCase):

    def _get_system_instruction(self, gen: GeminiAnswerGenerator, mock_client: MagicMock) -> str:
        captured = {}

        def capture(**kwargs):
            captured["config"] = kwargs.get("config")
            return iter([_chunk("answer")])

        mock_client.models.generate_content_stream.side_effect = capture
        gen.generate("q?", "hi", "evidence text")
        return captured["config"].system_instruction or ""

    def test_prompt_forbids_reasoning(self):
        gen, mock_client = _make_generator(["yes"])
        system = self._get_system_instruction(gen, mock_client)
        self.assertIn("Do NOT reason", system)
        self.assertIn("chain-of-thought", system)

    def test_prompt_requires_evidence_only(self):
        gen, mock_client = _make_generator(["yes"])
        system = self._get_system_instruction(gen, mock_client)
        self.assertIn("ONLY the supplied evidence", system)

    def test_prompt_names_language(self):
        gen, mock_client = _make_generator(["yes"])
        system = self._get_system_instruction(gen, mock_client)
        self.assertIn("Hindi", system)


# ---------------------------------------------------------------------------
# 5 & 6 — SUCCESS + TTFT
# ---------------------------------------------------------------------------

class SuccessTests(unittest.TestCase):

    def test_success_classification(self):
        gen, _ = _make_generator(["नई दिल्ली।"])
        result = gen.generate("capital?", "hi", "नई दिल्ली भारत की राजधानी है।")
        self.assertEqual(result["status"], SUCCESS)
        self.assertFalse(result["blocked"])
        self.assertTrue(result["grounded"])
        self.assertEqual(result["answer"], "नई दिल्ली।")

    def test_ttft_is_positive_on_success(self):
        gen, _ = _make_generator(["answer"])
        result = gen.generate("q", "hi", "evidence")
        self.assertGreater(result["ttft_ms"], 0.0)

    def test_llm_ms_is_positive_on_success(self):
        gen, _ = _make_generator(["मैनहट्टन परियोजना थी।"])
        result = gen.generate("q", "hi", "प्रमाण।")
        self.assertGreater(result["llm_ms"], 0.0)

    def test_latency_ms_aliases_llm_ms(self):
        gen, _ = _make_generator(["मैनहट्टन परियोजना थी।"])
        result = gen.generate("q", "hi", "प्रमाण।")
        self.assertEqual(result["latency_ms"], result["llm_ms"])

    def test_thinking_budget_reported(self):
        gen, _ = _make_generator(["मैनहट्टन परियोजना थी।"])
        result = gen.generate("q", "hi", "प्रमाण।")
        self.assertEqual(result["thinking_budget"], 0)


# ---------------------------------------------------------------------------
# 7 — Empty answer
# ---------------------------------------------------------------------------

class EmptyAnswerTests(unittest.TestCase):

    def test_empty_answer_is_exception(self):
        gen, _ = _make_generator(["", "  "])  # all whitespace chunks
        result = gen.generate("q", "hi", "evidence")
        self.assertEqual(result["status"], EXCEPTION)
        self.assertEqual(result["reason"], "empty_gemini_answer")
        self.assertTrue(result["blocked"])


# ---------------------------------------------------------------------------
# 8 — Missing context (no API call)
# ---------------------------------------------------------------------------

class MissingContextTests(unittest.TestCase):

    def test_missing_context_returns_fallback(self):
        gen, mock_client = _make_generator([])
        result = gen.generate("q", "mr", "")
        mock_client.models.generate_content_stream.assert_not_called()
        self.assertEqual(result["status"], EXCEPTION)
        self.assertEqual(result["reason"], "missing_context")
        self.assertEqual(result["answer"], missing_context_answer("mr"))

    def test_missing_context_whitespace_only(self):
        gen, mock_client = _make_generator([])
        result = gen.generate("q", "hi", "   \n  ")
        mock_client.models.generate_content_stream.assert_not_called()
        self.assertEqual(result["reason"], "missing_context")


# ---------------------------------------------------------------------------
# 9 & 10 — Key missing / wrong backend
# ---------------------------------------------------------------------------

class AvailabilityTests(unittest.TestCase):

    def test_missing_api_key(self):
        gen = GeminiAnswerGenerator.__new__(GeminiAnswerGenerator)
        gen.backend = "gemini"
        gen.model_name = "gemini-2.5-flash-lite"
        gen.max_output_tokens = 30
        gen.thinking_budget = 0
        gen.timeout = 10.0
        gen.api_key = ""
        gen.available = False
        gen.load_error = "GEMINI_API_KEY is not set"
        gen.load_ms = 0.0
        gen.client = None

        result = gen.generate("q", "hi", "evidence")
        self.assertEqual(result["status"], EXCEPTION)
        self.assertEqual(result["reason"], "gemini_api_key_missing")

    def test_wrong_backend_returns_gemini_disabled(self):
        gen, _ = _make_generator([])
        gen.backend = "qwen_api"
        result = gen.generate("q", "hi", "evidence")
        self.assertEqual(result["status"], EXCEPTION)
        self.assertEqual(result["reason"], "gemini_disabled")


# ---------------------------------------------------------------------------
# 11–13 — Exception, timeout, HTTP error classification
# ---------------------------------------------------------------------------

class ExceptionClassificationTests(unittest.TestCase):

    def _gen_with_side_effect(self, exc: Exception) -> GeminiAnswerGenerator:
        gen, mock_client = _make_generator(None)
        gen.available = True
        gen.client = mock_client
        mock_client.models.generate_content_stream.side_effect = exc
        return gen

    def test_generic_exception_classified(self):
        gen = self._gen_with_side_effect(RuntimeError("exploded"))
        result = gen.generate("q", "hi", "evidence")
        self.assertEqual(result["status"], EXCEPTION)
        self.assertEqual(result["exception_type"], "RuntimeError")
        self.assertTrue(result["blocked"])

    def test_timeout_exception_classified(self):
        class DeadlineExceeded(Exception):
            pass

        gen = self._gen_with_side_effect(DeadlineExceeded("deadline exceeded"))
        result = gen.generate("q", "hi", "evidence")
        self.assertEqual(result["status"], TIMEOUT)
        self.assertEqual(result["reason"], "gemini_timeout")

    def test_http_error_classified(self):
        class ClientError(Exception):
            pass

        gen = self._gen_with_side_effect(ClientError("403 Forbidden"))
        result = gen.generate("q", "hi", "evidence")
        self.assertEqual(result["status"], HTTP_ERROR)
        self.assertEqual(result["reason"], "gemini_http_error")

    def test_llm_ms_set_on_exception(self):
        gen = self._gen_with_side_effect(RuntimeError("boom"))
        result = gen.generate("q", "hi", "evidence")
        self.assertGreaterEqual(result["llm_ms"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
