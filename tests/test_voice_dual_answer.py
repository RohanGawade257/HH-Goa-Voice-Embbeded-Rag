import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

os.environ.setdefault("SARVAM_API_KEY", "")
os.environ.setdefault("SARVAM_STT_MOCK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from app.answer_generator import generate_extractive_answer  # noqa: E402
from app.generation.llm import missing_context_answer  # noqa: E402


WAV_BYTES = (
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


TRANSCRIPTS = {
    "hi-IN": "\u092e\u0948\u0928\u0939\u091f\u094d\u091f\u0928 \u092a\u0930\u093f\u092f\u094b\u091c\u0928\u093e \u0915\u094d\u092f\u093e \u0925\u0940?",
    "mr-IN": "\u092e\u0945\u0928\u0939\u0945\u091f\u0928 \u092a\u094d\u0930\u0915\u0932\u094d\u092a\u093e\u091a\u093e \u0924\u093e\u0924\u094d\u0915\u093e\u0933 \u092a\u0930\u093f\u0923\u093e\u092e \u0915\u093e\u092f \u091d\u093e\u0932\u093e?",
    "bn-IN": "\u09ae\u09cd\u09af\u09be\u09a8\u09b9\u09be\u099f\u09a8 \u09aa\u09cd\u09b0\u0995\u09b2\u09cd\u09aa\u09c7\u09b0 \u09a4\u09be\u09ce\u0995\u09cd\u09b7\u09a3\u09bf\u0995 \u09aa\u09cd\u09b0\u09ad\u09be\u09ac \u0995\u09c0?",
}


def short_code(language: str) -> str:
    return language.split("-", 1)[0]


def make_stt_result(language: str, requested_language: str | None = None) -> dict:
    return {
        "transcript": TRANSCRIPTS[language],
        "language_code": language,
        "requested_language_code": requested_language or language,
        "latency_ms": 12.5,
        "provider": "sarvam",
    }


def make_engine(fail_llm: bool = False):
    engine = MagicMock()
    engine.process_dual_calls = []

    def process_dual(query, language=None):
        engine.process_dual_calls.append((query, language))
        lang = short_code(language or "")
        direct = {
            "query": query,
            "answer": f"direct:{language}",
            "grounded": True,
            "blocked": False,
            "reason": "extractive_answer",
            "retrieved_chunks": 1,
            "sources": [
                {
                    "rank": 1,
                    "chunk_id": "c1",
                    "passage_id": "p1",
                    "query_id": "q1",
                    "text": "Evidence chunk",
                    "language": lang,
                    "score": 0.9,
                    "vector_score": 0.8,
                }
            ],
            "timings": {
                "embedding_ms": 1.0,
                "qdrant_ms": 2.0,
                "rerank_ms": 3.0,
                "compression_ms": 4.0,
                "llm_ms": 0.0,
                "answer_ms": 5.0,
                "total_ms": 15.0,
            },
        }
        state = {
            "query": query,
            "language_code": lang,
            "compression_result": {"context": "Evidence context"},
        }
        return direct, state

    engine.process_dual = MagicMock(side_effect=process_dual)
    engine.answer_generator = MagicMock()
    engine.answer_generator.available = True
    if fail_llm:
        engine.answer_generator.generate.side_effect = RuntimeError("simulated Gemini failure")
    else:
        engine.answer_generator.generate.side_effect = (
            lambda query, language, context: {
                "status": "SUCCESS",
                "answer": f"llm:{language}",
                "grounded": True,
                "blocked": False,
                "reason": "gemini_grounded_answer",
                "latency_ms": 21.0,
            }
        )
    return engine


class VoiceDualAnswerTests(unittest.TestCase):
    def post_voice(self, selected_language: str, stt_language: str | None = None, engine=None):
        from app.api import app

        if engine is None:
            engine = make_engine()
        stt_language = stt_language or selected_language

        with patch("app.api.get_engine", return_value=engine), patch(
            "app.api.transcribe_audio",
            return_value=make_stt_result(stt_language, selected_language),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/voice",
                files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
                data={"language": selected_language},
            )
        return response, engine

    def test_voice_hindi_uses_stt_language_for_direct_and_llm(self):
        response, engine = self.post_voice("hi-IN")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["language_code"], "hi-IN")
        self.assertEqual(payload["answer_language"], "hi")
        self.assertEqual(payload["direct_answer"]["answer"], "direct:hi-IN")
        self.assertEqual(payload["llm_answer"]["answer"], "llm:hi")
        engine.process_dual.assert_called_once()
        engine.answer_generator.generate.assert_called_once()

    def test_voice_marathi_uses_stt_language_for_direct_and_llm(self):
        response, _ = self.post_voice("mr-IN")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["language_code"], "mr-IN")
        self.assertEqual(payload["answer_language"], "mr")
        self.assertEqual(payload["direct_answer"]["answer"], "direct:mr-IN")
        self.assertEqual(payload["llm_answer"]["answer"], "llm:mr")

    def test_voice_bengali_uses_stt_language_for_direct_and_llm(self):
        response, _ = self.post_voice("bn-IN")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["language_code"], "bn-IN")
        self.assertEqual(payload["answer_language"], "bn")
        self.assertEqual(payload["direct_answer"]["answer"], "direct:bn-IN")
        self.assertEqual(payload["llm_answer"]["answer"], "llm:bn")

    def test_stt_detected_language_overrides_selected_language_for_answer(self):
        response, engine = self.post_voice("hi-IN", stt_language="mr-IN")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["requested_language_code"], "hi-IN")
        self.assertEqual(payload["language_code"], "mr-IN")
        self.assertEqual(payload["answer_language"], "mr")
        engine.process_dual.assert_called_once()
        _, kwargs = engine.process_dual.call_args
        self.assertEqual(kwargs["language"], "mr-IN")
        _, llm_language, _ = engine.answer_generator.generate.call_args.args
        self.assertEqual(llm_language, "mr")

    def test_gemini_failure_preserves_direct_answer_and_returns_llm_error(self):
        engine = make_engine(fail_llm=True)
        response, _ = self.post_voice("hi-IN", engine=engine)
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["direct_answer"]["answer"], "direct:hi-IN")
        self.assertIsNone(payload["llm_answer"]["answer"])
        self.assertIn("simulated Gemini failure", payload["llm_answer"]["error"])

    def test_voice_response_contains_both_answer_payloads(self):
        response, _ = self.post_voice("hi-IN")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("direct_answer", payload)
        self.assertIn("llm_answer", payload)
        self.assertIn("answer", payload["direct_answer"])
        self.assertIn("answer", payload["llm_answer"])

    def test_direct_fallback_messages_follow_voice_language(self):
        for language in ("hi", "mr", "bn"):
            with self.subTest(language=language):
                result = generate_extractive_answer(
                    "unmatched query",
                    [],
                    language=language,
                )
                self.assertEqual(result["answer"], missing_context_answer(language))
                self.assertTrue(result["blocked"])
                self.assertEqual(result["reason"], "no_context")


if __name__ == "__main__":
    unittest.main()
