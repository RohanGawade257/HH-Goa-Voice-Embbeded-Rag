import json
import os
import unittest

import httpx

os.environ["ANSWER_BACKEND"] = "qwen_api"
os.environ["LLM_API_KEY"] = "unit-test-key"

from app.generation.llm import QwenAnswerGenerator, missing_context_answer  # noqa: E402


class LLMGenerationTests(unittest.TestCase):
    def make_generator(self, handler):
        generator = QwenAnswerGenerator()
        generator.client.close()
        generator.backend = "qwen_api"
        generator.available = True
        generator.api_key = "unit-test-key"
        generator.client = httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=0.2,
            headers={"Authorization": "Bearer unit-test-key"},
        )
        return generator

    def test_generation_uses_grounded_prompt_and_qwen_model(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "\u092f\u0939 \u0909\u0924\u094d\u0924\u0930 \u0938\u0902\u0926\u0930\u094d\u092d \u092a\u0930 \u0906\u0927\u093e\u0930\u093f\u0924 \u0939\u0948\u0964",
                            }
                        }
                    ]
                },
            )

        generator = self.make_generator(handler)
        try:
            result = generator.generate(
                "\u092a\u094d\u0930\u0936\u094d\u0928?",
                "hi",
                "\u092f\u0939 \u092a\u094d\u0930\u092e\u093e\u0923 \u0939\u0948\u0964",
            )
        finally:
            generator.close()

        self.assertFalse(result["blocked"])
        self.assertTrue(result["grounded"])
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(seen["payload"]["model"], generator.model_name)
        system_prompt = seen["payload"]["messages"][0]["content"]
        user_prompt = seen["payload"]["messages"][1]["content"]
        # Phase 2: prompt uses "Use ONLY the supplied evidence"
        self.assertIn("supplied evidence", system_prompt)
        self.assertIn("Evidence:", user_prompt)
        self.assertIn("Question:", user_prompt)

    def test_missing_context_returns_language_specific_answer_without_api_call(self):
        called = False

        def handler(request):
            nonlocal called
            called = True
            return httpx.Response(200, json={})

        generator = self.make_generator(handler)
        try:
            result = generator.generate("anything", "mr", "")
        finally:
            generator.close()

        self.assertFalse(called)
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "missing_context")
        self.assertEqual(result["status"], "EXCEPTION")
        self.assertEqual(result["answer"], missing_context_answer("mr"))

    def test_http_failure_is_reported_without_secret_output(self):
        def handler(request):
            return httpx.Response(500, json={"error": "upstream failure"})

        generator = self.make_generator(handler)
        try:
            result = generator.generate("question", "hi", "evidence")
        finally:
            generator.close()

        self.assertTrue(result["blocked"])
        self.assertEqual(result["status"], "HTTP_ERROR")
        self.assertEqual(result["reason"], "qwen_api_http_error")
        self.assertEqual(result["http_status"], 500)
        self.assertNotIn("unit-test-key", result.get("error", ""))

    def test_timeout_is_reported(self):
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)

        generator = self.make_generator(handler)
        try:
            result = generator.generate("question", "hi", "evidence")
        finally:
            generator.close()

        self.assertTrue(result["blocked"])
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(result["reason"], "qwen_api_timeout")

    def test_unexpected_exception_is_reported(self):
        def handler(request):
            raise RuntimeError("mock transport exploded")

        generator = self.make_generator(handler)
        try:
            result = generator.generate("question", "hi", "evidence")
        finally:
            generator.close()

        self.assertTrue(result["blocked"])
        self.assertEqual(result["status"], "EXCEPTION")
        self.assertEqual(result["reason"], "qwen_api_error")
        self.assertEqual(result["exception_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
