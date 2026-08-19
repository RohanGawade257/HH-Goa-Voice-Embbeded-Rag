import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch


class ConfigTests(unittest.TestCase):
    def test_qwen_default_model_is_configured(self):
        source = Path("app/config.py").read_text(encoding="utf-8")
        self.assertIn('os.getenv("QWEN_MODEL", "Qwen/Qwen3-0.6B")', source)

    def test_llm_provider_and_key_can_be_loaded_from_environment(self):
        import app.config as config

        env = {
            "ANSWER_BACKEND": "qwen_api",
            "LLM_PROVIDER": "custom",
            "LLM_API_KEY": "unit-test-key",
            "LLM_CHAT_COMPLETIONS_URL": "https://example.test/v1/chat/completions",
            "QWEN_MODEL": "custom/qwen",
            "MAX_NEW_TOKENS": "32",
            "LLM_TIMEOUT_SECONDS": "1.5",
        }
        with patch.dict(os.environ, env, clear=False):
            reloaded = importlib.reload(config)
            self.assertEqual(reloaded.ANSWER_BACKEND, "qwen_api")
            self.assertEqual(reloaded.LLM_PROVIDER, "custom")
            self.assertEqual(reloaded.LLM_API_KEY, "unit-test-key")
            self.assertEqual(
                reloaded.LLM_CHAT_COMPLETIONS_URL,
                "https://example.test/v1/chat/completions",
            )
            self.assertEqual(reloaded.QWEN_MODEL, "custom/qwen")
            self.assertEqual(reloaded.MAX_NEW_TOKENS, 32)
            self.assertEqual(reloaded.LLM_TIMEOUT_SECONDS, 1.5)

        importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
