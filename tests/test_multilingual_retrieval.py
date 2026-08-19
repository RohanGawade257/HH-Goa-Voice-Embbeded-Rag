import importlib
import json
import os
import unittest
from pathlib import Path

os.environ["ANSWER_BACKEND"] = "extractive"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


PASSAGES_FILE = Path("data/processed/multilingual/passages.jsonl")
SUPPORTED_LANGUAGES = (
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "ta",
    "ur",
)


def representative_queries():
    found = {}
    with PASSAGES_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            language = item.get("language")
            if language in SUPPORTED_LANGUAGES and language not in found:
                found[language] = {
                    "language": language,
                    "query": item["query"],
                    "query_id": item["query_id"],
                }
            if len(found) == len(SUPPORTED_LANGUAGES):
                break
    return found


class MultilingualRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app.config as config

        importlib.reload(config)
        from app import pipeline

        importlib.reload(pipeline)
        cls.engine = pipeline.RAGEngine()
        cls.queries = representative_queries()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def assert_language_retrieves(self, language):
        item = self.queries[language]
        result = self.engine.process(item["query"], language=language)
        top20 = result["retrieval"]["top20"]
        timings = result["timings"]

        self.assertGreater(len(top20), 0)
        self.assertLessEqual(len(top20), 10)
        self.assertEqual(result["retrieved_chunks"], 3)
        self.assertIn("embedding_ms", timings)
        self.assertIn("qdrant_ms", timings)
        self.assertIn("rerank_ms", timings)
        self.assertIn("compression_ms", timings)

    def test_hindi_retrieval(self):
        self.assert_language_retrieves("hi")

    def test_marathi_retrieval(self):
        self.assert_language_retrieves("mr")

    def test_additional_supported_language_retrieval(self):
        for language in ("as", "bn", "gu", "kn", "ml", "ne", "or", "pa", "sa", "ta", "ur"):
            with self.subTest(language=language):
                self.assert_language_retrieves(language)


if __name__ == "__main__":
    unittest.main()
