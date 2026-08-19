import unittest

from app.context_compressor import compress_context


class ContextCompressorTests(unittest.TestCase):
    def test_removes_duplicate_text_and_respects_budget(self):
        result = compress_context(
            "alpha beta",
            [
                {
                    "text": "Alpha beta fact. Irrelevant tail sentence.",
                    "score": 0.9,
                    "language": "hi",
                    "chunk_id": "one",
                },
                {
                    "text": "Alpha beta fact. Other detail sentence.",
                    "score": 0.8,
                    "language": "hi",
                    "chunk_id": "two",
                },
                {
                    "text": "Alpha beta fact. Third detail sentence.",
                    "score": 0.7,
                    "language": "hi",
                    "chunk_id": "three",
                },
            ],
            top_context_chunks=3,
            max_context_chars=90,
        )

        self.assertLessEqual(result["after_chars"], 90)
        self.assertEqual(result["context"].count("Alpha beta fact."), 1)
        for snippet in result["snippets"]:
            self.assertEqual(set(snippet.keys()), {"text", "score"})

    def test_empty_results_produce_empty_context(self):
        result = compress_context("question", [], top_context_chunks=3, max_context_chars=100)
        self.assertEqual(result["context"], "")
        self.assertEqual(result["snippets"], [])
        self.assertEqual(result["after_chars"], 0)


if __name__ == "__main__":
    unittest.main()
