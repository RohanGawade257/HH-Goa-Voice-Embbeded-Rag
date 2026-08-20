import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.benchmark_qwen_api_manual import (
    EXCEPTION,
    HTTP_ERROR,
    SUCCESS,
    TIMEOUT,
    build_report,
    run_requests,
    write_report,
)


QUERY_POOL = [
    {"query": "q1", "query_id": "id-1", "language": "hi"},
    {"query": "q2", "query_id": "id-2", "language": "mr"},
]


def result(status, total_seed=1.0):
    blocked = status != SUCCESS
    reason = {
        SUCCESS: "qwen_api_grounded_answer",
        HTTP_ERROR: "qwen_api_http_error",
        TIMEOUT: "qwen_api_timeout",
        EXCEPTION: "qwen_api_error",
    }[status]
    return {
        "answer": "answer" if status == SUCCESS else "",
        "grounded": status == SUCCESS,
        "blocked": blocked,
        "reason": reason,
        "answer_generation": {
            "status": status,
            "reason": reason,
            "http_status": 500 if status == HTTP_ERROR else 200 if status == SUCCESS else None,
            "exception_type": "RuntimeError" if status == EXCEPTION else None,
            "timeout_seconds": 5 if status == TIMEOUT else None,
            "error": "failure" if status != SUCCESS else None,
            "prompt_chars": 20,
            "context_chars": 10,
        },
        "timings": {
            "embedding_ms": total_seed,
            "qdrant_ms": total_seed,
            "rerank_ms": total_seed,
            "compression_ms": total_seed,
            "llm_ms": total_seed,
        },
        "retrieval": {"top20": [{"language": "hi", "query_id": "id-1"}]},
    }


class FakeEngine:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def process(self, query, language=None, max_tokens=None):
        self.calls += 1
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class QwenBenchmarkTests(unittest.TestCase):
    def test_successful_latency_percentiles_use_success_only(self):
        measured = run_requests(
            FakeEngine(
                [
                    result(SUCCESS, 10),
                    result(HTTP_ERROR, 1000),
                    result(TIMEOUT, 1000),
                    result(EXCEPTION, 1000),
                    result(SUCCESS, 20),
                ]
            ),
            QUERY_POOL,
            5,
            "measured",
        )
        report = build_report(measured, [], QUERY_POOL, requested_requests=5, min_successful_requests=2)

        self.assertEqual(report["successful_requests"], 2)
        self.assertEqual(report["http_errors"], 1)
        self.assertEqual(report["timeouts"], 1)
        self.assertEqual(report["exceptions"], 1)
        self.assertEqual(report["stage_latency_ms"]["llm"]["samples"], 2)
        self.assertLess(report["latency_ms"]["total"]["p100_ms"], 200)

    def test_benchmark_continues_after_engine_exception(self):
        measured = run_requests(
            FakeEngine([RuntimeError("boom"), result(SUCCESS, 5)]),
            QUERY_POOL,
            2,
            "measured",
        )
        self.assertEqual([record["status"] for record in measured], [EXCEPTION, SUCCESS])

    def test_warmup_records_are_excluded_from_measured_statistics(self):
        engine = FakeEngine([result(SUCCESS, 1000), result(SUCCESS, 5), result(SUCCESS, 10)])
        warmup = run_requests(engine, QUERY_POOL, 1, "warmup")
        measured = run_requests(engine, QUERY_POOL, 2, "measured")
        report = build_report(measured, warmup, QUERY_POOL, requested_requests=2, min_successful_requests=2)

        self.assertEqual(report["warmup"], 1)
        self.assertEqual(report["total_requests"], 2)
        self.assertEqual(report["latency_ms"]["total"]["p100_ms"], 50.0)

    def test_request_count_cycles_query_pool(self):
        measured = run_requests(FakeEngine([result(SUCCESS, 1)] * 5), QUERY_POOL, 5, "measured")
        self.assertEqual(len(measured), 5)
        self.assertEqual(
            [record["query_id"] for record in measured],
            ["id-1", "id-2", "id-1", "id-2", "id-1"],
        )

    def test_report_generation_writes_json(self):
        measured = run_requests(FakeEngine([result(SUCCESS, 1)]), QUERY_POOL, 1, "measured")
        report = build_report(measured, [], QUERY_POOL, requested_requests=1, min_successful_requests=20)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            write_report(report, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["target_status"], "INCONCLUSIVE")
        self.assertEqual(len(loaded["requests"]), 1)


if __name__ == "__main__":
    unittest.main()
