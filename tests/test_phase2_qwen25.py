"""Phase 2 unit tests — Qwen2.5-0.5B-Instruct model replacement.

Tests cover:
  1.  Default model is Qwen2.5-0.5B-Instruct
  2.  Model name flows correctly into the API payload
  3.  System prompt enforces answer-only / no-reasoning behaviour
  4.  max_tokens override flows into the API payload
  5.  Default max_tokens (config fallback) is used when no override given
  6.  API success parsing (SUCCESS status + non-empty answer)
  7.  HTTP error classification (HTTP_ERROR status)
  8.  Timeout classification (TIMEOUT status)
  9.  Exception classification (EXCEPTION status)
  10. Successful latency percentiles exclude failed requests
  11. Benchmark continues after failed requests
  12. Token-experiment produces one report per config
  13. Comparison table covers all four token configs
  14. Warmup records are never in measured statistics
  15. Request-count cycles query pool

All tests run without a live API key.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import httpx

# Inject dummy credentials so QwenAnswerGenerator.available == True
os.environ["ANSWER_BACKEND"] = "qwen_api"
os.environ["LLM_API_KEY"] = "unit-test-key"
os.environ["QWEN_MODEL"] = "Qwen/Qwen2.5-0.5B-Instruct"

from app.generation.llm import QwenAnswerGenerator, missing_context_answer  # noqa: E402
from benchmarks.benchmark_qwen_api_manual import (  # noqa: E402
    EXCEPTION,
    HTTP_ERROR,
    SUCCESS,
    TIMEOUT,
    TOKEN_EXPERIMENT_CONFIGS,
    build_report,
    print_comparison_table,
    run_requests,
    run_token_experiment,
    write_report,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

QUERY_POOL = [
    {"query": "q1", "query_id": "id-1", "language": "hi"},
    {"query": "q2", "query_id": "id-2", "language": "mr"},
]


def pipeline_result(status: str, total_seed: float = 1.0, max_tokens: int = 64) -> dict:
    """Build a fake pipeline result as returned by RAGEngine.process()."""
    blocked = status != SUCCESS
    reason_map = {
        SUCCESS: "qwen_api_grounded_answer",
        HTTP_ERROR: "qwen_api_http_error",
        TIMEOUT: "qwen_api_timeout",
        EXCEPTION: "qwen_api_error",
    }
    return {
        "answer": "answer" if status == SUCCESS else "",
        "grounded": status == SUCCESS,
        "blocked": blocked,
        "reason": reason_map[status],
        "answer_generation": {
            "status": status,
            "reason": reason_map[status],
            "http_status": 200 if status == SUCCESS else (500 if status == HTTP_ERROR else None),
            "exception_type": "RuntimeError" if status == EXCEPTION else None,
            "timeout_seconds": 5 if status == TIMEOUT else None,
            "error": "failure" if status != SUCCESS else None,
            "prompt_chars": 20,
            "context_chars": 10,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "provider": "huggingface",
        },
        "timings": {
            "embedding_ms": total_seed,
            "qdrant_ms": total_seed,
            "rerank_ms": total_seed,
            "compression_ms": total_seed,
            "llm_ms": total_seed,
        },
        "retrieval": {"top20": [{"language": "hi", "query_id": "id-1"}]},
        "_max_tokens_used": max_tokens,  # captured by FakeEngine for assertions
    }


class FakeEngine:
    """Minimal RAGEngine stand-in that records calls and yields preset results."""

    def __init__(self, outputs: list):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def process(self, query: str, language: str | None = None, max_tokens: int | None = None):
        self.calls.append({"query": query, "language": language, "max_tokens": max_tokens})
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        pass


def make_generator(handler) -> QwenAnswerGenerator:
    """Return a QwenAnswerGenerator wired to a mock HTTP transport."""
    gen = QwenAnswerGenerator()
    gen.client.close()
    gen.backend = "qwen_api"
    gen.available = True
    gen.api_key = "unit-test-key"
    gen.client = httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=0.5,
        headers={"Authorization": "Bearer unit-test-key"},
    )
    return gen


# ===========================================================================
# 1 & 2 — Model configuration and payload
# ===========================================================================

class ModelConfigurationTests(unittest.TestCase):

    def test_default_model_is_qwen25_instruct(self):
        gen = QwenAnswerGenerator()
        gen.close()
        self.assertEqual(gen.model_name, "Qwen/Qwen2.5-0.5B-Instruct")

    def test_model_name_sent_in_api_payload(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        gen = make_generator(handler)
        try:
            gen.generate("question", "hi", "evidence")
        finally:
            gen.close()

        self.assertEqual(seen["payload"]["model"], "Qwen/Qwen2.5-0.5B-Instruct")


# ===========================================================================
# 3 — Non-thinking / instruct prompt enforcement
# ===========================================================================

class InstructPromptTests(unittest.TestCase):

    def test_system_prompt_forbids_reasoning(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "yes"}}]})

        gen = make_generator(handler)
        try:
            gen.generate("question?", "hi", "some evidence")
        finally:
            gen.close()

        system = seen["payload"]["messages"][0]["content"]
        # Must explicitly forbid chain-of-thought
        self.assertIn("Do NOT reason", system)
        self.assertIn("chain-of-thought", system)
        # Must require direct answer
        self.assertIn("Use ONLY the supplied evidence", system)
        # "step by step" must only appear in a prohibition context, not as an instruction
        self.assertIn("Do NOT", system)
        lower = system.lower()
        # The word "step by step" is present only inside "Do NOT … step by step"
        if "step by step" in lower:
            self.assertIn("do not", lower[:lower.index("step by step")])

    def test_system_prompt_requires_language_match(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ja"}}]})

        gen = make_generator(handler)
        try:
            gen.generate("q", "ta", "context")
        finally:
            gen.close()

        system = seen["payload"]["messages"][0]["content"]
        self.assertIn("Tamil", system)

    def test_stream_is_false(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "a"}}]})

        gen = make_generator(handler)
        try:
            gen.generate("q", "hi", "ctx")
        finally:
            gen.close()

        self.assertFalse(seen["payload"]["stream"])


# ===========================================================================
# 4 & 5 — max_tokens override
# ===========================================================================

class MaxTokensConfigTests(unittest.TestCase):

    def test_max_tokens_override_flows_into_payload(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        gen = make_generator(handler)
        try:
            gen.generate("q", "hi", "evidence", max_tokens=16)
        finally:
            gen.close()

        self.assertEqual(seen["payload"]["max_tokens"], 16)

    def test_each_experiment_config_sent_correctly(self):
        """All four token configs (16/32/48/64) must arrive in the payload."""
        for expected_tokens in (16, 32, 48, 64):
            seen = {}

            def handler(request, _t=expected_tokens):
                seen["payload"] = json.loads(request.content)
                return httpx.Response(200, json={"choices": [{"message": {"content": "a"}}]})

            gen = make_generator(handler)
            try:
                gen.generate("q", "hi", "evidence", max_tokens=expected_tokens)
            finally:
                gen.close()

            self.assertEqual(
                seen["payload"]["max_tokens"],
                expected_tokens,
                f"Expected max_tokens={expected_tokens} in payload",
            )

    def test_no_override_uses_config_default(self):
        """When max_tokens is not given, the config default must be used."""
        from app.config import MAX_NEW_TOKENS
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        gen = make_generator(handler)
        try:
            gen.generate("q", "hi", "evidence")  # no max_tokens kwarg
        finally:
            gen.close()

        self.assertEqual(seen["payload"]["max_tokens"], MAX_NEW_TOKENS)

    def test_max_tokens_override_via_run_requests(self):
        """run_requests() must pass max_tokens through to engine.process()."""
        engine = FakeEngine([pipeline_result(SUCCESS)] * 2)
        run_requests(engine, QUERY_POOL, 2, "measured", max_tokens=32)
        for call in engine.calls:
            self.assertEqual(call["max_tokens"], 32)

    def test_no_max_tokens_in_run_requests_passes_none(self):
        engine = FakeEngine([pipeline_result(SUCCESS)] * 2)
        run_requests(engine, QUERY_POOL, 2, "measured")
        for call in engine.calls:
            self.assertIsNone(call["max_tokens"])


# ===========================================================================
# 6–9 — API response classification
# ===========================================================================

class ResponseClassificationTests(unittest.TestCase):

    def test_success_parsing(self):
        def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": "हाँ"}}]})

        gen = make_generator(handler)
        try:
            result = gen.generate("q", "hi", "evidence")
        finally:
            gen.close()

        self.assertEqual(result["status"], SUCCESS)
        self.assertFalse(result["blocked"])
        self.assertTrue(result["grounded"])
        self.assertEqual(result["answer"], "हाँ")

    def test_http_error_classification(self):
        def handler(request):
            return httpx.Response(503, json={"error": "overloaded"})

        gen = make_generator(handler)
        try:
            result = gen.generate("q", "hi", "evidence")
        finally:
            gen.close()

        self.assertEqual(result["status"], HTTP_ERROR)
        self.assertEqual(result["http_status"], 503)
        self.assertTrue(result["blocked"])

    def test_timeout_classification(self):
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)

        gen = make_generator(handler)
        try:
            result = gen.generate("q", "hi", "evidence")
        finally:
            gen.close()

        self.assertEqual(result["status"], TIMEOUT)
        self.assertEqual(result["reason"], "qwen_api_timeout")

    def test_exception_classification(self):
        def handler(request):
            raise RuntimeError("something broke")

        gen = make_generator(handler)
        try:
            result = gen.generate("q", "hi", "evidence")
        finally:
            gen.close()

        self.assertEqual(result["status"], EXCEPTION)
        self.assertEqual(result["exception_type"], "RuntimeError")

    def test_empty_answer_classified_as_exception(self):
        def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

        gen = make_generator(handler)
        try:
            result = gen.generate("q", "hi", "evidence")
        finally:
            gen.close()

        self.assertEqual(result["status"], EXCEPTION)
        self.assertEqual(result["reason"], "empty_llm_answer")


# ===========================================================================
# 10 — Latency percentiles use SUCCESS only
# ===========================================================================

class LatencyPercentileTests(unittest.TestCase):

    def test_failed_requests_excluded_from_latency(self):
        measured = run_requests(
            FakeEngine([
                pipeline_result(SUCCESS, 10),
                pipeline_result(HTTP_ERROR, 9999),
                pipeline_result(TIMEOUT, 9999),
                pipeline_result(EXCEPTION, 9999),
                pipeline_result(SUCCESS, 20),
            ]),
            QUERY_POOL,
            5,
            "measured",
        )
        report = build_report(measured, [], QUERY_POOL, requested_requests=5, min_successful_requests=2)

        self.assertEqual(report["successful_requests"], 2)
        self.assertEqual(report["http_errors"], 1)
        self.assertEqual(report["timeouts"], 1)
        self.assertEqual(report["exceptions"], 1)
        # Only the 2 SUCCESS timings (seed=10 and 20, total = seed*5 each) must appear
        self.assertEqual(report["stage_latency_ms"]["llm"]["samples"], 2)
        # p100 must be the larger success (seed=20 → total=100 ms), not 9999*5
        self.assertLess(report["latency_ms"]["total"]["p100_ms"], 200)

    def test_zero_successes_yields_zero_percentiles(self):
        measured = run_requests(
            FakeEngine([pipeline_result(HTTP_ERROR)] * 3),
            QUERY_POOL,
            3,
            "measured",
        )
        report = build_report(measured, [], QUERY_POOL, requested_requests=3, min_successful_requests=1)
        self.assertEqual(report["successful_requests"], 0)
        self.assertEqual(report["latency_ms"]["total"]["p95_ms"], 0.0)
        self.assertEqual(report["target_status"], "INCONCLUSIVE")


# ===========================================================================
# 11 — Benchmark continues after failed requests
# ===========================================================================

class BenchmarkContinuationTests(unittest.TestCase):

    def test_continues_after_engine_exception(self):
        measured = run_requests(
            FakeEngine([RuntimeError("boom"), pipeline_result(SUCCESS, 5)]),
            QUERY_POOL,
            2,
            "measured",
        )
        self.assertEqual(len(measured), 2)
        self.assertEqual(measured[0]["status"], EXCEPTION)
        self.assertEqual(measured[1]["status"], SUCCESS)

    def test_continues_through_mixed_failures(self):
        outputs = [
            pipeline_result(SUCCESS, 1),
            pipeline_result(HTTP_ERROR),
            pipeline_result(TIMEOUT),
            pipeline_result(EXCEPTION),
            pipeline_result(SUCCESS, 2),
        ]
        measured = run_requests(FakeEngine(outputs), QUERY_POOL, 5, "measured")
        statuses = [r["status"] for r in measured]
        self.assertEqual(statuses, [SUCCESS, HTTP_ERROR, TIMEOUT, EXCEPTION, SUCCESS])


# ===========================================================================
# 12 & 13 — Token experiment and comparison table
# ===========================================================================

class TokenExperimentTests(unittest.TestCase):

    def _make_engine_factory(self, per_config_results: dict[int, list]) -> FakeEngine:
        """Return an engine that cycles correct results regardless of max_tokens."""
        all_outputs = []
        for outputs in per_config_results.values():
            all_outputs.extend(outputs)
        return FakeEngine(all_outputs)

    def test_token_experiment_produces_report_per_config(self):
        # 3 success results per config
        outputs_per_config = {t: [pipeline_result(SUCCESS, float(t))] * 3 for t in TOKEN_EXPERIMENT_CONFIGS}
        all_outputs = [item for lst in outputs_per_config.values() for item in lst]
        # warmup = 2 (uses first config's tokens), then 3 per config
        warmup_outputs = [pipeline_result(SUCCESS, 1.0)] * 2

        engine = FakeEngine(warmup_outputs + all_outputs)

        with tempfile.TemporaryDirectory() as tmp:
            results = run_token_experiment(
                query_pool=QUERY_POOL,
                token_configs=TOKEN_EXPERIMENT_CONFIGS,
                warmup=2,
                requests=3,
                min_successes=2,
                report_dir=Path(tmp),
            )
            # One JSON file per config
            for t in TOKEN_EXPERIMENT_CONFIGS:
                p = Path(tmp) / f"qwen25_benchmark_tokens{t}.json"
                self.assertTrue(p.exists(), f"Missing report for max_tokens={t}")
                loaded = json.loads(p.read_text(encoding="utf-8"))
                self.assertEqual(loaded["max_new_tokens"], t)

        self.assertEqual(set(results.keys()), set(TOKEN_EXPERIMENT_CONFIGS))

    def test_comparison_table_covers_all_configs(self):
        """print_comparison_table must not crash and must include all four configs."""
        results = {
            t: {
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_new_tokens": t,
                "successful_requests": 3,
                "success_rate": 100.0,
                "latency_ms": {
                    "llm": {"p50_ms": 100.0, "p95_ms": 150.0, "p99_ms": 180.0, "p100_ms": 200.0, "avg_ms": 120.0, "samples": 3},
                    "total": {"p50_ms": 120.0, "p95_ms": 170.0, "p99_ms": 190.0, "p100_ms": 210.0, "avg_ms": 140.0, "samples": 3},
                },
                "target_status": "FAIL",
            }
            for t in TOKEN_EXPERIMENT_CONFIGS
        }
        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_comparison_table(results)
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        for t in TOKEN_EXPERIMENT_CONFIGS:
            self.assertIn(str(t), output)


# ===========================================================================
# 14 — Warmup exclusion
# ===========================================================================

class WarmupExclusionTests(unittest.TestCase):

    def test_warmup_not_in_measured_statistics(self):
        engine = FakeEngine([
            pipeline_result(SUCCESS, 1000),  # warmup
            pipeline_result(SUCCESS, 5),     # measured
            pipeline_result(SUCCESS, 10),    # measured
        ])
        warmup = run_requests(engine, QUERY_POOL, 1, "warmup")
        measured = run_requests(engine, QUERY_POOL, 2, "measured")
        report = build_report(measured, warmup, QUERY_POOL, requested_requests=2, min_successful_requests=2)

        self.assertEqual(report["warmup"], 1)
        self.assertEqual(report["total_requests"], 2)
        # p100 should reflect only the 2 measured records (seed 5 and 10 → totals 25 and 50)
        self.assertLessEqual(report["latency_ms"]["total"]["p100_ms"], 50.0)


# ===========================================================================
# 15 — Request count cycles query pool
# ===========================================================================

class RequestCountTests(unittest.TestCase):

    def test_cycles_through_query_pool(self):
        measured = run_requests(
            FakeEngine([pipeline_result(SUCCESS)] * 5),
            QUERY_POOL,
            5,
            "measured",
        )
        self.assertEqual(len(measured), 5)
        self.assertEqual(
            [r["query_id"] for r in measured],
            ["id-1", "id-2", "id-1", "id-2", "id-1"],
        )

    def test_report_json_written_correctly(self):
        measured = run_requests(
            FakeEngine([pipeline_result(SUCCESS, 1)]),
            QUERY_POOL,
            1,
            "measured",
        )
        report = build_report(measured, [], QUERY_POOL, requested_requests=1, min_successful_requests=20)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            write_report(report, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["target_status"], "INCONCLUSIVE")
        self.assertIn("behavior", loaded)
        self.assertEqual(loaded["behavior"], "concise-instruct-no-thinking")
        self.assertEqual(len(loaded["requests"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
