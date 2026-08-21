"""Unit tests for app/generation/harness.py

Tests cover:
  validate_output  — length checks, boundary values, script presence, mixed-script
  is_retryable     — all status/http_status combinations
  with_retry       — first-try success, retry on TIMEOUT, retry on 429,
                     no retry on HTTP 500, exhausted retries, wall-clock cap
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, call, patch

from app.generation.harness import (
    RetryConfig,
    SCRIPT_RANGES,
    is_retryable,
    validate_output,
    with_retry,
)


# ===========================================================================
# validate_output
# ===========================================================================

class ValidateOutputLengthTests(unittest.TestCase):

    def test_empty_string_is_too_short(self):
        ok, reason = validate_output("", "hi")
        self.assertFalse(ok)
        self.assertEqual(reason, "answer_too_short")

    def test_none_treated_as_empty(self):
        ok, reason = validate_output(None, "hi")  # type: ignore[arg-type]
        self.assertFalse(ok)
        self.assertEqual(reason, "answer_too_short")

    def test_4_chars_is_too_short(self):
        ok, reason = validate_output("abcd", "en")
        self.assertFalse(ok)
        self.assertEqual(reason, "answer_too_short")

    def test_5_chars_is_valid_boundary(self):
        """5 chars is the minimum — must pass."""
        ok, reason = validate_output("abcde", "en")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_500_chars_is_valid_boundary(self):
        """500 chars is the maximum — must pass."""
        ok, reason = validate_output("a" * 500, "en")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_501_chars_is_too_long(self):
        ok, reason = validate_output("a" * 501, "en")
        self.assertFalse(ok)
        self.assertEqual(reason, "answer_too_long")

    def test_whitespace_stripped_before_length_check(self):
        """Leading/trailing whitespace must not count toward length."""
        ok, reason = validate_output("   ab   ", "en")
        self.assertFalse(ok)
        self.assertEqual(reason, "answer_too_short")


class ValidateOutputScriptTests(unittest.TestCase):

    def test_valid_hindi_answer_passes(self):
        """Answer with Devanagari chars for language=hi must pass."""
        hindi = "मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान थी।"
        ok, reason = validate_output(hindi, "hi")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_latin_only_for_hindi_fails_wrong_script(self):
        ok, reason = validate_output("This is an answer.", "hi")
        self.assertFalse(ok)
        self.assertEqual(reason, "wrong_script")

    def test_mixed_script_passes_for_hindi(self):
        """Mixed-script answer must pass — script PRESENCE not purity."""
        mixed = "यह answer is correct"          # Devanagari + Latin
        ok, reason = validate_output(mixed, "hi")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_valid_marathi_passes(self):
        marathi = "मॅनहॅटन प्रकल्पाचा परिणाम झाला."
        ok, reason = validate_output(marathi, "mr")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_valid_tamil_passes(self):
        tamil = "மன்ஹாட்டன் திட்டம் இரண்டாம் உலகப் போரில் தொடங்கியது."
        ok, reason = validate_output(tamil, "ta")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_latin_for_tamil_fails_wrong_script(self):
        ok, reason = validate_output("Some latin answer.", "ta")
        self.assertFalse(ok)
        self.assertEqual(reason, "wrong_script")

    def test_english_with_language_en_passes(self):
        """English has no script range — no script check performed."""
        ok, reason = validate_output("This is a valid English answer.", "en")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_unknown_language_passes(self):
        """Unknown language code → no script check → pass if length ok."""
        ok, reason = validate_output("valid answer here", "xx")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_all_script_ranges_present(self):
        """Ensure all 13 LANGUAGE_NAMES languages are covered in SCRIPT_RANGES."""
        from app.generation.llm import LANGUAGE_NAMES
        for code in LANGUAGE_NAMES:
            self.assertIn(
                code, SCRIPT_RANGES,
                f"Language '{code}' missing from SCRIPT_RANGES",
            )


# ===========================================================================
# is_retryable
# ===========================================================================

class IsRetryableTests(unittest.TestCase):

    def test_timeout_is_retryable(self):
        self.assertTrue(is_retryable({"status": "TIMEOUT"}))

    def test_http_429_is_retryable(self):
        self.assertTrue(is_retryable({"status": "HTTP_ERROR", "http_status": 429}))

    def test_http_500_is_not_retryable(self):
        self.assertFalse(is_retryable({"status": "HTTP_ERROR", "http_status": 500}))

    def test_http_503_is_not_retryable(self):
        self.assertFalse(is_retryable({"status": "HTTP_ERROR", "http_status": 503}))

    def test_http_400_is_not_retryable(self):
        self.assertFalse(is_retryable({"status": "HTTP_ERROR", "http_status": 400}))

    def test_exception_is_not_retryable(self):
        self.assertFalse(is_retryable({"status": "EXCEPTION"}))

    def test_success_is_not_retryable(self):
        self.assertFalse(is_retryable({"status": "SUCCESS", "answer": "ok"}))

    def test_http_error_without_http_status_is_not_retryable(self):
        self.assertFalse(is_retryable({"status": "HTTP_ERROR"}))


# ===========================================================================
# with_retry
# ===========================================================================

class WithRetryFirstSuccessTests(unittest.TestCase):

    def test_success_on_first_call_returns_attempt_count_1(self):
        fn = MagicMock(return_value={"status": "SUCCESS", "answer": "ok"})
        result = with_retry(fn, RetryConfig())
        self.assertEqual(fn.call_count, 1)
        self.assertEqual(result["attempt_count"], 1)
        self.assertEqual(result["retry_delays_ms"], [])

    def test_success_result_keys_are_preserved(self):
        fn = MagicMock(return_value={"status": "SUCCESS", "answer": "ok", "grounded": True})
        result = with_retry(fn, RetryConfig())
        self.assertEqual(result["grounded"], True)
        self.assertEqual(result["answer"], "ok")


class WithRetryTimeoutTests(unittest.TestCase):

    def test_retry_on_timeout_then_success(self):
        """TIMEOUT on attempt 1, SUCCESS on attempt 2 → attempt_count=2."""
        responses = [
            {"status": "TIMEOUT", "reason": "gemini_timeout"},
            {"status": "SUCCESS", "answer": "Paris"},
        ]
        fn = MagicMock(side_effect=responses)
        cfg = RetryConfig(max_retries=2, base_delay_s=0.0)
        result = with_retry(fn, cfg)
        self.assertEqual(fn.call_count, 2)
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(len(result["retry_delays_ms"]), 1)
        self.assertEqual(result["status"], "SUCCESS")

    def test_exhausted_retries_returns_last_timeout(self):
        """All 3 attempts TIMEOUT → last result returned, attempt_count=3."""
        fn = MagicMock(return_value={"status": "TIMEOUT", "reason": "gemini_timeout"})
        cfg = RetryConfig(max_retries=2, base_delay_s=0.0)
        result = with_retry(fn, cfg)
        self.assertEqual(fn.call_count, 3)
        self.assertEqual(result["attempt_count"], 3)
        self.assertEqual(len(result["retry_delays_ms"]), 2)
        self.assertEqual(result["status"], "TIMEOUT")


class WithRetry429Tests(unittest.TestCase):

    def test_retry_on_429_twice_then_success(self):
        """HTTP 429 on attempts 1 & 2, SUCCESS on attempt 3 → attempt_count=3."""
        responses = [
            {"status": "HTTP_ERROR", "http_status": 429, "reason": "rate_limit"},
            {"status": "HTTP_ERROR", "http_status": 429, "reason": "rate_limit"},
            {"status": "SUCCESS", "answer": "answer"},
        ]
        fn = MagicMock(side_effect=responses)
        cfg = RetryConfig(max_retries=2, base_delay_s=0.0)
        result = with_retry(fn, cfg)
        self.assertEqual(fn.call_count, 3)
        self.assertEqual(result["attempt_count"], 3)
        self.assertEqual(len(result["retry_delays_ms"]), 2)
        self.assertEqual(result["status"], "SUCCESS")


class WithRetryNonRetryableTests(unittest.TestCase):

    def test_http_500_is_not_retried(self):
        """HTTP 500 is permanent — only 1 attempt, no retry."""
        fn = MagicMock(return_value={
            "status": "HTTP_ERROR", "http_status": 500, "reason": "server_error",
        })
        result = with_retry(fn, RetryConfig(base_delay_s=0.0))
        self.assertEqual(fn.call_count, 1)
        self.assertEqual(result["attempt_count"], 1)
        self.assertEqual(result["retry_delays_ms"], [])

    def test_exception_is_not_retried(self):
        fn = MagicMock(return_value={
            "status": "EXCEPTION", "reason": "sdk_error",
        })
        result = with_retry(fn, RetryConfig(base_delay_s=0.0))
        self.assertEqual(fn.call_count, 1)
        self.assertEqual(result["attempt_count"], 1)


class WithRetryWallClockCapTests(unittest.TestCase):

    def test_wall_clock_cap_prevents_second_retry(self):
        """max_wall_s=0.0: elapsed >= cap immediately after attempt 1,
        so no retry is started even though result is retryable."""
        fn = MagicMock(return_value={"status": "TIMEOUT", "reason": "timeout"})
        cfg = RetryConfig(max_retries=2, base_delay_s=0.0, max_wall_s=0.0)
        result = with_retry(fn, cfg)
        # After attempt 1: elapsed is certainly >= 0.0, cap is hit, no retry.
        self.assertEqual(fn.call_count, 1)
        self.assertEqual(result["attempt_count"], 1)
        self.assertEqual(result["retry_delays_ms"], [])


class WithRetryLoggingTests(unittest.TestCase):

    def test_logger_called_on_retry(self):
        responses = [
            {"status": "TIMEOUT", "reason": "timeout"},
            {"status": "SUCCESS", "answer": "ok"},
        ]
        fn = MagicMock(side_effect=responses)
        mock_logger = MagicMock()
        cfg = RetryConfig(max_retries=2, base_delay_s=0.0)
        with_retry(fn, cfg, mock_logger)
        self.assertTrue(mock_logger.info.called)

    def test_no_log_on_first_success(self):
        fn = MagicMock(return_value={"status": "SUCCESS", "answer": "ok"})
        mock_logger = MagicMock()
        with_retry(fn, RetryConfig(), mock_logger)
        mock_logger.info.assert_not_called()


class WithRetryBackoffValuesTests(unittest.TestCase):

    def test_backoff_delay_doubles(self):
        """First retry delay ≈ base_delay_s ms, second ≈ 2× base_delay_s ms."""
        responses = [
            {"status": "TIMEOUT"},
            {"status": "TIMEOUT"},
            {"status": "SUCCESS", "answer": "ok"},
        ]
        fn = MagicMock(side_effect=responses)
        cfg = RetryConfig(max_retries=2, base_delay_s=0.001)  # 1 ms for speed
        result = with_retry(fn, cfg)
        delays = result["retry_delays_ms"]
        self.assertEqual(len(delays), 2)
        # delay[0] = 1ms, delay[1] = 2ms
        self.assertAlmostEqual(delays[0], 1.0, places=0)
        self.assertAlmostEqual(delays[1], 2.0, places=0)


if __name__ == "__main__":
    unittest.main()
