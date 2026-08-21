import os
import unittest
from unittest.mock import Mock, patch

import httpx

from app.voice import stt


TEST_KEY = "sk_test_unit_secret_1234567890"


class FakeResponse:
    def __init__(self, status_code, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else str(payload or "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class SarvamSTTTests(unittest.TestCase):
    def test_environment_key_selects_real_provider(self):
        with patch.dict(os.environ, {"SARVAM_API_KEY": TEST_KEY}, clear=False):
            with patch.object(stt, "_transcribe_sarvam", return_value={"provider": "sarvam"}) as call:
                result = stt.transcribe_audio(b"wav", "hi-IN")

        self.assertEqual(result["provider"], "sarvam")
        call.assert_called_once()

    def test_missing_key_requires_explicit_mock_mode(self):
        with patch.dict(os.environ, {"SARVAM_API_KEY": ""}, clear=False):
            with patch.object(stt, "SARVAM_STT_MOCK", False):
                with self.assertRaises(stt.STTConfigurationError):
                    stt.transcribe_audio(b"wav", "hi-IN")

            with patch.object(stt, "SARVAM_STT_MOCK", True):
                result = stt.transcribe_audio(b"wav", "hi-IN")

        self.assertEqual(result["provider"], "mock")
        self.assertTrue(result["mock"])

    def test_header_uses_configured_key_without_leaking_it(self):
        response = FakeResponse(
            200,
            {"transcript": "hello", "language_code": "en-IN"},
        )
        client = Mock()
        client.post.return_value = response
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=None)

        with patch.dict(os.environ, {"SARVAM_API_KEY": TEST_KEY}, clear=False):
            with patch("httpx.Client", return_value=client):
                stt._transcribe_sarvam(b"wav", "en-IN")

        _, kwargs = client.post.call_args
        self.assertEqual(kwargs["headers"][stt.AUTH_HEADER], TEST_KEY)
        self.assertNotIn(TEST_KEY, str(stt.stt_status()))

    def test_duplicate_assignment_prefix_is_removed_before_request(self):
        response = FakeResponse(200, {"transcript": "hello", "language_code": "en-IN"})
        client = Mock()
        client.post.return_value = response
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=None)

        with patch.dict(
            os.environ,
            {"SARVAM_API_KEY": f"SARVAM_API_KEY={TEST_KEY}"},
            clear=False,
        ):
            with patch("httpx.Client", return_value=client):
                stt._transcribe_sarvam(b"wav", "en-IN")

        _, kwargs = client.post.call_args
        self.assertEqual(kwargs["headers"][stt.AUTH_HEADER], TEST_KEY)

    def test_successful_sarvam_response(self):
        response = FakeResponse(
            200,
            {"transcript": "namaste", "language_code": "hi-IN"},
        )
        client = Mock()
        client.post.return_value = response
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=None)

        with patch.dict(os.environ, {"SARVAM_API_KEY": TEST_KEY}, clear=False):
            with patch("httpx.Client", return_value=client):
                result = stt._transcribe_sarvam(b"wav", "hi-IN")

        self.assertEqual(result["transcript"], "namaste")
        self.assertEqual(result["language_code"], "hi-IN")
        self.assertEqual(result["provider"], "sarvam")
        self.assertIsInstance(result["latency_ms"], float)

    def test_authentication_failure_is_classified(self):
        response = FakeResponse(
            403,
            {"error": {"code": "invalid_api_key_error", "message": "Invalid API key"}},
        )
        with self.assertRaises(stt.STTAuthenticationError) as ctx:
            stt._raise_for_provider_failure(response, stt.SarvamCredential(TEST_KEY))

        safe = ctx.exception.to_safe_dict()
        self.assertEqual(safe["category"], "authentication")
        self.assertEqual(safe["provider_status_code"], 403)
        self.assertEqual(safe["provider_error_code"], "invalid_api_key_error")

    def test_other_http_failures_are_distinguished(self):
        cases = [
            (400, stt.STTBadRequestError, "bad_request"),
            (422, stt.STTBadRequestError, "bad_request"),
            (429, stt.STTRateLimitError, "rate_limit"),
            (500, stt.STTProviderUnavailableError, "provider_unavailable"),
            (503, stt.STTProviderUnavailableError, "provider_unavailable"),
        ]

        for status_code, exc_type, category in cases:
            with self.subTest(status_code=status_code):
                response = FakeResponse(
                    status_code,
                    {"error": {"code": "provider_error", "message": "failure"}},
                )
                with self.assertRaises(exc_type) as ctx:
                    stt._raise_for_provider_failure(response, stt.SarvamCredential(TEST_KEY))
                self.assertEqual(ctx.exception.to_safe_dict()["category"], category)

    def test_timeout_is_classified(self):
        client = Mock()
        client.post.side_effect = httpx.TimeoutException("slow")
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=None)

        with patch.dict(os.environ, {"SARVAM_API_KEY": TEST_KEY}, clear=False):
            with patch("httpx.Client", return_value=client):
                with self.assertRaises(stt.STTTimeoutError):
                    stt._transcribe_sarvam(b"wav", "hi-IN")

    def test_errors_do_not_contain_full_api_key(self):
        response = FakeResponse(
            400,
            {"error": {"code": "invalid_request_error", "message": f"bad {TEST_KEY}"}},
            text=f"bad {TEST_KEY}",
        )
        with self.assertRaises(stt.STTBadRequestError) as ctx:
            stt._raise_for_provider_failure(response, stt.SarvamCredential(TEST_KEY))

        self.assertNotIn(TEST_KEY, str(ctx.exception))
        self.assertNotIn(TEST_KEY, str(ctx.exception.to_safe_dict()))


if __name__ == "__main__":
    unittest.main()
