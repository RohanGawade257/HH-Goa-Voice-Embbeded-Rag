"""Live connectivity test for the configured LLM API key.

This test verifies:
  1. An API key is present in the environment.
  2. The key is accepted by the configured endpoint (no 401/403).
  3. The model responds with a valid chat completion (non-empty answer).

Run:
    python -m pytest tests/test_llm_api_connectivity.py -v

The test is skipped automatically when LLM_API_KEY / HF_API_KEY / HF_TOKEN is
not set, so it is safe to include in a normal CI run — it only executes when
credentials are available.

To force-run against the live API:
    LLM_API_KEY=hf_... python -m pytest tests/test_llm_api_connectivity.py -v
"""

from __future__ import annotations

import os
import time
import unittest

import httpx


# ---------------------------------------------------------------------------
# Load .env early so the test works when run directly from the project root.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Pull config from environment (mirrors app/config.py logic exactly so we
# never need to import the full app stack in this standalone test).
# ---------------------------------------------------------------------------
_API_KEY: str = (
    os.getenv("LLM_API_KEY")
    or os.getenv("HF_API_KEY")
    or os.getenv("HF_TOKEN")
    or ""
)
_CHAT_URL: str = os.getenv(
    "LLM_CHAT_COMPLETIONS_URL",
    os.getenv(
        "HF_CHAT_COMPLETIONS_URL",
        "https://router.huggingface.co/v1/chat/completions",
    ),
)
_MODEL: str = os.getenv("QWEN_MODEL", "Qwen/Qwen3-0.6B")
_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "10"))

# Minimal probe payload — single short turn so the call is fast and cheap.
_PROBE_MESSAGES = [
    {"role": "system", "content": "Reply with the single word: OK"},
    {"role": "user", "content": "Respond with: OK"},
]
_PROBE_PAYLOAD = {
    "model": _MODEL,
    "messages": _PROBE_MESSAGES,
    "max_tokens": 8,
    "temperature": 0.0,
    "stream": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> httpx.Client:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_API_KEY}",
    }
    return httpx.Client(timeout=_TIMEOUT, headers=headers)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(_API_KEY, "LLM_API_KEY / HF_API_KEY / HF_TOKEN is not set — skipping live connectivity test")
class LLMApiConnectivityTests(unittest.TestCase):
    """Live tests that require a real API key.  Skipped in offline CI."""

    # ------------------------------------------------------------------
    # 1. API key is present
    # ------------------------------------------------------------------

    def test_api_key_is_present(self):
        """The API key env var must be non-empty before any network call."""
        self.assertTrue(
            bool(_API_KEY),
            "No API key found in LLM_API_KEY / HF_API_KEY / HF_TOKEN.",
        )
        # Must not be the placeholder from .env.example
        self.assertNotEqual(
            _API_KEY,
            "your_huggingface_api_key_here",
            "API key is still the example placeholder — set a real key.",
        )
        print(f"\n  Key prefix : {_API_KEY[:8]}{'*' * max(0, len(_API_KEY) - 8)}")
        print(f"  Endpoint   : {_CHAT_URL}")
        print(f"  Model      : {_MODEL}")

    # ------------------------------------------------------------------
    # 2. Endpoint is reachable (HTTP-level check)
    # ------------------------------------------------------------------

    def test_endpoint_is_reachable(self):
        """A POST to the chat completions endpoint must not time out."""
        start = time.perf_counter()
        try:
            with _make_client() as client:
                response = client.post(_CHAT_URL, json=_PROBE_PAYLOAD)
            elapsed_ms = (time.perf_counter() - start) * 1000
        except httpx.TimeoutException as exc:
            self.fail(
                f"Connection timed out after {_TIMEOUT}s — "
                f"check LLM_CHAT_COMPLETIONS_URL or network: {exc}"
            )
        except httpx.ConnectError as exc:
            self.fail(
                f"Could not reach {_CHAT_URL} — "
                f"check LLM_CHAT_COMPLETIONS_URL or network: {exc}"
            )

        print(f"\n  HTTP status : {response.status_code}")
        print(f"  Round-trip  : {elapsed_ms:.0f} ms")
        # Any response (including 4xx) proves the endpoint is reachable.
        self.assertIsNotNone(response.status_code)

    # ------------------------------------------------------------------
    # 3. API key is accepted (no 401 / 403)
    # ------------------------------------------------------------------

    def test_api_key_is_accepted(self):
        """The server must not reject the key with 401 or 403."""
        try:
            with _make_client() as client:
                response = client.post(_CHAT_URL, json=_PROBE_PAYLOAD)
        except httpx.TimeoutException:
            self.skipTest(
                "Request timed out — cannot verify key acceptance. "
                "Check network or increase LLM_TIMEOUT_SECONDS."
            )
        except httpx.ConnectError as exc:
            self.skipTest(f"Cannot reach {_CHAT_URL}: {exc}")

        status = response.status_code
        if status == 401:
            self.fail(
                "401 Unauthorized — API key is missing or malformed. "
                "Set LLM_API_KEY / HF_API_KEY to a valid Hugging Face token."
            )
        if status == 403:
            self.fail(
                "403 Forbidden — API key does not have permission for this "
                f"endpoint or model ({_MODEL}). Check token scopes."
            )
        # Any non-auth error (400, 429, 500, 200 …) means the key itself is valid.
        print(f"\n  API key accepted — HTTP {status}")

    # ------------------------------------------------------------------
    # 4. Model returns a valid chat completion
    # ------------------------------------------------------------------

    def test_model_returns_valid_completion(self):
        """A successful generation must return a non-empty answer string."""
        try:
            with _make_client() as client:
                response = client.post(_CHAT_URL, json=_PROBE_PAYLOAD)
        except httpx.TimeoutException:
            self.skipTest(
                "Request timed out — cannot verify model completion. "
                "Check network or increase LLM_TIMEOUT_SECONDS."
            )
        except httpx.ConnectError as exc:
            self.skipTest(f"Cannot reach {_CHAT_URL}: {exc}")

        # Skip transient infra issues — these are not key or model problems.
        if response.status_code == 429:
            self.skipTest("Rate-limited (429) — retry later.")
        if response.status_code in (502, 503, 504):
            self.skipTest(f"Service temporarily unavailable ({response.status_code}).")

        # 400 with model_not_supported is a configuration issue, not a key issue.
        if response.status_code == 400:
            body = response.text[:400]
            if "model_not_supported" in body or "not supported" in body.lower():
                self.fail(
                    f"400 Bad Request — model '{_MODEL}' is not supported by "
                    "the configured endpoint/provider. "
                    "Update QWEN_MODEL in .env to a model that the provider supports. "
                    f"Provider response: {body}"
                )
            self.fail(f"400 Bad Request. Body: {body}")

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200, got {response.status_code}. Body: {response.text[:300]}",
        )

        data = response.json()
        choices = data.get("choices", [])
        self.assertTrue(choices, f"Response has no 'choices'. Full body: {data}")

        answer = choices[0].get("message", {}).get("content", "").strip()
        self.assertTrue(
            bool(answer),
            f"Answer content is empty. Full body: {data}",
        )
        print(f"\n  Model answer : {answer!r}")

    # ------------------------------------------------------------------
    # 5. End-to-end latency is within the configured timeout
    # ------------------------------------------------------------------

    def test_latency_is_within_timeout(self):
        """Round-trip must complete within the configured LLM_TIMEOUT_SECONDS."""
        start = time.perf_counter()
        try:
            with _make_client() as client:
                response = client.post(_CHAT_URL, json=_PROBE_PAYLOAD)
        except httpx.TimeoutException:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.fail(
                f"Request timed out at {elapsed_ms:.0f} ms "
                f"(timeout={_TIMEOUT}s). LLM is too slow for the current budget."
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Skip latency assertion for non-200 to avoid false failures
        if response.status_code != 200:
            self.skipTest(f"Non-200 response ({response.status_code}) — skipping latency check.")

        print(f"\n  Round-trip : {elapsed_ms:.0f} ms  (budget: {_TIMEOUT * 1000:.0f} ms)")
        self.assertLess(
            elapsed_ms,
            _TIMEOUT * 1000,
            f"Latency {elapsed_ms:.0f} ms exceeded timeout budget {_TIMEOUT * 1000:.0f} ms.",
        )


# ---------------------------------------------------------------------------
# Offline guard — runs even without a key, verifies skip behaviour is correct
# ---------------------------------------------------------------------------

class LLMApiKeyPresenceTest(unittest.TestCase):
    """Always runs.  Provides a clear diagnostic when the key is missing."""

    def test_reports_key_status(self):
        """Print a clear message about whether the key is configured."""
        if not _API_KEY:
            print(
                "\n  [INFO] No LLM API key found in LLM_API_KEY / HF_API_KEY / HF_TOKEN.\n"
                "         Set one in .env to enable the live connectivity tests."
            )
        else:
            masked = _API_KEY[:8] + "*" * max(0, len(_API_KEY) - 8)
            print(f"\n  [INFO] LLM API key detected: {masked}")
        # This test always passes — it's purely informational.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
