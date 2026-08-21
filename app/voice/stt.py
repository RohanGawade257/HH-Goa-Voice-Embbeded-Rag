"""
HH Goa 2026 - Voice RAG
Sarvam Speech-to-Text integration.

The production path uses the real Sarvam REST STT API. Mock STT is available
only when SARVAM_STT_MOCK=1 is set for local/offline tests.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from app.config import (
    SARVAM_API_KEY_ENV,
    SARVAM_STT_MOCK,
    SARVAM_STT_MODE,
    SARVAM_STT_MODEL,
    SARVAM_STT_TIMEOUT_SECONDS,
    SARVAM_STT_URL,
    get_sarvam_api_key,
)

logger = logging.getLogger(__name__)


# Current Sarvam STT language support for /speech-to-text.
# These are STT capabilities, independent from the RAG corpus language set.
SUPPORTED_LANGUAGES = {
    "unknown": "Auto-detect",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "en-IN": "English",
    "gu-IN": "Gujarati",
    "as-IN": "Assamese",
    "ur-IN": "Urdu",
    "ne-IN": "Nepali",
    "kok-IN": "Konkani",
    "ks-IN": "Kashmiri",
    "sd-IN": "Sindhi",
    "sa-IN": "Sanskrit",
    "sat-IN": "Santali",
    "mni-IN": "Manipuri",
    "brx-IN": "Bodo",
    "mai-IN": "Maithili",
    "doi-IN": "Dogri",
}

# Frontend/RAG commonly use ISO 639 "or" for Odia, while Sarvam's STT
# contract currently uses "od-IN".
LANGUAGE_ALIASES = {
    "or-IN": "od-IN",
    "or": "od-IN",
    "od": "od-IN",
}

DEFAULT_LANGUAGE = "hi-IN"
AUTH_HEADER = "api-subscription-key"
SUPPORTED_HTTP_FAILURES = {400, 403, 422, 429, 500, 503}
_ASSIGNMENT_PREFIX_RE = re.compile(r"^\s*SARVAM_API_KEY\s*=\s*", re.IGNORECASE)


@dataclass
class SarvamCredential:
    key: str
    normalized: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.key)

    @property
    def fingerprint(self) -> dict[str, Any]:
        return {
            "configured": bool(self.key),
            "length": len(self.key),
            "prefix": self.key[:8] if self.key else "",
            "suffix": self.key[-4:] if self.key else "",
            "normalized_assignment_prefix": self.normalized,
        }


class STTError(RuntimeError):
    """Base error with safe, structured STT diagnostic information."""

    category = "stt_error"
    provider_status_code: int | None = None
    provider_error_code: str | None = None

    def __init__(
        self,
        message: str,
        *,
        category: str | None = None,
        provider_status_code: int | None = None,
        provider_error_code: str | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message)
        if category is not None:
            self.category = category
        self.provider_status_code = provider_status_code
        self.provider_error_code = provider_error_code
        self.provider_message = provider_message

    def to_safe_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "message": str(self),
            "category": self.category,
        }
        if self.provider_status_code is not None:
            payload["provider_status_code"] = self.provider_status_code
        if self.provider_error_code:
            payload["provider_error_code"] = self.provider_error_code
        if self.provider_message:
            payload["provider_message"] = self.provider_message
        return payload


class STTConfigurationError(STTError):
    category = "configuration"


class STTAuthenticationError(STTError):
    category = "authentication"


class STTBadRequestError(STTError):
    category = "bad_request"


class STTRateLimitError(STTError):
    category = "rate_limit"


class STTProviderUnavailableError(STTError):
    category = "provider_unavailable"


class STTTimeoutError(STTError):
    category = "timeout"


class STTNetworkError(STTError):
    category = "network"


def _clean_api_key(raw_key: str) -> SarvamCredential:
    key = (raw_key or "").strip().strip('"').strip("'")
    normalized = False
    if _ASSIGNMENT_PREFIX_RE.match(key):
        key = _ASSIGNMENT_PREFIX_RE.sub("", key, count=1).strip().strip('"').strip("'")
        normalized = True
    return SarvamCredential(key=key, normalized=normalized)


def _get_credential() -> SarvamCredential:
    return _clean_api_key(get_sarvam_api_key())


def _normalize_language_code(language_code: str) -> str:
    code = (language_code or DEFAULT_LANGUAGE).strip()
    return LANGUAGE_ALIASES.get(code, code)


def _sanitize_text(text: str, credential: SarvamCredential | None = None) -> str:
    sanitized = text or ""
    if credential and credential.key:
        sanitized = sanitized.replace(credential.key, "[REDACTED_SARVAM_API_KEY]")
    raw_key = get_sarvam_api_key()
    if raw_key:
        sanitized = sanitized.replace(raw_key, "[REDACTED_SARVAM_API_KEY]")
    return sanitized[:500]


def _parse_provider_error(response: Any, credential: SarvamCredential) -> tuple[str | None, str | None]:
    text = _sanitize_text(getattr(response, "text", "") or "", credential)
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError, AttributeError):
        payload = None

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
    elif isinstance(payload, dict):
        code = payload.get("code") or payload.get("error_code")
        message = payload.get("message") or payload.get("detail")
    else:
        code = None
        message = None

    safe_code = str(code) if code else None
    safe_message = _sanitize_text(str(message), credential) if message else text
    return safe_code, safe_message


def _raise_for_provider_failure(response: Any, credential: SarvamCredential) -> None:
    status_code = int(response.status_code)
    error_code, error_message = _parse_provider_error(response, credential)

    if status_code == 403:
        raise STTAuthenticationError(
            "Sarvam STT authentication failed. Verify SARVAM_API_KEY and Sarvam account/API access.",
            provider_status_code=status_code,
            provider_error_code=error_code,
            provider_message=error_message,
        )
    if status_code in {400, 422}:
        raise STTBadRequestError(
            "Sarvam STT rejected the audio or request parameters.",
            provider_status_code=status_code,
            provider_error_code=error_code,
            provider_message=error_message,
        )
    if status_code == 429:
        raise STTRateLimitError(
            "Sarvam STT rate limit or quota was exceeded.",
            provider_status_code=status_code,
            provider_error_code=error_code,
            provider_message=error_message,
        )
    if status_code in {500, 503}:
        raise STTProviderUnavailableError(
            "Sarvam STT provider is temporarily unavailable.",
            provider_status_code=status_code,
            provider_error_code=error_code,
            provider_message=error_message,
        )
    raise STTProviderUnavailableError(
        "Sarvam STT provider returned an unexpected HTTP error.",
        provider_status_code=status_code,
        provider_error_code=error_code,
        provider_message=error_message,
    )


def _transcribe_sarvam(
    audio_bytes: bytes,
    language_code: str = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    """
    Call the real Sarvam STT REST API.

    Sarvam's current contract uses:
      - POST https://api.sarvam.ai/speech-to-text
      - header api-subscription-key: <key>
      - multipart field file
      - model saaras:v3 with mode=transcribe
    """

    try:
        import httpx
    except ImportError as exc:
        raise STTConfigurationError(
            "httpx is required for Sarvam STT. Install it with: pip install httpx"
        ) from exc

    credential = _get_credential()
    if not credential.configured:
        raise STTConfigurationError(
            f"{SARVAM_API_KEY_ENV} is not configured. Set it to enable real Sarvam STT."
        )

    if credential.normalized:
        logger.warning(
            "Sarvam API key value included a SARVAM_API_KEY= assignment prefix; "
            "normalized it before sending the provider request."
        )

    key_info = credential.fingerprint
    logger.info(
        "Sarvam STT configured=%s key_length=%s key_prefix=%s normalized=%s endpoint=%s model=%s",
        key_info["configured"],
        key_info["length"],
        key_info["prefix"],
        key_info["normalized_assignment_prefix"],
        SARVAM_STT_URL,
        SARVAM_STT_MODEL,
    )

    start = time.perf_counter()
    normalized_language = _normalize_language_code(language_code)

    headers = {
        AUTH_HEADER: credential.key,
    }
    files = {
        "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
    }
    data = {
        "language_code": normalized_language,
        "model": SARVAM_STT_MODEL,
        "mode": SARVAM_STT_MODE,
        "with_timestamps": "false",
    }

    try:
        with httpx.Client(timeout=SARVAM_STT_TIMEOUT_SECONDS) as client:
            response = client.post(
                SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
            )
    except httpx.TimeoutException as exc:
        raise STTTimeoutError("Sarvam STT request timed out.") from exc
    except httpx.RequestError as exc:
        raise STTNetworkError(
            "Sarvam STT network request failed.",
            provider_message=_sanitize_text(str(exc), credential),
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        _raise_for_provider_failure(response, credential)

    result = response.json()
    transcript = result.get("transcript", "")
    provider_language = result.get("language_code") or normalized_language

    return {
        "transcript": transcript,
        "language_code": provider_language,
        "requested_language_code": language_code,
        "latency_ms": round(latency_ms, 2),
        "provider": "sarvam",
        "model": SARVAM_STT_MODEL,
    }


MOCK_TRANSCRIPTS: dict[str, str] = {
    "hi-IN": "\u092e\u0948\u0928\u0939\u091f\u094d\u091f\u0928 \u092a\u0930\u093f\u092f\u094b\u091c\u0928\u093e \u0915\u094d\u092f\u093e \u0925\u0940?",
    "bn-IN": "\u09ae\u09cd\u09af\u09be\u09a8\u09b9\u09be\u099f\u09a8 \u09aa\u09cd\u09b0\u0995\u09b2\u09cd\u09aa\u09c7\u09b0 \u09b8\u09be\u09ab\u09b2\u09cd\u09af\u09c7\u09b0 \u09a4\u09be\u09ce\u0995\u09cd\u09b7\u09a3\u09bf\u0995 \u09aa\u09cd\u09b0\u09ad\u09be\u09ac \u0995\u09c0 \u099b\u09bf\u09b2?",
    "gu-IN": "\u0aae\u0ac7\u0aa8\u0ab9\u0a9f\u0aa8 \u0aaa\u0acd\u0ab0\u0acb\u0a9c\u0ac7\u0a95\u0acd\u0a9f\u0aa8\u0ac0 \u0ab8\u0aab\u0ab3\u0aa4\u0abe\u0aa8\u0ac0 \u0aa4\u0abe\u0aa4\u0acd\u0a95\u0abe\u0ab2\u0abf\u0a95 \u0a85\u0ab8\u0ab0 \u0ab6\u0ac1\u0a82 \u0ab9\u0aa4\u0ac0?",
    "kn-IN": "\u0cae\u0ccd\u0caf\u0cbe\u0ca8\u0ccd\u200c\u0cb9\u0ccd\u0caf\u0cbe\u0c9f\u0ca8\u0ccd \u0caf\u0ccb\u0c9c\u0ca8\u0cc6\u0caf \u0caf\u0cb6\u0cb8\u0ccd\u0cb8\u0cbf\u0ca8 \u0ca4\u0c95\u0ccd\u0cb7\u0ca3\u0ca6 \u0caa\u0cb0\u0cbf\u0ca3\u0cbe\u0cae \u0c8f\u0ca8\u0cc1 \u0c86\u0c97\u0cbf\u0ca4\u0ccd\u0ca4\u0cc1?",
    "ml-IN": "\u0d2e\u0d3e\u0d7b\u0d39\u0d3e\u0d31\u0d4d\u0d31\u0d7b \u0d2a\u0d26\u0d4d\u0d27\u0d24\u0d3f\u0d2f\u0d41\u0d1f\u0d46 \u0d35\u0d3f\u0d1c\u0d2f\u0d24\u0d4d\u0d24\u0d3f\u0d28\u0d4d\u0d31\u0d46 \u0d09\u0d1f\u0d28\u0d1f\u0d3f \u0d06\u0d18\u0d3e\u0d24\u0d02 \u0d0e\u0d28\u0d4d\u0d24\u0d3e\u0d2f\u0d3f\u0d30\u0d41\u0d28\u0d4d\u0d28\u0d41?",
    "mr-IN": "\u092e\u0945\u0928\u0939\u0945\u091f\u0928 \u092a\u094d\u0930\u0915\u0932\u094d\u092a\u093e\u091a\u094d\u092f\u093e \u092f\u0936\u093e\u091a\u093e \u0924\u093e\u0924\u094d\u0915\u093e\u0933 \u0915\u093e\u092f \u092a\u0930\u093f\u0923\u093e\u092e \u091d\u093e\u0932\u093e?",
    "od-IN": "\u0b2e\u0b4d\u0b5f\u0b3e\u0b28\u0b39\u0b3e\u0b1f\u0b28 \u0b2a\u0b4d\u0b30\u0b15\u0b33\u0b4d\u0b2a\u0b30 \u0b38\u0b2b\u0b33\u0b24\u0b3e\u0b30 \u0b24\u0b24\u0b4d\u200c\u0b15\u0b4d\u0b37\u0b23\u0b3e\u0b24 \u0b2a\u0b4d\u0b30\u0b2d\u0b3e\u0b2c \u0b15'\u0b23 \u0b25\u0b3f\u0b32\u0b3e?",
    "pa-IN": "\u0a2e\u0a48\u0a28\u0a39\u0a48\u0a1f\u0a28 \u0a2a\u0a4d\u0a30\u0a4b\u0a1c\u0a48\u0a15\u0a1f \u0a26\u0a40 \u0a38\u0a2b\u0a32\u0a24\u0a3e \u0a26\u0a3e \u0a24\u0a41\u0a30\u0a70\u0a24 \u0a2a\u0a4d\u0a30\u0a2d\u0a3e\u0a35 \u0a15\u0a40 \u0a38\u0a40?",
    "ta-IN": "\u0bae\u0ba9\u0bcd\u0bb9\u0bbe\u0b9f\u0bcd\u0b9f\u0ba9\u0bcd \u0ba4\u0bbf\u0b9f\u0bcd\u0b9f\u0ba4\u0bcd\u0ba4\u0bbf\u0ba9\u0bcd \u0bb5\u0bc6\u0bb1\u0bcd\u0bb1\u0bbf\u0baf\u0bbf\u0ba9\u0bcd \u0b89\u0b9f\u0ba9\u0b9f\u0bbf \u0bb5\u0bbf\u0bb3\u0bc8\u0bb5\u0bc1 \u0b8e\u0ba9\u0bcd\u0ba9?",
    "te-IN": "\u0c2e\u0c3e\u0c28\u0c4d\u0c39\u0c3e\u0c1f\u0c28\u0c4d \u0c2a\u0c4d\u0c30\u0c3e\u0c1c\u0c46\u0c15\u0c4d\u0c1f\u0c4d \u0c35\u0c3f\u0c1c\u0c2f\u0c02 \u0c2f\u0c4a\u0c15\u0c4d\u0c15 \u0c24\u0c15\u0c4d\u0c37\u0c23 \u0c2a\u0c4d\u0c30\u0c2d\u0c3e\u0c35\u0c02 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?",
    "en-IN": "What was the Manhattan Project?",
}

_MOCK_FALLBACK = MOCK_TRANSCRIPTS["hi-IN"]


def _transcribe_mock(
    audio_bytes: bytes,
    language_code: str = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    logger.warning("Using mock STT because SARVAM_STT_MOCK=1.")
    normalized_language = _normalize_language_code(language_code)
    transcript = MOCK_TRANSCRIPTS.get(normalized_language, _MOCK_FALLBACK)

    return {
        "transcript": transcript,
        "language_code": normalized_language,
        "requested_language_code": language_code,
        "latency_ms": 0.0,
        "provider": "mock",
        "mock": True,
    }


def transcribe_audio(
    audio_bytes: bytes,
    language_code: str = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    """
    Transcribe audio bytes to text.

    Production requires SARVAM_API_KEY. Mock mode is explicit and only enabled
    with SARVAM_STT_MOCK=1.
    """

    if not audio_bytes:
        return {
            "transcript": "",
            "language_code": language_code,
            "latency_ms": 0.0,
            "provider": "none",
            "error": "empty_audio",
        }

    normalized_language = _normalize_language_code(language_code)
    if normalized_language not in SUPPORTED_LANGUAGES:
        logger.warning(
            "Language code %s is not in the Sarvam STT supported list; attempting anyway.",
            language_code,
        )

    credential = _get_credential()
    if credential.configured:
        return _transcribe_sarvam(audio_bytes, language_code)
    if SARVAM_STT_MOCK:
        return _transcribe_mock(audio_bytes, language_code)

    raise STTConfigurationError(
        f"{SARVAM_API_KEY_ENV} is missing. Real Sarvam STT is disabled and SARVAM_STT_MOCK is not enabled."
    )


def stt_status() -> dict[str, Any]:
    credential = _get_credential()
    has_key = credential.configured

    return {
        "provider": "sarvam" if has_key else ("mock" if SARVAM_STT_MOCK else "unconfigured"),
        "configured": has_key,
        "mock_enabled": SARVAM_STT_MOCK,
        "language_default": DEFAULT_LANGUAGE,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "endpoint": SARVAM_STT_URL,
        "model": SARVAM_STT_MODEL,
        "mode": SARVAM_STT_MODE,
        "key_length": len(credential.key) if has_key else 0,
        "key_prefix": credential.key[:8] if has_key else "",
        "key_normalized_assignment_prefix": credential.normalized,
        "note": (
            "Sarvam STT API key is configured."
            if has_key
            else "Set SARVAM_API_KEY or enable SARVAM_STT_MOCK=1 for local tests."
        ),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("STT MODULE STATUS")
    print("=" * 60)
    status = stt_status()
    print(f"Provider    : {status['provider']}")
    print(f"Configured  : {status['configured']}")
    print(f"Key length  : {status['key_length']}")
    print(f"Key prefix  : {status['key_prefix']}")
    print(f"Endpoint    : {status['endpoint']}")
    print(f"Model       : {status['model']}")
    print(f"Mode        : {status['mode']}")
    print(f"Default lang: {status['language_default']}")
    print(f"Note        : {status['note']}")
