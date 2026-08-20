"""
HH Goa 2026 — Voice RAG
STT Module — Sarvam Speech-to-Text Integration

Supports:
  - Real Sarvam STT API (when SARVAM_API_KEY is set)
  - Mock mode for development/testing (when key is absent)

Usage:
    from stt import transcribe_audio

    transcript = transcribe_audio(audio_bytes, language_code="hi-IN")
"""

import os
import io
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

# Supported language codes for Sarvam STT
SUPPORTED_LANGUAGES = {
    "hi-IN": "Hindi",
    "en-IN": "English (India)",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
}

DEFAULT_LANGUAGE = "hi-IN"


# ============================================================
# SARVAM STT
# ============================================================

def _transcribe_sarvam(
    audio_bytes: bytes,
    language_code: str = DEFAULT_LANGUAGE,
) -> dict:
    """
    Call the real Sarvam STT API.

    Requires SARVAM_API_KEY environment variable.

    Returns:
        {
            "transcript": str,
            "language_code": str,
            "latency_ms": float,
            "provider": "sarvam",
        }
    """

    try:
        import httpx
    except ImportError:
        raise RuntimeError(
            "httpx is required for Sarvam STT. "
            "Install it with: pip install httpx"
        )

    start = time.perf_counter()

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
    }

    files = {
        "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
    }

    data = {
        "language_code": language_code,
        "model": "saarika:v2",
        "with_timestamps": "false",
        "with_disfluencies": "false",
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            SARVAM_STT_URL,
            headers=headers,
            files=files,
            data=data,
        )

    latency_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        raise RuntimeError(
            f"Sarvam STT API error: "
            f"HTTP {response.status_code} — {response.text[:200]}"
        )

    result = response.json()

    transcript = result.get("transcript", "")

    return {
        "transcript": transcript,
        "language_code": language_code,
        "latency_ms": round(latency_ms, 2),
        "provider": "sarvam",
    }


# ============================================================
# MOCK STT
# ============================================================

# Representative test queries per BCP-47 language code.
# These are used when SARVAM_API_KEY is not set so that
# dev/test runs produce language-appropriate mock transcripts.
MOCK_TRANSCRIPTS: dict[str, str] = {
    "hi-IN": "मैनहट्टन परियोजना क्या थी?",
    "bn-IN": "ম্যানহাটন প্রকল্পের সাফল্যের তাৎক্ষণিক প্রভাব কী ছিল?",
    "gu-IN": "મેનહટન પ્રોજેક્ટની સફળતાની તાત્કાલિક અસર શું હતી?",
    "kn-IN": "ಮ್ಯಾನ್\u200cಹ್ಯಾಟನ್ ಯೋಜನೆಯ ಯಶಸ್ಸಿನ ತಕ್ಷಣದ ಪರಿಣಾಮ ಏನು ಆಗಿತ್ತು?",
    "ml-IN": "മാൻഹാട്ടൻ പദ്ധതിയുടെ വിജയത്തിന്റെ ഉടനടി ആഘാതം എന്തായിരുന്നു?",
    "mr-IN": "मॅनहॅटन प्रकल्पाच्या यशाचा तात्काळ काय परिणाम झाला?",
    "od-IN": "ମ୍ୟାନହାଟନ ପ୍ରକଳ୍ପର ସଫଳତାର ତତ୍\u200cକ୍ଷଣାତ ପ୍ରଭାବ କ'ଣ ଥିଲା?",
    "pa-IN": "ਮੈਨਹੈਟਨ ਪ੍ਰੋਜੈਕਟ ਦੀ ਸਫਲਤਾ ਦਾ ਤੁਰੰਤ ਪ੍ਰਭਾਵ ਕੀ ਸੀ?",
    "ta-IN": "மன்ஹாட்டன் திட்டத்தின் வெற்றியின் உடனடி விளைவு என்ன?",
    "te-IN": "మాన్హాటన్ ప్రాజెక్ట్ విజయం యొక్క తక్షణ ప్రభావం ఏమిటి?",
    "en-IN": "What was the Manhattan Project?",
}

# Hindi fallback for any language not in MOCK_TRANSCRIPTS
_MOCK_FALLBACK = MOCK_TRANSCRIPTS["hi-IN"]


def _transcribe_mock(
    audio_bytes: bytes,
    language_code: str = DEFAULT_LANGUAGE,
) -> dict:
    """
    Mock STT for development/testing when Sarvam API is unavailable.

    Returns a language-appropriate test query based on language_code.
    Falls back to Hindi if the language is not in MOCK_TRANSCRIPTS.
    """

    logger.warning(
        "SARVAM_API_KEY not set. Using mock STT. "
        "Set SARVAM_API_KEY environment variable for real STT."
    )

    transcript = MOCK_TRANSCRIPTS.get(language_code, _MOCK_FALLBACK)

    return {
        "transcript": transcript,
        "language_code": language_code,
        "latency_ms": 0.0,
        "provider": "mock",
        "mock": True,
    }


# ============================================================
# PUBLIC API
# ============================================================

def transcribe_audio(
    audio_bytes: bytes,
    language_code: str = DEFAULT_LANGUAGE,
) -> dict:
    """
    Transcribe audio bytes to text.

    If SARVAM_API_KEY is set, uses Sarvam STT.
    Otherwise uses mock mode (development only).

    Args:
        audio_bytes: WAV audio bytes
        language_code: BCP-47 language code (default: "hi-IN")

    Returns:
        {
            "transcript": str,          # The transcribed text
            "language_code": str,       # Language used
            "latency_ms": float,        # STT latency (excluded from RAG target)
            "provider": str,            # "sarvam" | "mock"
        }
    """

    if not audio_bytes:
        return {
            "transcript": "",
            "language_code": language_code,
            "latency_ms": 0.0,
            "provider": "none",
            "error": "empty_audio",
        }

    if language_code not in SUPPORTED_LANGUAGES:
        logger.warning(
            f"Language code '{language_code}' not in supported list. "
            f"Attempting anyway."
        )

    if SARVAM_API_KEY:
        return _transcribe_sarvam(audio_bytes, language_code)
    else:
        return _transcribe_mock(audio_bytes, language_code)


def stt_status() -> dict:
    """
    Return the current STT configuration status.
    Used by the /health endpoint.
    """

    has_key = bool(SARVAM_API_KEY)

    return {
        "provider": "sarvam" if has_key else "mock",
        "configured": has_key,
        "language_default": DEFAULT_LANGUAGE,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "note": (
            "Set SARVAM_API_KEY environment variable to enable real STT."
            if not has_key
            else "Sarvam STT API key is configured."
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STT MODULE STATUS")
    print("=" * 60)

    status = stt_status()

    print(f"Provider    : {status['provider']}")
    print(f"Configured  : {status['configured']}")
    print(f"Default lang: {status['language_default']}")
    print(f"Note        : {status['note']}")

    if not SARVAM_API_KEY:
        print()
        print("Testing mock STT...")
        result = transcribe_audio(b"fake_audio_bytes")
        print(f"Transcript  : {result['transcript']}")
        print(f"Provider    : {result['provider']}")
        print(f"Mock        : {result.get('mock', False)}")
