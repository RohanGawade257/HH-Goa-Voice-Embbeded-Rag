"""
HH Goa 2026 — Voice RAG
Model Harness: retry engine + structured output validator.

This module provides the shared harness used by both GeminiAnswerGenerator
and QwenAnswerGenerator.  It has no imports beyond the standard library.

Exports
-------
RetryConfig          — frozen dataclass controlling retry behaviour
is_retryable         — classifies a result dict as retryable or not
with_retry           — wraps a callable with exponential-backoff retry
SCRIPT_RANGES        — Unicode ordinal ranges per language code
validate_output      — checks answer length and script plausibility

Retry policy (harness-internal — callers do NOT configure this)
---------------------------------------------------------------
  TIMEOUT             → retry
  HTTP_ERROR + 429    → retry
  everything else     → no retry   (HTTP 4xx/5xx, EXCEPTION, SUCCESS)

Output validation
-----------------
  len < 5 chars       → answer_too_short
  len > 500 chars     → answer_too_long
  no expected-script  → wrong_script   (script PRESENCE check, not purity)
  otherwise           → ok

The script check is deliberately weak: it requires at least one character
from the language's Unicode block.  Mixed-script answers ("यह answer is
correct" for language="hi") pass because Devanagari chars are present.
Languages without a dedicated non-Latin block (en, unknown) skip the check.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

_logger = logging.getLogger(__name__)


# ============================================================
# RETRY CONFIG
# ============================================================

@dataclass(frozen=True)
class RetryConfig:
    """Parameters controlling the retry loop in with_retry().

    max_retries  — number of additional attempts after the first (so total
                   attempts = max_retries + 1).
    base_delay_s — sleep duration before the 2nd attempt; doubles each time.
    max_wall_s   — post-attempt wall-clock guard: if this much time has already
                   elapsed after an attempt, no further retry is started.
                   This is NOT an in-flight abort — fn() runs to completion.
    """

    max_retries: int = 2
    base_delay_s: float = 0.2   # 200 ms before attempt 2
    max_wall_s: float = 8.0     # give up after 8 s total elapsed


# ============================================================
# RETRY CLASSIFIER
# ============================================================

def is_retryable(result: dict[str, Any]) -> bool:
    """Return True if a generator result dict represents a transient failure.

    Only two conditions are retried:
      - TIMEOUT  (network / model overload — transient)
      - HTTP 429 (rate limit — wait and retry)

    Everything else — including HTTP 500/502/503 and EXCEPTION — is not
    retried because it is either permanent, a code error, or outside the
    harness's scope to fix by waiting.
    """
    status = result.get("status", "")
    if status == "TIMEOUT":
        return True
    if status == "HTTP_ERROR" and result.get("http_status") == 429:
        return True
    return False


# ============================================================
# RETRY ENGINE
# ============================================================

def with_retry(
    fn: Callable[[], dict[str, Any]],
    retry_config: RetryConfig = RetryConfig(),
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Call fn() and retry on transient failures with exponential backoff.

    Parameters
    ----------
    fn           — zero-argument callable returning a generator result dict.
    retry_config — controls max_retries, base_delay_s, max_wall_s.
    logger       — optional logger for retry INFO messages; falls back to
                   the module-level logger if None.

    Returns
    -------
    The last result dict from fn(), with two extra keys injected:
      attempt_count   (int)        — how many times fn() was called (1–3)
      retry_delays_ms (list[float])— sleep durations between attempts (ms)

    Notes
    -----
    max_wall_s is a post-attempt guard: after fn() returns, if the elapsed
    wall time exceeds max_wall_s, no further retry is started.  It does NOT
    abort an in-flight fn() call — that would require async cancellation or
    a separate thread.
    """
    log = logger or _logger
    retry_delays_ms: list[float] = []
    result: dict[str, Any] = {}
    wall_start = time.monotonic()

    for attempt in range(retry_config.max_retries + 1):
        result = fn()

        # Not retryable — done immediately
        if not is_retryable(result):
            break

        # Last allowed attempt — done
        if attempt >= retry_config.max_retries:
            break

        # Post-attempt wall-clock guard
        elapsed = time.monotonic() - wall_start
        if elapsed >= retry_config.max_wall_s:
            log.info(
                "LLM harness: wall-clock cap reached (%.2f s >= %.2f s), "
                "skipping further retries",
                elapsed,
                retry_config.max_wall_s,
            )
            break

        # Schedule next retry
        delay_s = retry_config.base_delay_s * (2 ** attempt)
        delay_ms = delay_s * 1000
        retry_delays_ms.append(round(delay_ms, 2))

        reason = result.get("reason", result.get("status", "unknown"))
        log.info(
            "LLM harness: attempt %d failed (%s), retrying in %.0f ms",
            attempt + 1,
            reason,
            delay_ms,
        )
        time.sleep(delay_s)

    result["attempt_count"] = len(retry_delays_ms) + 1
    result["retry_delays_ms"] = retry_delays_ms
    return result


# ============================================================
# SCRIPT RANGES
# ============================================================

# Maps the short language code (as used in LANGUAGE_NAMES in llm.py) to a
# (lo, hi) tuple of Unicode ordinal boundaries for the primary script used
# by that language.  None means "no dedicated non-Latin block — skip check."
#
# Boundaries mirror app/api.py::SCRIPT_LANGUAGE_RANGES for consistency.

SCRIPT_RANGES: dict[str, tuple[int, int] | None] = {
    # Devanagari  U+0900–U+097F
    "hi": (0x0900, 0x097F),
    "mr": (0x0900, 0x097F),
    "ne": (0x0900, 0x097F),
    "sa": (0x0900, 0x097F),
    # Bengali     U+0980–U+09FF
    "bn": (0x0980, 0x09FF),
    "as": (0x0980, 0x09FF),
    # Gujarati    U+0A80–U+0AFF
    "gu": (0x0A80, 0x0AFF),
    # Gurmukhi / Punjabi  U+0A00–U+0A7F
    "pa": (0x0A00, 0x0A7F),
    # Kannada     U+0C80–U+0CFF
    "kn": (0x0C80, 0x0CFF),
    # Malayalam   U+0D00–U+0D7F
    "ml": (0x0D00, 0x0D7F),
    # Odia        U+0B00–U+0B7F
    "or": (0x0B00, 0x0B7F),
    # Tamil       U+0B80–U+0BFF
    "ta": (0x0B80, 0x0BFF),
    # Arabic block U+0600–U+06FF (Urdu plausibility — script presence only)
    "ur": (0x0600, 0x06FF),
    # No script check for Latin-based or unknown languages
    "en": None,
}


# ============================================================
# OUTPUT VALIDATOR
# ============================================================

def validate_output(answer: str, language: str) -> tuple[bool, str]:
    """Check that an LLM answer meets minimum quality requirements.

    Checks performed (in order):
      1. Minimum length  — answer must have at least 5 non-whitespace characters
      2. Maximum length  — answer must not exceed 500 characters
      3. Script presence — if the language has an expected Unicode block, at
                          least one character must fall within it

    The script check is a PRESENCE test, not a purity test.  Mixed-script
    answers (e.g. "यह answer is correct" for language="hi") pass because at
    least one Devanagari character is found.  The check is deliberately weak:
    it catches "model answered in entirely the wrong language" without
    penalising natural code-switching or proper nouns in Latin script.

    Returns
    -------
    (True, "ok")                 — answer is acceptable
    (False, "answer_too_short")  — stripped length < 5
    (False, "answer_too_long")   — stripped length > 500
    (False, "wrong_script")      — no expected-script chars found
    """
    text = (answer or "").strip()

    if len(text) < 5:
        return False, "answer_too_short"

    if len(text) > 500:
        return False, "answer_too_long"

    script_range = SCRIPT_RANGES.get(language)
    if script_range is not None:
        lo, hi = script_range
        if not any(lo <= ord(c) <= hi for c in text):
            return False, "wrong_script"

    return True, "ok"
