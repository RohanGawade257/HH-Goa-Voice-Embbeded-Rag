# Model Harness — Implementation Plan (v2)

## Top-Level Overview

**Goal:** Satisfy the hackathon "harness your model" requirement by adding two
missing capabilities to the LLM generation layer:

1. **Retry with exponential backoff** — on transient failures only (TIMEOUT,
   HTTP 429), up to 2 retries, 200 ms → 400 ms delay, 8 s hard post-attempt cap.
2. **Structured output validation** — after every successful LLM call, the
   returned answer text is checked for: minimum length, maximum length, and
   script plausibility (the answer must contain at least some characters from
   the expected language's Unicode block).

**CRITICAL LATENCY REQUIREMENT:**
The harness must never be placed on the critical path of the <200 ms
direct-answer response. The extractive/direct answer must remain immediately
available after retrieval/compression. LLM generation and any retry/backoff
execute only on the existing asynchronous/enhancement path (SSE stream step 3,
voice endpoint after direct answer). A failed or delayed LLM call must never
delay, replace, or block the direct answer shown within the latency budget.

```
retrieve → compress ──→ DIRECT ANSWER → USER  (<200 ms, always)
                    └──→ LLM harness (async, best-effort)
                           ├── success → AI answer
                           ├── timeout → retry (200ms delay, attempt 2)
                           └── 429     → retry (200ms delay, attempt 2)
```

**Non-goals:**
- Tool calls / agent routing
- Changes to the extractive (non-LLM) answer path
- Changes to the RAG retrieval, reranking, or chunking pipeline
- Adding latency on the happy path

**Scope:** 6 files change:
- `app/generation/harness.py` — NEW shared module
- `app/generation/gemini.py` — wrap with harness
- `app/generation/llm.py` — wrap with harness
- `tests/test_harness.py` — NEW unit tests
- `tests/test_gemini_generation.py` — +1 assertion
- `tests/test_llm_generation.py` — +1 assertion

---

## Retry decision table

| LLM result status | http_status | Retry? | Reasoning |
|---|---|---|---|
| `TIMEOUT` | — | YES | Transient — network/model overload |
| `HTTP_ERROR` | 429 | YES | Rate limit — wait and retry |
| `HTTP_ERROR` | 400/401/403/500/502/503 | NO | Permanent or server fault |
| `EXCEPTION` | — | NO | SDK/code error, not transient |
| `SUCCESS` + invalid output | — | NO | Validation fail ≠ transient error |

---

## Sub-Tasks

---

### Sub-Task 1 — Create `app/generation/harness.py`

**Status:** [x] done

**Intent:**
Centralise all harness logic in one small module so both `GeminiAnswerGenerator`
and `QwenAnswerGenerator` share exactly the same retry engine and output
validator. No imports beyond stdlib (`time`, `dataclasses`, `typing`, `logging`).

**Expected Outcomes:**
- `app/generation/harness.py` exists and exports:
  - `RetryConfig` dataclass
  - `is_retryable(result: dict) -> bool` — dedicated classifier function
  - `with_retry(fn, retry_config, logger=None)` — retry engine
  - `SCRIPT_RANGES` dict — Unicode block ranges per language
  - `validate_output(answer, language) -> tuple[bool, str]`

**Todo List:**

1. **`RetryConfig` dataclass** (frozen):
   - `max_retries: int = 2`
   - `base_delay_s: float = 0.2`  (200 ms first sleep)
   - `max_wall_s: float = 8.0`    (post-attempt wall cap — prevents a second retry if time already exceeded)

2. **`is_retryable(result: dict) -> bool`** — a small dedicated function:
   - Return `True` if `result.get("status") == "TIMEOUT"`
   - Return `True` if `result.get("status") == "HTTP_ERROR"` AND
     `result.get("http_status") == 429`
   - Return `False` for everything else
   - Retryable conditions are harness policy — callers do NOT pass them

3. **`with_retry(fn, retry_config=RetryConfig(), logger=None)`**:
   - Record wall-clock start with `time.monotonic()`
   - Loop `retry_config.max_retries + 1` times (attempts 0, 1, 2)
   - Each iteration: call `fn()`, record result
   - Check `is_retryable(result)` — if not retryable, break immediately
   - Check remaining wall time: `elapsed = time.monotonic() - start_time`.
     If `elapsed >= retry_config.max_wall_s`, break (no more retries — this is
     a post-attempt guard, not an in-flight abort; `fn()` may already have
     completed by the time we check)
   - If this is the last allowed attempt (index == max_retries), break
   - Compute `delay = retry_config.base_delay_s * (2 ** attempt_index)`
   - Log at INFO: `"LLM attempt {n} failed ({reason}), retrying in {delay*1000:.0f}ms"`
   - `time.sleep(delay)`, append delay*1000 to `retry_delays_ms` list, continue
   - After loop: inject `attempt_count` (int) and `retry_delays_ms` (list[float])
     into the result dict and return it

4. **`SCRIPT_RANGES` dict** — maps short language code to `(lo, hi)` Unicode
   ordinal range. Languages with no dedicated non-Latin block map to `None`
   (no script check performed):
   - `"hi"`, `"mr"`, `"ne"`, `"sa"` → `(0x0900, 0x097F)` Devanagari
   - `"bn"`, `"as"` → `(0x0980, 0x09FF)` Bengali
   - `"gu"` → `(0x0A80, 0x0AFF)` Gujarati
   - `"pa"` → `(0x0A00, 0x0A7F)` Gurmukhi
   - `"kn"` → `(0x0C80, 0x0CFF)` Kannada
   - `"ml"` → `(0x0D00, 0x0D7F)` Malayalam
   - `"or"` → `(0x0B00, 0x0B7F)` Odia
   - `"ta"` → `(0x0B80, 0x0BFF)` Tamil
   - `"ur"` → `(0x0600, 0x06FF)` Arabic block (Urdu plausibility — weak but
     sufficient for script presence check; described as "script plausibility",
     not language validation)
   - `"en"`, unknown codes → `None` (no script check)

5. **`validate_output(answer: str, language: str) -> tuple[bool, str]`**:
   - `text = (answer or "").strip()`
   - `len(text) < 5` → return `(False, "answer_too_short")`
   - `len(text) > 500` → return `(False, "answer_too_long")`
   - Look up `SCRIPT_RANGES.get(language)` — if a range `(lo, hi)` is found,
     check `any(lo <= ord(c) <= hi for c in text)`. If no character in range →
     return `(False, "wrong_script")`
   - Note: script presence check, NOT script purity. A mixed-script answer like
     `"यह answer is correct"` with `language="hi"` must PASS (Devanagari chars
     present, Latin code-switching allowed).
   - Otherwise return `(True, "ok")`

**Relevant Context:**
- `app/generation/llm.py::LANGUAGE_NAMES` — the 13 supported language codes
- `app/api.py::SCRIPT_LANGUAGE_RANGES` — same Unicode block boundaries already
  used for voice language detection; use identical boundaries

---

### Sub-Task 2 — Update `app/generation/gemini.py`

**Status:** [x] done

**Intent:**
Wrap the Gemini streaming call in the harness without changing the public
`generate()` interface or adding latency on the happy path.

**Expected Outcomes:**
- `GeminiAnswerGenerator.generate()` retries on TIMEOUT or HTTP 429 (up to 2 retries)
- On success, `validate_output()` is called; invalid answers return
  `status=EXCEPTION, reason="invalid_output"` — the invalid text is NOT returned
- Return dict gains `attempt_count` (int) and `retry_delays_ms` (list[float])
- All existing return dict keys preserved; method signature unchanged
- The LLM path is NOT on the critical path of the direct answer (no architectural
  change needed — already true by design in SSE + voice endpoints)

**Todo List:**

1. Import `RetryConfig`, `with_retry`, `validate_output` from
   `app.generation.harness` at the top of `gemini.py`

2. Extract the entire try/except streaming block from `generate()` into a new
   private method `_call_once(self, query, language, context, max_tokens)`:
   - Same body as the current try/except in `generate()`
   - Returns the same dict shapes (SUCCESS, EXCEPTION, TIMEOUT, HTTP_ERROR)
   - Does NOT call `with_retry` — single raw attempt only

3. In `generate()`, replace the extracted block with:
   ```
   fn = lambda: self._call_once(query, language, context, max_tokens)
   result = with_retry(fn, RetryConfig(), logger)
   ```

4. After `with_retry` returns, if `result["status"] == SUCCESS`:
   - Call `is_valid, val_reason = validate_output(result["answer"], language)`
   - If `not is_valid`: return a new dict:
     `status=EXCEPTION, reason="invalid_output", error=val_reason`
     plus `ttft_ms`, `llm_ms`, `latency_ms`, `attempt_count`, `retry_delays_ms`
     carried from result — do NOT include the rejected answer text

5. All pre-call guards (backend disabled, empty context, API key missing) stay
   at the top of `generate()` BEFORE `with_retry` — they return immediately, no
   retry overhead

6. Timing fields (`ttft_ms`, `llm_ms`, `latency_ms`) in `_call_once` continue
   to reflect the wall-clock time of that single attempt

**Relevant Context:**
- `app/generation/gemini.py` lines 148–324 — current `generate()` body to extract
- `app/generation/harness.py` — created in Sub-Task 1

---

### Sub-Task 3 — Update `app/generation/llm.py`

**Status:** [x] done

**Intent:**
Apply the identical harness pattern to `QwenAnswerGenerator`. The Qwen path uses
`httpx` directly; the error dict already includes `"http_status"` which the
harness uses to detect 429s.

**Expected Outcomes:**
- Same as Sub-Task 2 but for Qwen
- `attempt_count` and `retry_delays_ms` added to return dict
- Existing keys preserved; method signature unchanged

**Todo List:**

1. Import `RetryConfig`, `with_retry`, `validate_output` from
   `app.generation.harness`

2. Extract the try/except httpx block into `_call_once(self, query, language,
   context, max_tokens)`. Verify the HTTP_ERROR dict includes
   `"http_status": response.status_code` (already present at line ~239) so that
   `is_retryable()` can detect 429.

3. In `generate()`, replace the block with `with_retry` call, same pattern as
   Gemini Sub-Task 2 step 3.

4. Post-success `validate_output()` call — identical to Gemini step 4.

5. Pre-call guards (backend disabled, empty context, API key missing) stay at
   top before `with_retry`.

**Relevant Context:**
- `app/generation/llm.py` lines 126–270 — current `generate()` body
- `app/generation/harness.py` — created in Sub-Task 1
- Qwen HTTP_ERROR dict at line ~239: `"http_status": response.status_code`

---

### Sub-Task 4 — Tests

**Status:** [x] done

**Intent:**
Full unit coverage of the harness module in isolation, plus minimal regression
check on the generators. No live network calls anywhere.

**Expected Outcomes:**
- `tests/test_harness.py` all tests pass
- Existing generator tests continue to pass
- `attempt_count == 1` asserted on first-try success in both generator test files

**Todo List:**

1. **`tests/test_harness.py` — `validate_output` tests:**
   - Empty string → `(False, "answer_too_short")`
   - 4-char string → `(False, "answer_too_short")`
   - **5-char string → `(True, "ok")`** (boundary: 5 is valid)
   - **500-char string → `(True, "ok")`** (boundary: 500 is valid)
   - 501-char string → `(False, "answer_too_long")`
   - Valid Hindi answer containing Devanagari chars, `language="hi"` → `(True, "ok")`
   - Latin-only string, `language="hi"` → `(False, "wrong_script")`
   - **Mixed-script answer `"यह answer is correct"`, `language="hi"` → `(True, "ok")`**
     (script PRESENCE not purity — Devanagari chars present, code-switching allowed)
   - Valid Tamil answer, `language="ta"` → `(True, "ok")`
   - Valid English answer, `language="en"` → `(True, "ok")` (no script check for Latin)
   - Unknown language code → `(True, "ok")` (no script check)

2. **`tests/test_harness.py` — `with_retry` tests (all using `unittest.mock`):**
   - `fn` returns `{"status": "SUCCESS", ...}` on first call →
     `attempt_count == 1`, `retry_delays_ms == []`
   - `fn` returns TIMEOUT on call 1, SUCCESS on call 2 →
     `attempt_count == 2`, `len(retry_delays_ms) == 1`
   - `fn` returns HTTP 429 on calls 1–2, SUCCESS on call 3 →
     `attempt_count == 3`, `len(retry_delays_ms) == 2`
   - `fn` returns HTTP 500 on call 1 →
     `attempt_count == 1`, no retry (HTTP 500 not retryable)
   - `fn` always returns TIMEOUT →
     `attempt_count == 3`, result status is still TIMEOUT (last result returned)
   - **Wall-clock cap test:** configure `RetryConfig(max_wall_s=0.0)`, mock `fn`
     returns TIMEOUT. Because `elapsed >= max_wall_s` immediately after attempt 1,
     no retry occurs → `attempt_count == 1`. (This correctly tests the
     post-attempt guard without requiring in-flight abort.)

3. **`tests/test_gemini_generation.py`:** Add assertion on the existing mocked
   success test that `result["attempt_count"] == 1`.

4. **`tests/test_llm_generation.py`:** Same assertion for Qwen.

**Relevant Context:**
- Use `unittest.mock.patch` or `MagicMock` to mock `_call_once` — no live API
- Follow existing test file patterns (both files use `unittest.TestCase`)

---

## Implementation Notes

### Latency impact
- **Happy path:** `validate_output()` scans at most 500 characters — negligible
  relative to LLM/network latency. `with_retry` adds only a single wrapper call.
  P50 latency of 38 ms for the direct-answer path is completely unaffected
  because the harness only wraps the LLM generation step, which is on the
  async/enhancement path.
- **Retry path:** Only fires after the LLM already returned a failure. The 200/400 ms
  sleeps only count against LLM response time, never against the direct answer.

### Files changed summary

| File | Change type |
|---|---|
| `app/generation/harness.py` | NEW |
| `app/generation/gemini.py` | MODIFIED — extract `_call_once`, add harness |
| `app/generation/llm.py` | MODIFIED — extract `_call_once`, add harness |
| `tests/test_harness.py` | NEW |
| `tests/test_gemini_generation.py` | MINOR — +1 assertion |
| `tests/test_llm_generation.py` | MINOR — +1 assertion |
| Everything else | UNCHANGED |
