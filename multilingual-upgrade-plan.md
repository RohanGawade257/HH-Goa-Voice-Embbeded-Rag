# Phase 3 — Multilingual Integration, Hardening & Validation
## HH-Goa-Rag Corrected Plan

---

## Critical Engineering Requirement

> After STT completes, the **first useful grounded answer must reach the user in under 200 ms** in the deployed environment. STT latency is excluded.

The direct (extractive) answer is the **performance-critical path**. The Gemini AI answer is a secondary enhancement that must never block or replace the direct answer.

The deployed dual-answer SSE contract is:

```
STT complete → START
  embed → Qdrant → rerank → compress → direct_answer emitted  ← must be < 200 ms
  (same compressed context) → Gemini → llm_answer emitted     ← separate, no target
END
```

`time_to_direct_answer` and `time_to_llm_answer` must always be reported **separately**.

---

## What Already Exists (Do Not Touch)

| Component | Status |
|---|---|
| 13-language Qdrant corpus (6,046 chunks) | ✅ Done |
| `paraphrase-multilingual-MiniLM-L12-v2` embedder | ✅ Done |
| Hybrid reranker (vector 0.70 + lexical 0.20 + phrase 0.10) | ✅ Done |
| Context compressor | ✅ Done |
| Dual-answer pipeline (`process_dual()` + `_retrieve_and_compress()`) | ✅ Done |
| `/query/stream` SSE endpoint | ✅ Done |
| `GeminiAnswerGenerator` (Gemini 2.5 Flash-Lite) | ✅ Done |
| SSE client in `frontend/lib/api.ts` | ✅ Done |
| `AnswerCard` with direct + AI answer panels | ✅ Done |

---

## What Is Broken or Missing

| Gap | Severity |
|---|---|
| Language always hardcoded to `hi-IN` in frontend | High |
| No language selector in UI | High |
| STT ≠ RAG language capability (not separated in UI) | High |
| No integration test for `/query/stream` SSE path | High |
| SSE `event_generator` has no top-level exception guard | Medium |
| STT `RuntimeError` propagates as raw 500 | Medium |
| `onError` in SSE client never calls `onDone` → stuck loading state | Medium |
| Hero/UI still says "Hindi" everywhere | Medium |
| `TOP_K=10` — may limit recall but change must be benchmark-driven | Medium |
| Language-affinity boost unvalidated — must not be added blindly | Medium |
| No benchmark separating `time_to_direct_answer` vs `time_to_llm_answer` | High |
| Old 38 ms Hindi baseline incorrectly cited as current performance | Medium |

---

## Non-Goals (Strictly Forbidden)

- Switching back to Qwen or adding another LLM
- Redesigning the dual-answer architecture
- Re-running retrieval/embedding/compression for Gemini
- Blindly increasing `QDRANT_TOP_K` without measurement
- Blindly adding language-affinity weights without A/B test
- Changing the visual design (colors, fonts, layout, animations)
- Redesigning `AnswerCard`, `Hero` layout structure
- Changing the embedding model
- Adding new npm or Python packages beyond what is necessary

---

## Sub-Tasks

---

### Sub-Task 1 — Retrieval Benchmark: Measure Before Changing

**Status:** [x] done — Recall@3=42.3% at all K values (10/20/30); TOP_K stays 10; language-affinity not adopted (delta=0%). Direct-answer P95=57ms at K=10. Report saved to `data/processed/multilingual/retrieval_benchmark.json`.

**Intent**
`QDRANT_TOP_K=10` may be limiting recall, but that is unconfirmed. Before changing any production config, measure actual retrieval quality at K=10, K=20, K=30, and also test a language-affinity-boosted reranker. Only adopt the configuration that measurements justify. All changes must also be evaluated against `time_to_direct_answer` latency, not just recall.

**Expected Outcomes**
- A benchmark script `benchmarks/benchmark_retrieval.py` runs offline (no API key needed)
- It measures Recall@K and mean reciprocal rank for K ∈ {10, 20, 30}
- It measures reranking latency and `time_to_direct_answer` for each K
- It tests language-affinity scoring separately (A vs B) across 13 languages
- Results printed as a comparison table and saved to `data/processed/multilingual/retrieval_benchmark.json`
- `QDRANT_TOP_K` in `app/config.py` is **only changed** if K=20 or K=30 shows a meaningful recall gain with an acceptable latency cost (≤10 ms additional)
- Language-affinity weight is **only added to `rerank()`** if the A/B test shows improvement
- If no change is justified, `QDRANT_TOP_K` stays at 10 and reranker weights stay unchanged
- Document findings in `data/processed/multilingual/retrieval_benchmark.json`

**Todo List**
1. Create `benchmarks/benchmark_retrieval.py`:
   - Load `RAGEngine` with `ANSWER_BACKEND=extractive` (no LLM needed)
   - Define a representative query set: 2 queries per language × 13 languages = 26 queries (use known queries from the dataset)
   - For each K in (10, 20, 30):
     - Run all 26 queries through `_retrieve_and_compress()`
     - Record: `qdrant_ms`, `rerank_ms`, `compression_ms`, `time_to_direct_answer`, retrieved chunk IDs
     - Compute Recall@3 (fraction of queries where at least one relevant chunk is in top 3) using known passage IDs from query metadata
     - Compute Recall@K (same, within the Qdrant pool before reranking)
   - Separately, run queries with a temporary language-affinity-boosted reranker (inject a `language_bonus` into `score_document()` at 0.05 weight) and compare Recall@3 vs baseline
   - Print comparison table; save to JSON
2. Run the benchmark and inspect results
3. Apply `QDRANT_TOP_K` change in `app/config.py` **only if justified**
4. Apply language-affinity weight to `rerank()` in `app/pipeline.py` **only if justified**
5. Record the decision and rationale in the JSON report

**Relevant Context**
- `app/config.py` line 41: `QDRANT_TOP_K = get_int("QDRANT_TOP_K", 10)`
- `app/pipeline.py` lines 56–58: `VECTOR_WEIGHT=0.70, LEXICAL_WEIGHT=0.20, PHRASE_WEIGHT=0.10`
- `app/pipeline.py` line 301: `def rerank(query, hits)` — no language param today
- `app/pipeline.py` `_retrieve_and_compress()` — reusable for benchmarking
- Pipeline current Recall@10 ≈ 18.8% (weak — but root cause unconfirmed)

---

### Sub-Task 2 — Language Contract: Explicit Capability Table

**Status:** [ ] pending

**Intent**
Define and encode the exact language capabilities of the system. There are two independent capabilities — RAG support and STT support — and they do not overlap perfectly. The frontend must reflect the true state, not an idealized one. Telugu is STT-supported but NOT in the RAG corpus; Nepali/Sanskrit/Urdu are in the RAG corpus but NOT STT-supported.

**Expected Outcomes**
- `frontend/lib/api.ts` exports a `LANGUAGES` constant covering all languages the UI may display
- Each entry has: `code` (BCP-47 e.g. `"hi-IN"`), `name` (English), `nativeName` (script), `ragSupported: boolean`, `sttSupported: boolean`
- Telugu (`te-IN`) has `ragSupported: false, sttSupported: true` — not shown in language selector
- Nepali (`ne-IN`), Sanskrit (`sa-IN`), Urdu (`ur-IN`) have `ragSupported: true, sttSupported: false`
- The `queryStream()` and `queryVoice()` functions accept the selected language code
- `app/voice/stt.py` `_transcribe_mock()` updated to return a language-appropriate test query for each supported language (so dev testing is language-aware)
- No backend API changes needed for language routing — it already flows through the pipeline

**Todo List**
1. In `frontend/lib/api.ts`:
   - Add `LANGUAGES` array (14 entries: 13 corpus + English): `{ code, name, nativeName, ragSupported, sttSupported }`
   - Mark Telugu `ragSupported: false` (not in corpus)
   - Mark Nepali, Sanskrit, Urdu `sttSupported: false`
   - English: `ragSupported: false, sttSupported: true` (useful for voice input only)
   - `queryStream` and `queryVoice` signatures already accept `language` — no change needed
2. In `app/voice/stt.py`:
   - Add a `MOCK_TRANSCRIPTS` dict: language_code → representative test query in that language
   - Update `_transcribe_mock()` to look up the language code and return the appropriate query
   - Keep Hindi fallback for any unmapped code
3. Verify `app/api.py` `/voice` endpoint already passes `language` to `transcribe_audio()` — it does (line 409) — no change needed
4. Verify `app/pipeline.py` `normalize_language_code()` handles all 13 language codes correctly by stripping the `-IN` suffix — it already does — no change needed

**Relevant Context**
- `app/voice/stt.py` lines 32–44: 11 STT-supported language codes
- `app/pipeline.py` lines 60–74: 13 RAG corpus language codes (`as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, ur`)
- `app/voice/stt.py` line 144: mock returns hardcoded Hindi query only
- `frontend/lib/api.ts` line 139: `queryVoice(audioBlob, language = "hi-IN")` — signature already accepts language
- `frontend/lib/api.ts` line 163: `queryStream(..., language = "hi-IN")` — already accepts language

---

### Sub-Task 3 — Frontend Language Selector

**Status:** [ ] pending

**Intent**
The frontend currently hardcodes `"hi-IN"` for all queries and has no UI for language selection. This sub-task wires a compact language selector into `QuerySection` and propagates the selection to both the stream and voice endpoints. The design must not change — same colors, same layout, same brutalist style.

**Expected Outcomes**
- `QuerySection.tsx` renders a horizontal scrollable pill row of RAG-supported languages above the input area
- Selected language pill is highlighted with `var(--foreground)` background
- Non-selected pills use border-only style consistent with existing mode toggle buttons
- `queryStream()` is called with the selected language code
- `queryVoice()` is called with the selected language code
- When a language with `sttSupported: false` is selected, voice mode shows an inline badge "Voice not available for this language" and the record button is disabled
- Subtitle text updated: "Hindi or English — voice or text" → "13 Indian languages — voice or text"
- Placeholder queries in the textarea rotate across at least 5 different languages (not all Hindi)
- No new npm packages

**Todo List**
1. In `frontend/components/QuerySection.tsx`:
   - Import `LANGUAGES` from `@/lib/api`
   - Add `selectedLang` state (default: first entry with `ragSupported: true, sttSupported: true`)
   - Render pill row: `LANGUAGES.filter(l => l.ragSupported)` — one pill per language
   - Pass `selectedLang.code` to `queryStream()` and `queryVoice()`
   - In voice mode: if `!selectedLang.sttSupported`, show a warning chip and disable record button
   - Update subtitle text
   - Update `PLACEHOLDER_QUERIES` to cover 5+ languages with appropriate script
2. Accept optional `defaultQuery` and `defaultLanguage` props (both optional, for Hero click-through)
3. Apply `defaultQuery`/`defaultLanguage` to state on mount using a `useEffect`

**Relevant Context**
- `frontend/components/QuerySection.tsx` line 19–24: current Hindi-only placeholders
- `frontend/components/QuerySection.tsx` line 229–231: subtitle "Hindi or English"
- `frontend/components/QuerySection.tsx` line 87: `queryStream(q, { ... })` — missing language arg
- `frontend/components/QuerySection.tsx` line 140: `queryVoice(blob)` — missing language arg
- Existing mode toggle at lines 236–256 shows the correct pill button style to reuse

---

### Sub-Task 4 — Frontend Multilingual Branding

**Status:** [ ] pending

**Intent**
The Hero and surrounding copy still say "Hindi" everywhere. Update text content to reflect the 13-language system. Do NOT change visual design, layout, colors, fonts, or animations. Also update the Hero example queries to show multiple languages, with click-through that pre-fills both the query and the language selector in `QuerySection`.

**Expected Outcomes**
- `Hero.tsx`: tag "Voice · RAG · Hindi AI" → "Voice · RAG · 13 Languages"
- `Hero.tsx`: heading "Hindi / Retrieval / Intelligence." → "Multilingual / Retrieval / Intelligence."
- `Hero.tsx`: Devanagari subtitle updated to reflect multilingual theme (can keep Devanagari script but broaden meaning)
- `Hero.tsx`: English description paragraph updated — remove "Hindi" references
- `Hero.tsx`: stats row updated — "10K+" → "6K+ Chunks", add "13 Languages" stat
- `Hero.tsx`: example queries span 5 languages (Hindi, Bengali, Tamil, Marathi, Gujarati)
- Example query click pre-fills QuerySection with query text AND correct language
- `app/page.tsx`: adds `onExampleSelect(query: string, language: string)` callback threading Hero → QuerySection
- `QuerySection.tsx`: accepts `defaultQuery` / `defaultLanguage` props (from Sub-Task 3)
- All visual design tokens unchanged

**Todo List**
1. Update `frontend/components/Hero.tsx`:
   - Change tag, heading, Devanagari subtitle, English description, stats
   - Update `EXAMPLE_QUERIES` to `EXAMPLE_QUERIES: { query: string; language: string; label: string }[]` — 5 entries with different languages
   - `HeroProps` gains `onExampleSelect?: (query: string, language: string) => void`
   - Example query buttons call `onExampleSelect(q.query, q.language)` then `onQueryClick()`
2. Update `frontend/app/page.tsx`:
   - Add `exampleQuery` / `exampleLanguage` state
   - Thread `onExampleSelect` from Hero → state update
   - Pass `defaultQuery={exampleQuery}` and `defaultLanguage={exampleLanguage}` to `<QuerySection />`
3. `frontend/components/QuerySection.tsx` already handles these props (Sub-Task 3)

**Relevant Context**
- `frontend/components/Hero.tsx` line 64: "Voice · RAG · Hindi AI"
- `frontend/components/Hero.tsx` lines 75–80: "Hindi / Retrieval / Intelligence."
- `frontend/components/Hero.tsx` lines 93–97: Hindi-only Devanagari text
- `frontend/components/Hero.tsx` lines 127–131: stats (42ms, 79.1%, 10K+)
- `frontend/app/page.tsx` line 28: `<QuerySection />` — no props today

---

### Sub-Task 5 — SSE Integration Test

**Status:** [ ] pending

**Intent**
No integration test exists for the `/query/stream` SSE endpoint — the most important new endpoint. This sub-task adds a comprehensive test covering the dual-answer contract, Gemini failure handling, and stream termination. All tests use `ANSWER_BACKEND=extractive` so no live API key is required.

**Expected Outcomes**
- `tests/test_integration_stream.py` exists and all tests pass with `pytest`
- Tests use `TestClient` with `with client.stream(...)` pattern to consume SSE
- `direct_answer` always arrives before `llm_answer` in the event sequence
- Gemini failure (`llm_unavailable`) does NOT cause `direct_answer` to be absent or empty
- `direct_answer` is present and non-empty even when `llm_answer` contains an error
- No second retrieval/embedding call occurs for the LLM path (verified by mock call count)
- `sources` event contains a list (may be empty)
- `timing` event contains all required fields: `embedding_ms`, `qdrant_ms`, `rerank_ms`, `compression_ms`, `time_to_direct_ms`, `time_to_llm_ms`, `total_ms`
- Empty query → `error` event then `done`
- Stream always terminates with `done` event (even on error paths)
- All tests run without Qdrant file-lock issues (mock the engine)

**Todo List**
1. Create `tests/test_integration_stream.py`:
   - Set `os.environ["ANSWER_BACKEND"] = "extractive"` at top (before app import)
   - Patch `app.api.get_engine()` to return a mock engine that wraps `process_dual()` with a call-count tracker
   - Use `fastapi.testclient.TestClient` with `with client.stream("GET", "/query/stream?q=..."):`
   - Write SSE frame parser: split on `\n\n`, extract `event:` and `data:` lines
2. Test cases:
   - `test_direct_answer_arrives_first`: parse all events, assert `direct_answer` index < `llm_answer` index
   - `test_direct_answer_not_empty`: `direct_answer.answer` is a non-empty string
   - `test_gemini_failure_preserves_direct_answer`: mock generator raises exception; assert `direct_answer` present AND `llm_answer.error` set
   - `test_no_second_retrieval_for_gemini`: confirm `process_dual` is called exactly once per request
   - `test_sources_event_is_list`: `sources` event data has a `sources` key with a list value
   - `test_timing_fields_present`: `timing` event has all 7 required fields
   - `test_empty_query_returns_error_then_done`: event list ends with `done`, contains `error` before `done`
   - `test_stream_always_terminates`: even with a pipeline exception, final event is `done`
   - `test_multilingual_query_succeeds`: Bengali/Tamil query returns a `direct_answer` event
3. Confirm all existing tests still pass after adding this file

**Relevant Context**
- `app/api.py` lines 306–436: `event_generator()` inside `/query/stream`
- `tests/test_api.py`: shows `TestClient` usage pattern for this project
- `tests/test_dual_answer.py`: shows mock engine construction pattern to reuse
- SSE event format: `"event: {name}\ndata: {json}\n\n"`

---

### Sub-Task 6 — Error Hardening

**Status:** [ ] pending

**Intent**
Several failure paths currently produce incorrect behavior: the SSE generator has no top-level guard so an unexpected exception silently closes the stream without a `done` event; the STT `RuntimeError` propagates as a raw 500; the SSE client's `onError` handler never calls `onDone` so the frontend can get stuck in `"direct"` state; and the AnswerCard shows cryptic internal error codes to the user.

**Expected Outcomes**
- `app/api.py` `event_generator()`: a top-level `try/except` wraps the entire generator body; if an unexpected exception escapes after `direct_answer` has already been emitted, an `error` event is emitted followed by `done`
- `app/api.py` `/voice` endpoint: `transcribe_audio()` call wrapped in `try/except RuntimeError` → `HTTPException(status_code=502, detail="STT provider error: ...")`
- `frontend/lib/api.ts` `queryStream()`: `case "error"` branch calls `onDone()` after `onError()` if `done` has not already fired
- `frontend/components/AnswerCard.tsx`: friendly error message map — `"llm_unavailable"` → "AI answer requires a configured API key", `"empty_query"` → "No query was provided", all others show the raw error string as-is
- All existing tests continue to pass

**Todo List**
1. `app/api.py` `event_generator()`:
   - Track `direct_sent = False` flag
   - Set `direct_sent = True` immediately after yielding the `direct_answer` event
   - Wrap the entire generator in `try/except Exception as exc`: if exception raised after `direct_sent`, yield `_sse_event("error", {"message": f"internal error: {exc}"})` then `_sse_event("done", {})`
   - If exception before `direct_sent`, yield `error` + `done` and return
2. `app/api.py` `/voice`:
   - Wrap `transcribe_audio(audio_bytes=audio_bytes, language_code=language)` in `try/except RuntimeError as e: raise HTTPException(status_code=502, detail=f"STT provider error: {e}")`
3. `frontend/lib/api.ts`:
   - Add `let doneCalled = false` tracker before the read loop
   - Set `doneCalled = true` in the `case "done"` branch before calling `callbacks.onDone()`
   - In `case "error"` branch: after `callbacks.onError(...)`, check `if (!doneCalled) { doneCalled = true; callbacks.onDone(); }`
4. `frontend/components/AnswerCard.tsx`:
   - Replace the inline ternary in the llm-failed block with a `getErrorMessage(code)` helper function
   - Map: `"llm_unavailable"` → `"AI answer requires a configured API key"`, `"empty_query"` → `"No query was provided"`, others → raw error string

**Relevant Context**
- `app/api.py` lines 324–427: `event_generator()` — no top-level guard currently
- `app/api.py` lines 408–412: `transcribe_audio()` call — no try/except
- `frontend/lib/api.ts` lines 236–241: `case "error"` — no `onDone` call
- `frontend/components/AnswerCard.tsx` lines ~175–195: hardcoded error string for `"llm_unavailable"`

---

### Sub-Task 7 — Direct-Answer Latency Benchmark

**Status:** [ ] pending

**Intent**
There is currently no clean benchmark that measures `time_to_direct_answer` as the primary metric. The old 38 ms Hindi benchmark is a historical baseline only. The current Qwen benchmark measured a different architecture (no dual-answer, remote LLM, mostly API errors). A new benchmark must measure the direct-answer fast path correctly, with warmup, proper success/failure separation, and P50/P95/P99/P100 reporting. It must be run in the environment closest to production (not relying on localhost claims).

**Expected Outcomes**
- `benchmarks/benchmark_direct_answer.py` exists
- Measures `time_to_direct_ms` per request: embed + Qdrant + rerank + compress + extractive answer
- Separate from `time_to_llm_ms` (Gemini not called in this benchmark)
- Configurable warmup (default 10) and request count (default 50)
- Each request classified: SUCCESS / EXCEPTION
- Latency stats computed on SUCCESS requests only: P50, P95, P99, P100
- Prints clean human-readable report
- Saves auditable JSON to `data/processed/multilingual/direct_answer_benchmark.json`
- Benchmark covers at least 2 queries per language (26 queries cycled to reach 50 requests)
- Report includes: warmup count, measured requests, unique queries, success rate, stage breakdowns

**Todo List**
1. Create `benchmarks/benchmark_direct_answer.py`:
   - Import `RAGEngine` with `ANSWER_BACKEND=extractive`
   - Define 26 representative queries (2 per language)
   - Warmup loop: run `N_WARMUP` requests, discard timings
   - Measured loop: run `N_REQUESTS` requests cycling through query pool
   - Each iteration: call `engine.process_dual(query, language)`, record `timings["total_ms"]` as `time_to_direct_ms`
   - Classify: SUCCESS if no exception AND `result["retrieved_chunks"] > 0`; EXCEPTION otherwise
   - Compute P50/P95/P99/P100 over SUCCESS timings only
   - Evaluate against 200 ms target (and note 150 ms engineering target)
   - Print report and save JSON
2. Add CLI args: `--warmup`, `--requests`
3. Run the benchmark locally, record results
4. Note in the report that this is a localhost measurement; deployed benchmark is a separate concern

**Relevant Context**
- `app/pipeline.py` `process_dual()` returns `(result, state)` — use `result["timings"]` for stage breakdown
- `benchmarks/benchmark_qwen_api_manual.py`: shows the existing benchmark pattern to follow
- Historical reference: Hindi extractive P50 ≈ 38 ms (different collection, not current architecture)
- Current multilingual embedding P50 ≈ 40 ms (single unvalidated measurement)

---

## Implementation Order

Sub-tasks are ordered by dependency and risk:

```
1 — Retrieval Benchmark  (measure before touching production config)
2 — Language Contract    (defines the truth table all UI depends on)
3 — Frontend Language Selector  (uses Sub-Task 2's LANGUAGES constant)
4 — Frontend Multilingual Branding  (uses Sub-Task 3's new props)
5 — SSE Integration Test  (validates the existing implementation)
6 — Error Hardening  (targeted fixes, minimal risk)
7 — Direct-Answer Benchmark  (measurement — no code changes to pipeline)
```

## Files Touched Per Sub-Task

| # | Sub-Task | Files Changed |
|---|---|---|
| 1 | Retrieval Benchmark | `benchmarks/benchmark_retrieval.py` (new), `app/config.py` (conditional), `app/pipeline.py` (conditional) |
| 2 | Language Contract | `frontend/lib/api.ts`, `app/voice/stt.py` |
| 3 | Frontend Selector | `frontend/components/QuerySection.tsx` |
| 4 | Multilingual Branding | `frontend/components/Hero.tsx`, `frontend/app/page.tsx`, `frontend/components/QuerySection.tsx` |
| 5 | SSE Integration Test | `tests/test_integration_stream.py` (new) |
| 6 | Error Hardening | `app/api.py`, `frontend/lib/api.ts`, `frontend/components/AnswerCard.tsx` |
| 7 | Direct-Answer Benchmark | `benchmarks/benchmark_direct_answer.py` (new) |

## Acceptance Criteria (Phase 3 Complete When)

- [ ] 13 RAG languages and 11 STT languages are correctly represented in the language contract
- [ ] Telugu excluded from RAG language selector (not in corpus)
- [ ] Nepali/Sanskrit/Urdu show text-only mode in UI (no voice button)
- [ ] Language selection flows to `queryStream` and `queryVoice`
- [ ] `TOP_K` and language-affinity changes are benchmark-driven only
- [ ] SSE integration test exists covering all listed scenarios
- [ ] `direct_answer` arrives before `llm_answer` (guaranteed by implementation)
- [ ] Gemini failure does not remove or replace direct answer (tested)
- [ ] Gemini path does not re-run retrieval/compression (tested)
- [ ] SSE stream always terminates with `done` (even on error paths)
- [ ] STT `RuntimeError` returns 502 (not 500 with traceback)
- [ ] SSE client `onError` always followed by `onDone` (no stuck loading state)
- [ ] Hero copy updated to reflect 13-language system
- [ ] Direct-answer benchmark exists: P50/P95/P99/P100 for `time_to_direct_ms`
- [ ] `time_to_direct_answer` and `time_to_llm_answer` reported separately
- [ ] Old 38 ms Hindi baseline labeled as historical reference in benchmark report
- [ ] All existing tests continue to pass
