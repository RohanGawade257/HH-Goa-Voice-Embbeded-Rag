HH-Goa-Rag: Total Context
1. Project Purpose
HH-Goa-Rag is a voice-enabled multilingual RAG system for Indian-language questions.

The system accepts:

Text queries through POST /query
Audio queries through POST /voice
Health checks through GET /health
The original project was Hindi-focused, but the active implementation has now moved toward a 13-language multilingual corpus.

2. Languages
Active multilingual corpus: 13 languages
The corpus contains 1,000 source records per language:

Assamese
Bengali
Gujarati
Hindi
Kannada
Malayalam
Marathi
Nepali
Odia
Punjabi
Sanskrit
Tamil
Urdu
Current multilingual data statistics:

Input records: 13,000
Cleaned passages: 5,919
Indexed chunks: 6,046
Qdrant collection: hh_goa_rag_multilingual
The language list is defined in process_multilingual.py, pipeline.py, and llm.py.

Voice/STT language support
Sarvam STT currently lists 11 voice language codes in stt.py:

Hindi
English
Bengali
Gujarati
Kannada
Malayalam
Marathi
Odia
Punjabi
Tamil
Telugu
There is a small mismatch here: Telugu is supported by the STT module, but it is not part of the 13-language indexed RAG corpus. Nepali, Sanskrit, and Urdu are in the corpus but are not listed in the Sarvam STT language map.

3. Models and Services Used
Embedding model
The active embedding model is:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Configuration:

Library: sentence-transformers
Vector size: 384
Distance: cosine similarity
Embedding threads: 4
Batch size during ingestion: 128
This model is used both when indexing documents and when embedding user queries. The current code uses SentenceTransformer, not fastembed, despite the older README describing it as fastembed.

Vector database
The system uses:Qdrant local storage
Path: data/qdrant/

The active collection is:hh_goa_rag_multilingual

It contains:6,046 vectors

There is also an older Hindi collection:hh_goa_rag_hindi

That collection belongs to the previous Hindi baseline.

Answer-generation model
The current configured answer backend is:
Provider: Hugging Face
Model: Qwen/Qwen3-0.6B
Endpoint: Hugging Face OpenAI-compatible chat API

It is implemented in llm.py.

Important configuration:

Maximum output tokens: 64
Timeout: 5 seconds
Temperature: 0
Answers are prompted to use only retrieved evidence
The response should be generated in the user’s language
The Qwen stage is optional at code level, but it is the active default because ANSWER_BACKEND defaults to qwen_api in config.py.

Speech-to-text model/service
Voice input uses Sarvam’s API:Sarvam Saarika v2

The STT call is made remotely and requires SARVAM_API_KEY.

Without the key, the project uses a mock Hindi transcript for development.

Text-to-speech
TTS is not implemented yet. tts.py is currently only a placeholder for future Sarvam multilingual TTS integration.

4. End-to-End Pipeline
Voice request
Audio upload
    ↓
Sarvam STT
    ↓
Transcript
    ↓
Multilingual query embedding
    ↓
Qdrant vector retrieval
    ↓
Lexical and phrase reranking
    ↓
Context compression
    ↓
Qwen answer generation
    ↓
FastAPI response
    ↓
Next.js frontend

Detailed processing
POST /voice receives an audio file.
stt.py sends the audio to Sarvam or uses mock mode.
The transcript is passed to RAGEngine.process() in pipeline.py.
The query is embedded using paraphrase-multilingual-MiniLM-L12-v2.
Qdrant retrieves the nearest multilingual chunks.
Retrieved chunks are reranked using:
Vector similarity: 70%
Lexical overlap:   20%
Phrase matching:   10%

The top results are reduced by context_compressor.py.
The context is sent to Qwen through llm.py.
Guardrails handle empty queries, missing context, low relevance, and failed generation.
FastAPI returns the answer, sources, and timing breakdown.
Text request
POST /query skips the STT phase and begins directly at query embedding.

5. Data-Ingestion Pipeline
The ingestion flow is:
Raw multilingual JSONL
    ↓
Repair corrupted multilingual text
    ↓
process_multilingual.py
    ↓
Unified passages.jsonl
    ↓
chunk_multilingual.py
    ↓
chunks.jsonl
    ↓
embed_multilingual.py
    ↓
Qdrant multilingual collection

Chunking uses adaptive rules:

Up to 80 words: keep passage intact
81 to 180 words: sentence-based grouping
Longer passages: adaptive windows of roughly 100 words
Maximum chunk size: 140 words
One-sentence overlap
Long-sentence fallback: word-based chunks
The generated multilingual reports are in:

process_report.json
multilingual_processing_report.json
embed_report.json
6. Latency: What We Know Now
There are two different latency baselines in the repository.

Older Hindi extractive baseline
The frozen Hindi baseline reported:

Stage	P50
Embedding	17 ms
Qdrant	19 ms
Reranking	1.8 ms
Answer extraction	0.2 ms
Total	38.1 ms
This baseline used:

Hindi-only data
hh_goa_rag_hindi
No remote LLM
Extractive sentence selection
It passed the post-STT target of 200 ms.

Current multilingual Qwen path
The latest live benchmark reported:

Stage	P50	P95
Embedding	40.15 ms	58.41 ms
Qdrant	12.58 ms	19.37 ms
Reranking	0.26 ms	0.45 ms
Compression	0.11 ms	0.21 ms
Qwen API	331.90 ms	551.39 ms
Total	394.83 ms	590.15 ms
The report is qwen_api_benchmark_report.json.

Current major bottleneck
The major bottleneck is now the remote Qwen API call:
Qwen API: approximately 332 ms P50
Qwen API: approximately 551 ms P95

The second-largest cost is multilingual embedding:
Embedding: approximately 40 ms P50

The current total does not meet the 200 ms target:
P95 total: 590.15 ms
Target:    200 ms
Status:    FAIL

However, all 13 benchmark requests were reported as qwen_api_error. Therefore, this is not yet a clean successful-answer latency measurement. The next reliable benchmark should separately report:

Successful Qwen requests
API errors
Timeouts
Successful LLM P50/P95/P99
Successful total P50/P95/P99
The old 38 ms number should not be presented as the current multilingual production latency.

7. Current Folder Structure
Runtime application
app

pipeline.py: central RAG orchestration
api.py: FastAPI endpoints
config.py: environment and runtime configuration
context_compressor.py: context reduction
answer_generator.py: older extractive answer and guardrails
generation/llm.py: Qwen API generation
retrieval/: older standalone retrieval utilities
voice/stt.py: Sarvam STT
voice/tts.py: future TTS placeholder
Ingestion
ingestion

Contains one-time data preparation:

Processing and cleaning
Multilingual repair
Chunking
Embedding
Qdrant upload
Validation and inspection
Data
data

Contains:

Original Hindi sample
Per-language multilingual JSONL files
Repaired multilingual data
Processed passages and chunks
Qdrant local database
Benchmark reports
Benchmarks
benchmarks

Contains separate measurements for:

Hindi baseline latency
Multilingual retrieval
Post-STT pipeline
Qwen live API latency
Reranking
Embedding and Qdrant breakdown
Tests
tests

Contains tests for:

API behavior
Configuration
Context compression
Imports
LLM generation
Multilingual retrieval
Frontend
frontend

This is a separate Next.js application:

Next.js 16
React 19
Tailwind CSS 4
Motion
Lucide icons
Typed API client in api.ts
The frontend calls the FastAPI backend using NEXT_PUBLIC_API_URL.

Archive
archive

Contains historical experiments and debugging scripts. These should remain separate from active runtime code.

8. Folder-Alignment Status
The broad folder separation is sensible:
app/         active runtime
ingestion/   data preparation
data/        datasets, indexes, reports
benchmarks/  measurements
tests/       verification
frontend/    web UI
archive/     historical work

The main alignment issue is that the documentation still describes the old Hindi architecture while the active code uses the newer multilingual Qwen architecture.

The following items are currently inconsistent:

README describes Hindi-only behavior, but the active pipeline uses 13 languages.
README says the answer is extractive, but the active default is Qwen API generation.
README mentions fastembed, but active code uses sentence-transformers.
README describes Top-20 retrieval, while current config defaults to QDRANT_TOP_K=10.
Old Hindi collection names remain in legacy utilities.
Frontend timing types do not yet fully represent compression_ms and llm_ms.
STT supports a different language set from the indexed RAG corpus.
TTS is structurally present but functionally unimplemented.
The 200 ms performance claim is valid for the old extractive Hindi baseline, not yet for the live multilingual Qwen path.
The clean target structure is therefore already mostly present. The remaining work is primarily to align the project’s documentation, benchmark interpretation, frontend API types, language contracts, and active configuration around the multilingual Qwen pipeline


| Component               | Status                           | Notes                                      |
| ----------------------- | -------------------------------- | ------------------------------------------ |
| 13-language corpus      | ✅ Done                           | 13,000 records → 6,046 chunks              |
| Multilingual processing | ✅ Done                           | UTF-8/language metadata validated          |
| Multilingual embeddings | ✅ Done                           | MiniLM-L12-v2, 384d                        |
| Multilingual Qdrant     | ✅ Done                           | 6,046 vectors                              |
| Query embedding         | ✅ Done                           | SentenceTransformers                       |
| Vector retrieval        | ✅ Done                           | Qdrant                                     |
| Reranking               | ✅ Done                           | Optimized reranker                         |
| Context compression     | ✅ Done                           | Very low latency                           |
| Qwen API integration    | ✅ Implemented                    | `Qwen/Qwen3-0.6B` via API                  |
| Grounded prompting      | ✅ Done                           | Missing-context handling included          |
| Offline latency         | ✅ Excellent                      | P95 ≈ 31 ms                                |
| Live Qwen latency       | ❌ Not measured                   | Current blocker                            |
| Full post-STT latency   | ❌ Not proven                     | Must include Qwen                          |
| Retrieval quality       | ⚠️ Weak                          | Recall@10 ≈ 18.8%                          |
| STT integration         | ⚠️ Needs final integration test  | Pipeline currently starts after text/query |
| TTS                     | ⚠️ Needs implementation/decision | Depends on task requirements               |
| End-to-end voice test   | ❌ Not proven                     | Must test actual voice → answer            |
| Production API          | ✅ Mostly done                    | Needs end-to-end validation                |
| Frontend                | ⚠️ Needs final integration       | Need connect to real backend               |
| Error handling          | ⚠️ Needs final audit             | API/network/model failures                 |
| Deployment              | ❌ Not final                      | Need deploy and benchmark remotely         |
| Final benchmark/report  | ❌ Not final                      | Need actual evidence                       |


**WHAT SHOULD WE DO NEXT**

Phase 1 — Fix the benchmark

We need the benchmark to distinguish:

SUCCESS
API error
timeout
HTTP error

and report successful Qwen latency separately.

Run something like:

13 queries
5–10 warmups
30–50 measured requests

Then report:

successful requests
failed requests
timeouts


LLM success P50
LLM success P95
LLM success P99
LLM success P100


TOTAL success P50
TOTAL success P95
TOTAL success P99
TOTAL success P100

Otherwise we're mixing failures into the latency statistics.

Phase 2 — Phase 2 — replace Qwen/Qwen3-0.6B with Qwen2.5-0.5B-Instruct and force to be in non thinking mode And Test whether Qwen is fundamentally too slow

We should benchmark different output limits:

MAX_NEW_TOKENS=16
MAX_NEW_TOKENS=32
MAX_NEW_TOKENS=48
MAX_NEW_TOKENS=64

If 16 tokens is still ~300–500 ms, then token reduction isn't the solution.

If 16 tokens gets close to ~100–150 ms, then aggressive answer constraints become worthwhile.

Phase 3 — Test another inference provider

This is where I'd spend serious effort.

Your architecture should remain:

                    ┌─ multilingual embedding
Voice → STT → query ┤
                    └─ Qdrant
                         ↓
                       rerank
                         ↓
                    compression
                         ↓
                  FAST LLM API
                         ↓
                      answer

The LLM provider should be configurable, which your current architecture already supports.

Don't hard-code the project around Hugging Face inference.

One more important issue

Your embedding latency jumped from approximately:

~10–15 ms P95

in your earlier benchmark to:

58.41 ms P95

in this benchmark.

That's significant.

The reason is probably not the multilingual model itself. The live benchmark has:

Warmup: 0

and only 13 queries.

So we need to benchmark the deployed-style process with proper warmup before changing EMBEDDING_THREADS.

Your earlier experiment showed:

threads=4
P95 ≈ 12–14 ms

That's the number I'd currently trust more for the warmed local retrieval pipeline.

The target architecture I would aim for

Ideally:

Embedding       10–15 ms
Qdrant          10–15 ms
Rerank           <1 ms
Compression      <1 ms
LLM             60–100 ms
--------------------------------
TOTAL           ~82–132 ms

That gives you some safety margin under 200 ms.

Don't aim for exactly 200 ms. Aim for ≤150 ms P95 so deployment/network variance doesn't kill you.

Your current status

I'd mark the project:

Retrieval: ✅ Good
Multilingual corpus: ✅ Good
Multilingual embedding: ✅ Good
Reranking: ✅ Good
Context compression: ✅ Good
Qwen integration: ✅ Implemented
Qwen latency: ❌ Fails target
Production deployment: ⏳ Not finished
Real STT: ⏳ Not finished
Production end-to-end benchmark: ⏳ Not finished



*MAIN AIM*

After STT the whole Pipeline must get run within 200ms
it is gonna get deployed so not local models








HH GOA RAG — GEMINI LLM LATENCY BENCHMARK
============================================================
Provider         : gemini
Model            : gemini-2.5-flash-lite
Thinking budget  : 0  (0 = non-thinking)
Max output tokens: 30
Temperature      : 0.0
Warmup           : 5
Measured requests: 30

------------------------------------------------------------
REQUEST RESULTS
------------------------------------------------------------
SUCCESS   : 5
HTTP_ERROR: 25
TIMEOUT   : 0
EXCEPTION : 0
SUCCESS RATE: 16.7%

------------------------------------------------------------
TTFT  (time-to-first-token, successful requests only)
------------------------------------------------------------
P50 : 730 ms
P95 : 982 ms
P99 : 1014 ms
P100: 1023 ms
Avg : 798 ms

------------------------------------------------------------
TOTAL LLM GENERATION TIME  (successful requests only)
------------------------------------------------------------
P50 : 742 ms
P95 : 982 ms
P99 : 1015 ms
P100: 1023 ms
Avg : 801 ms

------------------------------------------------------------
TARGET  (LLM generation P95 contribution to post-STT budget)
------------------------------------------------------------
POST-STT P95 BUDGET: <= 200 ms  (full pipeline)
LLM TOTAL P95      : 982 ms
STATUS             : INCONCLUSIVE
Report             : data\processed\multilingual\gemini_llm_benchmark_report.json