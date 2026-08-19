That's just an environment dependency issue. Your code isn't failing yet. The active .venv doesn't have sentence-transformers installed, even though your project was migrated to it.

Run this inside the same activated .venv:

python -m pip install "sentence-transformers>=2.7.0"

Then verify:

python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformers OK')"

You should get:

SentenceTransformers OK

Then rerun:

python benchmarks\benchmark_qwen_api_manual.py --allow-live-api
If it still fails

Check that python and pip are pointing to the same .venv:

python -c "import sys; print(sys.executable)"
python -m pip --version

Both should point somewhere under:

HH-Goa-Rag\.venv\
One important thing

Your earlier agent report claimed:

sentence-transformers>=2.7.0 was added to requirements.txt

but your current environment clearly doesn't have it. That means the dependency migration was committed to the project files but wasn't installed into this virtual environment.

After installation, don't change anything else yet. Run the Qwen benchmark and give me the output. That's the measurement we actually need now.

Manual live API benchmark file: benchmarks/benchmark_qwen_api_manual.py; requires --allow-live-api.









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

------------------------------------------------------------------------------------------------------------

| Area                                  | Status                             | Priority     |
| ------------------------------------- | ---------------------------------- | ------------ |
| 13-language corpus                    | ✅ Done                             | —            |
| Multilingual embedding                | ✅ Done                             | —            |
| Multilingual Qdrant                   | ✅ Done                             | —            |
| Query embedding                       | ✅ Done                             | —            |
| Retrieval                             | ✅ Done                             | —            |
| Optimized reranking                   | ✅ Done                             | —            |
| Context compression                   | ✅ Done                             | —            |
| Qwen API integration                  | ✅ Implemented                      | 🔴 Test      |
| **Live Qwen latency**                 | ❌ Not measured                     | 🔴 Critical  |
| **Full post-STT P95/P100**            | ❌ Not measured                     | 🔴 Critical  |
| Production deployment                 | ❌ Not done                         | 🔴 Critical  |
| Production Qdrant                     | ❌ Not finalized                    | 🔴 Critical  |
| Production API/server                 | ⚠️ Local only                      | 🔴 Critical  |
| Load/concurrency testing              | ❌                                  | 🟠 Important |
| 13-language end-to-end quality test   | ⚠️ Retrieval only                  | 🟠 Important |
| Failure/timeout handling              | Implemented, needs live validation | 🟠 Important |
| Frontend → production API integration | ⚠️                                 | 🟠 Important |
| Final benchmark/report                | ❌                                  | 🔴 Critical  |




*HH GOA RAG - LIVE QWEN API MULTILINGUAL BENCHMARK*
======================================================================
Collection : hh_goa_rag_multilingual
Provider   : huggingface
Model      : Qwen/Qwen3-0.6B
Tokens     : 64
Timeout    : 5.0s
Queries    : 13
Warmup     : 0
Loading engine once...
Embedding threads: 4
Loading embedding model...
Loading weights: 100%|████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 8314.49it/s]
Qwen API client ready in 7 ms
Opening Qdrant...
RAG engine ready.
Running live benchmark...
[13/13] llm=291.81ms total=360.87ms reason=qwen_api_error

Stage       | P50    | P95    | P99    | P100
------------------------------------------------------
embedding   |  40.15 |  58.41 |  59.19 |   59.39
qdrant      |  12.58 |  19.37 |  20.83 |   21.19
rerank      |   0.26 |   0.45 |   0.46 |    0.46
compression |   0.11 |   0.21 |   0.22 |    0.22
llm         | 331.90 | 551.39 | 757.03 |  808.44
total       | 394.83 | 590.15 | 804.96 |  858.66

Recall:
Recall@1  : 23.08%
Recall@5  : 38.46%
Recall@10 : 38.46%

P95 total < 200 ms: False
Report: data\processed\multilingual\qwen_api_benchmark_report.json





**What we should do next**
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

Phase 2 — Test whether Qwen is fundamentally too slow

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