


*This was Half Done by codex*



You are working on the HH Goa 2026 multilingual Voice RAG project.

## OBJECTIVE

Our hard requirement is:

POST-STT END-TO-END LATENCY < 200 ms

Current multilingual retrieval benchmark:

* Embedding P50: ~10.9 ms
* Embedding P95: ~14.4 ms
* Qdrant P50: ~13.6 ms
* Qdrant P95: ~15 ms
* Reranker optimized P50: ~0.3 ms
* Reranker optimized P95: ~0.6 ms
* Retrieval pipeline P95: roughly 48 ms in the latest experiment

Do NOT optimize blindly. Preserve retrieval quality while reducing latency.

## CURRENT STACK

Embedding model:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Embedding dimension:

384

Normalization:

normalize_embeddings=True

Vector database:

Qdrant

Collection:

hh_goa_rag_multilingual

Corpus:

13 languages

Current corpus:

6046 chunks

Languages:

as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, ur

Current retrieval architecture:

query
→ SentenceTransformer embedding
→ Qdrant dense retrieval
→ lightweight reranking
→ answer generation

Target final architecture:

Voice
→ Sarvam multilingual STT
→ language identification
→ multilingual query embedding
→ Qdrant dense retrieval
→ lightweight reranking
→ context compression
→ small multilingual LLM
→ multilingual TTS

## IMPORTANT: DO NOT REVERT SENTENCETRANSFORMERS

FastEmbed has already been removed from the active system.

Continue using:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Do not reintroduce FastEmbed.

## STEP 1 — LOCK EMBEDDING CONFIGURATION

Analyze the existing embedding implementation.

Benchmark results show that 4 CPU threads currently provide the best balance.

Set the production embedding configuration to:

4 CPU threads

Do this through a configurable setting/environment variable rather than hardcoding it throughout the project.

For example:

EMBEDDING_THREADS=4

Do not create a new model instance for every request.

The SentenceTransformer model MUST be loaded once during application startup and reused.

Verify that request-time embedding only calls encode() on the already-loaded model.

## STEP 2 — DO NOT CHANGE THE EMBEDDING MODEL

Do NOT experiment with another embedding model yet.

Do NOT downgrade dimensions.

Do NOT introduce a second embedding model.

Keep:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

384 dimensions

COSINE similarity

normalized vectors

The priority now is system optimization, not another model migration.

## STEP 3 — OPTIMIZE QDRANT RETRIEVAL

Inspect the current Qdrant implementation.

Measure:

Top-20
Top-10
Top-8
Top-5

using the SAME benchmark methodology.

Do not select a lower Top-K merely because it is faster.

Compare:

* Recall@1
* Recall@5
* Recall@10
* P50
* P95
* P100

The previous experiment showed:

Top-20:
Recall@1 7.60%
Recall@5 15.40%
Recall@10 18.80%

Top-10:
Recall@1 7.60%
Recall@5 15.40%
Recall@10 18.80%

Therefore Top-10 may be sufficient, but verify it using a stable benchmark before changing production.

If Top-10 preserves retrieval quality and produces a reliable latency improvement, use:

QDRANT_TOP_K=10

Otherwise keep Top-20.

Do NOT choose Top-5 simply for latency.

## STEP 4 — KEEP RERANKER OPTIMIZED

The optimized reranker is substantially better than the legacy implementation.

Previous result:

Legacy:
P50 ~0.7 ms
P95 ~1.3 ms
P100 up to ~57 ms

Optimized:
P50 ~0.3 ms
P95 ~0.6 ms
P100 ~6–10 ms

Keep the optimized implementation.

Investigate the remaining P100 outliers.

The goal is deterministic low tail latency.

Do NOT replace the reranker with a heavyweight cross-encoder.

The reranker must remain lightweight.

## STEP 5 — ADD CONTEXT COMPRESSION

Do NOT immediately reduce the database chunk size.

Current chunks are approximately <=120 words.

Keep the retrieval corpus unchanged for now.

Instead implement a post-reranking context compression stage:

Qdrant Top-K
→ reranker
→ select top 2–3 chunks
→ extract only relevant sentences
→ construct compact LLM context

The LLM should NOT receive all retrieved chunks.

Default target:

TOP_CONTEXT_CHUNKS=2 or 3

The context should contain only the highest-value evidence.

Preserve enough information to answer the query correctly.

Do not perform expensive semantic compression using another transformer model.

Use lightweight CPU operations only:

* sentence splitting
* relevance scoring using existing retrieval/reranker information
* deduplication
* maximum character/token budget

## STEP 6 — CREATE A HARD LLM CONTEXT BUDGET

Add a configurable limit such as:

MAX_CONTEXT_CHARS

or

MAX_CONTEXT_TOKENS

Start conservatively.

The goal is to minimize the number of tokens sent to the LLM.

The LLM should receive:

* user query
* language
* 2–3 highly relevant evidence snippets
* strict instruction to answer only from supplied evidence

Do NOT send:

* all Top-20 chunks
* metadata
* chunk IDs
* unnecessary language metadata
* benchmark information
* retrieval scores unless actually needed
* duplicated text

## STEP 7 — ADD QWEN 0.6B AS ANSWER GENERATION

Add the Hugging Face Qwen3-0.6B model as the initial answer-generation model.

Do NOT put model loading inside the request path.

The model must be loaded once at application startup if running locally.

If using an external Hugging Face inference API instead:

* reuse the HTTP connection
* use a short timeout
* minimize prompt size
* minimize max_new_tokens
* do not request unnecessary reasoning
* stop generation as soon as the answer is complete

Use a configurable value:

MAX_NEW_TOKENS

Start small, for example 64 tokens, and benchmark.

Do not generate long answers.

The answer should be concise and grounded.

IMPORTANT:

Do not assume the Hugging Face API latency will be <200 ms.

Benchmark the actual API latency separately.

If the remote API cannot reliably meet the latency budget, report that clearly instead of pretending it does.

## STEP 8 — MULTILINGUAL ANSWER GENERATION

The LLM must preserve the user's language.

Example:

Hindi query → Hindi answer
Marathi query → Marathi answer
Tamil query → Tamil answer
Bengali query → Bengali answer

Do not translate everything into English unless absolutely necessary.

The prompt should explicitly instruct the model to answer in the query language.

## STEP 9 — DO NOT ADD TTS YET

Do NOT implement TTS in this optimization pass.

First establish:

STT excluded
→ embedding
→ retrieval
→ rerank
→ context compression
→ LLM

Then benchmark this complete post-STT text pipeline.

Only after that should Sarvam TTS be added and benchmarked separately.

This prevents multiple moving parts from hiding the actual latency bottleneck.

## STEP 10 — BUILD A REAL LATENCY BREAKDOWN

Create/update a benchmark that reports:

stage             P50     P95     P99     P100

embedding
qdrant
rerank
compression
LLM
total

Run at least 500 queries.

Use warmup queries.

Exclude model loading from request latency.

Do NOT average away outliers.

The PRIMARY GATE is:

P95 total < 200 ms

Also report P100.

## STEP 11 — RETRIEVAL QUALITY MUST REMAIN VISIBLE

Every optimization benchmark must report:

Recall@1
Recall@5
Recall@10

Never optimize latency while hiding retrieval degradation.

If latency improves but recall falls substantially, do NOT automatically accept the change.

## STEP 12 — DO NOT TOUCH THE MULTILINGUAL CORPUS

The following corpus is already complete:

13 languages
13,000 input records
5,919 passages
6,046 chunks
6,046 embeddings

Qdrant collection:

hh_goa_rag_multilingual

Do not reprocess or re-embed the corpus unless a genuine compatibility problem is discovered.

The old Hindi collection:

hh_goa_rag_hindi

must remain untouched.

## STEP 13 — PRODUCE A BEFORE/AFTER REPORT

At the end provide:

1. Files changed
2. Configuration changes
3. Retrieval latency before/after
4. Reranker latency before/after
5. Context size before/after
6. LLM latency
7. Total post-STT latency
8. Recall@1
9. Recall@5
10. Recall@10
11. P50
12. P95
13. P99
14. P100
15. Any remaining bottlenecks

## SUCCESS CRITERIA

Minimum:

P95 post-STT < 200 ms

Preferred:

P95 post-STT < 150 ms

Excellent:

P95 post-STT < 100 ms

Do NOT claim production readiness merely because the retrieval portion is <50 ms.

The final production latency must include:

embedding

* Qdrant
* reranking
* context preparation
* LLM generation

STT and TTS should be measured separately because they are external voice stages.

## MOST IMPORTANT RULE

Do not make unnecessary architectural changes.

Optimize one stage at a time.

Benchmark before and after every meaningful change.

Preserve retrieval quality.

Do not sacrifice correctness for an artificial latency number.
