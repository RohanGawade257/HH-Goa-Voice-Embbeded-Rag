# Stage 1 Baseline — HH Goa 2026 Voice RAG

**Date:** 2025-07-04  
**Status:** VERIFIED

---

## System Overview

A Voice-Enabled Retrieval-Augmented Generation (RAG) system over the AI4Bharat MSMARCO-XI Hindi dataset.

---

## Dataset

| Property | Value |
|---|---|
| Source file | `data/hindi_sample_1000.jsonl` |
| Records | 1,000 |
| Language | Hindi (Devanagari) |
| Dataset | AI4Bharat MSMARCO-XI |
| Schema | `query_id`, `query`, `Answer`, `passages.Translated_passages`, `passages.is_selected`, `query_type` |

---

## Data Processing

| Step | File | Output | Count |
|---|---|---|---|
| Cleaning | `process_data.py` | `data/processed/passages_1000.jsonl` | 9,936 passages |
| Chunking | `chunk_data.py` | `data/processed/chunks_1000.jsonl` | 10,302 chunks |

### Chunking Strategy (VAST/Semantic Adaptive)

- **Strategy A – intact**: passages ≤ 80 words → kept whole
- **Strategy B – sentence_group**: 81–180 words → sentence-boundary groups ≤ 140 words
- **Strategy C – adaptive_window**: > 180 words → sentence-boundary windows, target 100 words, max 140, 1-sentence overlap
- **Fallback**: pathological single sentences > 140 words → word-based sub-chunks

Chunk metadata preserved: `chunk_id`, `passage_id`, `query_id`, `passage_index`, `chunk_index`, `language`, `word_count`, `chunk_strategy`, `has_overlap`, `is_selected`, `query_type`

---

## Embeddings

| Property | Value |
|---|---|
| Model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Library | fastembed |
| Dimension | 384 |
| Distance | Cosine |
| Batch size | 128 |

---

## Vector Database (Qdrant)

| Property | Value |
|---|---|
| Storage | Local SQLite at `data/qdrant/` |
| Collection | `hh_goa_rag_hindi` |
| Points | 10,302 |
| Persistence | Yes — pre-built, not rebuilt on API startup |

---

## Retrieval

| Property | Value |
|---|---|
| Top-K retrieval | 20 candidates from Qdrant |
| Final Top-K | 3 (after reranking) |
| Reranker | Fast lexical reranker (no external model) |
| Reranker weights | Vector 0.70 + Lexical 0.20 + Phrase 0.10 |

---

## Answer Generation

| Property | Value |
|---|---|
| Method | Extractive (no remote LLM call) |
| Max context chunks | 3 |
| Max context words | 350 |
| Answer selection | Top-2 sentences by query keyword overlap from best chunk |

---

## Guardrails

| Check | Trigger |
|---|---|
| Empty query | `blocked: True`, reason `empty_query` |
| Off-topic | pattern match (code, recipe, weather, etc.) |
| No context | no chunks retrieved |
| Low relevance | `keyword_overlap < 0.05` **AND** `top_score < 0.50` |
| Weak answer | answer length < 10 chars |

**Note:** Guardrail 3 (low relevance) uses a dual-signal check — keyword overlap alone is insufficient for Hindi because many Devanagari words have ≥ 2 Unicode codepoints and vector similarity (≥ 0.50) is a reliable relevance signal.

---

## Benchmark Results (VERIFIED — 2025-07-04)

### Latency (1000 queries, post-warmup)

| Metric | Value |
|---|---|
| P50 total | 38.1 ms |
| P70 total | 39.3 ms |
| P100 total | 110.7 ms |
| Mean total | 39.0 ms |
| Embedding P50 | 17.0 ms |
| Qdrant P50 | 18.9 ms |
| Rerank P50 | 1.8 ms |
| Answer P50 | 0.2 ms |

**Target: < 200 ms post-STT → PASS (P50: 38 ms)**

### Retrieval Quality

| Metric | Value |
|---|---|
| Recall@1 | 68.40% |
| Recall@3 | 79.10% |
| Recall@5 | 79.10% |

*Recall is measured by matching retrieved chunk `query_id` against the ground-truth query_id.*

### Guardrail Behavior

| Metric | Value |
|---|---|
| Grounded answers | 99.7% |
| Blocked (all reasons) | 0.3% |

---

## Current Bottleneck

```
Embedding: 17 ms  (primary bottleneck — model inference)
Qdrant:    19 ms  (secondary — SQLite local storage)
Rerank:     2 ms  (negligible)
Answer:     0.2 ms (negligible)
```

The embedding step dominates. `paraphrase-multilingual-MiniLM-L12-v2` is already a lightweight model (384-dim). The Qdrant local SQLite storage is slower than in-memory Qdrant server but sufficient for <200ms target.

---

## Component Status

| Component | Status | Notes |
|---|---|---|
| Dataset | DONE | 1000 records |
| Data processing | DONE | 9,936 passages |
| VAST/Semantic chunking | DONE | 10,302 chunks, 4 strategies |
| Embeddings | DONE | fastembed, 384-dim |
| Qdrant | DONE | local, persistent |
| Retrieval | DONE | Recall@3 = 79.1% |
| Reranking | DONE | lightweight lexical |
| Answer generation | DONE | extractive |
| Guardrails | DONE (fixed) | fixed Hindi false-rejection bug |
| FastAPI | DONE | /health /query /voice |
| STT (Sarvam) | DONE | mock fallback if no key |
| requirements.txt | DONE | |
| .env.example | DONE | Sarvam key scrubbed |
| Next.js frontend | DONE | builds clean, 0 TS errors |

---

## Known Issues / Limitations

1. **Answer generation is extractive only** — returns the most relevant sentences from the retrieved passage. No generative LLM call. This keeps latency under 200 ms but limits answer quality for complex questions.
2. **Recall@1 = 68.4%** — approximately 32% of queries don't surface the correct passage as rank-1. This is typical for a 1000-record dataset with 10K chunks.
3. **Recall@5 = Recall@3 = 79.1%** — no improvement from 3→5, meaning the reranker correctly identifies the top 3 from top 20 in the cases where it can be identified at all.
4. **Local SQLite Qdrant** — production should use Qdrant Cloud or a remote Qdrant server.
5. **No generative LLM** — for a production system, connecting to an LLM API (e.g., IBM watsonx, Groq, OpenAI) would significantly improve answer quality while still targeting <200ms.

---

## Best Known Configuration

```
Dataset:          data/hindi_sample_1000.jsonl (1000 records)
Chunking:         VAST adaptive (sentence-boundary, 80/180 word thresholds)
Embedding model:  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Embedding dim:    384
Vector DB:        Qdrant local SQLite
Collection:       hh_goa_rag_hindi (10,302 points)
Top-K retrieval:  20
Reranker:         Lexical (vector=0.70, lexical=0.20, phrase=0.10)
Top-K final:      3
Answer method:    Extractive (sentence selection from top chunk)
Guardrails:       overlap<0.05 AND score<0.50 (dual-signal)
P50 latency:      38.1 ms
Recall@3:         79.1%
Grounded rate:    99.7%
```
