# Experiment Log — HH Goa 2026 Voice RAG

---

## EXP-001

**Date:** 2025-07-04  
**Change:** Fixed Hindi guardrail false-rejection bug  
**File:** `answer_generator.py`

### Problem

65% of all queries were blocked with reason `low_context_relevance` despite having retrieved chunks with vector similarity scores of 0.6–0.8.

### Root Cause

The `keyword_overlap` guard in `generate_extractive_answer` used:
1. `re.findall(r"\w+", query)` without `re.UNICODE` flag
2. `len(w) > 2` filter — this excluded most short Hindi Devanagari words (2-char codepoints)
3. Single-signal threshold `overlap < 0.05` — when all query words are filtered out, overlap = 0.0 regardless of retrieval quality

The result: `query_words` was always empty for Hindi queries → `keyword_overlap` returned 0.0 → ALWAYS blocked.

### Fix

Two changes:
1. Added `re.UNICODE` flag and changed `len(w) > 2` to `len(w) >= 2` to correctly tokenize Hindi
2. Changed guard from `overlap < 0.05` (single signal) to `overlap < 0.05 AND top_score < 0.50` (dual signal) — if vector retrieval score ≥ 0.50, trust the semantic retrieval

### Results

| Metric | Before | After |
|---|---|---|
| Grounded | 34.8% | 99.7% |
| Blocked | 65.2% | 0.3% |
| P50 latency | 38.8 ms | 38.1 ms |
| Recall@3 | 79.1% | 79.1% |

**Decision: KEEP** — significant improvement in answer rate, no regression in latency or recall.

---

## EXP-002

**Date:** 2025-07-04  
**Change:** Added FastAPI layer, STT module, requirements.txt, .env.example  

See `api.py`, `stt.py`, `requirements.txt`, `.env.example`.

**Decision: KEEP** — adds required API and STT infrastructure without touching the core pipeline.

---
## EXP-003

**Date:** 2025-07-04  
**Change:** Pre-rerank deduplication — limit each `query_id` to max 2 chunks before reranking  
**File:** `pipeline.py` (added `MAX_CHUNKS_PER_SOURCE = 2` + dedup loop between Qdrant and rerank)

### Hypothesis

93.2% of queries have ≥3 duplicate chunks from the same source in Top-20. 68 of 82 reranking losses were associated with duplicate monopoly. Deduplication should free Top-3 slots for diverse candidates.

### Results

| Metric | Baseline | Dedup (exp) | Change |
|---|---|---|---|
| Recall@3 | 79.1% | 78.5% | **−0.6pp** |
| Top-20 recall | 87.3% | 87.3% | 0 |
| Grounded | 99.7% | — | — |
| P95 latency | ~42 ms | ~41 ms | negligible |

### Decision: **REVERTED**

The dedup caused a net regression of 6 queries (79.1% → 78.5%).

**Why it failed:** The "duplicate" chunks retrieved by Qdrant are not noise — they are multiple excerpt windows from the correct passage, each genuinely scoring high on cosine similarity. The reranker correctly selects the single best one among them, which does reach Top-3. Discarding 2nd and 3rd copies of the same source reduced the effective candidate pool from ~20 to ~12–14 unique sources, which sometimes caused the single correct chunk to be ranked below 3 strong unrelated candidates. The monopoly problem diagnosed in Phase 2 is real, but the fix must work at the **source level** (avoiding indexing redundant passage chunks in the first place), not by filtering at retrieval time. That is a re-chunking/re-indexing concern, which belongs to a future phase.

---

