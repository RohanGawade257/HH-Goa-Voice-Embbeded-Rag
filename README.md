# HH-Goa-Rag — Hindi Voice RAG

**Hacker House Goa 2026 — Task 2**

A complete, production-quality Voice-Enabled Hindi Retrieval-Augmented Generation system built on the AI4Bharat MSMARCO-XI dataset.

---

## What it does

Ask a question in Hindi (by voice or text). The system retrieves semantically relevant passages from a knowledge base of 10,302 chunks, reranks them, assembles an extractive grounded answer, and returns it in under 55 ms (P100).

---

## Architecture

```
Audio (Hindi)
      ↓
Sarvam STT          [app/voice/stt.py]
      ↓
Query embedding     [app/pipeline.py — fastembed MiniLM-L12-v2, 384-dim]
      ↓
Qdrant Top-20       [local SQLite, cosine similarity]
      ↓
Lexical Reranker     [vector×0.70 + overlap×0.20 + phrase×0.10 → Top-3]
      ↓
Extractive answer   [app/answer_generator.py — sentence selection + guardrails]
      ↓
FastAPI response    [app/api.py — /query /voice /health]
      ↓
Next.js frontend    [frontend/ — React + Tailwind v4 + Motion]
```

---

## Quick start

### Prerequisites

- Python 3.10+
- Node.js 18+ (tested on 22.20.0)
- The pre-built Qdrant index at `data/qdrant/` (already in repo — do **not** re-index unless necessary)

### 1. Python backend

```bash
# Install dependencies
pip install -r requirements.txt

# Copy env and set your Sarvam API key
cp .env.example .env
# Edit .env and set SARVAM_API_KEY if you have one

# Start the API server
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload 
```

The server starts at **http://localhost:8000**. The Qdrant index loads once at startup (~2 s).

Verify it's running:
```bash
curl http://localhost:8000/health
```

### 2. Next.js frontend

```bash
cd frontend

# Install dependencies (already done if node_modules exists)
npm install

# Development server
npm run dev
```

Open **http://localhost:3000** in your browser.

### Environment variables

| File | Variable | Description |
|---|---|---|
| `.env` | `SARVAM_API_KEY` | Sarvam STT key for real voice input |
| `.env` | `SARVAM_STT_MOCK` | Set to `1` only for offline/dev mock STT tests |
| `frontend/.env.local` | `NEXT_PUBLIC_API_URL` | FastAPI base URL (default: `http://localhost:8000`) |

---

## API endpoints

### `GET /health`

Returns system status, Qdrant info, STT config, and benchmark P50.

### `POST /query`

```json
{ "query": "मैनहट्टन परियोजना क्या थी?", "language": "hi-IN" }
```

Returns:
```json
{
  "query": "...",
  "answer": "...",
  "grounded": true,
  "blocked": false,
  "reason": "ok",
  "retrieved_chunks": 3,
  "sources": [{ "rank": 1, "chunk_id": "...", "text": "...", "score": 0.87, "vector_score": 0.85 }],
  "timings": { "embedding_ms": 17.2, "qdrant_ms": 18.5, "rerank_ms": 1.8, "answer_ms": 0.2, "total_ms": 37.7 }
}
```

### `POST /voice`

Multipart form: `file` (audio blob) + `language` (default `hi-IN`).

Returns same as `/query` plus `transcript`, `stt_latency_ms`, `stt_provider`, `rag_timings`.

---

## Benchmark — Phase 2 frozen baseline

| Metric | Value |
|---|---|
| Dataset | 1,000 Hindi queries, AI4Bharat MSMARCO-XI |
| Chunks indexed | 10,302 (VAST semantic chunking) |
| Recall@3 | **79.1%** |
| Recall@1 | 68.4% |
| Top-20 Recall | 87.3% |
| Grounded rate | **99.7%** |
| P50 latency (post-STT) | **38.1 ms** |
| P95 latency | **42 ms** |
| P100 latency | **55 ms** |
| 200 ms target | **PASS — 4.5× headroom** |

### Latency breakdown (P50)

```
Embedding (MiniLM inference):  17 ms   ← primary bottleneck
Qdrant (local SQLite search):  19 ms   ← secondary
Lexical reranker:               2 ms
Extractive answer:              0.2 ms
─────────────────────────────────────
Total P50:                     38 ms
```

Run the benchmark yourself:

```bash
python -m benchmarks.benchmark
```

---

## Project structure

```
HH-Goa-Rag/
│
├── app/                        Production application package
│   ├── pipeline.py             RAG core — embedding + Qdrant + reranking  [FROZEN]
│   ├── api.py                  FastAPI — /health /query /voice
│   ├── answer_generator.py     Extractive answer + guardrails              [FROZEN]
│   ├── retrieval/
│   │   ├── retrieve.py         Standalone retrieval benchmark utility
│   │   └── query_embedding.py  Query embedding utility (sentence-transformers)
│   ├── voice/
│   │   └── stt.py              Sarvam STT integration + explicit dev mock mode
│   └── generation/             (placeholder for future LLM integration)
│
├── ingestion/                  One-time data preparation scripts
│   ├── process_data.py         Raw data → passages
│   ├── chunk_data.py           Passages → chunks (VAST semantic chunking)
│   ├── embed_data.py           Chunks → Qdrant vectors
│   ├── validate_chunks.py      Chunk quality validation
│   ├── inspect_lengths.py      Passage length analysis
│   └── sample.py               Extract sample from Parquet dataset
│
├── benchmarks/                 Performance measurement
│   ├── benchmark.py            Official benchmark (p50/p95/p99/p100)
│   ├── benchmark_pipeline.py   Full post-STT pipeline benchmark
│   ├── rerank_benchmark.py     Reranking-specific benchmark
│   └── latency_breakdown.py    Low-level embedding + search latency
│
├── tests/
│   ├── test_api.py             FastAPI endpoint tests
│   └── test_imports.py         Import smoke tests
│
├── archive/                    Historical experiments (not active)
│   ├── old_experiments/        Step 9–20 calibration and diagnostic scripts
│   └── old_scripts/            Ad-hoc debug/analysis scripts
│
├── data/
│   ├── hindi_sample_1000.jsonl     Primary dataset (1,000 records)
│   ├── processed/
│   │   ├── passages_1000.jsonl     9,936 passages
│   │   └── chunks_1000.jsonl       10,302 chunks
│   └── qdrant/                     Local Qdrant SQLite vector store
│
├── requirements.txt
├── .env.example                    Environment variable template
├── STAGE1_BASELINE.md              Verified Phase 1+2 benchmark results
├── EXPERIMENT_LOG.md               Experiment history log
│
└── frontend/                   Next.js 16 + React 19 + Tailwind v4 UI
    ├── app/
    │   ├── globals.css
    │   ├── layout.tsx
    │   └── page.tsx
    ├── components/
    │   ├── Navbar.tsx, Hero.tsx, QuerySection.tsx
    │   ├── AnswerCard.tsx, BenchmarkSection.tsx
    │   ├── HowItWorks.tsx, Footer.tsx
    │   └── ui/button.tsx
    └── lib/
        ├── api.ts              Typed API client
        └── utils.ts
```

---

## Re-indexing (only if needed)

The Qdrant index at `data/qdrant/` is already built. Do **not** re-index unnecessarily — it takes ~5 minutes and regenerates the same result.

If you need to rebuild (e.g., after changing chunk strategy):

```bash
python -m ingestion.process_data    # re-generate passages
python -m ingestion.chunk_data      # re-generate chunks
python -m ingestion.embed_data      # re-build vector index
```

---

## Known limitations

1. **Extractive answers only** — no generative LLM. Answer quality is bounded by passage quality.
2. **Local Qdrant SQLite** — fine for Stage 1 latency targets. Production should use Qdrant Cloud.
3. **STT requires Sarvam credentials** — set `SARVAM_API_KEY` in `.env`. Use `SARVAM_STT_MOCK=1` only for offline/dev mock transcription.
4. **Recall@1 = 68.4%** — ~32% of queries don't surface the correct passage as rank-1. Typical for a 1K-record dataset.

---

## Stage roadmap

| Stage | Status | Description |
|---|---|---|
| Stage 1 | **COMPLETE** | Core RAG pipeline, benchmarked, optimized, <200ms |
| Phase 2 | **COMPLETE** | Retrieval diagnostics, baseline frozen |
| Phase 3 | **COMPLETE** | Next.js frontend |
| Stage 2 | Deferred | Dataset expansion (5K→200K), scaling, re-embedding |
