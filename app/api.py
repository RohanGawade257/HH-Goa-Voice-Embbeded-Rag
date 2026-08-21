"""
HH Goa 2026 — Voice RAG
FastAPI Application Layer

Endpoints:
    GET  /health     — System health and component status
    POST /query      — Text query → RAG answer
    POST /voice      — Audio upload → STT → RAG answer

Architecture:
    FastAPI → RAG Service → pipeline.RAGEngine

The RAG core (pipeline.py) is independent of this HTTP layer.
"""

import asyncio
import io
import json
import os
import time
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

from app.pipeline import RAGEngine
from app.config import (
    ANSWER_BACKEND,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_THREADS,
    MAX_CONTEXT_CHARS,
    MAX_NEW_TOKENS,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    QWEN_MODEL,
    QDRANT_COLLECTION,
    QDRANT_TOP_K,
    TOP_CONTEXT_CHUNKS,
    TOP_K_FINAL,
)
from app.voice.stt import STTConfigurationError, STTError, transcribe_audio, stt_status


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="HH Goa 2026 — Voice RAG",
    description=(
        "Voice-Enabled RAG system over AI4Bharat MSMARCO-XI Hindi dataset. "
        "Supports text and voice queries with <200ms post-STT latency."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend SPA)
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount(
    "/static",
    StaticFiles(directory=str(_static_dir)),
    name="static",
)


@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend SPA."""
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "HH Goa 2026 Voice RAG API — frontend not found"}


# ============================================================
# RAG ENGINE — SINGLETON
# Loaded once on startup, NOT on every request.
# ============================================================

_engine: Optional[RAGEngine] = None
_engine_load_ms: float = 0.0


def get_engine() -> RAGEngine:
    global _engine, _engine_load_ms

    if _engine is None:

        logger.info("Loading RAG engine...")

        start = time.perf_counter()

        _engine = RAGEngine()

        _engine_load_ms = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            f"RAG engine ready in {_engine_load_ms:.0f} ms"
        )

    return _engine


@app.on_event("startup")
async def startup_event():
    """Pre-load the RAG engine on startup."""
    get_engine()


@app.on_event("shutdown")
async def shutdown_event():
    """Release the persistent Qdrant and HTTP clients."""
    global _engine
    if _engine is not None:
        _engine.close()
        _engine = None


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class QueryRequest(BaseModel):

    query: str = Field(
        ...,
        description="The question to answer (Hindi or English)",
        min_length=1,
        max_length=2000,
        example="मैनहट्टन परियोजना क्या थी?",
    )

    language: Optional[str] = Field(
        default="hi-IN",
        description="Query language code (e.g. 'hi-IN', 'en-IN')",
    )


class TimingsResponse(BaseModel):
    embedding_ms: float
    qdrant_ms: float
    rerank_ms: float
    compression_ms: float = 0.0
    llm_ms: float = 0.0
    answer_ms: float
    total_ms: float


class SourceItem(BaseModel):
    rank: int
    chunk_id: Optional[str]
    passage_id: Optional[str]
    query_id: Optional[str]
    text: str
    language: Optional[str] = None
    score: float
    vector_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    blocked: bool
    reason: str
    retrieved_chunks: int
    sources: list[SourceItem]
    timings: TimingsResponse


class VoiceResponse(BaseModel):
    transcript: str
    stt_latency_ms: float
    stt_provider: str
    query: str
    answer: str
    grounded: bool
    blocked: bool
    reason: str
    retrieved_chunks: int
    sources: list[SourceItem]
    rag_timings: TimingsResponse


# ============================================================
# SSE HELPERS
# ============================================================

def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _build_sources(items: list) -> list[dict]:
    """Convert pipeline source items to serialisable dicts."""
    return [
        {
            "rank": item.get("rank", 0),
            "chunk_id": item.get("chunk_id"),
            "passage_id": item.get("passage_id"),
            "query_id": str(item.get("query_id", "")),
            "text": item.get("text", ""),
            "language": item.get("language"),
            "score": float(item.get("score", 0.0)),
            "vector_score": float(item.get("vector_score", 0.0)),
        }
        for item in items
    ]


# ============================================================
# GET /health
# ============================================================

@app.get("/health")
async def health():
    """
    Return system health status and component configuration.
    """

    engine = get_engine()

    # Verify Qdrant is accessible — use the engine's existing client
    # to avoid opening a second connection (SQLite file lock conflict).
    try:
        collection_info = engine.client.get_collection(
            QDRANT_COLLECTION
        )

        qdrant_status = {
            "status": "ok",
            "collection": QDRANT_COLLECTION,
            "points": collection_info.points_count,
        }

    except Exception as e:
        qdrant_status = {
            "status": "error",
            "error": str(e),
        }

    return {
        "status": "ok",
        "version": "1.0.0",

        "rag_engine": {
            "status": "loaded",
            "embedding_model": (
                EMBEDDING_MODEL
            ),
            "embedding_dim": EMBEDDING_DIM,
            "embedding_threads": EMBEDDING_THREADS,
            "top_k_retrieval": QDRANT_TOP_K,
            "top_k_final": TOP_K_FINAL,
            "top_context_chunks": TOP_CONTEXT_CHUNKS,
            "max_context_chars": MAX_CONTEXT_CHARS,
            "answer_backend": ANSWER_BACKEND,
            "llm_provider": LLM_PROVIDER,
            "llm_model": QWEN_MODEL,
            "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
            "llm_available": engine.answer_generator.available,
            "max_new_tokens": MAX_NEW_TOKENS,
            "load_time_ms": round(_engine_load_ms, 0),
        },

        "qdrant": qdrant_status,

        "stt": stt_status(),

        "performance_target": {
            "post_stt_p95_target_ms": 200,
            "requires_live_llm_benchmark": ANSWER_BACKEND in {"qwen", "qwen_api"},
        },
    }


# ============================================================
# GET /query/stream  — dual-answer SSE endpoint
# ============================================================

@app.get("/query/stream")
async def query_stream(q: str, language: str = "hi-IN"):
    """
    Stream the RAG pipeline answer as Server-Sent Events.

    Events emitted (in order):
        direct_answer  — extractive answer, ready after retrieve+compress (~50–70 ms)
        llm_answer     — Gemini LLM answer (or error if Gemini unavailable)
        sources        — retrieved source chunks
        timing         — full stage latency breakdown
        done           — stream complete

    The direct_answer is always emitted before llm_answer so the
    frontend can render immediately without waiting for the LLM.
    """

    engine = get_engine()

    async def event_generator():

        t0 = time.perf_counter()
        direct_sent = False

        try:
            if not q.strip():
                yield _sse_event("error", {"message": "empty_query"})
                yield _sse_event("done", {})
                return

            # --------------------------------------------------
            # STEP 1 — retrieve + compress  (runs in thread to
            # avoid blocking the event loop)
            # --------------------------------------------------
            try:
                direct_result, state = await asyncio.to_thread(
                    engine.process_dual, q, language
                )
            except Exception as exc:
                yield _sse_event("error", {"message": f"pipeline error: {exc}"})
                yield _sse_event("done", {})
                return

            time_to_direct_ms = (time.perf_counter() - t0) * 1000

            # --------------------------------------------------
            # STEP 2 — emit direct (extractive) answer
            # --------------------------------------------------
            yield _sse_event("direct_answer", {
                "answer": direct_result["answer"],
                "grounded": direct_result["grounded"],
                "blocked": direct_result["blocked"],
                "reason": direct_result["reason"],
                "retrieved_chunks": direct_result["retrieved_chunks"],
                "time_to_direct_ms": round(time_to_direct_ms, 2),
            })
            direct_sent = True  # direct answer is now delivered

            # --------------------------------------------------
            # STEP 3 — LLM answer  (Gemini via thread)
            # --------------------------------------------------
            llm_answer = None
            llm_error = None
            llm_ms = 0.0

            if state is not None and engine.answer_generator.available:
                try:
                    llm_start = time.perf_counter()
                    llm_result = await asyncio.to_thread(
                        engine.answer_generator.generate,
                        state["query"],
                        state["language_code"] or "",
                        state["compression_result"]["context"],
                    )
                    llm_ms = (time.perf_counter() - llm_start) * 1000
                    llm_answer = llm_result.get("answer", "")
                    llm_grounded = bool(llm_result.get("grounded", False))
                    llm_blocked = bool(llm_result.get("blocked", False))
                    llm_reason = llm_result.get("reason", "")
                except Exception as exc:
                    llm_error = str(exc)
            elif state is None:
                llm_error = "empty_query"
            else:
                llm_error = "llm_unavailable"

            time_to_llm_ms = (time.perf_counter() - t0) * 1000

            if llm_error:
                yield _sse_event("llm_answer", {
                    "answer": None,
                    "error": llm_error,
                    "time_to_llm_ms": round(time_to_llm_ms, 2),
                })
            else:
                yield _sse_event("llm_answer", {
                    "answer": llm_answer,
                    "grounded": llm_grounded,
                    "blocked": llm_blocked,
                    "reason": llm_reason,
                    "time_to_llm_ms": round(time_to_llm_ms, 2),
                })

            # --------------------------------------------------
            # STEP 4 — sources
            # --------------------------------------------------
            yield _sse_event("sources", {
                "sources": _build_sources(direct_result.get("sources", [])),
            })

            # --------------------------------------------------
            # STEP 5 — timing
            # --------------------------------------------------
            base_timings = direct_result.get("timings", {})
            yield _sse_event("timing", {
                "embedding_ms": base_timings.get("embedding_ms", 0),
                "qdrant_ms": base_timings.get("qdrant_ms", 0),
                "rerank_ms": base_timings.get("rerank_ms", 0),
                "compression_ms": base_timings.get("compression_ms", 0),
                "llm_ms": round(llm_ms, 2),
                "time_to_direct_ms": round(time_to_direct_ms, 2),
                "time_to_llm_ms": round(time_to_llm_ms, 2),
                "total_ms": round(time_to_llm_ms, 2),
            })

            yield _sse_event("done", {})

        except Exception as exc:
            # Top-level guard: catch any unexpected generator exception.
            # If direct_answer was already sent, the user already has an answer.
            # Always ensure the stream terminates cleanly.
            logger.error(f"/query/stream unexpected error: {exc}", exc_info=True)
            yield _sse_event("error", {
                "message": f"internal error: {exc}",
                "direct_delivered": direct_sent,
            })
            yield _sse_event("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# POST /query
# ============================================================

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a text query through the RAG pipeline.

    Returns a grounded answer extracted from the knowledge base,
    with timing breakdown and source attribution.

    Post-STT latency target: < 200 ms
    """

    engine = get_engine()

    result = engine.process(
        request.query,
        language=request.language,
    )

    # Build sources list
    sources = []
    for item in result.get("sources", []):
        sources.append(
            SourceItem(
                rank=item.get("rank", 0),
                chunk_id=item.get("chunk_id"),
                passage_id=item.get("passage_id"),
                query_id=str(item.get("query_id", "")),
                text=item.get("text", ""),
                language=item.get("language"),
                score=float(item.get("score", 0.0)),
                vector_score=float(item.get("vector_score", 0.0)),
            )
        )

    timings = result.get("timings", {})

    return QueryResponse(
        query=result.get("query", request.query),
        answer=result.get("answer", ""),
        grounded=bool(result.get("grounded", False)),
        blocked=bool(result.get("blocked", False)),
        reason=result.get("reason", ""),
        retrieved_chunks=int(result.get("retrieved_chunks", 0)),
        sources=sources,
        timings=TimingsResponse(
            embedding_ms=float(timings.get("embedding_ms", 0)),
            qdrant_ms=float(timings.get("qdrant_ms", 0)),
            rerank_ms=float(timings.get("rerank_ms", 0)),
            compression_ms=float(timings.get("compression_ms", 0)),
            llm_ms=float(timings.get("llm_ms", 0)),
            answer_ms=float(timings.get("answer_ms", 0)),
            total_ms=float(timings.get("total_ms", 0)),
        ),
    )


# ============================================================
# POST /voice
# ============================================================

@app.post("/voice", response_model=VoiceResponse)
async def voice(
    file: UploadFile = File(
        ...,
        description=(
            "Audio file (WAV recommended). "
            "The audio is transcribed with Sarvam STT, "
            "then passed to the RAG pipeline."
        ),
    ),
    language: str = Form("hi-IN"),
):
    """
    Process a voice query through STT → RAG pipeline.

    Flow:
        Audio upload → Sarvam STT → transcript → RAG → answer

    Note: STT latency is NOT included in the <200ms RAG target.
    The RAG pipeline starts after transcription is complete.

    If SARVAM_API_KEY is not set, mock STT is used (dev mode).
    """

    # --------------------------------------------------------
    # VALIDATE FILE
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded.",
        )

    allowed_types = {
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp4", "audio/ogg",
        "application/octet-stream",
    }

    content_type = file.content_type or ""

    # We accept any audio/* type plus octet-stream
    if not (
        content_type.startswith("audio/")
        or content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {content_type}. "
                   f"Please upload an audio file.",
        )

    # --------------------------------------------------------
    # READ AUDIO
    # --------------------------------------------------------

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # STT (Sarvam or mock)
    #
    # STT latency is NOT counted in the <200ms RAG target.
    # --------------------------------------------------------

    try:
        stt_result = transcribe_audio(
            audio_bytes=audio_bytes,
            language_code=language,
        )
    except STTConfigurationError as stt_exc:
        raise HTTPException(
            status_code=503,
            detail=stt_exc.to_safe_dict(),
        )
    except STTError as stt_exc:
        raise HTTPException(
            status_code=502,
            detail=stt_exc.to_safe_dict(),
        )

    transcript = stt_result.get("transcript", "").strip()

    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="STT produced an empty transcript. "
                   "Please check the audio quality.",
        )

    # --------------------------------------------------------
    # RAG (post-STT — this is the <200ms window)
    # --------------------------------------------------------

    engine = get_engine()

    rag_result = engine.process(
        transcript,
        language=language,
    )

    # Build sources list
    sources = []
    for item in rag_result.get("sources", []):
        sources.append(
            SourceItem(
                rank=item.get("rank", 0),
                chunk_id=item.get("chunk_id"),
                passage_id=item.get("passage_id"),
                query_id=str(item.get("query_id", "")),
                text=item.get("text", ""),
                language=item.get("language"),
                score=float(item.get("score", 0.0)),
                vector_score=float(item.get("vector_score", 0.0)),
            )
        )

    timings = rag_result.get("timings", {})

    return VoiceResponse(
        transcript=transcript,
        stt_latency_ms=float(
            stt_result.get("latency_ms", 0.0)
        ),
        stt_provider=stt_result.get("provider", "unknown"),
        query=rag_result.get("query", transcript),
        answer=rag_result.get("answer", ""),
        grounded=bool(rag_result.get("grounded", False)),
        blocked=bool(rag_result.get("blocked", False)),
        reason=rag_result.get("reason", ""),
        retrieved_chunks=int(
            rag_result.get("retrieved_chunks", 0)
        ),
        sources=sources,
        rag_timings=TimingsResponse(
            embedding_ms=float(
                timings.get("embedding_ms", 0)
            ),
            qdrant_ms=float(
                timings.get("qdrant_ms", 0)
            ),
            rerank_ms=float(
                timings.get("rerank_ms", 0)
            ),
            compression_ms=float(
                timings.get("compression_ms", 0)
            ),
            llm_ms=float(
                timings.get("llm_ms", 0)
            ),
            answer_ms=float(
                timings.get("answer_ms", 0)
            ),
            total_ms=float(
                timings.get("total_ms", 0)
            ),
        ),
    )
