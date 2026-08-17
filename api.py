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

import io
import os
import time
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from pipeline import RAGEngine
from stt import transcribe_audio, stt_status


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
    answer_ms: float
    total_ms: float


class SourceItem(BaseModel):
    rank: int
    chunk_id: Optional[str]
    passage_id: Optional[str]
    query_id: Optional[str]
    text: str
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
            "hh_goa_rag_hindi"
        )

        qdrant_status = {
            "status": "ok",
            "collection": "hh_goa_rag_hindi",
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
                "sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2"
            ),
            "embedding_dim": 384,
            "top_k_retrieval": 20,
            "top_k_final": 3,
            "load_time_ms": round(_engine_load_ms, 0),
        },

        "qdrant": qdrant_status,

        "stt": stt_status(),

        "performance_target": {
            "post_stt_p50_target_ms": 200,
            "last_benchmark_p50_ms": 38.1,
            "status": "PASS",
        },
    }


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

    result = engine.process(request.query)

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
    language: str = "hi-IN",
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

    stt_result = transcribe_audio(
        audio_bytes=audio_bytes,
        language_code=language,
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

    rag_result = engine.process(transcript)

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
            answer_ms=float(
                timings.get("answer_ms", 0)
            ),
            total_ms=float(
                timings.get("total_ms", 0)
            ),
        ),
    )
