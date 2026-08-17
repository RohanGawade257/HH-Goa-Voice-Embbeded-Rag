/**
 * HH-Goa-Rag API client
 * All communication with the Python FastAPI backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Timings {
  embedding_ms: number;
  qdrant_ms: number;
  rerank_ms: number;
  answer_ms: number;
  total_ms: number;
}

export interface Source {
  rank: number;
  chunk_id: string | null;
  passage_id: string | null;
  query_id: string | null;
  text: string;
  score: number;
  vector_score: number;
}

export interface QueryResponse {
  query: string;
  answer: string;
  grounded: boolean;
  blocked: boolean;
  reason: string;
  retrieved_chunks: number;
  sources: Source[];
  timings: Timings;
}

export interface VoiceResponse extends QueryResponse {
  transcript: string;
  stt_latency_ms: number;
  stt_provider: string;
  rag_timings: Timings;
}

export interface HealthResponse {
  status: string;
  version: string;
  rag_engine: {
    status: string;
    embedding_model: string;
    embedding_dim: number;
    top_k_retrieval: number;
    top_k_final: number;
    load_time_ms: number;
  };
  qdrant: {
    status: string;
    collection: string;
    points: number;
  };
  stt: {
    provider: string;
    configured: boolean;
    language_default: string;
  };
  performance_target: {
    post_stt_p50_target_ms: number;
    last_benchmark_p50_ms: number;
    status: string;
  };
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export async function queryText(query: string, language = "hi-IN"): Promise<QueryResponse> {
  return post<QueryResponse>("/query", { query, language });
}

export async function queryVoice(audioBlob: Blob, language = "hi-IN"): Promise<VoiceResponse> {
  const form = new FormData();
  form.append("file", audioBlob, "recording.wav");
  form.append("language", language);
  const res = await fetch(`${API_BASE}/voice`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new ApiError(res.status, "Health check failed");
  return res.json();
}
