/**
 * HH-Goa-Rag API client
 * All communication with the Python FastAPI backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Language capability table ────────────────────────────────
// ragSupported  — language is in the Qdrant multilingual corpus
// sttSupported  — language is supported by the Sarvam STT API
//
// These are INDEPENDENT capabilities. Do NOT conflate them.
//
// Telugu: STT-supported but NOT in the RAG corpus.
// Nepali/Sanskrit/Urdu: in RAG corpus but NOT STT-supported by Sarvam.
// English: STT-supported (voice input), but no RAG passages indexed.

export interface Language {
  code: string;        // BCP-47, e.g. "hi-IN"
  name: string;        // English display name
  nativeName: string;  // Script-native name
  ragSupported: boolean;
  sttSupported: boolean;
}

export const LANGUAGES: Language[] = [
  { code: "hi-IN", name: "Hindi",      nativeName: "हिंदी",      ragSupported: true,  sttSupported: true  },
  { code: "bn-IN", name: "Bengali",    nativeName: "বাংলা",      ragSupported: true,  sttSupported: true  },
  { code: "gu-IN", name: "Gujarati",   nativeName: "ગુજરાતી",    ragSupported: true,  sttSupported: true  },
  { code: "kn-IN", name: "Kannada",    nativeName: "ಕನ್ನಡ",      ragSupported: true,  sttSupported: true  },
  { code: "ml-IN", name: "Malayalam",  nativeName: "മലയാളം",     ragSupported: true,  sttSupported: true  },
  { code: "mr-IN", name: "Marathi",    nativeName: "मराठी",      ragSupported: true,  sttSupported: true  },
  { code: "or-IN", name: "Odia",       nativeName: "ଓଡ଼ିଆ",      ragSupported: true,  sttSupported: true  },
  { code: "pa-IN", name: "Punjabi",    nativeName: "ਪੰਜਾਬੀ",     ragSupported: true,  sttSupported: true  },
  { code: "ta-IN", name: "Tamil",      nativeName: "தமிழ்",      ragSupported: true,  sttSupported: true  },
  { code: "as-IN", name: "Assamese",   nativeName: "অসমীয়া",    ragSupported: true,  sttSupported: false },
  { code: "ne-IN", name: "Nepali",     nativeName: "नेपाली",     ragSupported: true,  sttSupported: false },
  { code: "sa-IN", name: "Sanskrit",   nativeName: "संस्कृतम्",  ragSupported: true,  sttSupported: false },
  { code: "ur-IN", name: "Urdu",       nativeName: "اردو",       ragSupported: true,  sttSupported: false },
  // Telugu: STT-supported but not in RAG corpus — available for voice transcription only
  { code: "te-IN", name: "Telugu",     nativeName: "తెలుగు",     ragSupported: false, sttSupported: true  },
  // English: voice input works, but no RAG passages indexed
  { code: "en-IN", name: "English",    nativeName: "English",    ragSupported: false, sttSupported: true  },
];

/** Lookup a language entry by BCP-47 code. Falls back to Hindi. */
export function getLanguage(code: string): Language {
  return LANGUAGES.find((l) => l.code === code) ?? LANGUAGES[0];
}

export interface Timings {
  embedding_ms: number;
  qdrant_ms: number;
  rerank_ms: number;
  answer_ms: number;
  total_ms: number;
}

// ── Dual-answer / SSE types ──────────────────────────────────

export interface StreamTimings {
  embedding_ms: number;
  qdrant_ms: number;
  rerank_ms: number;
  compression_ms: number;
  llm_ms: number;
  time_to_direct_ms: number;
  time_to_llm_ms: number;
  total_ms: number;
}

export interface DirectAnswerEvent {
  answer: string;
  grounded: boolean;
  blocked: boolean;
  reason: string;
  retrieved_chunks: number;
  time_to_direct_ms: number;
}

export interface LlmAnswerEvent {
  answer: string | null;
  grounded?: boolean;
  blocked?: boolean;
  reason?: string;
  error?: string;
  time_to_llm_ms: number;
}

export interface StreamCallbacks {
  onDirectAnswer: (data: DirectAnswerEvent) => void;
  onLlmAnswer: (data: LlmAnswerEvent) => void;
  onSources: (sources: Source[]) => void;
  onTiming: (timing: StreamTimings) => void;
  onError: (message: string) => void;
  onDone: () => void;
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

/**
 * Open the /query/stream SSE endpoint and call the appropriate
 * callback as each event arrives.
 *
 * Returns an AbortController so the caller can cancel the stream.
 */
export function queryStream(
  query: string,
  callbacks: StreamCallbacks,
  language = "hi-IN"
): AbortController {
  const controller = new AbortController();
  const url = `${API_BASE}/query/stream?q=${encodeURIComponent(query)}&language=${encodeURIComponent(language)}`;

  (async () => {
    let res: Response;
    try {
      res = await fetch(url, { signal: controller.signal });
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        callbacks.onError("Failed to reach the API. Is the Python server running?");
        callbacks.onDone();
      }
      return;
    }

    if (!res.ok) {
      const detail = await res.text().catch(() => res.statusText);
      callbacks.onError(`API error (${res.status}): ${detail}`);
      callbacks.onDone();
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      callbacks.onError("No response body from SSE stream.");
      callbacks.onDone();
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let doneCalled = false;

    const callDoneOnce = () => {
      if (!doneCalled) {
        doneCalled = true;
        callbacks.onDone();
      }
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const messages = buffer.split("\n\n");
        buffer = messages.pop() ?? "";

        for (const msg of messages) {
          if (!msg.trim()) continue;
          const eventMatch = msg.match(/^event: (\w+)/m);
          const dataMatch = msg.match(/^data: (.+)$/m);
          if (!eventMatch || !dataMatch) continue;

          const event = eventMatch[1];
          let data: Record<string, unknown>;
          try {
            data = JSON.parse(dataMatch[1]);
          } catch {
            continue;
          }

          switch (event) {
            case "direct_answer":
              callbacks.onDirectAnswer(data as unknown as DirectAnswerEvent);
              break;
            case "llm_answer":
              callbacks.onLlmAnswer(data as unknown as LlmAnswerEvent);
              break;
            case "sources":
              callbacks.onSources((data.sources as Source[]) ?? []);
              break;
            case "timing":
              callbacks.onTiming(data as unknown as StreamTimings);
              break;
            case "error":
              callbacks.onError((data.message as string) ?? "Unknown error");
              // Always follow error with done so the UI never gets stuck
              callDoneOnce();
              break;
            case "done":
              callDoneOnce();
              break;
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        callbacks.onError("Stream interrupted.");
        callDoneOnce();
      }
    }
  })();

  return controller;
}
