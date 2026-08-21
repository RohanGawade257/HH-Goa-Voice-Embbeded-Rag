"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Search, Mic, MicOff, Loader2, X, Send, MicOff as VoiceOff } from "lucide-react";
import clsx from "clsx";
import {
  queryStream,
  queryVoice,
  ApiError,
  Source,
  StreamTimings,
  DirectAnswerEvent,
  LlmAnswerEvent,
  LANGUAGES,
  Language,
} from "@/lib/api";
import DualAnswerCard from "./AnswerCard";

// ── Placeholder queries spanning multiple languages ──────────
const PLACEHOLDER_QUERIES = [
  "मैनहट्टन परियोजना क्या थी?",               // Hindi
  "ম্যানহাটন প্রকল্পের তাৎক্ষণিক প্রভাব কী?", // Bengali
  "மன்ஹாட்டன் திட்டத்தின் விளைவு என்ன?",     // Tamil
  "मॅनहॅटन प्रकल्पाचा परिणाम काय झाला?",     // Marathi
  "ਮੈਨਹੈਟਨ ਪ੍ਰੋਜੈਕਟ ਦਾ ਤੁਰੰਤ ਪ੍ਰਭਾਵ ਕੀ ਸੀ?",  // Punjabi
];

type State = "idle" | "loading" | "direct" | "complete" | "error";
type InputMode = "text" | "voice";

export interface DualResult {
  query: string;
  directAnswer: DirectAnswerEvent | null;
  llmAnswer: LlmAnswerEvent | null;
  sources: Source[];
  timing: StreamTimings | null;
}

interface QuerySectionProps {
  /** Pre-fill the query textarea (from Hero example click) */
  defaultQuery?: string;
  /** Pre-select a language (from Hero example click) */
  defaultLanguage?: string;
}

// RAG-supported languages shown in the selector
const RAG_LANGUAGES = LANGUAGES.filter((l) => l.ragSupported);

export default function QuerySection({ defaultQuery, defaultLanguage }: QuerySectionProps) {
  const [query, setQuery] = useState(defaultQuery ?? "");
  const [state, setState] = useState<State>("idle");
  const [mode, setMode] = useState<InputMode>("text");
  const [dualResult, setDualResult] = useState<DualResult | null>(null);
  const [error, setError] = useState<string>("");
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");

  // Language selection — default Hindi, or override from Hero
  const defaultLangEntry = LANGUAGES.find((l) => l.code === (defaultLanguage ?? "hi-IN")) ?? RAG_LANGUAGES[0];
  const [selectedLang, setSelectedLang] = useState<Language>(defaultLangEntry);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  // Apply defaults from Hero when they change (example query click)
  useEffect(() => {
    if (defaultQuery) setQuery(defaultQuery);
    if (defaultLanguage) {
      const lang = LANGUAGES.find((l) => l.code === defaultLanguage);
      if (lang) setSelectedLang(lang);
    }
  }, [defaultQuery, defaultLanguage]);

  // Rotate placeholders
  useEffect(() => {
    const id = setInterval(() => {
      setPlaceholderIdx((i) => (i + 1) % PLACEHOLDER_QUERIES.length);
    }, 3500);
    return () => clearInterval(id);
  }, []);

  // Cancel any in-flight stream on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  const handleSubmit = useCallback(() => {
    const q = query.trim();
    if (!q || state === "loading" || state === "direct") return;

    abortRef.current?.abort();
    setState("loading");
    setError("");
    setDualResult(null);

    const draft: DualResult = {
      query: q,
      directAnswer: null,
      llmAnswer: null,
      sources: [],
      timing: null,
    };

    const controller = queryStream(q, {
      onDirectAnswer(data) {
        draft.directAnswer = data;
        setDualResult({ ...draft });
        setState("direct");
      },
      onLlmAnswer(data) {
        draft.llmAnswer = data;
        setDualResult({ ...draft });
      },
      onSources(sources) {
        draft.sources = sources;
        setDualResult({ ...draft });
      },
      onTiming(timing) {
        draft.timing = timing;
        setDualResult({ ...draft });
      },
      onError(message) {
        setError(message);
        setState("error");
      },
      onDone() {
        setState("complete");
      },
    }, selectedLang.code);

    abortRef.current = controller;
  }, [query, state, selectedLang]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Voice recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setIsRecording(false);
        setState("loading");
        setError("");
        setDualResult(null);
        try {
          const data = await queryVoice(blob, selectedLang.code);
          const timings = data.rag_timings ?? data.timings;
          const directAnswer = data.direct_answer ?? {
            answer: data.answer,
            grounded: data.grounded,
            blocked: data.blocked,
            reason: data.reason,
            retrieved_chunks: data.retrieved_chunks,
            time_to_direct_ms: timings?.total_ms ?? 0,
          };
          const llmAnswer = data.llm_answer ?? {
            answer: null,
            error: "AI answer was not returned by the voice endpoint.",
            time_to_llm_ms: timings?.total_ms ?? 0,
          };

          // Auto-update the selected language to match the detected spoken
          // language returned by the backend. This ensures subsequent text
          // queries use the correct language and the UI badge reflects reality.
          if (data.answer_language) {
            // answer_language is a BCP-47 code like "mr-IN" or "hi-IN"
            const detectedLang = LANGUAGES.find((l) => l.code === data.answer_language);
            if (detectedLang && detectedLang.code !== selectedLang.code) {
              setSelectedLang(detectedLang);
              console.info("VOICE RESPONSE language auto-switched:", {
                from: selectedLang.code,
                to: detectedLang.code,
              });
            }
          }

          console.info("VOICE RESPONSE direct_answer received", {
            requestedLanguage: data.requested_language_code ?? selectedLang.code,
            sttLanguage: data.language_code,
            answerLanguage: data.answer_language,
            retrievedChunks: directAnswer.retrieved_chunks,
          });
          console.info("VOICE RESPONSE llm_answer received", {
            hasAnswer: Boolean(llmAnswer.answer),
            hasError: Boolean(llmAnswer.error),
            timeToLlmMs: llmAnswer.time_to_llm_ms,
          });
          setTranscript(data.transcript);
          setQuery(data.transcript);
          setDualResult({
            query: data.query,
            directAnswer,
            llmAnswer,
            sources: data.sources,
            timing: timings
              ? {
                  embedding_ms: timings.embedding_ms,
                  qdrant_ms: timings.qdrant_ms,
                  rerank_ms: timings.rerank_ms,
                  compression_ms: timings.compression_ms ?? 0,
                  llm_ms: timings.llm_ms ?? 0,
                  time_to_direct_ms: directAnswer.time_to_direct_ms,
                  time_to_llm_ms: llmAnswer.time_to_llm_ms,
                  total_ms: timings.total_ms,
                }
              : null,
          });
          setState("complete");
        } catch (e) {
          setError(
            e instanceof ApiError
              ? `STT/API error (${e.status}): ${e.message}`
              : "Voice query failed. Check the API server."
          );
          setState("error");
        }
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setIsRecording(true);
    } catch {
      setError("Microphone access denied.");
    }
  };

  const stopRecording = () => { mediaRecorderRef.current?.stop(); };

  const clearAll = () => {
    abortRef.current?.abort();
    setQuery("");
    setDualResult(null);
    setState("idle");
    setError("");
    setTranscript("");
    inputRef.current?.focus();
  };

  const isActive = state === "loading" || state === "direct";

  return (
    <section
      id="query"
      className="relative py-24 px-4"
      style={{ background: "var(--background)" }}
    >
      <div className="max-w-4xl mx-auto">
        {/* Section label */}
        <div className="flex items-center gap-4 mb-10">
          <span
            className="font-black text-7xl leading-none select-none"
            style={{ color: "var(--color-panel)" }}
          >
            02
          </span>
          <div>
            <h2
              className="font-black leading-none"
              style={{ fontSize: "clamp(1.6rem, 4vw, 2.8rem)", letterSpacing: "-0.03em" }}
            >
              Ask the knowledge base
            </h2>
            <p className="text-sm mt-1" style={{ color: "var(--color-ink-muted)" }}>
              13 Indian languages — voice or text
            </p>
          </div>
        </div>

        {/* ── Language selector ──────────────────────────────── */}
        <div className="mb-5">
          <p
            className="text-[11px] uppercase tracking-widest font-semibold mb-2"
            style={{ color: "var(--color-ink-muted)" }}
          >
            {mode === "voice" ? "Spoken language" : "Query language"}
          </p>
          <div
            className="flex flex-wrap gap-1.5"
            role="listbox"
            aria-label={mode === "voice" ? "Select spoken language" : "Select query language"}
          >
            {RAG_LANGUAGES.map((lang) => {
              const isSelected = selectedLang.code === lang.code;
              return (
                <button
                  key={lang.code}
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => setSelectedLang(lang)}
                  disabled={isActive}
                  className={clsx(
                    "px-2.5 py-1 text-xs font-semibold transition-all duration-100 disabled:opacity-50",
                    "hover:opacity-90 active:scale-95"
                  )}
                  style={{
                    background: isSelected ? "var(--foreground)" : "transparent",
                    border: "1.5px solid var(--foreground)",
                    color: isSelected ? "var(--background)" : "var(--foreground)",
                    opacity: isSelected ? 1 : 0.55,
                  }}
                  title={lang.name}
                >
                  {lang.nativeName}
                </button>
              );
            })}
          </div>
          {/* STT availability note */}
          {!selectedLang.sttSupported && (
            <p
              className="text-[11px] mt-2 flex items-center gap-1.5"
              style={{ color: "var(--color-ink-muted)" }}
            >
              <VoiceOff size={11} />
              Voice input not available for {selectedLang.name} — text only
            </p>
          )}
        </div>

        {/* Mode toggle */}
        <div className="flex gap-2 mb-6">
          {(["text", "voice"] as InputMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={clsx(
                "flex items-center gap-2 px-4 py-2 text-xs font-semibold uppercase tracking-widest transition-all duration-150",
                mode === m
                  ? "text-white"
                  : "text-[var(--foreground)] opacity-50 hover:opacity-80"
              )}
              style={{
                background: mode === m ? "var(--foreground)" : "transparent",
                border: "2px solid var(--foreground)",
              }}
            >
              {m === "text" ? <Search size={13} /> : <Mic size={13} />}
              {m}
            </button>
          ))}
        </div>

        {/* Input area */}
        <div className="relative mb-6">
          <div
            className={clsx("relative transition-all duration-200", isActive && "border-animated")}
            style={{ border: "2px solid var(--foreground)" }}
          >
            {mode === "text" ? (
              <>
                <textarea
                  ref={inputRef}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={PLACEHOLDER_QUERIES[placeholderIdx]}
                  rows={3}
                  disabled={isActive}
                  className={clsx(
                    "w-full resize-none bg-transparent px-5 pt-4 pb-12",
                    "text-base text-devanagari leading-relaxed",
                    "placeholder:text-[var(--color-ink-muted)] placeholder:opacity-60",
                    "focus:outline-none disabled:opacity-60"
                  )}
                  style={{ fontFamily: "'Noto Sans Devanagari', 'Inter', sans-serif" }}
                />
                {/* Bottom bar */}
                <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-4 py-2.5 border-t border-[var(--border)]">
                  <span className="text-xs" style={{ color: "var(--color-ink-muted)" }}>
                    {query.length > 0
                      ? `${query.length} chars · Enter to submit`
                      : "Enter ↵ or click Send"}
                  </span>
                  <div className="flex items-center gap-2">
                    {query.length > 0 && (
                      <button onClick={clearAll} className="p-1 opacity-40 hover:opacity-80 transition-opacity" title="Clear">
                        <X size={14} />
                      </button>
                    )}
                    <button
                      onClick={handleSubmit}
                      disabled={!query.trim() || isActive}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50 transition-all duration-150 hover:opacity-90 active:scale-95"
                      style={{ background: "var(--accent)" }}
                    >
                      {state === "loading" ? <Loader2 size={13} className="animate-spin" /> : <Send size={13} />}
                      Send
                    </button>
                  </div>
                </div>
              </>
            ) : (
              /* Voice mode */
              <div className="flex flex-col items-center justify-center py-10 gap-4">
                {transcript && (
                  <p className="text-sm text-devanagari px-6 text-center mb-2" style={{ color: "var(--color-ink-secondary)" }}>
                    "{transcript}"
                  </p>
                )}

                {/* Voice not supported for this language */}
                {!selectedLang.sttSupported ? (
                  <div
                    className="flex flex-col items-center gap-2 py-4 px-6 text-center"
                    style={{ color: "var(--color-ink-muted)" }}
                  >
                    <VoiceOff size={32} style={{ opacity: 0.35 }} />
                    <p className="text-sm font-semibold">
                      Voice not available for {selectedLang.name}
                    </p>
                    <p className="text-xs opacity-70">
                      Switch to text mode or choose a language with voice support.
                    </p>
                  </div>
                ) : (
                  <motion.button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isActive}
                    className="relative flex items-center justify-center rounded-full transition-all disabled:opacity-50"
                    style={{
                      width: 72,
                      height: 72,
                      background: isRecording ? "#CC2200" : "var(--foreground)",
                    }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {isRecording && (
                      <motion.span
                        className="absolute inset-0 rounded-full"
                        style={{ background: "#CC2200" }}
                        animate={{ scale: [1, 1.4], opacity: [0.4, 0] }}
                        transition={{ duration: 1.2, repeat: Infinity }}
                      />
                    )}
                    {isActive ? (
                      <Loader2 size={28} className="text-white animate-spin" />
                    ) : isRecording ? (
                      <MicOff size={28} className="text-white" />
                    ) : (
                      <Mic size={28} className="text-white" />
                    )}
                  </motion.button>
                )}

                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--color-ink-muted)" }}>
                  {!selectedLang.sttSupported
                    ? "Text input only"
                    : isRecording
                    ? "Recording — click to stop"
                    : isActive
                    ? "Processing…"
                    : "Click to record"}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Error state */}
        <AnimatePresence>
          {state === "error" && error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-start gap-3 p-4 mb-6 text-sm"
              style={{ background: "rgba(204,34,0,0.08)", border: "2px solid #CC2200", color: "#CC2200" }}
            >
              <X size={16} className="shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold mb-0.5">Query failed</p>
                <p className="opacity-80 text-xs">{error}</p>
              </div>
              <button onClick={clearAll} className="ml-auto opacity-60 hover:opacity-100">
                <X size={14} />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Loading skeleton — only before direct answer arrives */}
        <AnimatePresence>
          {state === "loading" && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-3">
              {[120, 96, 80].map((w, i) => (
                <div key={i} className="h-4 pulse-accent rounded-sm"
                  style={{ width: `${w}%`, maxWidth: `${w}%`, background: "var(--color-panel)" }} />
              ))}
              <div className="h-3" />
              {[100, 88].map((w, i) => (
                <div key={i} className="h-3 pulse-accent rounded-sm"
                  style={{ width: `${w}%`, maxWidth: `${w}%`, background: "var(--color-panel)", animationDelay: `${i * 0.1}s` }} />
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Dual result — renders as soon as directAnswer arrives */}
        <AnimatePresence>
          {(state === "direct" || state === "complete") && dualResult && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.45, ease: [0.23, 1, 0.32, 1] }}
            >
              <DualAnswerCard result={dualResult} onClear={clearAll} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
