"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Search, Mic, MicOff, Loader2, X, Send } from "lucide-react";
import clsx from "clsx";
import { queryText, queryVoice, QueryResponse, ApiError } from "@/lib/api";
import AnswerCard from "./AnswerCard";

const PLACEHOLDER_QUERIES = [
  "मैनहट्टन परियोजना क्या थी?",
  "डीएनए क्या है और यह कैसे काम करता है?",
  "सौर ऊर्जा के क्या फायदे हैं?",
  "पृथ्वी का वायुमंडल कैसे बना?",
];

type State = "idle" | "loading" | "success" | "error";
type InputMode = "text" | "voice";

export default function QuerySection() {
  const [query, setQuery] = useState("");
  const [state, setState] = useState<State>("idle");
  const [mode, setMode] = useState<InputMode>("text");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Rotate placeholders
  useEffect(() => {
    const id = setInterval(() => {
      setPlaceholderIdx((i) => (i + 1) % PLACEHOLDER_QUERIES.length);
    }, 3500);
    return () => clearInterval(id);
  }, []);

  const handleSubmit = useCallback(async () => {
    const q = query.trim();
    if (!q || state === "loading") return;
    setState("loading");
    setError("");
    setResult(null);
    try {
      const data = await queryText(q);
      setResult(data);
      setState("success");
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? `API error (${e.status}): ${e.message}`
          : "Failed to reach the API. Is the Python server running?";
      setError(msg);
      setState("error");
    }
  }, [query, state]);

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
        setResult(null);
        try {
          const data = await queryVoice(blob);
          setTranscript(data.transcript);
          setQuery(data.transcript);
          // VoiceResponse uses `rag_timings`; AnswerCard expects `timings`.
          // Normalise before storing so the shape is always QueryResponse.
          setResult({ ...data, timings: data.rag_timings ?? data.timings });
          setState("success");
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

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
  };

  const clearAll = () => {
    setQuery("");
    setResult(null);
    setState("idle");
    setError("");
    setTranscript("");
    inputRef.current?.focus();
  };

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
            <p
              className="text-sm mt-1"
              style={{ color: "var(--color-ink-muted)" }}
            >
              Hindi or English — voice or text
            </p>
          </div>
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
            className={clsx(
              "relative transition-all duration-200",
              state === "loading" && "border-animated"
            )}
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
                  disabled={state === "loading"}
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
                  <span
                    className="text-xs"
                    style={{ color: "var(--color-ink-muted)" }}
                  >
                    {query.length > 0
                      ? `${query.length} chars · Enter to submit`
                      : "Enter ↵ or click Send"}
                  </span>
                  <div className="flex items-center gap-2">
                    {query.length > 0 && (
                      <button
                        onClick={clearAll}
                        className="p-1 opacity-40 hover:opacity-80 transition-opacity"
                        title="Clear"
                      >
                        <X size={14} />
                      </button>
                    )}
                    <button
                      onClick={handleSubmit}
                      disabled={!query.trim() || state === "loading"}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50 transition-all duration-150 hover:opacity-90 active:scale-95"
                      style={{ background: "var(--accent)" }}
                    >
                      {state === "loading" ? (
                        <Loader2 size={13} className="animate-spin" />
                      ) : (
                        <Send size={13} />
                      )}
                      Send
                    </button>
                  </div>
                </div>
              </>
            ) : (
              /* Voice mode */
              <div className="flex flex-col items-center justify-center py-10 gap-4">
                {transcript && (
                  <p
                    className="text-sm text-devanagari px-6 text-center mb-2"
                    style={{ color: "var(--color-ink-secondary)" }}
                  >
                    "{transcript}"
                  </p>
                )}
                <motion.button
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={state === "loading"}
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
                  {state === "loading" ? (
                    <Loader2 size={28} className="text-white animate-spin" />
                  ) : isRecording ? (
                    <MicOff size={28} className="text-white" />
                  ) : (
                    <Mic size={28} className="text-white" />
                  )}
                </motion.button>
                <p
                  className="text-xs font-semibold uppercase tracking-widest"
                  style={{ color: "var(--color-ink-muted)" }}
                >
                  {isRecording
                    ? "Recording — click to stop"
                    : state === "loading"
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
              style={{
                background: "rgba(204,34,0,0.08)",
                border: "2px solid #CC2200",
                color: "#CC2200",
              }}
            >
              <X size={16} className="shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold mb-0.5">Query failed</p>
                <p className="opacity-80 text-xs">{error}</p>
              </div>
              <button
                onClick={clearAll}
                className="ml-auto opacity-60 hover:opacity-100"
              >
                <X size={14} />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Loading skeleton */}
        <AnimatePresence>
          {state === "loading" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-3"
            >
              {[120, 96, 80].map((w, i) => (
                <div
                  key={i}
                  className="h-4 pulse-accent rounded-sm"
                  style={{
                    width: `${w}%` ,
                    maxWidth: `${w}%`,
                    background: "var(--color-panel)",
                  }}
                />
              ))}
              <div className="h-3" />
              {[100, 88].map((w, i) => (
                <div
                  key={i}
                  className="h-3 pulse-accent rounded-sm"
                  style={{
                    width: `${w}%`,
                    maxWidth: `${w}%`,
                    background: "var(--color-panel)",
                    animationDelay: `${i * 0.1}s`,
                  }}
                />
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Result */}
        <AnimatePresence>
          {state === "success" && result && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.45, ease: [0.23, 1, 0.32, 1] }}
            >
              <AnswerCard result={result} onClear={clearAll} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
