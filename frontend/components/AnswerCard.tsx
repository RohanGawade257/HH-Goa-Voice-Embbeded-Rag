"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Layers,
  BarChart2,
  RefreshCw,
  Loader2,
  Sparkles,
  Zap,
} from "lucide-react";
import clsx from "clsx";
import type { Source } from "@/lib/api";
import type { DualResult } from "./QuerySection";

// ── Friendly error message map ───────────────────────────────
function getErrorMessage(code: string | undefined | null): string {
  if (!code) return "AI answer unavailable.";
  switch (code) {
    case "llm_unavailable":
      return "AI answer requires a configured API key (GEMINI_API_KEY).";
    case "empty_query":
      return "No query was provided.";
    case "gemini_disabled":
      return "Gemini backend is not enabled on this server.";
    case "missing_context":
      return "No relevant context was found to generate an AI answer.";
    default:
      return code;
  }
}

// ── Legacy prop type kept for backward compat (used by voice path) ──────────
// The component accepts DualResult (primary) only now.
interface AnswerCardProps {
  result: DualResult;
  onClear: () => void;
}

export default function AnswerCard({ result, onClear }: AnswerCardProps) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [timingsOpen, setTimingsOpen] = useState(false);

  const { directAnswer, llmAnswer, sources, timing } = result;

  // LLM answer section visibility: show if pending OR received
  const llmPending = directAnswer !== null && llmAnswer === null;
  const llmFailed = llmAnswer?.error !== undefined && llmAnswer.error !== null;
  const llmSuccess = llmAnswer !== null && !llmFailed && llmAnswer.answer;

  const directIsGrounded = directAnswer?.grounded && !directAnswer?.blocked;

  return (
    <div className="space-y-4">

      {/* ── DIRECT ANSWER (fast extractive) ───────────────── */}
      {directAnswer && (
        <div
          className="p-6 lg:p-8"
          style={{
            background: "var(--color-panel-dark)",
            border: "2px solid var(--foreground)",
          }}
        >
          {/* Header */}
          <div className="flex items-start justify-between gap-4 mb-5">
            <div className="flex items-center gap-2.5">
              <Zap size={16} style={{ color: "var(--accent)" }} />
              <span
                className="text-xs font-semibold uppercase tracking-widest"
                style={{ color: "var(--accent)" }}
              >
                Direct answer
              </span>
              <span
                className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                style={{ background: "rgba(232,93,4,0.14)", color: "var(--accent)" }}
              >
                {directAnswer.time_to_direct_ms.toFixed(0)} ms
              </span>
            </div>
            <button
              onClick={onClear}
              className="flex items-center gap-1.5 text-xs opacity-40 hover:opacity-80 transition-opacity"
              style={{ color: "#F5F3EE" }}
            >
              <RefreshCw size={12} />
              New query
            </button>
          </div>

          {/* Query echo */}
          <p
            className="text-xs mb-4 text-devanagari"
            style={{ color: "rgba(245,243,238,0.38)", lineHeight: 1.8 }}
          >
            Query: {result.query}
          </p>

          {/* Answer text */}
          <p
            className="text-devanagari leading-loose"
            style={{
              fontSize: "clamp(1rem, 2vw, 1.2rem)",
              color: "#F5F3EE",
              lineHeight: 2,
            }}
          >
            {directAnswer.answer}
          </p>

          {/* Meta chips */}
          <div
            className="flex flex-wrap items-center gap-3 mt-6 pt-5 border-t"
            style={{ borderColor: "rgba(245,243,238,0.10)" }}
          >
            <MetaChip icon={<Layers size={12} />} label={`${directAnswer.retrieved_chunks} sources`} />
            {directIsGrounded && (
              <MetaChip icon={<CheckCircle2 size={12} />} label="Grounded" accent />
            )}
            {!directIsGrounded && directAnswer.reason && (
              <MetaChip
                icon={<AlertTriangle size={12} />}
                label={directAnswer.reason.replace(/_/g, " ")}
                warn
              />
            )}
          </div>
        </div>
      )}

      {/* ── AI ANSWER (Gemini LLM) ─────────────────────────── */}
      <AnimatePresence>
        {/* Pending: shimmer while LLM is running */}
        {llmPending && (
          <motion.div
            key="llm-pending"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="p-6 lg:p-8"
            style={{
              border: "2px dashed var(--border)",
              background: "var(--color-panel)",
            }}
          >
            <div className="flex items-center gap-2.5 mb-4">
              <Loader2 size={15} className="animate-spin" style={{ color: "#7c5cd8" }} />
              <span
                className="text-xs font-semibold uppercase tracking-widest"
                style={{ color: "#7c5cd8" }}
              >
                AI answer
              </span>
              <span
                className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                style={{ background: "rgba(124,92,216,0.14)", color: "#7c5cd8" }}
              >
                generating…
              </span>
            </div>
            <div className="space-y-2">
              {[90, 75, 55].map((w, i) => (
                <div
                  key={i}
                  className="h-3 pulse-accent rounded-sm"
                  style={{
                    width: `${w}%`,
                    background: "var(--border)",
                    animationDelay: `${i * 0.12}s`,
                  }}
                />
              ))}
            </div>
          </motion.div>
        )}

        {/* Failed — shown as a proper card, not a tiny chip */}
        {llmFailed && llmAnswer && (
          <motion.div
            key="llm-error"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="p-6 lg:p-8"
            style={{
              background: "var(--color-panel)",
              border: "2px solid var(--border)",
            }}
          >
            <div className="flex items-center gap-2.5 mb-4">
              <Sparkles size={16} style={{ color: "#7c5cd8", opacity: 0.4 }} />
              <span
                className="text-xs font-semibold uppercase tracking-widest"
                style={{ color: "#7c5cd8", opacity: 0.6 }}
              >
                AI answer
              </span>
              <span
                className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                style={{ background: "rgba(124,92,216,0.10)", color: "#7c5cd8", opacity: 0.7 }}
              >
                {llmAnswer.time_to_llm_ms.toFixed(0)} ms
              </span>
            </div>
            <div className="flex items-center gap-2 text-sm" style={{ color: "var(--color-ink-muted)" }}>
              <AlertTriangle size={14} className="shrink-0" />
              <span>{getErrorMessage(llmAnswer.error)}</span>
            </div>
          </motion.div>
        )}

        {/* Success */}
        {llmSuccess && llmAnswer && (
          <motion.div
            key="llm-success"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.23, 1, 0.32, 1] }}
            className="p-6 lg:p-8"
            style={{
              background: "var(--color-panel)",
              border: "2px solid var(--border)",
            }}
          >
            {/* Header */}
            <div className="flex items-center gap-2.5 mb-5">
              <Sparkles size={16} style={{ color: "#7c5cd8" }} />
              <span
                className="text-xs font-semibold uppercase tracking-widest"
                style={{ color: "#7c5cd8" }}
              >
                AI answer
              </span>
              <span
                className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                style={{ background: "rgba(124,92,216,0.14)", color: "#7c5cd8" }}
              >
                {llmAnswer.time_to_llm_ms.toFixed(0)} ms total
              </span>
              {llmAnswer.grounded && (
                <MetaChip icon={<CheckCircle2 size={12} />} label="Grounded" accent />
              )}
            </div>

            <p
              className="text-devanagari leading-loose"
              style={{
                fontSize: "clamp(1rem, 2vw, 1.15rem)",
                color: "var(--foreground)",
                lineHeight: 2,
              }}
            >
              {llmAnswer.answer}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Sources ───────────────────────────────────────── */}
      {sources.length > 0 && (
        <div style={{ border: "2px solid var(--foreground)" }}>
          <button
            onClick={() => setSourcesOpen((o) => !o)}
            className="w-full flex items-center justify-between px-5 py-3.5 text-sm font-semibold transition-colors hover:bg-[var(--color-panel)]"
          >
            <div className="flex items-center gap-2">
              <Layers size={15} />
              Retrieved sources ({sources.length})
            </div>
            {sourcesOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </button>

          <AnimatePresence>
            {sourcesOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25, ease: "easeInOut" }}
                style={{ overflow: "hidden" }}
              >
                <div className="divide-y" style={{ borderColor: "var(--border)" }}>
                  {sources.map((src) => (
                    <SourceRow key={src.chunk_id ?? src.rank} source={src} />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* ── Latency breakdown ─────────────────────────────── */}
      {timing && (
        <div style={{ border: "2px solid var(--border)" }}>
          <button
            onClick={() => setTimingsOpen((o) => !o)}
            className="w-full flex items-center justify-between px-5 py-3 text-xs font-semibold uppercase tracking-widest transition-colors hover:bg-[var(--color-panel)] opacity-60 hover:opacity-100"
          >
            <div className="flex items-center gap-2">
              <BarChart2 size={13} />
              Latency breakdown
            </div>
            {timingsOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>

          <AnimatePresence>
            {timingsOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                style={{ overflow: "hidden" }}
              >
                <div className="px-5 pb-5 pt-2 grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {[
                    { label: "Embed", ms: timing.embedding_ms },
                    { label: "Qdrant", ms: timing.qdrant_ms },
                    { label: "Rerank", ms: timing.rerank_ms },
                    { label: "Compress", ms: timing.compression_ms },
                    { label: "Direct", ms: timing.time_to_direct_ms },
                    { label: "AI Answer", ms: timing.llm_ms },
                  ].map(({ label, ms }) => (
                    <LatencyBar key={label} label={label} ms={ms} total={timing.total_ms} />
                  ))}
                </div>
                <div className="px-5 pb-4 flex gap-6 text-xs font-black">
                  <span style={{ color: "var(--accent)" }}>
                    Direct: {timing.time_to_direct_ms.toFixed(1)} ms
                  </span>
                  <span style={{ color: "#7c5cd8" }}>
                    AI total: {timing.time_to_llm_ms.toFixed(1)} ms
                  </span>
                  <span style={{ color: "var(--color-ink-muted)", fontWeight: 400 }}>
                    (target &lt;200 ms)
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────

function SourceRow({ source }: { source: Source }) {
  const [expanded, setExpanded] = useState(false);
  const score = Math.round(source.score * 100);

  return (
    <div className="px-5 py-4" style={{ background: "var(--color-panel)" }}>
      <div className="flex items-start justify-between gap-4 mb-2">
        <div className="flex items-center gap-2.5">
          <span
            className="w-5 h-5 flex items-center justify-center rounded-full text-[11px] font-black text-white shrink-0"
            style={{ background: "var(--foreground)" }}
          >
            {source.rank}
          </span>
          <span className="text-[11px] font-mono" style={{ color: "var(--color-ink-muted)" }}>
            {source.chunk_id?.slice(0, 24) ?? "—"}
          </span>
        </div>
        {/* Score bar */}
        <div className="flex items-center gap-2 shrink-0">
          <span
            className="text-xs font-bold tabular-nums"
            style={{ color: score >= 70 ? "var(--accent)" : "var(--color-ink-muted)" }}
          >
            {score}%
          </span>
          <div className="w-16 h-1.5" style={{ background: "var(--border)" }}>
            <motion.div
              className="h-full"
              style={{ background: score >= 70 ? "var(--accent)" : "var(--color-ink-muted)" }}
              initial={{ width: 0 }}
              animate={{ width: `${score}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
          </div>
        </div>
      </div>

      <p
        className={clsx(
          "text-sm text-devanagari leading-relaxed cursor-pointer",
          !expanded && "line-clamp-2"
        )}
        style={{ color: "var(--color-ink-secondary)" }}
        onClick={() => setExpanded((e) => !e)}
      >
        {source.text}
      </p>
      <button
        onClick={() => setExpanded((e) => !e)}
        className="mt-1.5 text-[11px] flex items-center gap-1 opacity-40 hover:opacity-80 transition-opacity"
      >
        {expanded ? (
          <><ChevronUp size={11} /> Show less</>
        ) : (
          <><ChevronDown size={11} /> Show more</>
        )}
      </button>
    </div>
  );
}

function LatencyBar({ label, ms, total }: { label: string; ms: number; total: number }) {
  const pct = total > 0 ? (ms / total) * 100 : 0;
  return (
    <div>
      <div className="flex justify-between items-baseline mb-1">
        <span
          className="text-[11px] font-semibold uppercase tracking-wide"
          style={{ color: "var(--color-ink-muted)" }}
        >
          {label}
        </span>
        <span className="text-[11px] font-mono tabular-nums font-bold">
          {ms.toFixed(1)}
          <span style={{ color: "var(--color-ink-muted)" }}>ms</span>
        </span>
      </div>
      <div className="w-full h-1.5" style={{ background: "var(--border)" }}>
        <motion.div
          className="h-full"
          style={{ background: "var(--foreground)" }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

function MetaChip({
  icon,
  label,
  accent,
  warn,
}: {
  icon: React.ReactNode;
  label: string;
  accent?: boolean;
  warn?: boolean;
}) {
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-semibold rounded-full"
      style={{
        background: accent
          ? "rgba(232,93,4,0.18)"
          : warn
          ? "rgba(232,168,4,0.18)"
          : "rgba(245,243,238,0.10)",
        color: accent
          ? "var(--accent-light)"
          : warn
          ? "#E8A804"
          : "rgba(245,243,238,0.55)",
      }}
    >
      {icon}
      {label}
    </span>
  );
}
