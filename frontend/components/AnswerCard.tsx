"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Clock,
  Layers,
  BarChart2,
  RefreshCw,
} from "lucide-react";
import clsx from "clsx";
import type { QueryResponse, Source } from "@/lib/api";

interface AnswerCardProps {
  result: QueryResponse;
  onClear: () => void;
}

export default function AnswerCard({ result, onClear }: AnswerCardProps) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [timingsOpen, setTimingsOpen] = useState(false);

  const isGrounded = result.grounded && !result.blocked;

  return (
    <div className="space-y-4">
      {/* ── Main answer block ─────────────────────────────── */}
      <div
        className="p-6 lg:p-8"
        style={{
          background: "var(--color-panel-dark)",
          border: "2px solid var(--foreground)",
        }}
      >
        {/* Header row */}
        <div className="flex items-start justify-between gap-4 mb-5">
          <div className="flex items-center gap-2.5">
            {isGrounded ? (
              <CheckCircle2
                size={18}
                style={{ color: "var(--accent)" }}
              />
            ) : (
              <AlertTriangle size={18} style={{ color: "#E8A804" }} />
            )}
            <span
              className="text-xs font-semibold uppercase tracking-widest"
              style={{ color: isGrounded ? "var(--accent)" : "#E8A804" }}
            >
              {isGrounded ? "Grounded answer" : result.reason.replace(/_/g, " ")}
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
          {result.answer}
        </p>

        {/* Metadata chips */}
        <div className="flex flex-wrap items-center gap-3 mt-6 pt-5 border-t"
          style={{ borderColor: "rgba(245,243,238,0.10)" }}>
          <MetaChip icon={<Layers size={12} />} label={`${result.retrieved_chunks} sources`} />
          <MetaChip
            icon={<Clock size={12} />}
            label={`${result.timings.total_ms.toFixed(1)}ms total`}
          />
          {result.grounded && (
            <MetaChip
              icon={<CheckCircle2 size={12} />}
              label="Grounded"
              accent
            />
          )}
        </div>
      </div>

      {/* ── Sources ───────────────────────────────────────── */}
      {result.sources.length > 0 && (
        <div style={{ border: "2px solid var(--foreground)" }}>
          <button
            onClick={() => setSourcesOpen((o) => !o)}
            className="w-full flex items-center justify-between px-5 py-3.5 text-sm font-semibold transition-colors hover:bg-[var(--color-panel)]"
          >
            <div className="flex items-center gap-2">
              <Layers size={15} />
              Retrieved sources ({result.sources.length})
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
                <div
                  className="divide-y"
                  style={{ borderColor: "var(--border)" }}
                >
                  {result.sources.map((src) => (
                    <SourceRow key={src.chunk_id ?? src.rank} source={src} />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* ── Latency breakdown ─────────────────────────────── */}
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
              <div className="px-5 pb-5 pt-2 grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: "Embed", ms: result.timings.embedding_ms },
                  { label: "Qdrant", ms: result.timings.qdrant_ms },
                  { label: "Rerank", ms: result.timings.rerank_ms },
                  { label: "Answer", ms: result.timings.answer_ms },
                ].map(({ label, ms }) => (
                  <LatencyBar
                    key={label}
                    label={label}
                    ms={ms}
                    total={result.timings.total_ms}
                  />
                ))}
              </div>
              <div
                className="px-5 pb-3 text-xs font-black"
                style={{ color: "var(--accent)" }}
              >
                Total: {result.timings.total_ms.toFixed(2)}ms &nbsp;
                <span
                  className="font-normal"
                  style={{ color: "var(--color-ink-muted)" }}
                >
                  (target &lt;200ms)
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

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
          <span
            className="text-[11px] font-mono"
            style={{ color: "var(--color-ink-muted)" }}
          >
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
          <div
            className="w-16 h-1.5"
            style={{ background: "var(--border)" }}
          >
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

      {/* Preview / expanded */}
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
          <>
            <ChevronUp size={11} /> Show less
          </>
        ) : (
          <>
            <ChevronDown size={11} /> Show more
          </>
        )}
      </button>
    </div>
  );
}

function LatencyBar({
  label,
  ms,
  total,
}: {
  label: string;
  ms: number;
  total: number;
}) {
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
      <div
        className="w-full h-1.5"
        style={{ background: "var(--border)" }}
      >
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
}: {
  icon: React.ReactNode;
  label: string;
  accent?: boolean;
}) {
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-semibold rounded-full"
      style={{
        background: accent
          ? "rgba(232,93,4,0.18)"
          : "rgba(245,243,238,0.10)",
        color: accent ? "var(--accent-light)" : "rgba(245,243,238,0.55)",
      }}
    >
      {icon}
      {label}
    </span>
  );
}
