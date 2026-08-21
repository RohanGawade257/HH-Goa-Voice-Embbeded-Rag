"use client";

import { motion } from "motion/react";

const METRICS = [
  {
    label: "Recall@3",
    value: "79.1%",
    sub: "791 / 1000 queries",
    desc: "Relevant passage appears in top-3 results",
    color: "var(--accent)",
  },
  {
    label: "Top-20 Recall",
    value: "87.3%",
    sub: "873 / 1000 queries",
    desc: "Embedding retrieval before reranking",
    color: "var(--accent)",
  },
  {
    label: "Grounded Rate",
    value: "99.7%",
    sub: "997 / 1000 queries",
    desc: "Answers backed by retrieved evidence",
    color: "#4CAF50",
  },
  {
    label: "P50 Latency",
    value: "38ms",
    sub: "post-STT pipeline (direct)",
    desc: "Embed + Qdrant + Rerank + Extractive answer",
    color: "var(--accent)",
  },
  {
    label: "P100 Latency",
    value: "110ms",
    sub: "worst-case observed",
    desc: "Every query under 200ms target",
    color: "var(--accent)",
  },
  {
    label: "Chunks indexed",
    value: "10,302",
    sub: "from 1,000 passages",
    desc: "VAST semantic chunking — 4 strategies",
    color: "var(--foreground)",
  },
];

const STAGE_BREAKDOWN = [
  { label: "Embed",   pct: 44, ms: "17ms avg" },
  { label: "Qdrant",  pct: 49, ms: "19ms avg" },
  { label: "Rerank",  pct: 5,  ms: "2ms avg"  },
  { label: "Answer",  pct: 1,  ms: "0.2ms avg"},
];

export default function BenchmarkSection() {
  return (
    <section
      id="benchmark"
      className="py-24 px-4"
      style={{ background: "var(--background)" }}
    >
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-14">
          <span
            className="font-black text-7xl leading-none select-none block mb-4"
            style={{ color: "var(--color-panel)" }}
          >
            04
          </span>
          <div className="flex items-end justify-between gap-6 flex-wrap">
            <h2
              className="font-black leading-none"
              style={{ fontSize: "clamp(2rem, 5vw, 4rem)", letterSpacing: "-0.03em" }}
            >
              Benchmark
            </h2>
            <p
              className="text-sm max-w-xs leading-relaxed"
              style={{ color: "var(--color-ink-muted)" }}
            >
              Verified baseline. 13,000 queries
              from AI4Bharat MSMARCO-XI.
            </p>
          </div>
        </div>

        {/* Metrics grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-10">
          {METRICS.map((m, i) => (
            <motion.div
              key={m.label}
              className="p-5"
              style={{
                background: "var(--color-panel)",
                border: "2px solid var(--foreground)",
              }}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ delay: i * 0.06, duration: 0.4 }}
            >
              <div
                className="font-black tabular-nums mb-1"
                style={{ fontSize: "clamp(1.6rem, 4vw, 2.4rem)", color: m.color, letterSpacing: "-0.03em" }}
              >
                {m.value}
              </div>
              <div className="font-semibold text-sm mb-0.5">{m.label}</div>
              <div className="text-xs" style={{ color: "var(--color-ink-muted)" }}>
                {m.sub}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Stage latency bar chart */}
        <div
          className="p-6 lg:p-8"
          style={{
            background: "var(--color-panel)",
            border: "2px solid var(--foreground)",
          }}
        >
          <h3 className="font-black text-base mb-1">
            Pipeline stage breakdown
          </h3>
          <p
            className="text-xs mb-6"
            style={{ color: "var(--color-ink-muted)" }}
          >
            Where the 38ms P50 is spent (direct-answer path)
          </p>

          <div className="space-y-4">
            {STAGE_BREAKDOWN.map((s, i) => (
              <div key={s.label}>
                <div className="flex justify-between items-baseline mb-1.5">
                  <span className="text-sm font-semibold">{s.label}</span>
                  <span
                    className="text-xs font-mono"
                    style={{ color: "var(--color-ink-muted)" }}
                  >
                    {s.ms}
                  </span>
                </div>
                <div
                  className="w-full h-3"
                  style={{ background: "var(--border)" }}
                >
                  <motion.div
                    className="h-full"
                    style={{ background: "var(--foreground)" }}
                    initial={{ width: 0 }}
                    whileInView={{ width: `${s.pct}%` }}
                    viewport={{ once: true }}
                    transition={{
                      delay: 0.1 + i * 0.1,
                      duration: 0.6,
                      ease: [0.23, 1, 0.32, 1],
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          <p
            className="mt-6 text-xs leading-relaxed"
            style={{ color: "var(--color-ink-muted)" }}
          >
            All measurements on CPU (Windows, Intel). P50 = 38ms. P100 = 110ms. Budget = 200ms.{" "}
            <span className="font-semibold" style={{ color: "var(--accent)" }}>
              1.8× headroom at P100.
            </span>{" "}
            Gemini 3.5 Flash Lite AI answer streams in asynchronously after the direct answer.
          </p>
        </div>
      </div>
    </section>
  );
}
