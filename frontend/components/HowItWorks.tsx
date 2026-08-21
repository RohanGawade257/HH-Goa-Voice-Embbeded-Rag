"use client";

import { motion } from "motion/react";

const STEPS = [
  {
    n: "01",
    title: "Voice or text input",
    hindi: "आवाज़ या टेक्स्ट",
    desc: "Speak or type your question in any of 13 Indian languages. Sarvam STT converts speech to text with full Devanagari fidelity.",
  },
  {
    n: "02",
    title: "Semantic embedding",
    hindi: "अर्थ-आधारित एम्बेडिंग",
    desc: "The query is encoded into a 384-dimensional semantic vector using a multilingual MiniLM-L12 model — no translation needed.",
  },
  {
    n: "03",
    title: "Qdrant vector search",
    hindi: "Qdrant वेक्टर खोज",
    desc: "Top-20 semantically similar passages are retrieved from 10,302 indexed chunks via cosine similarity.",
  },
  {
    n: "04",
    title: "Lexical reranking",
    hindi: "पुनः क्रमांकन",
    desc: "A lightweight lexical reranker (vector 70% + overlap 20% + phrase 10%) selects the best 3 passages.",
  },
  {
    n: "05",
    title: "Dual-mode answer",
    hindi: "दोहरा उत्तर",
    desc: "An extractive answer returns in ~38ms P50. Gemini 3.5 Flash Lite then streams an AI-enhanced answer asynchronously.",
  },
  {
    n: "06",
    title: "≤38ms direct",
    hindi: "अत्यंत तेज़",
    desc: "The extractive post-STT pipeline completes at 38ms P50 / 110ms P100, well within the 200ms target.",
    accent: true,
  },
];

export default function HowItWorks() {
  return (
    <section
      id="how-it-works"
      className="py-24 px-4 overflow-hidden"
      style={{ background: "var(--color-panel-dark)" }}
    >
      <div className="max-w-6xl mx-auto">
        {/* Heading */}
        <div className="mb-16 flex items-end justify-between gap-6 flex-wrap">
          <div>
            <span
              className="font-black text-7xl leading-none select-none block mb-4"
              style={{ color: "rgba(245,243,238,0.07)" }}
            >
              03
            </span>
            <h2
              className="font-black text-white leading-none"
              style={{ fontSize: "clamp(2rem, 5vw, 4rem)", letterSpacing: "-0.03em" }}
            >
              How it works
            </h2>
          </div>
          <p
            className="text-sm max-w-xs leading-relaxed"
            style={{ color: "rgba(245,243,238,0.4)" }}
          >
            Six stages from spoken word to grounded answer.
            Each one measured, benchmarked, and verified.
          </p>
        </div>

        {/* Steps grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {STEPS.map((step, i) => (
            <motion.div
              key={step.n}
              className="p-6 relative overflow-hidden"
              style={{
                background: step.accent
                  ? "var(--accent)"
                  : "rgba(245,243,238,0.04)",
                border: step.accent
                  ? "2px solid var(--accent)"
                  : "1px solid rgba(245,243,238,0.10)",
              }}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ delay: i * 0.07, duration: 0.45, ease: [0.23, 1, 0.32, 1] }}
            >
              {/* Big number */}
              <span
                className="absolute top-4 right-5 font-black text-5xl leading-none select-none"
                style={{
                  color: step.accent
                    ? "rgba(255,255,255,0.18)"
                    : "rgba(245,243,238,0.06)",
                }}
              >
                {step.n}
              </span>

              <h3
                className="font-black text-base mb-1 relative z-10"
                style={{ color: step.accent ? "#fff" : "#F5F3EE" }}
              >
                {step.title}
              </h3>
              <p
                className="text-xs text-devanagari mb-3 relative z-10"
                style={{
                  color: step.accent
                    ? "rgba(255,255,255,0.7)"
                    : "rgba(245,243,238,0.35)",
                }}
              >
                {step.hindi}
              </p>
              <p
                className="text-sm leading-relaxed relative z-10"
                style={{
                  color: step.accent
                    ? "rgba(255,255,255,0.85)"
                    : "rgba(245,243,238,0.55)",
                }}
              >
                {step.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
