"use client";

import { motion } from "motion/react";
import { ArrowDown } from "lucide-react";

interface HeroProps {
  onQueryClick: () => void;
  onExampleSelect?: (query: string, language: string) => void;
}

// Multilingual example queries — 5 different languages
const EXAMPLE_QUERIES = [
  {
    query: "मैनहट्टन परियोजना क्या थी?",
    language: "hi-IN",
    label: "Hindi",
  },
  {
    query: "ম্যানহাটন প্রকল্পের সাফল্যের তাৎক্ষণিক প্রভাব কী ছিল?",
    language: "bn-IN",
    label: "Bengali",
  },
  {
    query: "மன்ஹாட்டன் திட்டத்தின் வெற்றியின் உடனடி விளைவு என்ன?",
    language: "ta-IN",
    label: "Tamil",
  },
  {
    query: "मॅनहॅटन प्रकल्पाच्या यशाचा तात्काळ काय परिणाम झाला?",
    language: "mr-IN",
    label: "Marathi",
  },
  {
    query: "ਮੈਨਹੈਟਨ ਪ੍ਰੋਜੈਕਟ ਦੀ ਸਫਲਤਾ ਦਾ ਤੁਰੰਤ ਪ੍ਰਭਾਵ ਕੀ ਸੀ?",
    language: "pa-IN",
    label: "Punjabi",
  },
];

export default function Hero({ onQueryClick, onExampleSelect }: HeroProps) {
  const handleExample = (q: typeof EXAMPLE_QUERIES[0]) => {
    onExampleSelect?.(q.query, q.language);
    onQueryClick();
  };

  return (
    <section
      className="relative min-h-screen flex flex-col overflow-hidden"
      style={{ background: "var(--background)" }}
    >
      {/* Scroll progress bar */}
      <div
        className="fixed top-0 left-0 right-0 h-[2px] z-[60] origin-left"
        style={{
          background: "var(--accent)",
          animationName: "scrollProgress",
          animationTimeline: "scroll()",
          animationTimingFunction: "linear",
          animationDuration: "1ms",
          animationFillMode: "both",
        }}
      />

      {/* ── Asymmetric layout ─────────────────────────────── */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_1.1fr] min-h-screen">

        {/* LEFT — dark panel */}
        <motion.div
          className="relative flex flex-col justify-between p-8 lg:p-12 xl:p-16 pt-28 lg:pt-32"
          style={{ background: "var(--color-panel-dark)", color: "#F5F3EE" }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7 }}
        >
          {/* Category tag */}
          <div>
            <motion.div
              className="inline-flex items-center gap-2 px-3 py-1.5 mb-8 rounded-full text-xs font-semibold tracking-widest uppercase"
              style={{
                background: "rgba(232,93,4,0.18)",
                color: "var(--accent-light)",
                border: "1px solid rgba(232,93,4,0.28)",
              }}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.5 }}
            >
              <span
                className="w-1.5 h-1.5 rounded-full animate-pulse"
                style={{ background: "var(--accent)" }}
              />
              Voice · RAG · 13 Languages
            </motion.div>

            {/* Main heading */}
            <motion.h1
              className="font-black leading-none mb-6"
              style={{ fontSize: "clamp(2.8rem, 6vw, 5.5rem)", letterSpacing: "-0.03em" }}
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
            >
              Multilingual
              <br />
              <span style={{ color: "var(--accent)" }}>Retrieval</span>
              <br />
              Intelligence.
            </motion.h1>

            {/* Multilingual subtitle */}
            <motion.p
              className="text-devanagari mb-8 font-medium"
              style={{
                fontSize: "clamp(1.1rem, 2vw, 1.4rem)",
                color: "rgba(245,243,238,0.55)",
                lineHeight: 1.9,
              }}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.45, duration: 0.5 }}
            >
              अपनी भाषा में प्रश्न पूछें।
              <br />
              ज्ञान-आधारित उत्तर पाएं।
            </motion.p>

            {/* English subtitle */}
            <motion.p
              className="text-sm leading-relaxed max-w-sm"
              style={{ color: "rgba(245,243,238,0.45)" }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.55, duration: 0.5 }}
            >
              Ask questions in 13 Indian languages. Get grounded answers from the
              AI4Bharat MSMARCO-XI knowledge base. Powered by multilingual semantic
              retrieval with{" "}
              <span style={{ color: "rgba(245,243,238,0.75)" }}>
                extractive answers at 38ms P50
              </span>{" "}
              and a{" "}
              <span style={{ color: "var(--accent-light)" }}>
                Gemini 3.5 Flash Lite AI answer
              </span>{" "}
              streaming shortly after.
            </motion.p>
          </div>

          {/* Bottom stats row */}
          <motion.div
            className="flex items-center gap-6 pt-12 mt-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7, duration: 0.5 }}
          >
            {[
              { value: "13",      label: "Languages" },
              { value: "38ms",    label: "P50 direct" },
              { value: "10,302",  label: "Chunks indexed" },
            ].map((stat) => (
              <div key={stat.label}>
                <div
                  className="text-xl font-black"
                  style={{ color: "var(--accent)" }}
                >
                  {stat.value}
                </div>
                <div
                  className="text-[11px] tracking-wide uppercase"
                  style={{ color: "rgba(245,243,238,0.4)" }}
                >
                  {stat.label}
                </div>
              </div>
            ))}
          </motion.div>
        </motion.div>

        {/* RIGHT — warm surface + query entry */}
        <motion.div
          className="flex flex-col justify-center p-8 lg:p-12 xl:p-16 pt-28 lg:pt-32"
          style={{ background: "var(--background)" }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.1 }}
        >
          {/* Large editorial number */}
          <motion.div
            className="font-black select-none mb-8 leading-none"
            style={{
              fontSize: "clamp(6rem, 14vw, 11rem)",
              color: "var(--color-panel)",
              letterSpacing: "-0.05em",
            }}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4, duration: 0.7, ease: [0.23, 1, 0.32, 1] }}
          >
            01
          </motion.div>

          {/* CTA block */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.55 }}
          >
            <h2
              className="font-black mb-3"
              style={{ fontSize: "clamp(1.4rem, 3vw, 2rem)", letterSpacing: "-0.02em" }}
            >
              Ask in any of 13 languages
            </h2>
            <p
              className="text-sm mb-6 max-w-xs leading-relaxed"
              style={{ color: "var(--color-ink-muted)" }}
            >
              Type or speak your question. The system retrieves the most
              relevant passages and returns an extractive answer at 38ms P50,
              followed by a Gemini AI-enhanced answer.
            </p>

            {/* Primary CTA */}
            <button
              onClick={onQueryClick}
              className="group flex items-center gap-3 px-6 py-3.5 font-bold text-sm text-white transition-all duration-150 hover:opacity-90 active:scale-[0.98] mb-4"
              style={{
                background: "var(--foreground)",
                border: "2px solid var(--foreground)",
              }}
            >
              <span>Start querying</span>
              <ArrowDown
                size={16}
                className="group-hover:translate-y-0.5 transition-transform duration-150"
              />
            </button>

            {/* Example queries */}
            <div className="flex flex-col gap-2 mt-6">
              <p
                className="text-[11px] uppercase tracking-widest font-semibold mb-1"
                style={{ color: "var(--color-ink-muted)" }}
              >
                Try an example
              </p>
              {EXAMPLE_QUERIES.map((q) => (
                <button
                  key={q.language}
                  onClick={() => handleExample(q)}
                  className="text-left text-sm py-2 px-3 border-l-2 text-devanagari transition-all duration-150 hover:pl-4 flex items-baseline gap-2"
                  style={{
                    borderColor: "var(--border)",
                    color: "var(--color-ink-secondary)",
                    background: "transparent",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--accent)";
                    (e.currentTarget as HTMLButtonElement).style.color = "var(--foreground)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border)";
                    (e.currentTarget as HTMLButtonElement).style.color = "var(--color-ink-secondary)";
                  }}
                >
                  <span className="shrink-0 text-[10px] uppercase tracking-wide opacity-40 w-14">
                    {q.label}
                  </span>
                  <span>{q.query}</span>
                </button>
              ))}
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
