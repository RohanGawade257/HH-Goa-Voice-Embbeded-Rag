"use client";

import { motion } from "motion/react";
import { ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer
      className="py-16 px-4 border-t-2"
      style={{
        background: "var(--color-panel-dark)",
        borderColor: "rgba(245,243,238,0.08)",
      }}
    >
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10 mb-12">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <span
                className="w-8 h-8 flex items-center justify-center rounded-full font-black text-sm text-white"
                style={{ background: "var(--accent)" }}
              >
                ह
              </span>
              <span className="font-black text-white tracking-tight">
                HH-Goa-Rag
              </span>
            </div>
            <p
              className="text-sm leading-relaxed"
              style={{ color: "rgba(245,243,238,0.4)" }}
            >
              Voice-Enabled Multilingual RAG — Hacker House Goa 2026, Task 2.
              13 Indian languages · AI4Bharat MSMARCO-XI knowledge base.
            </p>
          </div>

          {/* Pipeline */}
          <div>
            <h4
              className="font-black text-xs uppercase tracking-widest mb-4"
              style={{ color: "rgba(245,243,238,0.3)" }}
            >
              Pipeline
            </h4>
            <ul className="space-y-2">
              {[
                "MiniLM-L12 embeddings (384d)",
                "Qdrant vector DB (10,302 chunks)",
                "Lexical reranker (Top-20 → Top-3)",
                "Extractive direct answer (~38ms P50)",
                "Gemini 2.5 Flash Lite (AI answer, streaming)",
                "Sarvam STT (13 Indian languages)",
              ].map((item) => (
                <li
                  key={item}
                  className="text-sm"
                  style={{ color: "rgba(245,243,238,0.45)" }}
                >
                  — {item}
                </li>
              ))}
            </ul>
          </div>

          {/* Metrics */}
          <div>
            <h4
              className="font-black text-xs uppercase tracking-widest mb-4"
              style={{ color: "rgba(245,243,238,0.3)" }}
            >
              Frozen baseline
            </h4>
            <ul className="space-y-2">
              {[
                ["Recall@3",   "79.1%"],
                ["Grounded",   "99.7%"],
                ["P50 latency","38ms"],
                ["P100 latency","110ms"],
                ["Dataset",    "1,000 queries"],
              ].map(([k, v]) => (
                <li
                  key={k}
                  className="flex justify-between text-sm"
                >
                  <span style={{ color: "rgba(245,243,238,0.35)" }}>{k}</span>
                  <span
                    className="font-bold tabular-nums"
                    style={{ color: "var(--accent)" }}
                  >
                    {v}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div
          className="flex items-center justify-between flex-wrap gap-4 pt-8 border-t"
          style={{ borderColor: "rgba(245,243,238,0.08)" }}
        >
          <p
            className="text-xs"
            style={{ color: "rgba(245,243,238,0.25)" }}
          >
            HH-Goa-Rag © 2026 · Verified baseline · Built with Next.js + FastAPI + Gemini
          </p>
          <div className="flex items-center gap-4">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs transition-opacity opacity-40 hover:opacity-80"
              style={{ color: "#F5F3EE" }}
            >
              <ExternalLink size={12} />
              API Docs
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
