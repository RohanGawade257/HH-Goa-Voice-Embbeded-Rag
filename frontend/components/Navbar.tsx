"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useScroll, useMotionValueEvent } from "motion/react";
import { Search } from "lucide-react";
import Link from "next/link";
import clsx from "clsx";

interface NavbarProps {
  onQueryClick?: () => void;
}

export default function Navbar({ onQueryClick }: NavbarProps) {
  const [scrolled, setScrolled] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const { scrollY } = useScroll();

  useMotionValueEvent(scrollY, "change", (y) => {
    setScrolled(y > 48);
  });

  // Close on outside click
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setExpanded(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div className="fixed top-0 left-0 right-0 z-50 flex justify-center pt-4 px-4 pointer-events-none">
      <motion.nav
        ref={ref}
        className={clsx(
          "pointer-events-auto glass rounded-full",
          "transition-all duration-300 ease-out"
        )}
        initial={{ opacity: 0, y: -16, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1] }}
        style={{
          maxWidth: scrolled ? 520 : 640,
          width: "100%",
          background: scrolled
            ? "rgba(245,243,238,0.88)"
            : "rgba(245,243,238,0.72)",
        }}
      >
        {/* Main pill row */}
        <div className="flex items-center gap-2 px-3 py-2">
          {/* Logo mark */}
          <Link
            href="/"
            className="flex items-center gap-2 shrink-0 mr-1 group"
          >
            <span
              className="w-7 h-7 flex items-center justify-center rounded-full text-xs font-black text-white"
              style={{ background: "var(--accent)" }}
            >
              ह
            </span>
            <span
              className={clsx(
                "font-black tracking-tight text-sm transition-all duration-300",
                scrolled ? "opacity-100" : "opacity-100"
              )}
              style={{ color: "var(--foreground)" }}
            >
              HH-RAG
            </span>
          </Link>

          {/* Separator */}
          <div
            className="w-px h-4 shrink-0 mx-1"
            style={{ background: "var(--border)" }}
          />

          {/* Nav links */}
          <div className="flex items-center gap-1 flex-1 min-w-0">
            <NavPill href="#query" label="Query" />
            <NavPill href="#how-it-works" label="How it works" />
            <NavPill href="#benchmark" label="Benchmark" />
          </div>

          {/* CTA */}
          <button
            onClick={onQueryClick}
            className={clsx(
              "shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-full",
              "text-xs font-semibold text-white transition-all duration-150",
              "hover:opacity-90 active:scale-95"
            )}
            style={{ background: "var(--accent)" }}
          >
            <Search size={12} />
            <span className="hidden sm:inline">Ask</span>
          </button>
        </div>
      </motion.nav>
    </div>
  );
}

function NavPill({
  href,
  label,
}: {
  href: string;
  label: string;
}) {
  return (
    <a
      href={href}
      className={clsx(
        "px-3 py-1.5 rounded-full text-xs font-medium",
        "text-[var(--foreground)] opacity-70 hover:opacity-100",
        "transition-all duration-150 whitespace-nowrap",
        "hover:bg-[rgba(13,13,13,0.06)]"
      )}
    >
      {label}
    </a>
  );
}
