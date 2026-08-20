"use client";

import { useState } from "react";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import QuerySection from "@/components/QuerySection";
import HowItWorks from "@/components/HowItWorks";
import BenchmarkSection from "@/components/BenchmarkSection";
import Footer from "@/components/Footer";

export default function HomePage() {
  const [exampleQuery, setExampleQuery] = useState("");
  const [exampleLanguage, setExampleLanguage] = useState("");

  const scrollToQuery = () => {
    const el = document.getElementById("query");
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      setTimeout(() => {
        const ta = el.querySelector("textarea");
        ta?.focus();
      }, 600);
    }
  };

  const handleExampleSelect = (query: string, language: string) => {
    setExampleQuery(query);
    setExampleLanguage(language);
  };

  return (
    <>
      <Navbar onQueryClick={scrollToQuery} />
      <main>
        <Hero onQueryClick={scrollToQuery} onExampleSelect={handleExampleSelect} />
        <QuerySection
          defaultQuery={exampleQuery || undefined}
          defaultLanguage={exampleLanguage || undefined}
        />
        <HowItWorks />
        <BenchmarkSection />
      </main>
      <Footer />
    </>
  );
}
