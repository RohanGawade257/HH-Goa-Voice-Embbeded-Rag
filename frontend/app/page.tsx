"use client";

import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import QuerySection from "@/components/QuerySection";
import HowItWorks from "@/components/HowItWorks";
import BenchmarkSection from "@/components/BenchmarkSection";
import Footer from "@/components/Footer";

export default function HomePage() {
  const scrollToQuery = () => {
    const el = document.getElementById("query");
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      // Focus the textarea after scroll
      setTimeout(() => {
        const ta = el.querySelector("textarea");
        ta?.focus();
      }, 600);
    }
  };

  return (
    <>
      <Navbar onQueryClick={scrollToQuery} />
      <main>
        <Hero onQueryClick={scrollToQuery} />
        <QuerySection />
        <HowItWorks />
        <BenchmarkSection />
      </main>
      <Footer />
    </>
  );
}
