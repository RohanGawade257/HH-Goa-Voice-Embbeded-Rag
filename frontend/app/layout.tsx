import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HH-Goa-Rag — Hindi Voice RAG",
  description:
    "Voice-enabled Hindi RAG system. Ask questions in Hindi, get grounded answers from the AI4Bharat MSMARCO-XI knowledge base. 99.7% grounded, <55ms P100 latency.",
  openGraph: {
    title: "HH-Goa-Rag — Hindi Voice RAG",
    description:
      "Ask questions in Hindi. Get grounded answers in under 55ms.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="hi">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
