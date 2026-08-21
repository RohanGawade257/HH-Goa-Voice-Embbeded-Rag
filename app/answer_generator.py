import time
import re
from typing import List, Dict

from app.generation.llm import missing_context_answer


# ============================================================
# HH GOA RAG - STEP 10
# Answer Generation + Guardrails
# ============================================================

MAX_CONTEXT_CHUNKS = 3
MAX_CONTEXT_WORDS = 350

# Questions that are obviously unrelated to the indexed dataset.
OFF_TOPIC_PATTERNS = [
    r"\b(write|generate|create)\s+(a\s+)?code\b",
    r"\bpython\b",
    r"\bjavascript\b",
    r"\bprogramming\b",
    r"\brecipe\b",
    r"\bweather\b",
    r"\bstock\s+price\b",
    r"\bbitcoin\b",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_off_topic(query: str) -> bool:
    query_lower = query.lower()

    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, query_lower):
            return True

    return False


def normalize_answer_language(language: str | None) -> str:
    if not language:
        return ""
    code = str(language).strip().lower()
    if "-" in code:
        code = code.split("-", 1)[0]
    if "_" in code:
        code = code.split("_", 1)[0]
    return code


def localized_missing_context(language: str | None) -> str:
    return missing_context_answer(normalize_answer_language(language))


def clean_context(results: List[Dict]) -> List[Dict]:
    """
    Keep only the strongest retrieved chunks.
    This is important for latency and grounding.
    """

    cleaned = []

    total_words = 0

    for result in results:
        text = normalize(result.get("text", ""))

        if not text:
            continue

        words = text.split()

        if total_words + len(words) > MAX_CONTEXT_WORDS:
            remaining = MAX_CONTEXT_WORDS - total_words

            if remaining <= 0:
                break

            words = words[:remaining]
            text = " ".join(words)

        cleaned.append({
            "text": text,
            "score": result.get("score", 0.0),
            "chunk_id": result.get("chunk_id"),
        })

        total_words += len(words)

        if len(cleaned) >= MAX_CONTEXT_CHUNKS:
            break

        if total_words >= MAX_CONTEXT_WORDS:
            break

    return cleaned


def keyword_overlap(query: str, context: str) -> float:
    """
    Very cheap grounding signal.

    This is NOT the final hallucination detector.
    It is an additional safety signal.

    Uses len(w) >= 2 to correctly handle Devanagari/Hindi
    words which are often only 2-3 Unicode codepoints.
    """

    query_words = set(
        w.lower()
        for w in re.findall(r"\w+", query, re.UNICODE)
        if len(w) >= 2
    )

    context_words = set(
        w.lower()
        for w in re.findall(r"\w+", context, re.UNICODE)
        if len(w) >= 2
    )

    if not query_words:
        return 0.0

    return len(query_words & context_words) / len(query_words)


def generate_extractive_answer(
    query: str,
    retrieved_results: List[Dict],
    language: str | None = None,
) -> Dict:

    start = time.perf_counter()

    query = normalize(query)

    # --------------------------------------------------------
    # GUARDRAIL 1 - EMPTY QUERY
    # --------------------------------------------------------

    if not query:
        return {
            "answer": "Please provide a question.",
            "grounded": False,
            "blocked": True,
            "reason": "empty_query",
            "latency_ms": (
                time.perf_counter() - start
            ) * 1000,
        }

    # --------------------------------------------------------
    # GUARDRAIL 2 - OFF TOPIC
    # --------------------------------------------------------

    if is_off_topic(query):
        return {
            "answer": localized_missing_context(language),
            "grounded": False,
            "blocked": True,
            "reason": "off_topic",
            "latency_ms": (
                time.perf_counter() - start
            ) * 1000,
        }

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = clean_context(retrieved_results)

    if not context:
        return {
            "answer": localized_missing_context(language),
            "grounded": False,
            "blocked": True,
            "reason": "no_context",
            "latency_ms": (
                time.perf_counter() - start
            ) * 1000,
        }

    combined_context = " ".join(
        item["text"] for item in context
    )

    # --------------------------------------------------------
    # GUARDRAIL 3 - CONTEXT RELEVANCE
    #
    # Two signals are combined:
    # 1. keyword_overlap: lexical match between query and context
    # 2. top retrieval score: vector similarity already computed
    #
    # For Hindi/Devanagari, keyword overlap can be low even
    # when the vector retrieval is highly relevant (score > 0.5).
    # Therefore: if the top retrieval score >= 0.50, we trust
    # the semantic retrieval and skip the lexical block.
    # --------------------------------------------------------

    overlap = keyword_overlap(
        query,
        combined_context
    )

    # Get the top retrieval score for the fallback signal.
    top_score = 0.0
    if retrieved_results:
        top_score = float(
            retrieved_results[0].get("score", 0.0)
            or retrieved_results[0].get("vector_score", 0.0)
        )

    # Block only when BOTH signals indicate low relevance.
    if overlap < 0.05 and top_score < 0.50:
        return {
            "answer": localized_missing_context(language),
            "grounded": False,
            "blocked": True,
            "reason": "low_context_relevance",
            "latency_ms": (
                time.perf_counter() - start
            ) * 1000,
        }

    # --------------------------------------------------------
    # EXTRACTIVE ANSWER
    #
    # For the <200 ms requirement we deliberately avoid
    # making a remote LLM request here.
    # --------------------------------------------------------

    best_text = context[0]["text"]

    # Split into sentences.
    sentences = re.split(
        r"(?<=[.!?।])\s+",
        best_text
    )

    query_terms = set(
        w.lower()
        for w in re.findall(r"\w+", query)
        if len(w) > 2
    )

    scored_sentences = []

    for sentence in sentences:

        sentence_terms = set(
            w.lower()
            for w in re.findall(r"\w+", sentence)
            if len(w) > 2
        )

        overlap_count = len(
            query_terms & sentence_terms
        )

        scored_sentences.append(
            (overlap_count, sentence.strip())
        )

    scored_sentences.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = [
        sentence
        for score, sentence in scored_sentences[:2]
        if sentence
    ]

    if not selected:
        selected = [best_text]

    answer = " ".join(selected)

    # --------------------------------------------------------
    # FINAL SAFETY CHECK
    # --------------------------------------------------------

    if len(answer.strip()) < 10:
        return {
            "answer": localized_missing_context(language),
            "grounded": False,
            "blocked": True,
            "reason": "weak_answer",
            "latency_ms": (
                time.perf_counter() - start
            ) * 1000,
        }

    return {
        "answer": answer,
        "grounded": True,
        "blocked": False,
        "reason": "grounded_extractive_answer",
        "context_chunks": len(context),
        "context_overlap": round(overlap, 4),
        "latency_ms": (
            time.perf_counter() - start
        ) * 1000,
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    test_results = [
        {
            "chunk_id": "test_1",
            "score": 0.91,
            "text": (
                "मैनहट्टन परियोजना द्वितीय विश्व युद्ध "
                "के दौरान एक अनुसंधान और विकास परियोजना थी "
                "जिसने पहले परमाणु हथियारों का निर्माण किया।"
            ),
        }
    ]

    result = generate_extractive_answer(
        "मैनहट्टन परियोजना क्या थी?",
        test_results,
    )

    print("=" * 60)
    print("ANSWER GENERATOR TEST")
    print("=" * 60)

    print("\nAnswer:")
    print(result["answer"])

    print("\nGrounded:", result["grounded"])
    print("Blocked:", result["blocked"])
    print("Reason:", result["reason"])
    print(
        f"Latency: {result['latency_ms']:.2f} ms"
    )
