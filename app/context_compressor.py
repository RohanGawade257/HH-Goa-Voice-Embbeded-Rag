import re
import time
from typing import Any

from app.config import MAX_CONTEXT_CHARS, TOP_CONTEXT_CHUNKS


SENTENCE_RE = re.compile(r"(?<=[\u0964\u0965\u06d4\u061f.!?])\s+")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [
        sentence.strip()
        for sentence in SENTENCE_RE.split(text)
        if sentence.strip()
    ]


def query_tokens(query: str) -> set[str]:
    return set(WORD_RE.findall(normalize(query)))


def score_sentence(
    sentence: str,
    tokens: set[str],
    base_score: float,
) -> float:
    sentence_tokens = set(WORD_RE.findall(normalize(sentence)))
    if not sentence_tokens:
        return 0.0

    lexical = (
        len(tokens.intersection(sentence_tokens)) / len(tokens)
        if tokens
        else 0.0
    )

    return lexical + (0.15 * base_score)


def compress_context(
    query: str,
    reranked_results: list[dict[str, Any]],
    top_context_chunks: int = TOP_CONTEXT_CHUNKS,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> dict[str, Any]:
    start = time.perf_counter()
    tokens = query_tokens(query)
    snippets = []
    seen = set()
    before_chars = sum(
        len(str(result.get("text", "")).strip())
        for result in reranked_results
        if result.get("text")
    )

    for result in reranked_results[:top_context_chunks]:
        text = str(result.get("text", "")).strip()
        if not text:
            continue

        base_score = float(result.get("score", 0.0) or result.get("vector_score", 0.0))
        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]

        scored = [
            (
                score_sentence(sentence, tokens, base_score),
                index,
                sentence,
            )
            for index, sentence in enumerate(sentences)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        selected = sorted(scored[:2], key=lambda item: item[1])
        snippet = " ".join(sentence for _, _, sentence in selected).strip()
        if not snippet:
            snippet = text

        normalized = normalize(snippet)
        if normalized in seen:
            continue
        seen.add(normalized)

        remaining = max_context_chars - sum(len(item["text"]) for item in snippets)
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rsplit(" ", 1)[0].strip() or snippet[:remaining]

        snippets.append(
            {
                "text": snippet,
                "score": base_score,
            }
        )

        if sum(len(item["text"]) for item in snippets) >= max_context_chars:
            break

    context_text = "\n".join(
        snippet["text"]
        for snippet in snippets
    )
    after_chars = len(context_text)

    return {
        "context": context_text,
        "snippets": snippets,
        "before_chars": before_chars,
        "after_chars": after_chars,
        "compression_ratio": (
            round(after_chars / before_chars, 4)
            if before_chars
            else 0.0
        ),
        "latency_ms": (time.perf_counter() - start) * 1000,
    }
