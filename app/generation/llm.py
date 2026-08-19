import time
from typing import Any

import httpx

from app.config import (
    ANSWER_BACKEND,
    HF_API_KEY,
    HF_CHAT_COMPLETIONS_URL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MAX_NEW_TOKENS,
    QWEN_MODEL,
)


LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "ur": "Urdu",
}


class QwenAnswerGenerator:
    """Small Qwen answer generator backed by Hugging Face Inference API."""

    def __init__(self) -> None:
        start = time.perf_counter()
        self.backend = ANSWER_BACKEND
        self.model_name = QWEN_MODEL
        self.api_key = HF_API_KEY
        self.url = HF_CHAT_COMPLETIONS_URL
        self.available = self.backend in {"qwen", "qwen_api"} and bool(self.api_key)
        self.load_error = "" if self.available else "HF_API_KEY or HF_TOKEN is not set"
        self.client = httpx.Client(
            timeout=LLM_TIMEOUT_SECONDS,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        self.load_ms = (time.perf_counter() - start) * 1000

    def close(self) -> None:
        self.client.close()

    def messages(
        self,
        query: str,
        language: str,
        context: str,
    ) -> list[dict[str, str]]:
        language_name = LANGUAGE_NAMES.get(language, language or "the query language")
        return [
            {
                "role": "system",
                "content": (
                    "You answer questions using only supplied evidence. "
                    f"Answer concisely in {language_name}, matching the user's language. "
                    "If the evidence is insufficient, say so briefly. "
                    "Do not include reasoning, metadata, or citations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Evidence:\n{context}\n\n"
                    "Answer:"
                ),
            },
        ]

    def generate(
        self,
        query: str,
        language: str,
        context: str,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        if self.backend not in {"qwen", "qwen_api"}:
            return {
                "answer": "",
                "grounded": False,
                "blocked": True,
                "reason": "qwen_disabled",
                "latency_ms": (time.perf_counter() - start) * 1000,
            }

        if not self.available:
            return {
                "answer": "",
                "grounded": False,
                "blocked": True,
                "reason": "qwen_api_key_missing",
                "error": self.load_error,
                "latency_ms": (time.perf_counter() - start) * 1000,
            }

        messages = self.messages(query, language, context)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "stream": False,
        }

        try:
            response = self.client.post(
                self.url,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            answer = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not answer:
                return {
                    "answer": "",
                    "grounded": False,
                    "blocked": True,
                    "reason": "empty_llm_answer",
                    "latency_ms": (time.perf_counter() - start) * 1000,
                    "prompt_chars": sum(len(message["content"]) for message in messages),
                    "context_chars": len(context),
                }

            return {
                "answer": answer,
                "grounded": True,
                "blocked": False,
                "reason": "qwen_api_grounded_answer",
                "latency_ms": (time.perf_counter() - start) * 1000,
                "prompt_chars": sum(len(message["content"]) for message in messages),
                "context_chars": len(context),
            }
        except Exception as exc:
            return {
                "answer": "",
                "grounded": False,
                "blocked": True,
                "reason": "qwen_api_error",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.perf_counter() - start) * 1000,
                "prompt_chars": sum(len(message["content"]) for message in messages),
                "context_chars": len(context),
            }
