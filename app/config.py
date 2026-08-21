import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Environment variables still work without the optional helper.
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


QDRANT_PATH = os.getenv("QDRANT_PATH", "data/qdrant")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "hh_goa_rag_multilingual")
QDRANT_TOP_K = get_int("QDRANT_TOP_K", 10)
TOP_K_FINAL = get_int("TOP_K_FINAL", 3)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_DIM = get_int("EMBEDDING_DIM", 384)
EMBEDDING_THREADS = get_int("EMBEDDING_THREADS", 4)

TOP_CONTEXT_CHUNKS = get_int("TOP_CONTEXT_CHUNKS", 2)
MAX_CONTEXT_CHARS = get_int("MAX_CONTEXT_CHARS", 1200)

ANSWER_BACKEND = os.getenv("ANSWER_BACKEND", "gemini").strip().lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface").strip().lower()
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("HF_API_KEY")
    or os.getenv("HF_TOKEN")
    or ""
)
LLM_CHAT_COMPLETIONS_URL = os.getenv(
    "LLM_CHAT_COMPLETIONS_URL",
    os.getenv(
        "HF_CHAT_COMPLETIONS_URL",
        "https://router.huggingface.co/v1/chat/completions",
    ),
)
# Backward-compatible aliases for existing Hugging Face environment files.
HF_API_KEY = LLM_API_KEY
HF_CHAT_COMPLETIONS_URL = LLM_CHAT_COMPLETIONS_URL
# Phase 1 benchmark model: Qwen/Qwen3-0.6B via Hugging Face OpenAI-compatible API.
# Override with QWEN_MODEL env var.
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = get_int("MAX_NEW_TOKENS", 64)
LLM_TIMEOUT_SECONDS = get_float("LLM_TIMEOUT_SECONDS", 10.0)
LLM_TEMPERATURE = get_float("LLM_TEMPERATURE", 0.0)
HF_HUB_OFFLINE = get_bool("HF_HUB_OFFLINE", True)

# ── Gemini backend ──────────────────────────────────────────────────────────
# Active model: gemini-2.5-flash-lite
# Non-thinking enforced via thinking_budget=0 in generation_config.
# Get a key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_MAX_OUTPUT_TOKENS = get_int("GEMINI_MAX_OUTPUT_TOKENS", 30)
GEMINI_THINKING_BUDGET = get_int("GEMINI_THINKING_BUDGET", 0)
GEMINI_TIMEOUT_SECONDS = get_float("GEMINI_TIMEOUT_SECONDS", 10.0)

# Sarvam Speech-to-Text configuration. Importing app.config is the single
# project-level dotenv load point, so STT imports this module before reading env.
SARVAM_API_KEY_ENV = "SARVAM_API_KEY"
SARVAM_STT_URL = os.getenv(
    "SARVAM_STT_URL",
    "https://api.sarvam.ai/speech-to-text",
).strip()
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3").strip()
SARVAM_STT_MODE = os.getenv("SARVAM_STT_MODE", "transcribe").strip()
SARVAM_STT_TIMEOUT_SECONDS = get_float("SARVAM_STT_TIMEOUT_SECONDS", 30.0)
SARVAM_STT_MOCK = get_bool("SARVAM_STT_MOCK", False)


def get_sarvam_api_key() -> str:
    return os.getenv(SARVAM_API_KEY_ENV, "").strip()
