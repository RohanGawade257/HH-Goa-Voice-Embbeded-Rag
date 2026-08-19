import os


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

ANSWER_BACKEND = os.getenv("ANSWER_BACKEND", "extractive").strip().lower()
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN") or ""
HF_CHAT_COMPLETIONS_URL = os.getenv(
    "HF_CHAT_COMPLETIONS_URL",
    "https://router.huggingface.co/v1/chat/completions",
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = get_int("MAX_NEW_TOKENS", 64)
LLM_TIMEOUT_SECONDS = get_float("LLM_TIMEOUT_SECONDS", 5.0)
LLM_TEMPERATURE = get_float("LLM_TEMPERATURE", 0.0)
HF_HUB_OFFLINE = get_bool("HF_HUB_OFFLINE", True)
