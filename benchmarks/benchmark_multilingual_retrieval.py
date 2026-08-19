"""Benchmark multilingual retrieval through the production fast reranker.

Pipeline measured:
  multilingual Qdrant collection
      -> SentenceTransformer query embedding
      -> Qdrant Top-20
      -> app.pipeline.rerank

This intentionally excludes answer generation, LLM, STT, and TTS.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.pipeline import TOP_K_FINAL, TOP_K_RETRIEVAL, rerank


COLLECTION_NAME = "hh_goa_rag_multilingual"
QDRANT_PATH = "data/qdrant"
CHUNKS_FILE = Path("data/processed/multilingual/chunks.jsonl")
QUERY_DIR = Path("data/multilingual_repaired")
REPORT_FILE = Path("data/processed/multilingual/retrieval_benchmark_report.json")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_CHUNKS = 6046
VECTOR_SIZE = 384
WARMUP_QUERIES = 10

SUPPORTED_LANGUAGES = (
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "ta",
    "ur",
)


@dataclass
class StageStats:
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p100_ms: float


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]

    weight = k - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize(values: list[float]) -> StageStats:
    return StageStats(
        avg_ms=round(statistics.mean(values), 2) if values else 0.0,
        p50_ms=round(percentile(values, 50), 2),
        p95_ms=round(percentile(values, 95), 2),
        p100_ms=round(max(values), 2) if values else 0.0,
    )


def load_chunk_language_counts() -> Counter[str]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(CHUNKS_FILE)

    counts: Counter[str] = Counter()
    with CHUNKS_FILE.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            language = chunk.get("language")
            if language not in SUPPORTED_LANGUAGES:
                raise ValueError(f"unsupported language in chunks at line {line_number}: {language}")
            counts[language] += 1

    total = sum(counts.values())
    if total != EXPECTED_CHUNKS:
        raise ValueError(f"expected {EXPECTED_CHUNKS} chunks, found {total}")

    missing_languages = [language for language in SUPPORTED_LANGUAGES if counts[language] == 0]
    if missing_languages:
        raise ValueError(f"missing chunk languages: {missing_languages}")

    return counts


def load_queries(max_queries: int | None = None) -> list[dict[str, Any]]:
    queries_by_language: dict[str, list[dict[str, Any]]] = {
        language: []
        for language in SUPPORTED_LANGUAGES
    }
    seen_by_language: dict[str, set[Any]] = {
        language: set()
        for language in SUPPORTED_LANGUAGES
    }

    for language in SUPPORTED_LANGUAGES:
        path = QUERY_DIR / f"{language}_sample_1000.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                record = json.loads(line)
                query = str(record.get("query", "")).strip()
                query_id = record.get("query_id")
                if not query or query_id is None:
                    continue

                if query_id in seen_by_language[language]:
                    continue

                seen_by_language[language].add(query_id)
                queries_by_language[language].append(
                    {
                        "language": language,
                        "query_id": query_id,
                        "query": query,
                    }
                )

    queries = []
    max_language_size = max((len(items) for items in queries_by_language.values()), default=0)
    for index in range(max_language_size):
        for language in SUPPORTED_LANGUAGES:
            items = queries_by_language[language]
            if index >= len(items):
                continue
            queries.append(items[index])
            if max_queries is not None and len(queries) >= max_queries:
                return queries

    return queries


def validate_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        raise ValueError(f"missing Qdrant collection: {COLLECTION_NAME}")

    info = client.get_collection(COLLECTION_NAME)
    vectors = info.config.params.vectors
    size = getattr(vectors, "size", None)
    distance = getattr(vectors, "distance", None)

    if info.points_count != EXPECTED_CHUNKS:
        raise ValueError(f"expected {EXPECTED_CHUNKS} vectors, found {info.points_count}")
    if size != VECTOR_SIZE:
        raise ValueError(f"expected {VECTOR_SIZE} dimensions, found {size}")
    if distance != models.Distance.COSINE:
        raise ValueError(f"expected COSINE distance, found {distance}")


def benchmark(
    queries: list[dict[str, Any]],
    warmup_count: int,
) -> dict[str, Any]:
    chunk_counts = load_chunk_language_counts()

    print("=" * 70)
    print("HH GOA RAG - MULTILINGUAL RETRIEVAL + RERANK BENCHMARK")
    print("=" * 70)
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Chunks     : {sum(chunk_counts.values())}")
    print(f"Queries    : {len(queries)}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Top-K      : {TOP_K_RETRIEVAL} -> production rerank")

    print("\nLoading SentenceTransformer...")
    model_load_start = time.perf_counter()
    embedder = SentenceTransformer(MODEL_NAME)
    model_load_ms = (time.perf_counter() - model_load_start) * 1000
    print(f"Model load : {model_load_ms:.2f} ms")

    client = QdrantClient(path=QDRANT_PATH)

    try:
        validate_collection(client)

        warmups = queries[: max(0, min(warmup_count, len(queries)))]
        if warmups:
            print(f"\nWarmup queries: {len(warmups)}")
            for item in warmups:
                vector = embedder.encode(
                    item["query"],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                response = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    limit=TOP_K_RETRIEVAL,
                    with_payload=[
                        "chunk_id",
                        "passage_id",
                        "query_id",
                        "text",
                        "language",
                        "is_selected",
                        "chunk_strategy",
                        "word_count",
                    ],
                    with_vectors=False,
                )
                rerank(item["query"], response.points)

        embedding_ms: list[float] = []
        qdrant_ms: list[float] = []
        rerank_ms: list[float] = []
        total_ms: list[float] = []
        query_counts: Counter[str] = Counter()

        print("\nRunning benchmark...")
        for index, item in enumerate(queries, start=1):
            pipeline_start = time.perf_counter()

            start = time.perf_counter()
            vector = embedder.encode(
                item["query"],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embedding_elapsed = (time.perf_counter() - start) * 1000

            if len(vector) != VECTOR_SIZE:
                raise ValueError(f"query vector dimension mismatch: {len(vector)}")

            start = time.perf_counter()
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=TOP_K_RETRIEVAL,
                with_payload=[
                    "chunk_id",
                    "passage_id",
                    "query_id",
                    "text",
                    "language",
                    "is_selected",
                    "chunk_strategy",
                    "word_count",
                ],
                with_vectors=False,
            )
            hits = response.points
            qdrant_elapsed = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            rerank(item["query"], hits)
            rerank_elapsed = (time.perf_counter() - start) * 1000

            total_elapsed = (time.perf_counter() - pipeline_start) * 1000

            embedding_ms.append(embedding_elapsed)
            qdrant_ms.append(qdrant_elapsed)
            rerank_ms.append(rerank_elapsed)
            total_ms.append(total_elapsed)
            query_counts[item["language"]] += 1

            if index % 100 == 0:
                print(
                    f"[{index}/{len(queries)}] "
                    f"embedding={embedding_elapsed:.2f}ms "
                    f"qdrant={qdrant_elapsed:.2f}ms "
                    f"rerank={rerank_elapsed:.2f}ms "
                    f"total={total_elapsed:.2f}ms"
                )

        report = {
            "collection": COLLECTION_NAME,
            "qdrant_path": QDRANT_PATH,
            "chunks": sum(chunk_counts.values()),
            "chunk_counts_by_language": dict(chunk_counts),
            "queries": len(queries),
            "query_counts_by_language": dict(query_counts),
            "model": MODEL_NAME,
            "vector_size": VECTOR_SIZE,
            "top_k_retrieval": TOP_K_RETRIEVAL,
            "top_k_final": TOP_K_FINAL,
            "model_load_ms": round(model_load_ms, 2),
            "embedding": asdict(summarize(embedding_ms)),
            "qdrant": asdict(summarize(qdrant_ms)),
            "rerank": asdict(summarize(rerank_ms)),
            "total_retrieval": asdict(summarize(total_ms)),
        }

        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return report
    finally:
        client.close()


def print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("MULTILINGUAL RETRIEVAL BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Collection : {report['collection']}")
    print(f"Chunks     : {report['chunks']}")
    print(f"Queries    : {report['queries']}")
    print(f"Model      : {report['model']}")

    print("\nStage      | P50 ms | P95 ms | P100 ms")
    print("-" * 43)
    for key, label in (
        ("embedding", "Embedding"),
        ("qdrant", "Qdrant"),
        ("rerank", "Rerank"),
        ("total_retrieval", "Total"),
    ):
        stats = report[key]
        print(
            f"{label:<10} | "
            f"{stats['p50_ms']:>6.2f} | "
            f"{stats['p95_ms']:>6.2f} | "
            f"{stats['p100_ms']:>7.2f}"
        )

    print("\nLanguage | Chunks | Queries")
    print("-" * 29)
    chunk_counts = Counter(report["chunk_counts_by_language"])
    query_counts = Counter(report["query_counts_by_language"])
    for language in SUPPORTED_LANGUAGES:
        print(
            f"{language:2} | "
            f"{chunk_counts[language]:6} | "
            f"{query_counts[language]:7}"
        )

    print(f"\nReport: {REPORT_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multilingual Qdrant retrieval through production rerank.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=500,
        help=(
            "Maximum benchmark queries to run, balanced round-robin across languages. "
            "Use 0 for all available multilingual queries."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_QUERIES,
        help="Number of warmup queries before measuring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_queries = None if args.max_queries == 0 else args.max_queries
    queries = load_queries(max_queries=max_queries)

    if not queries:
        raise SystemExit("No benchmark queries found.")

    report = benchmark(queries=queries, warmup_count=args.warmup)
    print_report(report)


if __name__ == "__main__":
    main()
