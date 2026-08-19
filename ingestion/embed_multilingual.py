"""Embed multilingual chunks and upload them to a dedicated Qdrant collection."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/processed/multilingual/chunks.jsonl")
QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "hh_goa_rag_multilingual"
BASELINE_COLLECTION_NAME = "hh_goa_rag_hindi"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 128
UPLOAD_BATCH_SIZE = 128
VECTOR_SIZE = 384
EXPECTED_CHUNKS = 6046
TOP_K = 10

REPORT_FILE = Path("data/processed/multilingual/embed_report.json")

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

REQUIRED_FIELDS = {
    "chunk_id",
    "passage_id",
    "query_id",
    "passage_index",
    "chunk_index",
    "text",
    "language",
    "word_count",
    "chunk_strategy",
    "has_overlap",
    "is_selected",
    "query_type",
}

SANITY_QUERY_HI = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"
REPRESENTATIVE_LANGUAGES = ("hi", "mr", "ta", "bn", "ur")


@dataclass
class Timing:
    model_load_seconds: float = 0.0
    embedding_seconds: float = 0.0
    avg_document_embedding_ms: float = 0.0
    upload_seconds: float = 0.0


def point_id_for_chunk(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hh-goa-rag:{chunk_id}"))


def load_chunks() -> list[dict[str, Any]]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    chunks = []
    malformed = []
    chunk_ids = set()

    with INPUT_FILE.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed.append(
                    {
                        "line": line_number,
                        "error": f"invalid JSON: {exc}",
                    }
                )
                continue

            missing = sorted(REQUIRED_FIELDS - set(chunk))
            if missing:
                malformed.append(
                    {
                        "line": line_number,
                        "chunk_id": chunk.get("chunk_id"),
                        "error": f"missing fields: {missing}",
                    }
                )
                continue

            language = chunk.get("language")
            text = chunk.get("text")
            chunk_id = chunk.get("chunk_id")

            if language not in SUPPORTED_LANGUAGES:
                malformed.append(
                    {
                        "line": line_number,
                        "chunk_id": chunk_id,
                        "error": f"unsupported language: {language}",
                    }
                )
            if not isinstance(text, str) or not text.strip():
                malformed.append(
                    {
                        "line": line_number,
                        "chunk_id": chunk_id,
                        "error": "empty text",
                    }
                )
            if isinstance(text, str) and text.lower().startswith(("query:", "passage:")):
                malformed.append(
                    {
                        "line": line_number,
                        "chunk_id": chunk_id,
                        "error": "E5-style prefix detected",
                    }
                )
            if chunk_id in chunk_ids:
                malformed.append(
                    {
                        "line": line_number,
                        "chunk_id": chunk_id,
                        "error": "duplicate chunk_id",
                    }
                )

            chunk_ids.add(chunk_id)
            chunks.append(chunk)

    if malformed:
        raise ValueError(json.dumps(malformed[:20], ensure_ascii=False, indent=2))

    if len(chunks) != EXPECTED_CHUNKS:
        raise ValueError(f"expected {EXPECTED_CHUNKS} chunks, found {len(chunks)}")

    point_ids = [point_id_for_chunk(chunk["chunk_id"]) for chunk in chunks]
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("deterministic point IDs are not unique")

    return chunks


def create_or_validate_collection(client: QdrantClient) -> None:
    if COLLECTION_NAME == BASELINE_COLLECTION_NAME:
        raise ValueError("refusing to use the Hindi baseline collection")

    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        vector_params = info.config.params.vectors
        existing_size = getattr(vector_params, "size", None)
        existing_distance = getattr(vector_params, "distance", None)
        if existing_size != VECTOR_SIZE or existing_distance != models.Distance.COSINE:
            raise ValueError(
                f"existing {COLLECTION_NAME} config mismatch: "
                f"size={existing_size}, distance={existing_distance}"
            )
        print(f"Collection already exists, upserting into: {COLLECTION_NAME}")
        return

    print(f"Creating collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )


def build_points(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> list[models.PointStruct]:
    points = []
    for index, chunk in enumerate(chunks):
        payload = {
            "chunk_id": chunk["chunk_id"],
            "passage_id": chunk["passage_id"],
            "query_id": chunk["query_id"],
            "passage_index": chunk["passage_index"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "language": chunk["language"],
            "word_count": chunk["word_count"],
            "chunk_strategy": chunk["chunk_strategy"],
            "has_overlap": chunk["has_overlap"],
            "is_selected": chunk["is_selected"],
            "query_type": chunk["query_type"],
        }
        points.append(
            models.PointStruct(
                id=point_id_for_chunk(chunk["chunk_id"]),
                vector=embeddings[index].tolist(),
                payload=payload,
            )
        )
    return points


def upload_points(client: QdrantClient, points: list[models.PointStruct]) -> int:
    uploaded = 0
    for start in range(0, len(points), UPLOAD_BATCH_SIZE):
        batch = points[start : start + UPLOAD_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        uploaded += len(batch)
    return uploaded


def vector_norm(vector: list[float] | np.ndarray) -> float:
    array = np.asarray(vector, dtype=np.float32)
    return float(np.linalg.norm(array))


def validate_collection(
    client: QdrantClient,
    expected_count: int,
) -> dict[str, Any]:
    info = client.get_collection(COLLECTION_NAME)
    vector_params = info.config.params.vectors
    vector_size = getattr(vector_params, "size", None)
    distance = getattr(vector_params, "distance", None)

    sample_points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    if not sample_points:
        raise ValueError("collection contains no points")

    payload = sample_points[0].payload or {}
    vector = sample_points[0].vector

    checks = {
        "collection_exists": client.collection_exists(COLLECTION_NAME),
        "vector_dimension": vector_size,
        "distance": str(distance),
        "point_count": info.points_count,
        "payload_contains_language": bool(payload.get("language")),
        "payload_contains_text": bool(payload.get("text")),
        "sample_vector_dimension": len(vector),
        "sample_vector_norm": round(vector_norm(vector), 6),
    }

    if checks["point_count"] != expected_count:
        raise ValueError(f"point count mismatch: {checks['point_count']} != {expected_count}")
    if checks["vector_dimension"] != VECTOR_SIZE:
        raise ValueError(f"vector size mismatch: {checks['vector_dimension']}")
    if distance != models.Distance.COSINE:
        raise ValueError(f"distance mismatch: {distance}")
    if len(vector) != VECTOR_SIZE:
        raise ValueError(f"sample vector dimension mismatch: {len(vector)}")
    if not checks["payload_contains_language"] or not checks["payload_contains_text"]:
        raise ValueError("payload validation failed")

    return checks


def representative_vectors(client: QdrantClient) -> list[dict[str, Any]]:
    results = []
    for language in REPRESENTATIVE_LANGUAGES:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="language",
                        match=models.MatchValue(value=language),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            raise ValueError(f"no representative point found for {language}")
        point = points[0]
        results.append(
            {
                "language": language,
                "chunk_id": (point.payload or {}).get("chunk_id"),
                "vector_dimension": len(point.vector),
                "vector_norm": round(vector_norm(point.vector), 6),
            }
        )
    return results


def run_sanity_search(
    client: QdrantClient,
    model: SentenceTransformer,
) -> tuple[list[dict[str, Any]], float]:
    query_vector = model.encode(
        SANITY_QUERY_HI,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    ).points

    rows = []
    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}
        text = str(payload.get("text", ""))
        preview = " ".join(text.split())[:160]
        rows.append(
            {
                "rank": rank,
                "score": float(result.score),
                "language": payload.get("language"),
                "chunk_id": payload.get("chunk_id"),
                "preview": preview,
            }
        )

    return rows, vector_norm(query_vector)


def assert_normalized(embeddings: np.ndarray) -> list[float]:
    sample_indices = [0, len(embeddings) // 2, len(embeddings) - 1]
    norms = [vector_norm(embeddings[index]) for index in sample_indices]
    if not all(math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3) for norm in norms):
        raise ValueError(f"embedding norms are not approximately 1.0: {norms}")
    return [round(norm, 6) for norm in norms]


def main() -> None:
    print("=" * 70)
    print("HH GOA RAG - MULTILINGUAL EMBEDDING")
    print("=" * 70)

    chunks = load_chunks()
    language_counts = Counter(chunk["language"] for chunk in chunks)

    print(f"Loaded chunks: {len(chunks)}")
    print("Chunks per language:")
    for language in SUPPORTED_LANGUAGES:
        print(f"  {language}: {language_counts[language]}")

    print(f"\nLoading model: {MODEL_NAME}")
    timing = Timing()
    start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    timing.model_load_seconds = time.perf_counter() - start

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings...")
    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    timing.embedding_seconds = time.perf_counter() - start
    timing.avg_document_embedding_ms = (timing.embedding_seconds / len(chunks)) * 1000

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (len(chunks), VECTOR_SIZE):
        raise ValueError(f"unexpected embedding shape: {embeddings.shape}")

    sample_norms = assert_normalized(embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Representative vector norms: {sample_norms}")

    client = QdrantClient(path=QDRANT_PATH)
    try:
        create_or_validate_collection(client)

        print("\nUploading vectors...")
        points = build_points(chunks, embeddings)
        start = time.perf_counter()
        uploaded = upload_points(client, points)
        timing.upload_seconds = time.perf_counter() - start

        collection_validation = validate_collection(client, len(chunks))
        representative = representative_vectors(client)
        sanity_results, query_norm = run_sanity_search(client, model)

        report = {
            "model": MODEL_NAME,
            "dimension": VECTOR_SIZE,
            "batch_size": BATCH_SIZE,
            "collection": COLLECTION_NAME,
            "qdrant_path": QDRANT_PATH,
            "total_chunks": len(chunks),
            "vectors_generated": len(embeddings),
            "vectors_uploaded": uploaded,
            "language_counts": dict(language_counts),
            "timing": asdict(timing),
            "sample_vector_norms": sample_norms,
            "collection_validation": collection_validation,
            "representative_vectors": representative,
            "sanity_query": SANITY_QUERY_HI,
            "sanity_query_norm": round(query_norm, 6),
            "sanity_results": sanity_results,
        }
        REPORT_FILE.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\nSanity search top 10:")
        print("rank | score | language | chunk_id | preview")
        print("-" * 110)
        for row in sanity_results:
            print(
                f"{row['rank']:>4} | {row['score']:.6f} | "
                f"{row['language']:<8} | {row['chunk_id']} | {row['preview']}"
            )

        print("\n" + "=" * 70)
        print("MULTILINGUAL EMBEDDING COMPLETE")
        print("=" * 70)
        print(f"Model: {MODEL_NAME}")
        print(f"Dimension: {VECTOR_SIZE}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Vectors generated: {len(embeddings)}")
        print(f"Vectors uploaded: {uploaded}")
        print(f"Qdrant collection: {COLLECTION_NAME}")
        print(f"Vector dimension: {collection_validation['vector_dimension']}")
        print("Distance: COSINE")
        print(f"Languages: {len(SUPPORTED_LANGUAGES)}")
        print("\nLanguage | Chunks | Vectors")
        print("-" * 28)
        for language in SUPPORTED_LANGUAGES:
            count = language_counts[language]
            print(f"{language:2} | {count:6} | {count:7}")

        print("\nLatency:")
        print(f"Model loading time: {timing.model_load_seconds:.2f}s")
        print(f"Total embedding time: {timing.embedding_seconds:.2f}s")
        print(f"Average document embedding time: {timing.avg_document_embedding_ms:.4f}ms")
        print(f"Qdrant upload time: {timing.upload_seconds:.2f}s")
        print(f"\nReport: {REPORT_FILE}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
