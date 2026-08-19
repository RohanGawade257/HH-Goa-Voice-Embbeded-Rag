"""Chunk unified multilingual passages and validate the output corpus."""

from __future__ import annotations

import json
import re
import hashlib
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


INPUT_FILE = Path("data/processed/multilingual/passages.jsonl")
OUTPUT_DIR = Path("data/processed/multilingual")
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"
PROCESS_REPORT_FILE = OUTPUT_DIR / "process_report.json"
REPORT_JSON = OUTPUT_DIR / "multilingual_processing_report.json"
REPORT_MD = OUTPUT_DIR / "multilingual_processing_report.md"

SUPPORTED_LANGUAGES = {
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
}

SHORT_LIMIT = 80
MEDIUM_LIMIT = 180

TARGET_WORDS = 100
MAX_WORDS = 140

OVERLAP_SENTENCES = 1

FALLBACK_WORDS = 120
FALLBACK_OVERLAP_WORDS = 20

LOSSY_REPLACEMENT = "�"
SENTENCE_END_RE = re.compile(r"(?<=[।॥۔؟.!?])\s+")


@dataclass
class LanguageChunkStats:
    language: str
    input_records: int = 0
    input_passages: int = 0
    duplicates_removed: int = 0
    short_passages_removed: int = 0
    excluded_corrupted_records: int = 0
    output_passages: int = 0
    output_chunks: int = 0
    min_chunk_words: int = 0
    max_chunk_words: int = 0
    average_chunk_words: float = 0.0
    chunks_with_missing_language: int = 0
    chunks_containing_replacement: int = 0
    strategy_counts: dict[str, int] = field(default_factory=dict)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [sentence.strip() for sentence in SENTENCE_END_RE.split(text) if sentence.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def make_chunk_id(passage_id: str, strategy: str, index: int, text: str) -> str:
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{passage_id}_{strategy}_{index}_{text_hash}"


def split_long_sentence(sentence: str) -> list[dict[str, Any]]:
    words = sentence.split()
    if len(words) <= MAX_WORDS:
        return [
            {
                "text": sentence,
                "strategy": "sentence_group",
                "overlap": False,
            }
        ]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + FALLBACK_WORDS, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(
            {
                "text": chunk_text,
                "strategy": "long_sentence_fallback",
                "overlap": start > 0,
            }
        )

        if end >= len(words):
            break

        next_start = end - FALLBACK_OVERLAP_WORDS
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def intact_chunk(text: str) -> list[dict[str, Any]]:
    return [
        {
            "text": text,
            "strategy": "intact",
            "overlap": False,
        }
    ]


def sentence_group_chunk(text: str) -> list[dict[str, Any]]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []

    for sentence in sentences:
        sentence_words = word_count(sentence)

        if sentence_words > MAX_WORDS:
            if current:
                chunks.append(
                    {
                        "text": " ".join(current),
                        "strategy": "sentence_group",
                        "overlap": False,
                    }
                )
                current = []

            chunks.extend(split_long_sentence(sentence))
            continue

        candidate = " ".join(current + [sentence])
        if current and word_count(candidate) > MAX_WORDS:
            chunks.append(
                {
                    "text": " ".join(current),
                    "strategy": "sentence_group",
                    "overlap": False,
                }
            )
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        chunks.append(
            {
                "text": " ".join(current),
                "strategy": "sentence_group",
                "overlap": False,
            }
        )

    return chunks


def adaptive_window_chunk(text: str) -> list[dict[str, Any]]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    start = 0

    while start < len(sentences):
        current = []
        current_words = 0
        end = start

        while end < len(sentences):
            sentence = sentences[end]
            sentence_words = word_count(sentence)

            if sentence_words > MAX_WORDS:
                if current:
                    chunks.append(
                        {
                            "text": " ".join(current),
                            "strategy": "adaptive_window",
                            "overlap": start > 0,
                        }
                    )
                    current = []
                    current_words = 0

                chunks.extend(split_long_sentence(sentence))
                end += 1
                start = end
                break

            if current and current_words + sentence_words > MAX_WORDS:
                break

            current.append(sentence)
            current_words += sentence_words
            end += 1

            if current_words >= TARGET_WORDS:
                break
        else:
            end = len(sentences)

        if current:
            chunks.append(
                {
                    "text": " ".join(current),
                    "strategy": "adaptive_window",
                    "overlap": start > 0,
                }
            )

        if end >= len(sentences):
            break

        next_start = end - OVERLAP_SENTENCES
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def create_chunks(text: str) -> list[dict[str, Any]]:
    words = word_count(text)
    if words <= SHORT_LIMIT:
        return intact_chunk(text)
    if words <= MEDIUM_LIMIT:
        return sentence_group_chunk(text)
    return adaptive_window_chunk(text)


def load_process_stats() -> dict[str, LanguageChunkStats]:
    stats = {
        language: LanguageChunkStats(language=language)
        for language in sorted(SUPPORTED_LANGUAGES)
    }

    if not PROCESS_REPORT_FILE.exists():
        return stats

    report = json.loads(PROCESS_REPORT_FILE.read_text(encoding="utf-8"))
    for item in report.get("languages", []):
        language = item["language"]
        if language not in stats:
            continue
        stats[language].input_records = item.get("input_records", 0)
        stats[language].input_passages = item.get("input_passages", 0)
        stats[language].duplicates_removed = item.get("duplicates_removed", 0)
        stats[language].short_passages_removed = item.get("short_passages_removed", 0)
        stats[language].excluded_corrupted_records = item.get(
            "excluded_corrupted_records",
            0,
        )
        stats[language].output_passages = item.get("output_passages", 0)

    return stats


def build_chunk_record(
    passage: dict[str, Any],
    chunk: dict[str, Any],
    chunk_index: int,
) -> dict[str, Any]:
    chunk_text = chunk["text"]
    return {
        "chunk_id": make_chunk_id(
            passage["passage_id"],
            chunk["strategy"],
            chunk_index,
            chunk_text,
        ),
        "passage_id": passage["passage_id"],
        "query_id": passage["query_id"],
        "passage_index": passage["passage_index"],
        "chunk_index": chunk_index,
        "text": chunk_text,
        "language": passage["language"],
        "word_count": word_count(chunk_text),
        "chunk_strategy": chunk["strategy"],
        "has_overlap": chunk["overlap"],
        "is_selected": passage["is_selected"],
        "query_type": passage["query_type"],
    }


def update_chunk_stats(
    stats: LanguageChunkStats,
    chunk_record: dict[str, Any],
    chunk_lengths: list[int],
) -> None:
    stats.output_chunks += 1
    chunk_lengths.append(chunk_record["word_count"])
    if not chunk_record.get("language"):
        stats.chunks_with_missing_language += 1
    if LOSSY_REPLACEMENT in chunk_record["text"]:
        stats.chunks_containing_replacement += 1
    strategy = chunk_record["chunk_strategy"]
    stats.strategy_counts[strategy] = stats.strategy_counts.get(strategy, 0) + 1


def finalize_stats(
    stats_by_language: dict[str, LanguageChunkStats],
    chunk_lengths_by_language: dict[str, list[int]],
) -> None:
    for language, stats in stats_by_language.items():
        lengths = chunk_lengths_by_language[language]
        if not lengths:
            continue
        stats.min_chunk_words = min(lengths)
        stats.max_chunk_words = max(lengths)
        stats.average_chunk_words = round(statistics.mean(lengths), 2)


def validate_outputs(
    stats_by_language: dict[str, LanguageChunkStats],
    chunk_ids: set[str],
    passage_ids: set[str],
) -> dict[str, Any]:
    errors = []
    invalid_json_records = 0
    missing_language = 0
    unknown_language = 0
    oversized_chunks = 0
    empty_text = 0
    e5_prefixes = 0
    invalid_passage_refs = 0

    seen_chunk_ids: set[str] = set()

    with OUTPUT_FILE.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_records += 1
                continue

            chunk_id = chunk.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                errors.append(f"duplicate chunk_id at line {line_number}: {chunk_id}")
            seen_chunk_ids.add(chunk_id)

            language = chunk.get("language")
            if not language:
                missing_language += 1
            elif language not in SUPPORTED_LANGUAGES:
                unknown_language += 1

            text = str(chunk.get("text", ""))
            if not text.strip():
                empty_text += 1
            if text.lower().startswith(("query:", "passage:")):
                e5_prefixes += 1
            if int(chunk.get("word_count", 0)) > MAX_WORDS:
                oversized_chunks += 1
            if chunk.get("passage_id") not in passage_ids:
                invalid_passage_refs += 1

    if seen_chunk_ids != chunk_ids:
        errors.append("chunk_id accounting mismatch")

    return {
        "errors": errors,
        "invalid_json_records": invalid_json_records,
        "missing_language": missing_language,
        "unknown_language": unknown_language,
        "oversized_chunks": oversized_chunks,
        "empty_text": empty_text,
        "e5_prefixes": e5_prefixes,
        "invalid_passage_refs": invalid_passage_refs,
        "unique_chunk_ids": len(seen_chunk_ids),
        "valid_passage_ids": len(passage_ids),
        "missing_language_from_stats": sum(
            item.chunks_with_missing_language for item in stats_by_language.values()
        ),
    }


def write_reports(
    stats_by_language: dict[str, LanguageChunkStats],
    validation: dict[str, Any],
) -> None:
    languages = [asdict(stats_by_language[language]) for language in sorted(SUPPORTED_LANGUAGES)]
    totals = {
        "input_records": sum(item.input_records for item in stats_by_language.values()),
        "input_passages": sum(item.input_passages for item in stats_by_language.values()),
        "duplicates_removed": sum(item.duplicates_removed for item in stats_by_language.values()),
        "short_passages_removed": sum(
            item.short_passages_removed for item in stats_by_language.values()
        ),
        "excluded_corrupted_records": sum(
            item.excluded_corrupted_records for item in stats_by_language.values()
        ),
        "output_passages": sum(item.output_passages for item in stats_by_language.values()),
        "output_chunks": sum(item.output_chunks for item in stats_by_language.values()),
        "chunks_with_missing_language": sum(
            item.chunks_with_missing_language for item in stats_by_language.values()
        ),
        "chunks_containing_replacement": sum(
            item.chunks_containing_replacement for item in stats_by_language.values()
        ),
        "maximum_chunk_words": max(
            (item.max_chunk_words for item in stats_by_language.values()),
            default=0,
        ),
    }

    report = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "chunk_parameters": {
            "short_limit": SHORT_LIMIT,
            "medium_limit": MEDIUM_LIMIT,
            "target_words": TARGET_WORDS,
            "max_words": MAX_WORDS,
            "overlap_sentences": OVERLAP_SENTENCES,
            "fallback_words": FALLBACK_WORDS,
            "fallback_overlap_words": FALLBACK_OVERLAP_WORDS,
        },
        "languages": languages,
        "totals": totals,
        "validation": validation,
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "MULTILINGUAL PROCESSING REPORT",
        "===============================",
        "",
        (
            "Language | Input records | Input passages | Duplicates removed | "
            "Output passages | Output chunks | Min words | Max words | Avg words | "
            "Missing language | Chunks with replacement"
        ),
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for language in sorted(SUPPORTED_LANGUAGES):
        item = stats_by_language[language]
        lines.append(
            f"{language} | {item.input_records} | {item.input_passages} | "
            f"{item.duplicates_removed} | {item.output_passages} | "
            f"{item.output_chunks} | {item.min_chunk_words} | "
            f"{item.max_chunk_words} | {item.average_chunk_words:.2f} | "
            f"{item.chunks_with_missing_language} | {item.chunks_containing_replacement}"
        )

    lines.extend(
        [
            "",
            "Totals",
            f"- Input records: {totals['input_records']}",
            f"- Input passages: {totals['input_passages']}",
            f"- Duplicates removed: {totals['duplicates_removed']}",
            f"- Short passages removed: {totals['short_passages_removed']}",
            f"- Excluded corrupted records: {totals['excluded_corrupted_records']}",
            f"- Output passages: {totals['output_passages']}",
            f"- Output chunks: {totals['output_chunks']}",
            f"- Maximum chunk words: {totals['maximum_chunk_words']}",
            f"- Missing language metadata: {validation['missing_language']}",
            f"- Invalid JSON records: {validation['invalid_json_records']}",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("=" * 70)
    print("HH GOA RAG - MULTILINGUAL CHUNKING")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_by_language = load_process_stats()
    chunk_lengths_by_language = defaultdict(list)
    passage_ids = set()
    chunk_ids = set()

    with INPUT_FILE.open("r", encoding="utf-8") as input_file, OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        for line in input_file:
            if not line.strip():
                continue

            passage = json.loads(line)
            language = passage["language"]
            passage_ids.add(passage["passage_id"])

            chunks = create_chunks(passage["text"])
            for index, chunk in enumerate(chunks):
                chunk_record = build_chunk_record(passage, chunk, index)
                chunk_id = chunk_record["chunk_id"]
                if chunk_id in chunk_ids:
                    raise ValueError(f"duplicate chunk_id generated: {chunk_id}")
                chunk_ids.add(chunk_id)

                output_file.write(json.dumps(chunk_record, ensure_ascii=False))
                output_file.write("\n")
                update_chunk_stats(
                    stats_by_language[language],
                    chunk_record,
                    chunk_lengths_by_language[language],
                )

    finalize_stats(stats_by_language, chunk_lengths_by_language)
    validation = validate_outputs(stats_by_language, chunk_ids, passage_ids)
    write_reports(stats_by_language, validation)

    if validation["errors"]:
        raise SystemExit("\n".join(validation["errors"]))
    if any(
        validation[key] != 0
        for key in (
            "invalid_json_records",
            "missing_language",
            "unknown_language",
            "oversized_chunks",
            "empty_text",
            "e5_prefixes",
            "invalid_passage_refs",
        )
    ):
        raise SystemExit(f"validation failed: {validation}")

    print("\nLanguage | Records | Passages | Chunks")
    print("-" * 39)
    for language in sorted(SUPPORTED_LANGUAGES):
        item = stats_by_language[language]
        print(
            f"{language:2} | {item.input_records:7} | "
            f"{item.output_passages:8} | {item.output_chunks:6}"
        )

    print("-" * 39)
    totals = json.loads(REPORT_JSON.read_text(encoding="utf-8"))["totals"]
    print(
        f"TOTAL | {totals['input_records']:5} | "
        f"{totals['output_passages']:8} | {totals['output_chunks']:6}"
    )
    print("\nOutput:")
    print(OUTPUT_FILE)
    print(REPORT_JSON)
    print(REPORT_MD)


if __name__ == "__main__":
    main()
