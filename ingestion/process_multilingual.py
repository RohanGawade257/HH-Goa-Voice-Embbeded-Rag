"""Build a unified multilingual passage corpus from repaired JSONL samples."""

from __future__ import annotations

import json
import re
import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


INPUT_DIR = Path("data/multilingual_repaired")
OUTPUT_DIR = Path("data/processed/multilingual")
OUTPUT_FILE = OUTPUT_DIR / "passages.jsonl"
REPORT_FILE = OUTPUT_DIR / "process_report.json"

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

MIN_WORDS = 5
LOSSY_REPLACEMENT = "�"


@dataclass
class LanguageStats:
    language: str
    input_records: int = 0
    invalid_json_records: int = 0
    input_passages: int = 0
    duplicates_removed: int = 0
    short_passages_removed: int = 0
    excluded_corrupted_records: int = 0
    output_passages: int = 0


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def make_passage_id(language: str, query_id: Any, passage_index: int, text: str) -> str:
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{language}_{query_id}_{passage_index}_{text_hash}"


def source_files() -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for language in SUPPORTED_LANGUAGES:
        path = INPUT_DIR / f"{language}_sample_1000.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        files.append((language, path))
    return files


def build_passage_record(
    language: str,
    record: dict[str, Any],
    passage_index: int,
    passage: str,
) -> dict[str, Any]:
    query_id = record["query_id"]
    query = normalize_text(record.get("query"))
    query_type = record.get("query_type")
    word_count = len(passage.split())

    return {
        "passage_id": make_passage_id(language, query_id, passage_index, passage),
        "query_id": query_id,
        "query": query,
        "language": language,
        "passage_index": passage_index,
        "passage": passage,
        "text": passage,
        "word_count": word_count,
        "is_selected": True,
        "query_type": query_type,
    }


def main() -> None:
    print("=" * 70)
    print("HH GOA RAG - MULTILINGUAL PROCESSING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {language: LanguageStats(language=language) for language in SUPPORTED_LANGUAGES}
    seen_passages: set[tuple[str, str]] = set()
    schema_counts: dict[str, int] = defaultdict(int)
    total_invalid_json = 0

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as output:
        for language, path in source_files():
            language_stats = stats[language]
            print(f"Processing {language}: {path}")

            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        language_stats.invalid_json_records += 1
                        total_invalid_json += 1
                        continue

                    language_stats.input_records += 1
                    schema_counts["|".join(record.keys())] += 1

                    query = normalize_text(record.get("query"))
                    passage = normalize_text(record.get("Answer"))
                    language_stats.input_passages += 1

                    if not query or not passage:
                        language_stats.short_passages_removed += 1
                        continue

                    if LOSSY_REPLACEMENT in passage:
                        language_stats.excluded_corrupted_records += 1
                        continue

                    word_count = len(passage.split())
                    if word_count < MIN_WORDS:
                        language_stats.short_passages_removed += 1
                        continue

                    passage_hash = hashlib.sha1(passage.encode("utf-8")).hexdigest()
                    dedupe_key = (language, passage_hash)
                    if dedupe_key in seen_passages:
                        language_stats.duplicates_removed += 1
                        continue

                    seen_passages.add(dedupe_key)
                    passage_record = build_passage_record(
                        language=language,
                        record=record,
                        passage_index=0,
                        passage=passage,
                    )

                    output.write(json.dumps(passage_record, ensure_ascii=False))
                    output.write("\n")
                    language_stats.output_passages += 1

    report = {
        "source_dir": str(INPUT_DIR),
        "output_file": str(OUTPUT_FILE),
        "supported_languages": list(SUPPORTED_LANGUAGES),
        "min_words": MIN_WORDS,
        "schema_counts": dict(schema_counts),
        "invalid_json_records": total_invalid_json,
        "languages": [asdict(stats[language]) for language in SUPPORTED_LANGUAGES],
        "totals": {
            "input_records": sum(item.input_records for item in stats.values()),
            "invalid_json_records": total_invalid_json,
            "input_passages": sum(item.input_passages for item in stats.values()),
            "duplicates_removed": sum(item.duplicates_removed for item in stats.values()),
            "short_passages_removed": sum(item.short_passages_removed for item in stats.values()),
            "excluded_corrupted_records": sum(
                item.excluded_corrupted_records for item in stats.values()
            ),
            "output_passages": sum(item.output_passages for item in stats.values()),
        },
    }
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nLanguage | Records | Input Passages | Duplicates | Short | Corrupted | Output")
    print("-" * 78)
    for language in SUPPORTED_LANGUAGES:
        item = stats[language]
        print(
            f"{language:2} | {item.input_records:7} | {item.input_passages:14} | "
            f"{item.duplicates_removed:10} | {item.short_passages_removed:5} | "
            f"{item.excluded_corrupted_records:9} | {item.output_passages:6}"
        )

    print("\nOutput:")
    print(OUTPUT_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()
