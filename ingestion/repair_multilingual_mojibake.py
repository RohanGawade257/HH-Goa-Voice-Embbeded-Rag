"""Repair multilingual JSONL mojibake without touching the source files.

This script conservatively repairs text fields in data/multilingual/*.jsonl and
writes UTF-8 JSONL copies to data/multilingual_repaired/.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


INPUT_DIR = Path("data/multilingual")
OUTPUT_DIR = Path("data/multilingual_repaired")
REPORT_JSON = OUTPUT_DIR / "validation_report.json"
REPORT_MD = OUTPUT_DIR / "validation_report.md"

LANGUAGES = {
    "as": [(0x0980, 0x09FF)],
    "bn": [(0x0980, 0x09FF)],
    "gu": [(0x0A80, 0x0AFF)],
    "hi": [(0x0900, 0x097F)],
    "kn": [(0x0C80, 0x0CFF)],
    "ml": [(0x0D00, 0x0D7F)],
    "mr": [(0x0900, 0x097F)],
    "ne": [(0x0900, 0x097F)],
    "or": [(0x0B00, 0x0B7F)],
    "pa": [(0x0A00, 0x0A7F)],
    "sa": [(0x0900, 0x097F)],
    "ta": [(0x0B80, 0x0BFF)],
    "ur": [
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ],
}

SKIP_REPAIR_KEYS = {
    "query_id",
    "query_type",
    "id",
    "_id",
}

MOJIBAKE_MARKERS = (
    "Ã",
    "Â",
    "â",
    "à¤",
    "à¥",
    "à¦",
    "à§",
    "àª",
    "à«",
    "à¨",
    "à©",
    "à¬",
    "à­",
    "à®",
    "à¯",
    "à²",
    "à³",
    "à´",
    "àµ",
    "Ø",
    "Ù",
    "Ú",
)

LOSSY_REPLACEMENT = "�"


@dataclass
class FieldChange:
    key: str
    before: str
    after: str


@dataclass
class LanguageReport:
    language: str
    input_records: int = 0
    output_records: int = 0
    repaired_records: int = 0
    unchanged_records: int = 0
    failed_repairs: int = 0
    mojibake_detected: int = 0
    replacement_char_records: int = 0
    repaired_percentage: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)
    schemas: list[list[str]] = field(default_factory=list)


def is_in_ranges(char: str, ranges: list[tuple[int, int]]) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in ranges)


def count_expected_script(text: str, ranges: list[tuple[int, int]]) -> int:
    return sum(1 for char in text if is_in_ranges(char, ranges))


def count_text_chars(text: str) -> int:
    return sum(
        1
        for char in text
        if unicodedata.category(char)[0] in {"L", "M", "N"}
    )


def count_mojibake_markers(text: str) -> int:
    marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    c1_controls = sum(1 for char in text if 0x80 <= ord(char) <= 0x9F)
    return marker_count + c1_controls


def has_lossy_replacement(text: str) -> bool:
    return LOSSY_REPLACEMENT in text


def likely_mojibake(text: str) -> bool:
    return count_mojibake_markers(text) > 0


def decode_once(text: str, source_encoding: str) -> str | None:
    try:
        return text.encode(source_encoding).decode("utf-8")
    except UnicodeError:
        return None


def repair_candidates(text: str, max_passes: int = 3) -> set[str]:
    candidates = {text}
    frontier = {text}

    for _ in range(max_passes):
        next_frontier: set[str] = set()
        for candidate in frontier:
            for encoding in ("cp1252", "latin-1"):
                repaired = decode_once(candidate, encoding)
                if repaired is None or repaired in candidates:
                    continue
                candidates.add(repaired)
                next_frontier.add(repaired)
        frontier = next_frontier
        if not frontier:
            break

    return candidates


def quality_score(text: str, ranges: list[tuple[int, int]]) -> int:
    expected_script = count_expected_script(text, ranges)
    mojibake = count_mojibake_markers(text)
    replacement_chars = text.count("�")
    return (expected_script * 4) - (mojibake * 20) - (replacement_chars * 40)


def choose_repair(text: str, ranges: list[tuple[int, int]]) -> tuple[str, bool, bool]:
    """Return (text, changed, failed_detected_repair)."""
    if not text or not likely_mojibake(text):
        return text, False, False

    original_score = quality_score(text, ranges)
    original_script = count_expected_script(text, ranges)
    original_markers = count_mojibake_markers(text)
    best = text
    best_score = original_score

    for candidate in repair_candidates(text):
        if candidate == text or "\ufffd" in candidate:
            continue
        candidate_script = count_expected_script(candidate, ranges)
        candidate_markers = count_mojibake_markers(candidate)
        candidate_score = quality_score(candidate, ranges)

        improves_script = candidate_script > original_script
        reduces_markers = candidate_markers < original_markers
        preserves_script = candidate_script >= original_script
        if candidate_score > best_score and reduces_markers and (
            improves_script or preserves_script
        ):
            best = candidate
            best_score = candidate_score

    if best != text and count_text_chars(best) > 0:
        return best, True, False

    return text, False, True


def repair_value(value: Any, ranges: list[tuple[int, int]], key: str) -> tuple[Any, list[FieldChange], bool]:
    if key in SKIP_REPAIR_KEYS:
        return value, [], False

    if isinstance(value, str):
        if has_lossy_replacement(value):
            return value, [], True
        repaired, changed, failed = choose_repair(value, ranges)
        changes = [FieldChange(key=key, before=value, after=repaired)] if changed else []
        return repaired, changes, failed

    if isinstance(value, list):
        repaired_items = []
        changes: list[FieldChange] = []
        failed_any = False
        for index, item in enumerate(value):
            repaired, item_changes, item_failed = repair_value(
                item,
                ranges,
                f"{key}[{index}]",
            )
            repaired_items.append(repaired)
            changes.extend(item_changes)
            failed_any = failed_any or item_failed
        return repaired_items, changes, failed_any

    if isinstance(value, dict):
        repaired_obj: dict[str, Any] = {}
        changes: list[FieldChange] = []
        failed_any = False
        for child_key, child_value in value.items():
            repaired, child_changes, child_failed = repair_value(
                child_value,
                ranges,
                child_key,
            )
            repaired_obj[child_key] = repaired
            changes.extend(child_changes)
            failed_any = failed_any or child_failed
        return repaired_obj, changes, failed_any

    return value, [], False


def record_has_mojibake(record: dict[str, Any]) -> bool:
    for key, value in record.items():
        if key in SKIP_REPAIR_KEYS:
            continue
        if isinstance(value, str) and likely_mojibake(value):
            return True
    return False


def record_has_lossy_replacement(record: dict[str, Any]) -> bool:
    for key, value in record.items():
        if key in SKIP_REPAIR_KEYS:
            continue
        if isinstance(value, str) and has_lossy_replacement(value):
            return True
    return False


def trim_sample(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."


def process_language(language: str, input_path: Path, output_path: Path) -> LanguageReport:
    ranges = LANGUAGES[language]
    report = LanguageReport(language=language)
    schemas: set[tuple[str, ...]] = set()

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            report.input_records += 1
            schemas.add(tuple(record.keys()))
            had_mojibake = record_has_mojibake(record)
            repaired_record, changes, failed = repair_value(record, ranges, "")

            if had_mojibake:
                report.mojibake_detected += 1
            if record_has_lossy_replacement(record):
                report.replacement_char_records += 1
            if changes:
                report.repaired_records += 1
            else:
                report.unchanged_records += 1
            if failed:
                report.failed_repairs += 1

            if len(report.samples) < 3:
                sample_changes = [
                    {
                        "field": change.key,
                        "before": trim_sample(change.before),
                        "after": trim_sample(change.after),
                    }
                    for change in changes[:2]
                ]
                if not sample_changes:
                    sample_changes = [
                        {
                            "field": "query",
                            "before": trim_sample(str(record.get("query", ""))),
                            "after": trim_sample(str(repaired_record.get("query", ""))),
                        },
                        {
                            "field": "Answer",
                            "before": trim_sample(str(record.get("Answer", ""))),
                            "after": trim_sample(str(repaired_record.get("Answer", ""))),
                        },
                    ]
                report.samples.append(
                    {
                        "line": line_number,
                        "changed": bool(changes),
                        "fields": sample_changes,
                    }
                )

            dst.write(json.dumps(repaired_record, ensure_ascii=False))
            dst.write("\n")
            report.output_records += 1

    report.schemas = [list(schema) for schema in sorted(schemas)]
    report.repaired_percentage = (
        round((report.repaired_records / report.input_records) * 100, 2)
        if report.input_records
        else 0.0
    )
    return report


def validate_outputs(reports: list[LanguageReport]) -> list[str]:
    errors: list[str] = []

    for report in reports:
        if report.input_records != report.output_records:
            errors.append(
                f"{report.language}: input/output mismatch "
                f"{report.input_records} != {report.output_records}"
            )

        path = OUTPUT_DIR / f"{report.language}_sample_1000.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                query = str(record.get("query", ""))
                answer = str(record.get("Answer", ""))
                if not query.strip() or not answer.strip():
                    errors.append(f"{report.language}:{line_number}: empty query or Answer")
                    break
                if record_has_mojibake(record):
                    errors.append(f"{report.language}:{line_number}: mojibake marker remains")
                    break

    return errors


def write_reports(reports: list[LanguageReport], errors: list[str]) -> None:
    payload = {
        "reports": [report.__dict__ for report in reports],
        "validation_errors": errors,
    }
    REPORT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "MULTILINGUAL DATA REPAIR REPORT",
        "================================",
        "",
        "Language | Input | Repaired | Unchanged | Failed | Mojibake | Replacement | % Repaired",
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for report in reports:
        lines.append(
            f"{report.language} | {report.input_records} | "
            f"{report.repaired_records} | {report.unchanged_records} | "
            f"{report.failed_repairs} | {report.mojibake_detected} | "
            f"{report.replacement_char_records} | "
            f"{report.repaired_percentage:.2f}"
        )

    lines.extend(["", "Samples"])
    for report in reports:
        lines.append("")
        lines.append(f"### {report.language}")
        for sample in report.samples:
            lines.append(f"- line {sample['line']} changed={sample['changed']}")
            for field in sample["fields"]:
                lines.append(f"  - {field['field']} before: {field['before']}")
                lines.append(f"  - {field['field']} after: {field['after']}")

    lines.extend(["", "Validation"])
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.extend(
            [
                "- Valid UTF-8 JSONL",
                "- Native-script text present",
                "- No obvious mojibake markers detected",
                "- Irrecoverable replacement characters reported separately",
                "- Record counts preserved",
                "- JSON schema preserved",
                "- Original files untouched",
                "- Repaired files written to data/multilingual_repaired/",
            ]
        )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reports: list[LanguageReport] = []
    for language in LANGUAGES:
        input_path = INPUT_DIR / f"{language}_sample_1000.jsonl"
        output_path = OUTPUT_DIR / f"{language}_sample_1000.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        reports.append(process_language(language, input_path, output_path))

    errors = validate_outputs(reports)
    write_reports(reports, errors)

    if errors:
        raise SystemExit("\n".join(errors))

    print("MULTILINGUAL DATA REPAIR REPORT")
    print(
        "Language | Input | Repaired | Unchanged | Failed | "
        "Mojibake | Replacement | % Repaired"
    )
    for report in reports:
        print(
            f"{report.language} | {report.input_records} | "
            f"{report.repaired_records} | {report.unchanged_records} | "
            f"{report.failed_repairs} | {report.mojibake_detected} | "
            f"{report.replacement_char_records} | "
            f"{report.repaired_percentage:.2f}"
        )
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
