import json
import os
import re
import hashlib

INPUT_FILE = "data/hindi_sample_1000.jsonl"
OUTPUT_FILE = "data/processed/passages_1000.jsonl"

MIN_WORDS = 5


def normalize_text(text):
    """
    Clean whitespace and basic formatting without
    changing the actual meaning of the passage.
    """

    if not isinstance(text, str):
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def make_passage_id(query_id, passage_index, text):
    """
    Create a deterministic ID for each passage.
    """

    text_hash = hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:10]

    return f"{query_id}_{passage_index}_{text_hash}"


def main():

    print("=" * 60)
    print("HH GOA RAG - STEP 5")
    print("Cleaning and Normalizing Hindi Passages")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    total_records = 0
    total_passages = 0
    kept_passages = 0
    removed_short = 0
    removed_duplicate = 0

    seen_hashes = set()

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as input_file, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        for line in input_file:

            if not line.strip():
                continue

            record = json.loads(line)

            total_records += 1

            query_id = record["query_id"]
            query = normalize_text(record["query"])
            answer = normalize_text(record["Answer"])

            passages = record[
                "passages"
            ][
                "Translated_passages"
            ]

            selected = record[
                "passages"
            ][
                "is_selected"
            ]

            for index, passage in enumerate(passages):

                total_passages += 1

                text = normalize_text(passage)

                # ----------------------------------------
                # Remove empty / extremely short passages
                # ----------------------------------------

                word_count = len(text.split())

                if word_count < MIN_WORDS:
                    removed_short += 1
                    continue

                # ----------------------------------------
                # Remove exact duplicates
                # ----------------------------------------

                text_hash = hashlib.sha1(
                    text.encode("utf-8")
                ).hexdigest()

                if text_hash in seen_hashes:
                    removed_duplicate += 1
                    continue

                seen_hashes.add(text_hash)

                # ----------------------------------------
                # Create cleaned passage
                # ----------------------------------------

                passage_id = make_passage_id(
                    query_id,
                    index,
                    text
                )

                cleaned = {
                    "passage_id": passage_id,
                    "query_id": query_id,
                    "passage_index": index,
                    "text": text,
                    "language": "hi",
                    "word_count": word_count,
                    "is_selected": bool(selected[index]),
                    "query": query,
                    "answer": answer,
                    "query_type": record.get(
                        "query_type"
                    )
                }

                output_file.write(
                    json.dumps(
                        cleaned,
                        ensure_ascii=False
                    ) + "\n"
                )

                kept_passages += 1

    print("\n" + "=" * 60)
    print("STEP 5 COMPLETE")
    print("=" * 60)

    print(f"Input records       : {total_records}")
    print(f"Input passages      : {total_passages}")
    print(f"Kept passages       : {kept_passages}")
    print(f"Removed (<5 words)  : {removed_short}")
    print(f"Removed duplicates   : {removed_duplicate}")

    print(f"\nOutput:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()