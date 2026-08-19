import json
import os
import time

import fsspec
import pyarrow.parquet as pq


# ============================================================
# CONFIG
# ============================================================

REPO_ID = "ai4bharat/MSMARCO-XI"

SAMPLE_SIZE = 1000

OUTPUT_DIR = "data/multilingual"

HTTP_BLOCK_SIZE = 8 * 1024 * 1024


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {
   
    "bn": ("Bengali", "train/bentrain.parquet"),
    "gu": ("Gujarati", "train/gujtrain.parquet"),
    "hi": ("Hindi", "train/hintrain.parquet"),
    "kn": ("Kannada", "train/kantrain.parquet"),
    "ml": ("Malayalam", "train/maltrain.parquet"),
    "mr": ("Marathi", "train/martrain.parquet"),
    "ne": ("Nepali", "train/neptrain.parquet"),
    "or": ("Odia", "train/oritrain.parquet"),
    "pa": ("Punjabi", "train/pantrain.parquet"),
    "sa": ("Sanskrit", "train/santrain.parquet"),
    "ta": ("Tamil", "train/tamtrain.parquet"),
    "te": ("Telugu", "train/teltrain.parquet"),
    "ur": ("Urdu", "train/urdtrain.parquet"),
}


# ============================================================
# ONLY SMALL COLUMNS
# ============================================================

COLUMNS = [
    "query",
    "Answer",
    "query_id",
    "query_type",
]


# ============================================================
# OUTPUT
# ============================================================

def output_path(language_code):
    return os.path.join(
        OUTPUT_DIR,
        f"{language_code}_sample_{SAMPLE_SIZE}.jsonl"
    )


# ============================================================
# URL
# ============================================================

def make_url(filename):
    return (
        f"https://huggingface.co/datasets/"
        f"{REPO_ID}/resolve/main/{filename}"
    )


# ============================================================
# SAMPLE
# ============================================================

def sample_language(
    language_code,
    language_name,
    filename,
):

    print()
    print("=" * 70)
    print(f"{language_code.upper()} | {language_name}")
    print("=" * 70)

    url = make_url(filename)

    print(f"URL: {url}")
    print()
    print("Columns:")
    
    for column in COLUMNS:
        print(f"  - {column}")

    print()
    print(
        f"Target rows: {SAMPLE_SIZE:,}"
    )

    print()
    print("Opening remote Parquet...")

    start = time.perf_counter()

    try:

        # ----------------------------------------------------
        # HTTP FILESYSTEM
        # ----------------------------------------------------

        fs = fsspec.filesystem(
            "http",
            block_size=HTTP_BLOCK_SIZE,
            cache_type="bytes",
        )

        # ----------------------------------------------------
        # OPEN REMOTE FILE
        # ----------------------------------------------------

        with fs.open(url, "rb") as remote_file:

            parquet = pq.ParquetFile(
                remote_file
            )

            metadata_time = (
                time.perf_counter()
                - start
            )

            total_rows = (
                parquet.metadata.num_rows
            )

            row_groups = (
                parquet.metadata.num_row_groups
            )

            print()
            print(
                f"Metadata time : "
                f"{metadata_time:.2f}s"
            )

            print(
                f"Total rows    : "
                f"{total_rows:,}"
            )

            print(
                f"Row groups    : "
                f"{row_groups}"
            )

            print()
            print(
                "Reading ONLY selected columns..."
            )

            # ------------------------------------------------
            # ITERATE BATCHES
            # ------------------------------------------------

            read_start = time.perf_counter()

            rows = []

            for batch in parquet.iter_batches(
                batch_size=SAMPLE_SIZE,
                columns=COLUMNS,
                use_threads=True,
            ):

                batch_rows = batch.to_pylist()

                rows.extend(batch_rows)

                # Stop immediately after 1000.
                if len(rows) >= SAMPLE_SIZE:
                    rows = rows[:SAMPLE_SIZE]
                    break

            read_time = (
                time.perf_counter()
                - read_start
            )

            # ------------------------------------------------
            # WRITE
            # ------------------------------------------------

            os.makedirs(
                OUTPUT_DIR,
                exist_ok=True
            )

            output_file = output_path(
                language_code
            )

            print()
            print(
                f"Writing {len(rows):,} rows..."
            )

            with open(
                output_file,
                "w",
                encoding="utf-8",
            ) as f:

                for record in rows:

                    f.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )

            total_time = (
                time.perf_counter()
                - start
            )

            print()
            print("-" * 70)
            print("SUCCESS")
            print("-" * 70)

            print(
                f"Language       : {language_name}"
            )

            print(
                f"Rows           : {len(rows):,}"
            )

            print(
                f"Read time      : {read_time:.2f}s"
            )

            print(
                f"Total time     : {total_time:.2f}s"
            )

            print(
                f"Output         : {output_file}"
            )

            print(
                "Full 3+ GB file cached : NO"
            )

            print("-" * 70)

            return len(rows)

    except Exception as e:

        print()
        print("-" * 70)
        print("FAILED")
        print("-" * 70)

        print(
            f"{type(e).__name__}: {e}"
        )

        print(
            str(e)
        )

        print("-" * 70)

        return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HH GOA RAG - MINIMAL REMOTE PARQUET SAMPLER")
    print("=" * 70)

    print()
    print(
        f"Dataset   : {REPO_ID}"
    )

    print(
        f"Sample    : {SAMPLE_SIZE:,} rows/language"
    )

    print(
        f"Languages : {len(LANGUAGES)}"
    )

    print(
        f"Output    : {OUTPUT_DIR}"
    )

    print()
    print(
        "Strategy:"
    )

    print(
        "REMOTE PARQUET"
        " → HTTP RANGE"
        " → SMALL COLUMNS"
        " → FIRST 1000"
        " → JSONL"
    )

    print()
    print(
        "Selected columns:"
    )

    for column in COLUMNS:
        print(f"  - {column}")

    print()
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results = {}

    total_start = time.perf_counter()

    # ========================================================
    # PROCESS LANGUAGES
    # ========================================================

    for (
        language_code,
        (
            language_name,
            filename,
        )
    ) in LANGUAGES.items():

        count = sample_language(
            language_code,
            language_name,
            filename,
        )

        results[language_code] = count

    # ========================================================
    # FINAL REPORT
    # ========================================================

    total_time = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    total_rows = 0
    completed = 0

    print()

    for (
        language_code,
        (
            language_name,
            _,
        )
    ) in LANGUAGES.items():

        count = results[
            language_code
        ]

        total_rows += count

        if count == SAMPLE_SIZE:

            status = "COMPLETE"
            completed += 1

        elif count > 0:

            status = "PARTIAL"

        else:

            status = "FAILED"

        print(
            f"{language_code:2} | "
            f"{language_name:12} | "
            f"{status:8} | "
            f"{count:5} rows"
        )

    print()
    print("-" * 70)

    print(
        f"Languages completed : "
        f"{completed}/{len(LANGUAGES)}"
    )

    print(
        f"Total rows          : "
        f"{total_rows:,}"
    )

    print(
        f"Expected rows       : "
        f"{len(LANGUAGES) * SAMPLE_SIZE:,}"
    )

    print(
        f"Total time          : "
        f"{total_time:.2f}s"
    )

    print()
    print(
        f"Output directory: "
        f"{os.path.abspath(OUTPUT_DIR)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()