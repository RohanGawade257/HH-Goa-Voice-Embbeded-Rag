import json
import os
import time

import fsspec
import pyarrow.parquet as pq


# ============================================================
# CONFIG
# ============================================================

REPO_ID = "ai4bharat/MSMARCO-XI"

# ------------------------------------------------------------
# EXISTING DATA
# ------------------------------------------------------------
# Each language already has 1,000 rows locally.
# We DO NOT download those rows again.
# ------------------------------------------------------------

EXISTING_ROWS = 1000

# ------------------------------------------------------------
# NEW DATA TO ADD
# ------------------------------------------------------------
# Download rows:
#
# 1000 -> 4999
#
# = 4,000 new rows
# ------------------------------------------------------------

ROWS_TO_ADD = 4000

START_ROW = EXISTING_ROWS

END_ROW = START_ROW + ROWS_TO_ADD

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
        f"{language_code}_extra_{ROWS_TO_ADD}.jsonl"
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
# SAMPLE ADDITIONAL ROWS
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
        f"Existing rows     : {EXISTING_ROWS:,}"
    )

    print(
        f"Rows to add        : {ROWS_TO_ADD:,}"
    )

    print(
        f"Source rows        : "
        f"{START_ROW:,} -> {END_ROW - 1:,}"
    )

    print(
        f"Final total        : "
        f"{END_ROW:,}"
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

            # ------------------------------------------------
            # SAFETY CHECK
            # ------------------------------------------------

            if total_rows < END_ROW:

                print()
                print(
                    "ERROR:"
                )

                print(
                    f"Dataset contains only "
                    f"{total_rows:,} rows."
                )

                print(
                    f"Need at least "
                    f"{END_ROW:,} rows."
                )

                return 0

            print()
            print(
                "Reading ONLY selected columns..."
            )

            print(
                f"Skipping first "
                f"{START_ROW:,} rows..."
            )

            print(
                f"Collecting next "
                f"{ROWS_TO_ADD:,} rows..."
            )

            # ------------------------------------------------
            # ITERATE BATCHES
            # ------------------------------------------------

            read_start = time.perf_counter()

            rows = []

            rows_seen = 0

            # Use reasonably sized batches.
            batch_size = 1000

            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=COLUMNS,
                use_threads=True,
            ):

                batch_rows = batch.to_pylist()

                batch_start = rows_seen

                batch_end = (
                    rows_seen
                    + len(batch_rows)
                )

                rows_seen = batch_end

                # ------------------------------------------------
                # ENTIRE BATCH IS BEFORE START_ROW
                # ------------------------------------------------

                if batch_end <= START_ROW:
                    continue

                # ------------------------------------------------
                # BATCH CROSSES START_ROW
                # ------------------------------------------------

                if batch_start < START_ROW:

                    skip_inside_batch = (
                        START_ROW
                        - batch_start
                    )

                    batch_rows = batch_rows[
                        skip_inside_batch:
                    ]

                # ------------------------------------------------
                # ADD ROWS
                # ------------------------------------------------

                rows.extend(batch_rows)

                # ------------------------------------------------
                # STOP AFTER EXACTLY 4,000 NEW ROWS
                # ------------------------------------------------

                if len(rows) >= ROWS_TO_ADD:

                    rows = rows[
                        :ROWS_TO_ADD
                    ]

                    break

            read_time = (
                time.perf_counter()
                - read_start
            )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if len(rows) != ROWS_TO_ADD:

                print()
                print(
                    "FAILED:"
                )

                print(
                    f"Expected "
                    f"{ROWS_TO_ADD:,} rows."
                )

                print(
                    f"Received "
                    f"{len(rows):,} rows."
                )

                return 0

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
                f"Writing "
                f"{len(rows):,} rows..."
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

            # ------------------------------------------------
            # VERIFY OUTPUT
            # ------------------------------------------------

            written_rows = 0

            with open(
                output_file,
                "r",
                encoding="utf-8",
            ) as f:

                for _ in f:
                    written_rows += 1

            total_time = (
                time.perf_counter()
                - start
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            print()
            print("-" * 70)
            print("SUCCESS")
            print("-" * 70)

            print(
                f"Language       : "
                f"{language_name}"
            )

            print(
                f"Existing rows  : "
                f"{EXISTING_ROWS:,}"
            )

            print(
                f"Rows added     : "
                f"{written_rows:,}"
            )

            print(
                f"Source range   : "
                f"{START_ROW:,} - "
                f"{END_ROW - 1:,}"
            )

            print(
                f"Final total    : "
                f"{END_ROW:,}"
            )

            print(
                f"Read time      : "
                f"{read_time:.2f}s"
            )

            print(
                f"Total time     : "
                f"{total_time:.2f}s"
            )

            print(
                f"Output         : "
                f"{output_file}"
            )

            print(
                "Original 1,000 rows modified : NO"
            )

            print(
                "Full 3+ GB file cached       : NO"
            )

            print("-" * 70)

            return written_rows

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
    print(
        "HH GOA RAG - ADDITIONAL MULTILINGUAL DATA SAMPLER"
    )
    print("=" * 70)

    print()

    print(
        f"Dataset       : {REPO_ID}"
    )

    print(
        f"Existing rows : {EXISTING_ROWS:,}/language"
    )

    print(
        f"Rows to add   : {ROWS_TO_ADD:,}/language"
    )

    print(
        f"Final target  : {END_ROW:,}/language"
    )

    print(
        f"Languages     : {len(LANGUAGES)}"
    )

    print(
        f"New rows total: "
        f"{len(LANGUAGES) * ROWS_TO_ADD:,}"
    )

    print(
        f"Final corpus  : "
        f"{len(LANGUAGES) * END_ROW:,}"
    )

    print(
        f"Output        : {OUTPUT_DIR}"
    )

    print()

    print("Strategy:")

    print(
        "EXISTING 1,000"
        " → KEEP UNTOUCHED"
    )

    print(
        "REMOTE PARQUET"
        " → HTTP RANGE"
        " → SKIP FIRST 1,000"
        " → READ NEXT 4,000"
        " → JSONL"
    )

    print()

    print(
        "Source row ranges:"
    )

    print(
        f"  {START_ROW:,} -> "
        f"{END_ROW - 1:,}"
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

        if count == ROWS_TO_ADD:

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
            f"{count:5} new rows"
        )

    print()
    print("-" * 70)

    print(
        f"Languages completed : "
        f"{completed}/{len(LANGUAGES)}"
    )

    print(
        f"New rows added      : "
        f"{total_rows:,}"
    )

    print(
        f"Expected new rows   : "
        f"{len(LANGUAGES) * ROWS_TO_ADD:,}"
    )

    print(
        f"Final rows/language : "
        f"{END_ROW:,}"
    )

    print(
        f"Final corpus target : "
        f"{len(LANGUAGES) * END_ROW:,}"
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