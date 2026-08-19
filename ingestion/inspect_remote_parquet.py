import fsspec
import pyarrow.parquet as pq


URL = (
    "https://huggingface.co/datasets/"
    "ai4bharat/MSMARCO-XI/resolve/main/"
    "train/asmtrain.parquet"
)

BLOCK_SIZE = 8 * 1024 * 1024


print("=" * 70)
print("REMOTE PARQUET DIAGNOSTIC")
print("=" * 70)

print()
print("URL:")
print(URL)

print()
print("Opening remote file...")

fs = fsspec.filesystem(
    "http",
    block_size=BLOCK_SIZE,
    cache_type="bytes",
)

with fs.open(URL, "rb") as f:

    parquet = pq.ParquetFile(f)

    metadata = parquet.metadata

    print()
    print("=" * 70)
    print("FILE INFORMATION")
    print("=" * 70)

    print(
        f"Rows          : "
        f"{metadata.num_rows:,}"
    )

    print(
        f"Row groups    : "
        f"{metadata.num_row_groups:,}"
    )

    print(
        f"Columns       : "
        f"{metadata.num_columns:,}"
    )

    print()
    print("=" * 70)
    print("COLUMN SIZES")
    print("=" * 70)

    total_compressed = 0
    total_uncompressed = 0

    for row_group_index in range(
        metadata.num_row_groups
    ):

        row_group = metadata.row_group(
            row_group_index
        )

        print()
        print(
            f"ROW GROUP {row_group_index}"
        )

        print(
            f"Rows: "
            f"{row_group.num_rows:,}"
        )

        print()

        for column_index in range(
            row_group.num_columns
        ):

            column = row_group.column(
                column_index
            )

            name = column.path_in_schema

            compressed = (
                column.total_compressed_size
            )

            uncompressed = (
                column.total_uncompressed_size
            )

            total_compressed += compressed
            total_uncompressed += uncompressed

            print(
                f"{name:25} | "
                f"compressed: "
                f"{compressed / (1024**2):10.2f} MB | "
                f"uncompressed: "
                f"{uncompressed / (1024**2):10.2f} MB"
            )

    print()
    print("=" * 70)
    print("TOTALS")
    print("=" * 70)

    print(
        f"Compressed   : "
        f"{total_compressed / (1024**3):.3f} GB"
    )

    print(
        f"Uncompressed : "
        f"{total_uncompressed / (1024**3):.3f} GB"
    )

print()
print("=" * 70)
print("DONE")
print("=" * 70)