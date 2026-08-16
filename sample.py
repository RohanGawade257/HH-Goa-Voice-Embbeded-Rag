import pyarrow.parquet as pq
import json
import os

PARQUET_FILE = r"C:\Users\offic\.cache\huggingface\hub\datasets--ai4bharat--MSMARCO-XI\snapshots\bf5cdc1f26e581e519018e434db14edd1b77602b\train\hintrain.parquet"

SAMPLE_SIZE = 1000
OUTPUT_FILE = "data/hindi_sample_1000.jsonl"


print("=" * 60)
print("HH GOA RAG - STEP 3")
print("Extracting Hindi MSMARCO-XI Sample")
print("=" * 60)

print("\nOpening Parquet file...")
print(PARQUET_FILE)

parquet = pq.ParquetFile(PARQUET_FILE)

print(f"\nTotal rows: {parquet.metadata.num_rows}")

print("\nColumns:")

for field in parquet.schema_arrow:
    print(f"  {field.name}: {field.type}")

os.makedirs("data", exist_ok=True)

print(f"\nExtracting first {SAMPLE_SIZE} rows...")

count = 0

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as output:

    for batch in parquet.iter_batches(batch_size=100):

        records = batch.to_pylist()

        for record in records:

            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str
                ) + "\n"
            )

            count += 1

            if count % 100 == 0:
                print(f"Saved {count}/{SAMPLE_SIZE}")

            if count >= SAMPLE_SIZE:
                break

        if count >= SAMPLE_SIZE:
            break


print("\n" + "=" * 60)
print("STEP 3 COMPLETE")
print("=" * 60)

print(f"Rows saved : {count}")
print(f"Output     : {OUTPUT_FILE}")