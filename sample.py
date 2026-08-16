from datasets import load_dataset
import json

print("Loading Hindi dataset...")

dataset = load_dataset(
    "ai4bharat/MSMARCO-XI",
    "hi",
    split="train",
    streaming=True,
)

print("Getting first 1,000 rows...")

with open("data/hindi_sample_1000.jsonl", "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

        if (i + 1) % 100 == 0:
            print(f"Downloaded {i + 1} rows")

        if i + 1 >= 1000:
            break

print("Done!")
print("Saved to: data/hindi_sample_1000.jsonl")