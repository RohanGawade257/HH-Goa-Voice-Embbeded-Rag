import json
import statistics

INPUT_FILE = "data/hindi_sample_1000.jsonl"

lengths = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:
        record = json.loads(line)

        passages = record["passages"]["Translated_passages"]

        for passage in passages:
            lengths.append(len(passage.split()))


print("=" * 50)
print("MSMARCO-XI PASSAGE LENGTH ANALYSIS")
print("=" * 50)

print(f"\nTotal passages: {len(lengths)}")

print(f"Minimum words: {min(lengths)}")
print(f"Maximum words: {max(lengths)}")
print(f"Average words: {statistics.mean(lengths):.2f}")
print(f"Median words: {statistics.median(lengths):.2f}")

# Calculate percentiles correctly
sorted_lengths = sorted(lengths)


def percentile(data, p):
    index = int(len(data) * p)
    index = min(index, len(data) - 1)
    return data[index]


print("\nPercentiles:")

for p in [0.50, 0.75, 0.90, 0.95, 0.99]:
    print(
        f"P{int(p * 100)}: "
        f"{percentile(sorted_lengths, p)} words"
    )