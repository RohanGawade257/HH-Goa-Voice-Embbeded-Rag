import json
import os
import re
import hashlib


INPUT_FILE = "data/processed/passages_1000.jsonl"
OUTPUT_FILE = "data/processed/chunks_1000.jsonl"


# ---------------------------------------------------------
# CHUNKING PARAMETERS
# ---------------------------------------------------------

SHORT_LIMIT = 80
MEDIUM_LIMIT = 180

TARGET_WORDS = 100
MAX_WORDS = 140

OVERLAP_SENTENCES = 1

# Fallback for pathological individual sentences
FALLBACK_WORDS = 120
FALLBACK_OVERLAP_WORDS = 20


# ---------------------------------------------------------
# TEXT UTILITIES
# ---------------------------------------------------------

def split_sentences(text):
    """
    Sentence-aware segmentation for Hindi + English text.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[।!?])\s+|(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def word_count(text):
    return len(text.split())


def make_chunk_id(
    passage_id,
    strategy,
    index,
    text
):
    text_hash = hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:10]

    return (
        f"{passage_id}"
        f"_{strategy}"
        f"_{index}"
        f"_{text_hash}"
    )


# ---------------------------------------------------------
# PATHOLOGICAL SENTENCE FALLBACK
# ---------------------------------------------------------

def split_long_sentence(sentence):
    """
    Fallback for a single sentence that is itself
    longer than MAX_WORDS.

    We use a small word-based window only here.

    This is NOT our primary chunking strategy.
    It prevents pathological passages from producing
    multi-thousand-word chunks.
    """

    words = sentence.split()

    if len(words) <= MAX_WORDS:
        return [
            {
                "text": sentence,
                "strategy": "sentence_group",
                "overlap": False
            }
        ]

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + FALLBACK_WORDS,
            len(words)
        )

        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)

        chunks.append(
            {
                "text": chunk_text,
                "strategy": "long_sentence_fallback",
                "overlap": start > 0
            }
        )

        if end >= len(words):
            break

        next_start = (
            end - FALLBACK_OVERLAP_WORDS
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ---------------------------------------------------------
# STRATEGY A
# ---------------------------------------------------------

def intact_chunk(text):

    return [
        {
            "text": text,
            "strategy": "intact",
            "overlap": False
        }
    ]


# ---------------------------------------------------------
# STRATEGY B
# ---------------------------------------------------------

def sentence_group_chunk(text):
    """
    Medium passages.

    Groups complete sentences while respecting
    MAX_WORDS.

    Extremely long individual sentences use
    the fallback strategy.
    """

    sentences = split_sentences(text)

    if not sentences:
        return []

    chunks = []
    current = []

    for sentence in sentences:

        sentence_words = word_count(sentence)

        # -------------------------------------------------
        # Individual pathological sentence
        # -------------------------------------------------

        if sentence_words > MAX_WORDS:

            # Flush current chunk first
            if current:

                chunks.append(
                    {
                        "text": " ".join(current),
                        "strategy": "sentence_group",
                        "overlap": False
                    }
                )

                current = []

            chunks.extend(
                split_long_sentence(sentence)
            )

            continue

        # -------------------------------------------------
        # Normal sentence
        # -------------------------------------------------

        candidate = " ".join(
            current + [sentence]
        )

        if (
            current
            and word_count(candidate) > MAX_WORDS
        ):

            chunks.append(
                {
                    "text": " ".join(current),
                    "strategy": "sentence_group",
                    "overlap": False
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
                "overlap": False
            }
        )

    return chunks


# ---------------------------------------------------------
# STRATEGY C
# ---------------------------------------------------------

def adaptive_window_chunk(text):
    """
    Long passages.

    Primary strategy:

        sentence boundaries
        +
        target word budget
        +
        sentence overlap

    Pathological individual sentences use
    long_sentence_fallback.
    """

    sentences = split_sentences(text)

    if not sentences:
        return []

    chunks = []

    start = 0

    while start < len(sentences):

        current = []
        current_words = 0

        end = start

        # -------------------------------------------------
        # Build adaptive sentence window
        # -------------------------------------------------

        while end < len(sentences):

            sentence = sentences[end]

            sentence_words = word_count(
                sentence
            )

            # Pathological sentence
            if sentence_words > MAX_WORDS:

                # Flush sentences already accumulated
                if current:

                    chunks.append(
                        {
                            "text": " ".join(current),
                            "strategy": "adaptive_window",
                            "overlap": start > 0
                        }
                    )

                    current = []
                    current_words = 0

                # Add pathological sentence using
                # controlled fallback
                fallback_chunks = (
                    split_long_sentence(sentence)
                )

                chunks.extend(
                    fallback_chunks
                )

                end += 1

                start = end

                break

            # -------------------------------------------------
            # Normal sentence
            # -------------------------------------------------

            if (
                current
                and current_words + sentence_words
                > MAX_WORDS
            ):
                break

            current.append(sentence)

            current_words += sentence_words

            end += 1

            if current_words >= TARGET_WORDS:
                break

        else:
            # Reached end naturally
            end = len(sentences)

        # -------------------------------------------------
        # Write accumulated normal chunk
        # -------------------------------------------------

        if current:

            chunk_text = " ".join(current)

            chunks.append(
                {
                    "text": chunk_text,
                    "strategy": "adaptive_window",
                    "overlap": start > 0
                }
            )

        # -------------------------------------------------
        # Move window forward
        # -------------------------------------------------

        if end >= len(sentences):

            break

        next_start = (
            end - OVERLAP_SENTENCES
        )

        if next_start <= start:

            next_start = end

        start = next_start

    return chunks


# ---------------------------------------------------------
# ADAPTIVE CHUNKER
# ---------------------------------------------------------

def create_chunks(text):

    words = word_count(text)

    if words <= SHORT_LIMIT:

        return intact_chunk(text)

    elif words <= MEDIUM_LIMIT:

        return sentence_group_chunk(text)

    else:

        return adaptive_window_chunk(text)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("HH GOA RAG - STEP 6")
    print("VAST Adaptive Chunking")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    total_passages = 0
    total_chunks = 0

    strategy_counts = {
        "intact": 0,
        "sentence_group": 0,
        "adaptive_window": 0,
        "long_sentence_fallback": 0
    }

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

            passage = json.loads(line)

            total_passages += 1

            text = passage["text"]

            chunks = create_chunks(text)

            for index, chunk in enumerate(chunks):

                chunk_text = chunk["text"]

                chunk_record = {

                    "chunk_id": make_chunk_id(
                        passage["passage_id"],
                        chunk["strategy"],
                        index,
                        chunk_text
                    ),

                    "passage_id": passage[
                        "passage_id"
                    ],

                    "query_id": passage[
                        "query_id"
                    ],

                    "passage_index": passage[
                        "passage_index"
                    ],

                    "chunk_index": index,

                    "text": chunk_text,

                    "language": passage[
                        "language"
                    ],

                    "word_count": word_count(
                        chunk_text
                    ),

                    "chunk_strategy": chunk[
                        "strategy"
                    ],

                    "has_overlap": chunk[
                        "overlap"
                    ],

                    "is_selected": passage[
                        "is_selected"
                    ],

                    "query_type": passage[
                        "query_type"
                    ]
                }

                output_file.write(
                    json.dumps(
                        chunk_record,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_chunks += 1

                strategy_counts[
                    chunk["strategy"]
                ] += 1

    print("\n" + "=" * 60)
    print("STEP 6 COMPLETE")
    print("=" * 60)

    print(
        f"\nInput passages : {total_passages}"
    )

    print(
        f"Output chunks  : {total_chunks}"
    )

    print("\nChunk strategies:")

    for strategy, count in strategy_counts.items():

        print(
            f"  {strategy:<25} : {count}"
        )

    print("\nOutput:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()