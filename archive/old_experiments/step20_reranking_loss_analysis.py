"""
HH GOA RAG - STEP 20
RERANKING LOSS ANALYSIS

Deep-dives on:
  1. Score distributions: vector vs lexical vs phrase vs combined
  2. Reranking wins vs losses vs neutral (per query)
  3. Why lost queries were lost: score comparison
  4. Duplicate chunk analysis in Top-20
  5. Query-type / language breakdown
  6. Short vs long query behavior
"""

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.pipeline import RAGEngine, rerank, lexical_overlap, phrase_score
from app.pipeline import (
    VECTOR_WEIGHT, LEXICAL_WEIGHT, PHRASE_WEIGHT,
    TOP_K_RETRIEVAL, TOP_K_FINAL, COLLECTION_NAME
)

sys.stdout.reconfigure(encoding="utf-8")

QUERY_FILE = Path("data/hindi_sample_1000.jsonl")

# ============================================================
# LOAD QUERIES — with query_type and selected passage info
# ============================================================

def load_queries():
    queries = []
    seen = set()
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item.get("query_id")
            q = item.get("query", "").strip()
            if not qid or not q or qid in seen:
                continue
            seen.add(qid)
            queries.append({
                "query_id": str(qid),
                "query": q,
                "query_type": item.get("query_type", "unknown"),
                "has_selected": any(
                    s == 1
                    for s in item["passages"]["is_selected"]
                ),
            })
    return queries


# ============================================================
# PERCENTILE
# ============================================================

def percentile(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = (len(vals) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] + (k - f) * (vals[c] - vals[f])


# ============================================================
# LANGUAGE CLASSIFIER
# ============================================================

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[a-zA-Z]")


def classify_query_language(query: str) -> str:
    hindi_chars = len(DEVANAGARI_RE.findall(query))
    latin_chars = len(LATIN_RE.findall(query))
    total = hindi_chars + latin_chars
    if total == 0:
        return "other"
    ratio = hindi_chars / total
    if ratio >= 0.85:
        return "hindi"
    if ratio <= 0.15:
        return "english"
    return "mixed"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("HH GOA RAG - STEP 20")
    print("RERANKING LOSS ANALYSIS")
    print("=" * 70)

    print("\nLoading RAG engine...")
    engine = RAGEngine()

    queries = load_queries()
    print(f"Total queries    : {len(queries)}")
    print(f"With selected    : {sum(1 for q in queries if q['has_selected'])}")

    # Warmup
    print("\nWarming up...")
    for item in queries[:10]:
        engine.process(item["query"])
    print("Warmup complete.\n")

    # ============================================================
    # COUNTERS
    # ============================================================

    total = 0
    top20_hit = 0
    top3_hit = 0
    rerank_loss = 0
    rerank_gain = 0   # was NOT top3 before, IS after rerank (n/a here, rerank only sorts)

    # Score accumulators for rerank-loss cases
    loss_correct_vector_scores = []       # vector score of the correct-but-lost chunk
    loss_winner_vector_scores = []        # vector score of the chunk that displaced it
    loss_correct_combined_scores = []     # combined score of correct-but-lost
    loss_winner_combined_scores = []      # combined score of winner

    # Score accumulators for rerank-success cases
    success_correct_vector_scores = []
    success_winner_vector_scores = []

    # Rank position of correct result before and after rerank
    rank_before = []   # position in Top-20 (1-based)
    rank_after = []    # position in Top-3 after rerank (1-based, None if lost)

    # Combined score of correct result vs #1 result (in loss cases)
    score_gap_losses = []     # winner_combined - correct_combined (positive = winner ahead)

    # Duplicate counts per query
    dup_counts_top20 = []
    dup_present_loss = 0      # rerank-loss queries where duplicates filled Top-3
    dup_present_total = 0

    # Language breakdown
    lang_top20 = Counter()
    lang_top3 = Counter()
    lang_total = Counter()

    # Query type breakdown
    qtype_top20 = Counter()
    qtype_top3 = Counter()
    qtype_total = Counter()

    # Short vs long
    short_top20 = 0; short_total = 0
    long_top20 = 0;  long_total = 0
    short_top3 = 0;  long_top3 = 0

    # Examples
    loss_examples = []
    dup_monopoly_examples = []
    complete_failure_examples = []

    # ============================================================
    # BENCHMARK LOOP
    # ============================================================

    print("Running analysis...")
    benchmark_start = time.perf_counter()

    for idx, item in enumerate(queries, start=1):

        query = item["query"]
        qid = item["query_id"]
        qtype = item["query_type"]
        lang = classify_query_language(query)
        word_count = len(query.split())

        result = engine.process(query)
        total += 1

        top20 = result["retrieval"]["top20"]
        top3 = result["retrieval"]["top3"]

        top20_qids = [str(h.get("query_id", "")) for h in top20]
        top3_qids = [str(h.get("query_id", "")) for h in top3]

        # --------------------------------------------------------
        # Duplicate analysis in Top-20
        # --------------------------------------------------------
        dup_counter = Counter(top20_qids)
        max_dup = max(dup_counter.values()) if dup_counter else 0
        dup_counts_top20.append(max_dup)
        has_dup_monopoly = max_dup >= 3   # one source_id appears 3+ times

        if has_dup_monopoly:
            dup_present_total += 1

        # --------------------------------------------------------
        # Language / query-type
        # --------------------------------------------------------
        lang_total[lang] += 1
        qtype_total[qtype] += 1

        if word_count <= 4:
            short_total += 1
        else:
            long_total += 1

        # --------------------------------------------------------
        # Top-20 recall
        # --------------------------------------------------------
        pos20 = None
        for i, hid in enumerate(top20_qids, start=1):
            if hid == qid:
                pos20 = i
                break

        if pos20 is not None:
            top20_hit += 1
            rank_before.append(pos20)
            lang_top20[lang] += 1
            qtype_top20[qtype] += 1
            if word_count <= 4:
                short_top20 += 1
            else:
                long_top20 += 1
        else:
            if len(complete_failure_examples) < 20:
                complete_failure_examples.append({
                    "query": query,
                    "qid": qid,
                    "qtype": qtype,
                    "lang": lang,
                    "top20_qids": top20_qids[:5],
                    "dup_max": max_dup,
                })

        # --------------------------------------------------------
        # Top-3 recall
        # --------------------------------------------------------
        pos3 = None
        for i, hid in enumerate(top3_qids, start=1):
            if hid == qid:
                pos3 = i
                break

        if pos3 is not None:
            top3_hit += 1
            rank_after.append(pos3)
            lang_top3[lang] += 1
            qtype_top3[qtype] += 1
            if word_count <= 4:
                short_top3 += 1
            else:
                long_top3 += 1

        # --------------------------------------------------------
        # Reranking loss: correct in Top-20, not in Top-3
        # --------------------------------------------------------
        if pos20 is not None and pos3 is None:
            rerank_loss += 1

            # Pull the correct chunk's scores from top20 payload
            correct_hit = top20[pos20 - 1]  # dict from pipeline
            # Recompute scores for analysis
            correct_text = correct_hit.get("text", "")
            correct_vec = correct_hit.get("vector_score", 0.0)
            correct_lex = lexical_overlap(query, correct_text)
            correct_phr = phrase_score(query, correct_text)
            correct_combined = (
                VECTOR_WEIGHT * correct_vec
                + LEXICAL_WEIGHT * correct_lex
                + PHRASE_WEIGHT * correct_phr
            )

            # Pull the winner (rank-1 of top3) scores
            winner_hit = top3[0] if top3 else {}
            winner_vec = winner_hit.get("vector_score", 0.0)
            winner_text = winner_hit.get("text", "")
            winner_lex = lexical_overlap(query, winner_text)
            winner_phr = phrase_score(query, winner_text)
            winner_combined = (
                VECTOR_WEIGHT * winner_vec
                + LEXICAL_WEIGHT * winner_lex
                + PHRASE_WEIGHT * winner_phr
            )

            loss_correct_vector_scores.append(correct_vec)
            loss_winner_vector_scores.append(winner_vec)
            loss_correct_combined_scores.append(correct_combined)
            loss_winner_combined_scores.append(winner_combined)
            score_gap_losses.append(winner_combined - correct_combined)

            if has_dup_monopoly:
                dup_present_loss += 1

            if len(loss_examples) < 25:
                loss_examples.append({
                    "query": query,
                    "qid": qid,
                    "qtype": qtype,
                    "lang": lang,
                    "top20_rank": pos20,
                    "correct_vec": round(correct_vec, 4),
                    "correct_combined": round(correct_combined, 4),
                    "correct_lex": round(correct_lex, 4),
                    "winner_vec": round(winner_vec, 4),
                    "winner_combined": round(winner_combined, 4),
                    "winner_lex": round(winner_lex, 4),
                    "top3_qids": top3_qids,
                    "dup_monopoly": has_dup_monopoly,
                    "dup_max": max_dup,
                })

        # --------------------------------------------------------
        # Top-20 success
        # --------------------------------------------------------
        if pos20 is not None and pos3 is not None:
            correct_hit = top20[pos20 - 1]
            correct_vec = correct_hit.get("vector_score", 0.0)
            winner_hit = top3[0] if top3 else {}
            winner_vec = winner_hit.get("vector_score", 0.0)
            success_correct_vector_scores.append(correct_vec)
            success_winner_vector_scores.append(winner_vec)

        # --------------------------------------------------------
        # Dup monopoly examples
        # --------------------------------------------------------
        if has_dup_monopoly and len(dup_monopoly_examples) < 10:
            dominant_id = dup_counter.most_common(1)[0][0]
            dup_monopoly_examples.append({
                "query": query,
                "dominant_id": dominant_id,
                "count": dup_counter.most_common(1)[0][1],
                "correct_in_top20": pos20,
                "correct_in_top3": pos3,
            })

        if idx % 100 == 0:
            print(f"  [{idx}/1000]")

    benchmark_time = time.perf_counter() - benchmark_start

    # ============================================================
    # REPORT
    # ============================================================

    def pct(n): return n / total * 100 if total else 0.0

    print()
    print("=" * 70)
    print("STEP 20 COMPLETE — RERANKING LOSS ANALYSIS")
    print("=" * 70)

    # -------- RETRIEVAL FUNNEL --------
    print("\n1. RETRIEVAL FUNNEL")
    print("-" * 70)
    print(f"Total queries        : {total}")
    print(f"Top-20 recall        : {top20_hit} / {total}  = {pct(top20_hit):.2f}%")
    print(f"Top-3 recall         : {top3_hit} / {total}  = {pct(top3_hit):.2f}%")
    print(f"Complete failures    : {total - top20_hit} / {total}  = {pct(total - top20_hit):.2f}%")
    print(f"Reranking losses     : {rerank_loss} / {top20_hit}  = {rerank_loss/top20_hit*100:.2f}% of Top-20 hits")
    print()
    # Funnel contribution breakdown
    print("  Loss source breakdown:")
    print(f"    Embedding miss (not in Top-20): {total - top20_hit}  ({pct(total - top20_hit):.1f}% of all)")
    print(f"    Reranking loss (in Top-20 → not Top-3): {rerank_loss}  ({pct(rerank_loss):.1f}% of all)")
    total_miss = total - top3_hit
    print(f"    Total misses: {total_miss}  ({pct(total_miss):.1f}% of all)")
    if total_miss > 0:
        print(f"    Embedding share of misses: {(total-top20_hit)/total_miss*100:.1f}%")
        print(f"    Reranking share of misses: {rerank_loss/total_miss*100:.1f}%")

    # -------- CORRECT RESULT RANK DISTRIBUTION --------
    print("\n2. CORRECT RESULT RANK IN TOP-20 (for queries where it was found)")
    print("-" * 70)
    bins = [(1,1), (2,3), (4,5), (6,10), (11,15), (16,20)]
    for lo, hi in bins:
        cnt = sum(1 for r in rank_before if lo <= r <= hi)
        print(f"  Rank {lo:2d}-{hi:2d} : {cnt:4d}  ({cnt/total*100:.1f}% of all queries)")

    # -------- SCORE ANALYSIS --------
    print("\n3. SCORE ANALYSIS — RERANKING LOSS CASES")
    print("-" * 70)
    if loss_correct_vector_scores:
        print(f"  Correct chunk (correct but lost after rerank):")
        print(f"    Vector score  avg={sum(loss_correct_vector_scores)/len(loss_correct_vector_scores):.4f}  p50={percentile(loss_correct_vector_scores,50):.4f}  min={min(loss_correct_vector_scores):.4f}")
        print(f"    Combined score avg={sum(loss_correct_combined_scores)/len(loss_correct_combined_scores):.4f}  p50={percentile(loss_correct_combined_scores,50):.4f}")
        print(f"    Lexical score avg={sum(loss_correct_vector_scores)/len(loss_correct_vector_scores):.4f}")
        print()
        print(f"  Winner chunk (displaced the correct result):")
        print(f"    Vector score  avg={sum(loss_winner_vector_scores)/len(loss_winner_vector_scores):.4f}  p50={percentile(loss_winner_vector_scores,50):.4f}  min={min(loss_winner_vector_scores):.4f}")
        print(f"    Combined score avg={sum(loss_winner_combined_scores)/len(loss_winner_combined_scores):.4f}  p50={percentile(loss_winner_combined_scores,50):.4f}")
        print()
        print(f"  Score gap (winner_combined - correct_combined):")
        print(f"    avg={sum(score_gap_losses)/len(score_gap_losses):.4f}  p50={percentile(score_gap_losses,50):.4f}")
        print(f"    min={min(score_gap_losses):.4f}  max={max(score_gap_losses):.4f}")
        # How many losses had correct chunk with HIGHER vector score than winner?
        vec_higher = sum(1 for cv, wv in zip(loss_correct_vector_scores, loss_winner_vector_scores) if cv > wv)
        print(f"\n  Cases where correct chunk had HIGHER vector score but lost to reranker: {vec_higher} / {len(loss_correct_vector_scores)}")

    # -------- SCORE ANALYSIS SUCCESS --------
    print("\n4. SCORE ANALYSIS — RERANKING SUCCESS CASES")
    print("-" * 70)
    if success_correct_vector_scores:
        print(f"  Correct chunk vector score: avg={sum(success_correct_vector_scores)/len(success_correct_vector_scores):.4f}")
        print(f"  Winner vector score:         avg={sum(success_winner_vector_scores)/len(success_winner_vector_scores):.4f}")

    # -------- DUPLICATE ANALYSIS --------
    print("\n5. DUPLICATE ANALYSIS")
    print("-" * 70)
    dup_bins = {1: 0, 2: 0, 3: 0, 5: 0, 9: 0}
    for d in dup_counts_top20:
        if d == 1:   dup_bins[1] += 1
        elif d == 2: dup_bins[2] += 1
        elif d <= 4: dup_bins[3] += 1
        elif d <= 8: dup_bins[5] += 1
        else:        dup_bins[9] += 1
    print(f"  Max duplicates=1 (no dups) : {dup_bins[1]}")
    print(f"  Max duplicates=2           : {dup_bins[2]}")
    print(f"  Max duplicates=3-4         : {dup_bins[3]}")
    print(f"  Max duplicates=5-8         : {dup_bins[5]}")
    print(f"  Max duplicates=9+          : {dup_bins[9]}")
    queries_with_dup_monopoly = sum(1 for d in dup_counts_top20 if d >= 3)
    print(f"  Queries with 3+ same-ID    : {queries_with_dup_monopoly} ({queries_with_dup_monopoly/total*100:.1f}%)")
    print(f"  Rerank-loss queries with monopoly dup: {dup_present_loss} / {rerank_loss} ({dup_present_loss/max(rerank_loss,1)*100:.1f}%)")
    print()
    print("  Sample dup-monopoly queries:")
    for ex in dup_monopoly_examples[:5]:
        print(f"    Query: {ex['query'][:60]}")
        print(f"    Dominant ID: {ex['dominant_id']} appears {ex['count']} times")
        print(f"    Correct in Top-20: {ex['correct_in_top20']}  |  In Top-3: {ex['correct_in_top3']}")

    # -------- LANGUAGE ANALYSIS --------
    print("\n6. LANGUAGE ANALYSIS")
    print("-" * 70)
    for lang in ["hindi", "mixed", "english", "other"]:
        n = lang_total[lang]
        h20 = lang_top20[lang]
        h3 = lang_top3[lang]
        if n == 0:
            continue
        print(f"  {lang:<10}: total={n:4d}  Top-20={h20:4d} ({h20/n*100:.1f}%)  Top-3={h3:4d} ({h3/n*100:.1f}%)")

    # -------- QUERY TYPE ANALYSIS --------
    print("\n7. QUERY TYPE ANALYSIS")
    print("-" * 70)
    for qtype, n in sorted(qtype_total.items(), key=lambda x: -x[1]):
        h20 = qtype_top20[qtype]
        h3 = qtype_top3[qtype]
        print(f"  {str(qtype):<20}: total={n:4d}  Top-20={h20:4d} ({h20/n*100:.1f}%)  Top-3={h3:4d} ({h3/n*100:.1f}%)")

    # -------- SHORT VS LONG --------
    print("\n8. QUERY LENGTH ANALYSIS  (≤4 words = short)")
    print("-" * 70)
    print(f"  Short: total={short_total}  Top-20={short_top20} ({short_top20/max(short_total,1)*100:.1f}%)  Top-3={short_top3} ({short_top3/max(short_total,1)*100:.1f}%)")
    print(f"  Long:  total={long_total}   Top-20={long_top20} ({long_top20/max(long_total,1)*100:.1f}%)  Top-3={long_top3}  ({long_top3/max(long_total,1)*100:.1f}%)")

    # -------- RERANKING LOSS EXAMPLES --------
    print("\n9. RERANKING LOSS EXAMPLES (correct in Top-20 but dropped from Top-3)")
    print("-" * 70)
    for ex in loss_examples[:15]:
        print(f"\n  Query    : {ex['query'][:70]}")
        print(f"  QID      : {ex['qid']}  type={ex['qtype']}  lang={ex['lang']}")
        print(f"  Top-20 rank: {ex['top20_rank']}  correct_vec={ex['correct_vec']}  correct_combined={ex['correct_combined']}")
        print(f"  Lexical  : {ex['correct_lex']}")
        print(f"  Winner   : vec={ex['winner_vec']}  combined={ex['winner_combined']}  lex={ex['winner_lex']}")
        print(f"  Top-3 IDs: {ex['top3_qids']}")
        if ex['dup_monopoly']:
            print(f"  ⚠ DUP MONOPOLY: same ID appears {ex['dup_max']} times in Top-20")

    # -------- COMPLETE FAILURE EXAMPLES --------
    print("\n10. COMPLETE RETRIEVAL FAILURE EXAMPLES (not in Top-20)")
    print("-" * 70)
    for ex in complete_failure_examples[:10]:
        print(f"\n  Query: {ex['query'][:70]}")
        print(f"  QID={ex['qid']}  type={ex['qtype']}  lang={ex['lang']}")
        print(f"  Top retrieved IDs: {ex['top20_qids']}")

    # -------- DIAGNOSIS --------
    print()
    print("=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print(f"\n  Embedding miss    : {total-top20_hit:3d} queries ({pct(total-top20_hit):.1f}%)  — cannot be fixed by reranking")
    print(f"  Reranking loss    : {rerank_loss:3d} queries ({pct(rerank_loss):.1f}%)  — fixable only by changing reranker")
    print(f"  Total miss@3      : {total-top3_hit:3d} queries ({pct(total-top3_hit):.1f}%)")
    print()
    if rerank_loss > 0:
        dup_caused = dup_present_loss
        print(f"  Of {rerank_loss} rerank losses:")
        print(f"    Dup monopoly involved  : {dup_caused} ({dup_caused/rerank_loss*100:.1f}%)")
        print(f"    Pure score displacement: {rerank_loss - dup_caused} ({(rerank_loss-dup_caused)/rerank_loss*100:.1f}%)")
    print()
    print(f"  Benchmark time    : {benchmark_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
