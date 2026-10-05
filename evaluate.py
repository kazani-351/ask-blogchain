"""Step 4: MEASURE. How often does search put the right post in front of the LLM?

If retrieval misses, the LLM never sees the answer, however good the model is.
So we score retrieval on its own, before any LLM is involved:
  hit@k : share of questions where a right post appears in the top k chunks
  MRR   : mean of 1/rank of the first right chunk (1.0 = always ranked first)

Run: python3 evaluate.py [bm25|vector|hybrid|...] [-v]   (-v prints every question)
     Name several retrievers to compare them in one run, e.g. vector binary truncated
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import search

EVAL_FILE = Path("data/eval.json")
K = 5


def first_hit_rank(results, expected):
    for rank, r in enumerate(results, 1):
        if r["post_id"] in expected:
            return rank
    return None


def score_band(retriever, all_cases):
    """Stage 6: where do the scores sit? Ranking only needs right > wrong, but a
    cutoff ("nothing relevant, say so") needs a wide gap between the two.

    floor        : median score over all chunks = what "unrelated" scores
    answerable   : top-1 score when the right post is ranked first (min..max)
    unanswerable : top-1 score for questions the archive can't answer
    """
    n = len(retriever.chunks)
    floors, right, none = [], [], []
    for c in all_cases:
        hits = retriever.search(c["q"], n)
        floors.append(hits[n // 2]["score"])
        if not c["expect"]:
            none.append(hits[0]["score"])
        elif hits[0]["post_id"] in c["expect"]:
            right.append(hits[0]["score"])
    floor = sum(floors) / len(floors)
    print(f"\nscore band  floor {floor:.3f}   answerable top-1 {min(right):.3f}..{max(right):.3f}"
          f"   unanswerable top-1 {', '.join(f'{s:.3f}' for s in none)}")
    print(f"            weakest right answer sits {min(right) - floor:.3f} above the floor,"
          f" best wrong one {max(none) - floor:.3f}")


def main():
    verbose = "-v" in sys.argv
    kinds = [a for a in sys.argv[1:] if a in search.RETRIEVERS] or ["bm25"]
    for kind in kinds:
        evaluate(kind, verbose)


def evaluate(kind, verbose):
    retriever = search.load(kind)
    print(f"\nretriever: {kind}")
    all_cases = json.loads(EVAL_FILE.read_text())
    cases = [c for c in all_cases if c["expect"]]
    by_style = defaultdict(list)
    for c in cases:
        rank = first_hit_rank(retriever.search(c["q"], K), c["expect"])
        by_style[c["style"]].append(rank)
        by_style["ALL"].append(rank)
        if verbose:
            print(f"{'rank ' + str(rank) if rank else 'MISS  '}  [{c['style']}] {c['q']}")

    print(f"\n{'style':<11}{'n':>3}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>7}")
    for style in ("keyword", "paraphrase", "ALL"):
        ranks = by_style[style]
        hit = lambda k: sum(1 for r in ranks if r and r <= k) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        print(f"{style:<11}{len(ranks):>3}{hit(1):>8.0%}{hit(3):>8.0%}{hit(5):>8.0%}{mrr:>7.2f}")
    if isinstance(retriever, search.Vector):
        score_band(retriever, all_cases)


if __name__ == "__main__":
    main()
