"""Step 4: MEASURE. How often does search put the right post in front of the LLM?

If retrieval misses, the LLM never sees the answer, however good the model is.
So we score retrieval on its own, before any LLM is involved:
  hit@k : share of questions where a right post appears in the top k chunks
  MRR   : mean of 1/rank of the first right chunk (1.0 = always ranked first)

Run: python3 evaluate.py [-v]   (-v prints every question's result)
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


def main():
    verbose = "-v" in sys.argv
    retriever = search.load()
    cases = [c for c in json.loads(EVAL_FILE.read_text()) if c["expect"]]
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


if __name__ == "__main__":
    main()
