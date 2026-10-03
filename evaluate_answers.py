"""Step 6: MEASURE ANSWERS. evaluate.py checks search; this checks the final answer.

Structured output makes answers checkable by code, no human reading needed:
  found accuracy : says "found" for answerable questions, "not found" for the rest
  cited right    : at least one cited post is a correct post
  citation prec. : share of cited posts that are correct posts
  retries        : answers that needed a second attempt to pass validation

Costs real money: about $0.01 (pipeline) or $0.02 (agent) for the whole set.
Run: .venv/bin/python evaluate_answers.py [pipeline|agent] [-v]
"""
import json
import sys
from pathlib import Path

import agent
import ask
import search

EVAL_FILE = Path("data/eval.json")


def answer_with(mode):
    """Both modes return (Answer, the hits it cites from, tokens, retried-or-rewrote)."""
    if mode == "agent":
        app = agent.build()

        def run(q):
            s = agent.run(q, app)
            return s["answer"], s["relevant"], s["tokens"], s["rewrites"] > 0
    else:
        retriever = search.load("hybrid")

        def run(q):
            parsed, hits, usage, attempts = ask.answer(q, retriever)
            return parsed, hits, usage["prompt_tokens"] + usage["completion_tokens"], attempts > 1
    return run


def main():
    verbose = "-v" in sys.argv
    mode = "agent" if "agent" in sys.argv else "pipeline"
    run = answer_with(mode)
    print(f"mode: {mode}")
    cases = json.loads(EVAL_FILE.read_text())
    found_ok = cited_ok = retries = tokens = 0
    precisions = []
    answerable = [c for c in cases if c["expect"]]
    for c in cases:
        parsed, hits, used, looped = run(c["q"])
        tokens += used
        retries += looped
        should_find = bool(c["expect"])
        found_ok += parsed.found == should_find
        cited = {hits[s - 1]["post_id"] for cl in parsed.claims for s in cl.sources}
        if should_find and parsed.found:
            right = cited & set(c["expect"])
            cited_ok += bool(right)
            precisions.append(len(right) / len(cited))
        if verbose:
            mark = "ok " if parsed.found == should_find and (not should_find or cited & set(c["expect"])) else "BAD"
            print(f"{mark} found={parsed.found!s:<5} cited={sorted(cited)}  [{c['style']}] {c['q']}")

    print(f"\nfound accuracy : {found_ok}/{len(cases)}")
    print(f"cited right    : {cited_ok}/{len(answerable)} answerable")
    print(f"citation prec. : {sum(precisions) / max(len(precisions), 1):.0%}")
    print(f"{'rewrites' if mode == 'agent' else 'retries'}       : {retries} questions")
    print(f"tokens         : {tokens:,} (~${tokens * 0.15 / 1e6:.4f} at gpt-4o-mini input price)")


if __name__ == "__main__":
    main()
