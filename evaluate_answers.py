"""Step 6: MEASURE ANSWERS. evaluate.py checks search; this checks the final answer.

Structured output makes answers checkable by code, no human reading needed:
  found accuracy : says "found" for answerable questions, "not found" for the rest
  cited right    : at least one cited post is a correct post
  citation prec. : share of cited posts that are correct posts
  retries        : answers that needed a second attempt to pass validation

Costs real money: about $0.01 (pipeline) or $0.02 (agent) for the whole set.
With tracing on (agent mode), each question's trace gets a `correct` score and
the whole run shares one session id, so Langfuse can filter straight to failures.

Run: .venv/bin/python evaluate_answers.py [pipeline|agent] [-v]
"""

import json
import sys
import time
from pathlib import Path

import agent
import ask
import search
import tracing

EVAL_FILE = Path("data/eval.json")


def answer_with(mode):
    """Both modes return (Answer, the hits it cites from, tokens, retried-or-rewrote, trace id)."""
    if mode == "agent":
        app = agent.build()

        def run(q):
            s = agent.run(q, app)
            return s["answer"], s["relevant"], s["tokens"], s["rewrites"] > 0, s["trace_id"]
    else:
        retriever = search.load("hybrid")

        def run(q):
            parsed, hits, usage, attempts = ask.answer(q, retriever)
            return parsed, hits, usage["prompt_tokens"] + usage["completion_tokens"], attempts > 1, None
    return run


def main():
    verbose = "-v" in sys.argv
    mode = "agent" if "agent" in sys.argv else "pipeline"
    run = answer_with(mode)
    session = f"eval-{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"mode: {mode} | {tracing.status()} | session: {session}")
    cases = json.loads(EVAL_FILE.read_text())
    found_ok = cited_ok = retries = tokens = 0
    precisions = []
    answerable = [c for c in cases if c["expect"]]
    lf = tracing.client()
    with tracing.propagate_attributes(session_id=session, tags=["eval", mode]):
        for c in cases:
            parsed, hits, used, looped, trace_id = run(c["q"])
            tokens += used
            retries += looped
            should_find = bool(c["expect"])
            found_ok += parsed.found == should_find
            cited = {hits[s - 1]["post_id"] for cl in parsed.claims for s in cl.sources}
            if should_find and parsed.found:
                right = cited & set(c["expect"])
                cited_ok += bool(right)
                precisions.append(len(right) / len(cited))
            ok = parsed.found == should_find and (not should_find or bool(cited & set(c["expect"])))
            if trace_id:
                lf.create_score(trace_id=trace_id, name="correct", value=int(ok), data_type="BOOLEAN",
                                comment=f"[{c['style']}] expected {c['expect'] or 'not found'}, cited {sorted(cited)}")
            if verbose:
                print(f"{'ok ' if ok else 'BAD'} found={parsed.found!s:<5} cited={sorted(cited)}  [{c['style']}] {c['q']}")
    lf.flush()

    print(f"\nfound accuracy : {found_ok}/{len(cases)}")
    print(f"cited right    : {cited_ok}/{len(answerable)} answerable")
    print(f"citation prec. : {sum(precisions) / max(len(precisions), 1):.0%}")
    print(f"{'rewrites' if mode == 'agent' else 'retries'}       : {retries} questions")
    print(f"tokens         : {tokens:,} (~${tokens * 0.15 / 1e6:.4f} at gpt-4o-mini input price)")


if __name__ == "__main__":
    main()
