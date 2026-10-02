"""Step 5: ANSWER. Retrieve chunks, get a structured answer, validate it, retry on failure.

Stage 1 asked for free text and hoped the citations were right. Now:
  - the API is told to return JSON matching schemas.Answer (structured outputs)
  - Pydantic checks it, including rules the schema can't express
  - if validation fails, the error goes back to the model and it tries again

Run: .venv/bin/python ask.py "your question"
"""
import sys

from pydantic import ValidationError

import llm
import schemas
import search

SYSTEM = """You answer questions about the BlogChain newsletter using ONLY the numbered sources provided.
Split the answer into short claims, one fact each, and list the source numbers that support each claim.
If the sources do not contain the answer, set found to false and return no claims.
Use at most four claims."""

MAX_ATTEMPTS = 3


def answer(question, retriever, k=5, chat=llm.chat):
    """Returns (Answer, hits, usage, attempts). `chat` is swappable so tests can fake the model."""
    hits = retriever.search(question, k)
    sources = "\n\n".join(f"[{i}] {h['title']}\n{h['text']}" for i, h in enumerate(hits, 1))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"},
    ]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw, u = chat(messages, temperature=0, response_format=schemas.response_format())
        for key in usage:
            usage[key] += u.get(key, 0)
        try:
            parsed = schemas.Answer.model_validate_json(raw, context={"n_sources": len(hits)})
            return parsed, hits, usage, attempt
        except ValidationError as e:
            error = e
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That answer failed validation:\n{e}\nReturn corrected JSON."},
            ]
    raise SystemExit(f"No valid answer after {MAX_ATTEMPTS} attempts. Last error:\n{error}")


def render(parsed, hits):
    """Cite posts, not chunks: three chunks of one post become one numbered source."""
    if not parsed.found:
        return "I couldn't find that in BlogChain.", []
    posts = []  # unique post per citation number, in first-cited order
    lines = []
    for claim in parsed.claims:
        nums = []
        for s in claim.sources:
            hit = hits[s - 1]
            if hit["post_id"] not in [p["post_id"] for p in posts]:
                posts.append(hit)
            n = [p["post_id"] for p in posts].index(hit["post_id"]) + 1
            if n not in nums:
                nums.append(n)
        lines.append(f"{claim.text} " + "".join(f"[{n}]" for n in nums))
    return " ".join(lines), posts


def main():
    question = " ".join(sys.argv[1:]) or "Where did my trading bot finish?"
    parsed, hits, usage, attempts = answer(question, search.load("hybrid"))
    text, posts = render(parsed, hits)
    print(f"\n{text}\n")
    for i, p in enumerate(posts, 1):
        print(f"  [{i}] {p['title']}  {p['url']}")
    print(f"\n  tokens: {usage['prompt_tokens']} in, {usage['completion_tokens']} out, attempts: {attempts}")


if __name__ == "__main__":
    main()
