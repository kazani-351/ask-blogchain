"""Step 5: ANSWER. Retrieve the best chunks, hand them to the LLM, demand citations.

The prompt is the "augmented" part of retrieval-augmented generation: the model
answers from the numbered sources we give it, not from its own memory, and says
so when the sources don't contain the answer.

Run: .venv/bin/python ask.py "your question"
"""
import sys

import llm
import search

SYSTEM = """You answer questions about the BlogChain newsletter using ONLY the numbered sources provided.
Cite every claim with its source number in brackets, like [2].
If the sources do not contain the answer, reply exactly: I couldn't find that in BlogChain.
Be brief: two to four sentences."""


def answer(question, retriever, k=5):
    hits = retriever.search(question, k)
    sources = "\n\n".join(f"[{i}] {h['title']}\n{h['text']}" for i, h in enumerate(hits, 1))
    text, usage = llm.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return text, hits, usage


def main():
    question = " ".join(sys.argv[1:]) or "Where did my trading bot finish?"
    text, hits, usage = answer(question, search.load("hybrid"))
    print(f"\n{text}\n")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] {h['title']}  {h['url']}")
    print(f"\n  tokens: {usage.get('prompt_tokens')} in, {usage.get('completion_tokens')} out")


if __name__ == "__main__":
    main()
