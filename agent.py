"""Stage 3: agentic RAG as a LangGraph state machine (the "corrective RAG" pattern).

Stage 2 was a straight line: search once, answer once. If search missed, the
answer was "not found" even when the archive had it. Here the model checks its
own search results and tries again with different words:

    retrieve -> grade -> (relevant?) generate -> (found?) END
                  |  none relevant                |  not found
                  v                               v
               rewrite  <-------------------------+   (at most MAX_REWRITES)
                  |
                  +--> retrieve

LangGraph terms:
  State   : one dict every step reads from and writes to (AgentState below)
  Node    : a plain function that takes the state, returns the keys it changed
  Edge    : what runs next; a conditional edge is a function that picks the next node
  Reducer : how a returned value merges into the state; default is replace,
            `operator.add` appends (used for the step log and token count)

Run: .venv/bin/python agent.py "your question"
"""
import operator
import sys
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

import ask
import llm
import schemas
import search

MAX_REWRITES = 2
K = 5

GRADE_SYSTEM = """You check search results for a question about the BlogChain newsletter.
A source is relevant only if it contains information that directly helps answer the question.
Being about a similar topic is not enough."""

REWRITE_SYSTEM = """A search over the BlogChain newsletter archive found nothing that answers the question.
BlogChain covers AI tools and agents, crypto, privacy, football, and essays the author built or tested.
Write a new search query that uses different words from the earlier queries: the concrete terms,
numbers, or names the author would likely have used. Return only the query."""


class AgentState(TypedDict):
    question: str
    query: str                                   # what we search for; starts as the question
    queries: Annotated[list[str], operator.add]  # every query tried
    hits: list[dict]                             # latest search results
    relevant: list[dict]                         # hits the grader kept
    rewrites: int
    answer: schemas.Answer | None
    steps: Annotated[list[str], operator.add]    # readable log of what happened
    tokens: Annotated[int, operator.add]


def total(usage):
    return usage["prompt_tokens"] + usage["completion_tokens"]


def build(retriever=None, chat=llm.chat):
    """Wire the graph. `retriever` and `chat` are swappable so tests can fake them."""
    retriever = retriever or search.load("hybrid")

    def retrieve(state):
        hits = retriever.search(state["query"], K)
        titles = ", ".join(h["post_id"][:28] for h in hits)
        return {"hits": hits, "queries": [state["query"]], "steps": [f"retrieve '{state['query']}' -> {titles}"]}

    def grade(state):
        hits = state["hits"]
        messages = [
            {"role": "system", "content": GRADE_SYSTEM},
            {"role": "user", "content": f"Sources:\n\n{ask.format_sources(hits)}\n\nQuestion: {state['question']}"},
        ]
        grades, usage, _ = ask.structured(messages, schemas.Grades, {"n_sources": len(hits)}, chat)
        relevant = [hits[i - 1] for i in grades.relevant]
        return {"relevant": relevant, "tokens": total(usage), "steps": [f"grade -> {len(relevant)}/{len(hits)} relevant {grades.relevant}"]}

    def rewrite(state):
        tried = "\n".join(f"- {q}" for q in state["queries"])
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": f"Question: {state['question']}\n\nQueries already tried:\n{tried}"},
        ]
        new, usage, _ = ask.structured(messages, schemas.Rewrite, None, chat)
        return {"query": new.query, "rewrites": state["rewrites"] + 1, "tokens": total(usage), "steps": [f"rewrite -> '{new.query}'"]}

    def generate(state):
        parsed, usage, _ = ask.generate(state["question"], state["relevant"], chat)
        return {"answer": parsed, "tokens": total(usage), "steps": [f"generate -> found={parsed.found}"]}

    def after_grade(state):
        if state["relevant"]:
            return "generate"
        return "rewrite" if state["rewrites"] < MAX_REWRITES else "generate"

    def after_generate(state):
        if state["answer"].found or state["rewrites"] >= MAX_REWRITES:
            return END
        return "rewrite"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["generate", "rewrite"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("generate", after_generate, ["rewrite", END])
    return graph.compile()


def run(question, app=None):
    app = app or build()
    return app.invoke({
        "question": question, "query": question, "queries": [], "hits": [], "relevant": [],
        "rewrites": 0, "answer": None, "steps": [], "tokens": 0,
    })


def main():
    question = " ".join(sys.argv[1:]) or "Does giving an AI real-world results matter more than picking a smarter model?"
    state = run(question)
    for i, step in enumerate(state["steps"], 1):
        print(f"  {i}. {step}")
    text, posts = ask.render(state["answer"], state["relevant"])
    print(f"\n{text}\n")
    for i, p in enumerate(posts, 1):
        print(f"  [{i}] {p['title']}  {p['url']}")
    print(f"\n  tokens: {state['tokens']:,}, rewrites: {state['rewrites']}")


if __name__ == "__main__":
    main()
