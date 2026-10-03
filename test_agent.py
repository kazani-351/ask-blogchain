"""Offline tests for the agent's routing. Fake search, fake model, no cost.

Each test scripts what the fake grader says and checks the path the graph took.

Run: .venv/bin/python -m unittest -v test_agent
"""
import json
import os
import unittest

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"  # tests never send traces, even with keys in .env

import agent  # noqa: E402

GOOD = {"post_id": "good", "title": "Good", "url": "u/g", "text": "the answer"}
BAD = {"post_id": "bad", "title": "Bad", "url": "u/b", "text": "noise"}


class FakeRetriever:
    """Returns GOOD only when the query contains the magic word."""
    def __init__(self, magic="feedback"):
        self.magic, self.queries = magic, []

    def search(self, query, k):
        self.queries.append(query)
        return [GOOD, BAD] if self.magic in query else [BAD, BAD]


def fake_chat(rewrites=("feedback loop",)):
    """Plays grader, rewriter and answerer, picking the role from the requested schema."""
    calls = {"grades": 0, "rewrite": 0, "answer": 0}
    rewrites = list(rewrites)

    def chat(messages, **kw):
        role = kw["response_format"]["json_schema"]["name"]
        calls[role] += 1
        prompt = messages[-1]["content"]
        if role == "grades":
            out = {"relevant": [1] if "the answer" in prompt else []}
        elif role == "rewrite":
            out = {"query": rewrites.pop(0) if rewrites else "still nothing"}
        else:
            out = {"found": True, "claims": [{"text": "It's the feedback.", "sources": [1]}]}
        return json.dumps(out), {"prompt_tokens": 100, "completion_tokens": 10}

    chat.calls = calls
    return chat


def run(question, retriever, chat):
    return agent.run(question, agent.build(retriever, chat))


class Routing(unittest.TestCase):
    def test_relevant_first_try_goes_straight_to_answer(self):
        r, chat = FakeRetriever(), fake_chat()
        state = run("why feedback matters", r, chat)
        self.assertTrue(state["answer"].found)
        self.assertEqual(state["rewrites"], 0)
        self.assertEqual(chat.calls, {"grades": 1, "rewrite": 0, "answer": 1})
        self.assertEqual([h["post_id"] for h in state["relevant"]], ["good"])

    def test_rewrite_rescues_a_missed_search(self):
        r, chat = FakeRetriever(), fake_chat(rewrites=["feedback loop"])
        state = run("does real-world signal beat a smarter model?", r, chat)
        self.assertTrue(state["answer"].found)
        self.assertEqual(state["rewrites"], 1)
        self.assertEqual(r.queries, ["does real-world signal beat a smarter model?", "feedback loop"])
        self.assertEqual(state["queries"], r.queries)  # reducer appended, didn't overwrite

    def test_gives_up_after_max_rewrites_without_paying_for_an_answer(self):
        r, chat = FakeRetriever(magic="never"), fake_chat(rewrites=[])
        state = run("favourite cricket team?", r, chat)
        self.assertFalse(state["answer"].found)
        self.assertEqual(state["rewrites"], agent.MAX_REWRITES)
        self.assertEqual(len(r.queries), agent.MAX_REWRITES + 1)
        self.assertEqual(chat.calls["answer"], 0)  # empty sources skip the LLM

    def test_tokens_accumulate_across_steps(self):
        r, chat = FakeRetriever(), fake_chat(rewrites=["feedback loop"])
        state = run("question", r, chat)
        # grade, rewrite, grade, answer = 4 calls x 110 tokens
        self.assertEqual(state["tokens"], 440)


if __name__ == "__main__":
    unittest.main()
