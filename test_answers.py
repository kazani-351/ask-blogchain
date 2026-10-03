"""Offline tests for validation and the retry loop. No API key, no cost.

The model is replaced by a fake that returns scripted replies, so we can force
the failures that a real model only produces sometimes.

Run: .venv/bin/python -m unittest -v test_answers
"""
import json
import os
import unittest

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"  # tests never send traces, even with keys in .env

from pydantic import ValidationError  # noqa: E402

import ask  # noqa: E402
import schemas  # noqa: E402

HITS = [
    {"post_id": "post-a", "title": "Post A", "url": "u/a", "text": "alpha"},
    {"post_id": "post-a", "title": "Post A", "url": "u/a", "text": "alpha two"},
    {"post_id": "post-b", "title": "Post B", "url": "u/b", "text": "beta"},
]


class FakeRetriever:
    def search(self, question, k):
        return HITS


def fake_chat(*replies):
    """A chat function that returns the given replies in order and records what it was sent."""
    calls = []

    def chat(messages, **kwargs):
        calls.append([dict(m) for m in messages])
        reply = replies[len(calls) - 1]
        return (reply if isinstance(reply, str) else json.dumps(reply)), {"prompt_tokens": 10, "completion_tokens": 5}

    chat.calls = calls
    return chat


def check(data, n=3):
    return schemas.Answer.model_validate_json(json.dumps(data), context={"n_sources": n})


class Validation(unittest.TestCase):
    def test_valid_answer_dedupes_and_sorts_sources(self):
        a = check({"found": True, "claims": [{"text": " It worked. ", "sources": [3, 1, 3]}]})
        self.assertEqual(a.claims[0].sources, [1, 3])
        self.assertEqual(a.claims[0].text, "It worked.")

    def test_source_out_of_range(self):
        with self.assertRaisesRegex(ValidationError, r"\[7\] don't exist"):
            check({"found": True, "claims": [{"text": "x", "sources": [7]}]})

    def test_claim_without_sources(self):
        with self.assertRaisesRegex(ValidationError, "at least one source"):
            check({"found": True, "claims": [{"text": "x", "sources": []}]})

    def test_found_but_no_claims(self):
        with self.assertRaisesRegex(ValidationError, "no claims"):
            check({"found": True, "claims": []})

    def test_not_found_but_claims(self):
        with self.assertRaisesRegex(ValidationError, "found is false"):
            check({"found": False, "claims": [{"text": "x", "sources": [1]}]})

    def test_extra_field_rejected(self):
        with self.assertRaises(ValidationError):
            check({"found": False, "claims": [], "confidence": 0.9})


class RetryLoop(unittest.TestCase):
    def test_bad_then_good_retries_with_error(self):
        chat = fake_chat(
            {"found": True, "claims": [{"text": "x", "sources": [9]}]},
            {"found": True, "claims": [{"text": "x", "sources": [1]}]},
        )
        parsed, _, usage, attempts = ask.answer("q", FakeRetriever(), chat=chat)
        self.assertEqual(attempts, 2)
        self.assertEqual(usage["prompt_tokens"], 20)
        feedback = chat.calls[1][-1]["content"]
        self.assertIn("failed validation", feedback)
        self.assertIn("[9] don't exist", feedback)

    def test_broken_json_is_retried(self):
        chat = fake_chat("{not json", {"found": False, "claims": []})
        parsed, _, _, attempts = ask.answer("q", FakeRetriever(), chat=chat)
        self.assertEqual(attempts, 2)
        self.assertFalse(parsed.found)

    def test_gives_up_after_max_attempts(self):
        bad = {"found": True, "claims": []}
        chat = fake_chat(bad, bad, bad)
        with self.assertRaises(SystemExit):
            ask.answer("q", FakeRetriever(), chat=chat)
        self.assertEqual(len(chat.calls), ask.MAX_ATTEMPTS)


class Render(unittest.TestCase):
    def test_chunks_of_one_post_become_one_citation(self):
        parsed = check({"found": True, "claims": [
            {"text": "A.", "sources": [1, 2]},
            {"text": "B.", "sources": [3, 2]},
        ]})
        text, posts = ask.render(parsed, HITS)
        self.assertEqual(text, "A. [1] B. [1][2]")
        self.assertEqual([p["post_id"] for p in posts], ["post-a", "post-b"])

    def test_not_found(self):
        text, posts = ask.render(check({"found": False, "claims": []}), HITS)
        self.assertEqual(text, "I couldn't find that in BlogChain.")
        self.assertEqual(posts, [])


if __name__ == "__main__":
    unittest.main()
