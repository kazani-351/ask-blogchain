"""Stage 4: tracing with Langfuse.

A trace is the full record of one question: every step, every LLM call, its
exact prompt and reply, tokens, cost and time, nested the way they ran.

Langfuse builds on OpenTelemetry, the standard for tracing. Code marks the
interesting functions with @observe; the SDK records them in the background and
sends them in batches, so tracing never slows down or breaks an answer.

Tracing turns on only when both keys are in .env. Otherwise it's off, quietly,
and everything still runs. Tests force it off so they never send data.
"""
import logging
import os

import env  # noqa: F401  (loads .env before the Langfuse client reads it)

ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
if os.environ.get("LANGFUSE_TRACING_ENABLED", "").lower() == "false":
    ENABLED = False
if not ENABLED:
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)  # no "no key" warning on every call

from langfuse import get_client, observe, propagate_attributes  # noqa: E402

__all__ = ["ENABLED", "client", "observe", "propagate_attributes", "status"]


def client():
    return get_client()


def status():
    if not ENABLED:
        return "tracing: off (add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to .env)"
    return f"tracing: on -> {os.environ.get('LANGFUSE_BASE_URL') or os.environ.get('LANGFUSE_HOST') or 'https://cloud.langfuse.com'}"
