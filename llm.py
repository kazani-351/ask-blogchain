"""Tiny client for Mesh API (OpenAI-compatible), standard library only.

The key is read from .env into the process and never printed. Each call is
recorded as a Langfuse generation (chat) or embedding, with token usage.
"""
import json
import os
import urllib.error
import urllib.request

import tracing

BASE = "https://api.meshapi.ai/v1"
EMBED_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4o-mini"


def _key():
    key = os.environ.get("MESH_API_KEY")
    if not key:
        raise SystemExit("MESH_API_KEY not set. Add it to .env as MESH_API_KEY=...")
    return key


def _post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Mesh API {path} failed: HTTP {e.code} {e.read().decode()[:300]}")


def _model_name(model):
    # Langfuse prices known models by their plain name ("gpt-4o-mini"), not the gateway's "openai/..." id
    return model.split("/")[-1]


@tracing.observe(as_type="embedding", name="embed", capture_input=False)
def embed(texts):
    """One vector per text. Batched: one request covers many chunks."""
    resp = _post("/embeddings", {"model": EMBED_MODEL, "input": texts})
    tracing.client().update_current_generation(
        model=_model_name(EMBED_MODEL),
        input=texts if len(texts) == 1 else f"{len(texts)} texts",
        output=f"{len(resp['data'])} vectors",
        usage_details={"input": resp.get("usage", {}).get("prompt_tokens", 0)},
    )
    return [d["embedding"] for d in sorted(resp["data"], key=lambda d: d["index"])]


@tracing.observe(as_type="generation", name="chat", capture_input=False, capture_output=False)
def chat(messages, **extra):
    resp = _post("/chat/completions", {"model": CHAT_MODEL, "messages": messages, **extra})
    content, usage = resp["choices"][0]["message"]["content"], resp.get("usage", {})
    fmt = extra.get("response_format", {}).get("json_schema", {}).get("name")
    tracing.client().update_current_generation(
        model=_model_name(CHAT_MODEL),
        model_parameters={k: v for k, v in extra.items() if k != "response_format"},
        input=messages,
        output=content,
        usage_details={"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)},
        metadata={"gateway": "meshapi", "schema": fmt},
    )
    return content, usage
