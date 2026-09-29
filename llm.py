"""Tiny client for Mesh API (OpenAI-compatible), standard library only.

The key is read from .env into the process and never printed.
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://api.meshapi.ai/v1"
EMBED_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4o-mini"


def _key():
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() and value.strip():
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
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


def embed(texts):
    """One vector per text. Batched: one request covers many chunks."""
    data = _post("/embeddings", {"model": EMBED_MODEL, "input": texts})["data"]
    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


def chat(messages, **extra):
    resp = _post("/chat/completions", {"model": CHAT_MODEL, "messages": messages, **extra})
    return resp["choices"][0]["message"]["content"], resp.get("usage", {})
