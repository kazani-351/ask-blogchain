"""Step 3b: EMBED. Turn every chunk into a vector that captures its meaning.

An embedding model maps text to a list of 1,536 numbers. Texts with similar
meaning land close together, even with no words in common ("bot" ~ "agent").
Search then becomes: embed the question, find the nearest chunk vectors.

Embeddings are cached in data/embeddings.npy, so this runs once per chunking.

Run: .venv/bin/python embed.py
"""
import json
from pathlib import Path

import numpy as np

import llm

CHUNKS_FILE = Path("data/chunks.json")
VECS_FILE = Path("data/embeddings.npy")
META_FILE = Path("data/embeddings.json")


def main():
    chunks = json.loads(CHUNKS_FILE.read_text())
    # Title goes in front so each chunk knows which post it came from.
    texts = [f"{c['title']}\n\n{c['text']}" for c in chunks]
    vecs = []
    for i in range(0, len(texts), 32):
        vecs += llm.embed(texts[i:i + 32])
    arr = np.array(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)  # unit length: dot product = cosine
    np.save(VECS_FILE, arr)
    META_FILE.write_text(json.dumps({"model": llm.EMBED_MODEL, "ids": [c["id"] for c in chunks]}))
    print(f"{arr.shape[0]} vectors x {arr.shape[1]} dims ({llm.EMBED_MODEL}) -> {VECS_FILE}")


if __name__ == "__main__":
    main()
