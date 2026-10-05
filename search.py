"""Step 3: SEARCH. BM25 keyword search over the chunks, standard library only.

BM25 is the classic search-engine ranking formula. A chunk scores high when it:
  - contains the query's words (term frequency, with diminishing returns: k1)
  - those words are rare across all chunks (IDF: "the" counts for nothing)
  - adjusted for length, so long chunks don't win by just having more words (b)

Its blind spot is the reason embeddings exist: it only matches exact words.
"wearable tracker" will never match a chunk that says "ring".

Run: python3 search.py [bm25|vector|hybrid|truncated|binary|hybrid-binary] "your question"
"""
import json
import math
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

CHUNKS_FILE = Path("data/chunks.json")
STOPWORDS = set(
    "a an and are as at be but by can did do does for from has have how i if in "
    "is it its me my of on or so than that the their them there they this to up "
    "was we what when where which who why will with you your".split()
)


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c["title"] + " " + c["text"])) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.docs)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_terms, i):
        doc, length = self.docs[i], self.lengths[i]
        total = 0.0
        for t in query_terms:
            tf = doc.get(t, 0)
            if tf:
                norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return total

    def search(self, query, k=5):
        terms = tokenize(query)
        scored = sorted(((self.score(terms, i), i) for i in range(len(self.docs))), reverse=True)
        return [{**self.chunks[i], "score": round(s, 3)} for s, i in scored[:k] if s > 0]


class Vector:
    """Meaning search: cosine similarity between the question and every chunk.

    With 76 chunks, checking every one is instant. This is exactly what FAISS's
    IndexFlatIP does; a vector database only matters at millions of chunks.
    """

    def __init__(self, chunks):
        import numpy as np
        meta_file = Path("data/embeddings.json")
        ids = json.loads(meta_file.read_text())["ids"] if meta_file.exists() else None
        if ids != [c["id"] for c in chunks]:
            raise SystemExit("Embeddings missing or stale for these chunks. Run: .venv/bin/python embed.py")
        self.np, self.chunks = np, chunks
        self.vecs = self.encode(np.load("data/embeddings.npy"))

    def encode(self, vecs):
        return vecs

    def similarity(self, q):
        return self.vecs @ q

    def search(self, query, k=5):
        q = self.encode(self.np.array([embed_query(query)], dtype=self.np.float32))[0]
        sims = self.similarity(q)
        top = self.np.argsort(-sims)[:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 3)} for i in top]


class Truncated(Vector):
    """Stage 6: keep only the first 48 of 1,536 dimensions, at full precision.

    text-embedding-3-small is trained so a leading slice still works on its own.
    48 floats x 4 bytes = 192 bytes per vector, the same budget as Binary.
    """
    DIMS = 48

    def encode(self, vecs):
        cut = vecs[:, :self.DIMS]
        return cut / self.np.linalg.norm(cut, axis=1, keepdims=True)


class Binary(Vector):
    """Stage 6: keep all 1,536 dimensions, but only 1 bit each (the sign).

    1,536 bits = 192 bytes per vector, 32x smaller than float32. Cosine needs the
    magnitudes we threw away, so we compare by Hamming distance: XOR the bits and
    count the 1s (positions where the two vectors disagree). The score is the
    share of bits that agree, so 1.0 = identical and ~0.5 = unrelated.
    """

    def encode(self, vecs):
        return self.np.packbits(vecs >= 0, axis=1)

    def similarity(self, q):
        bits = self.vecs.shape[1] * 8
        return 1 - self.np.bitwise_count(self.vecs ^ q).sum(axis=1) / bits


@lru_cache(maxsize=None)
def embed_query(query):
    """One embedding call per question, shared by every vector variant."""
    import llm
    q = llm.embed([query])[0]
    norm = math.sqrt(sum(x * x for x in q))
    return tuple(x / norm for x in q)


class Hybrid:
    """Run both, merge with reciprocal rank fusion (RRF).

    RRF ignores the raw scores (BM25 and cosine aren't comparable) and uses rank
    only: each list gives a chunk 1/(60 + rank). Chunks both methods like win.
    Keeps only the best chunk per post, so one long post can't fill every slot.
    """

    def __init__(self, chunks, vector=Vector):
        self.parts = [BM25(chunks), vector(chunks)]

    def search(self, query, k=5):
        fused, by_id = {}, {}
        for part in self.parts:
            for rank, hit in enumerate(part.search(query, 20), 1):
                fused[hit["id"]] = fused.get(hit["id"], 0) + 1 / (60 + rank)
                by_id[hit["id"]] = hit
        top, seen = [], set()
        for i in sorted(fused, key=fused.get, reverse=True):
            if by_id[i]["post_id"] not in seen:
                seen.add(by_id[i]["post_id"])
                top.append(i)
            if len(top) == k:
                break
        return [{**by_id[i], "score": round(fused[i], 4)} for i in top]


RETRIEVERS = {
    "bm25": BM25, "vector": Vector, "hybrid": Hybrid,
    "truncated": Truncated, "binary": Binary,
    "hybrid-binary": lambda chunks: Hybrid(chunks, Binary),
}


def load(kind="bm25"):
    return RETRIEVERS[kind](json.loads(CHUNKS_FILE.read_text()))


def main():
    args = sys.argv[1:]
    kind = args.pop(0) if args and args[0] in RETRIEVERS else "bm25"
    for hit in load(kind).search(" ".join(args) or "trading bot leaderboard"):
        print(f"{hit['score']:7.3f}  {hit['id']}\n         {hit['text'][:120]}...")


if __name__ == "__main__":
    main()
