"""Step 3: SEARCH. BM25 keyword search over the chunks, standard library only.

BM25 is the classic search-engine ranking formula. A chunk scores high when it:
  - contains the query's words (term frequency, with diminishing returns: k1)
  - those words are rare across all chunks (IDF: "the" counts for nothing)
  - adjusted for length, so long chunks don't win by just having more words (b)

Its blind spot is the reason embeddings exist: it only matches exact words.
"wearable tracker" will never match a chunk that says "ring".

Run: python3 search.py "your question"
"""
import json
import math
import re
import sys
from collections import Counter
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


def load():
    return BM25(json.loads(CHUNKS_FILE.read_text()))


def main():
    for hit in load().search(" ".join(sys.argv[1:]) or "trading bot leaderboard"):
        print(f"{hit['score']:7.3f}  {hit['id']}\n         {hit['text'][:120]}...")


if __name__ == "__main__":
    main()
