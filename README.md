# Ask BlogChain

Ask a question, get an answer from the [BlogChain newsletter](https://paragraph.com/@kazani) archive, with citations to the exact posts.

Built in stages to learn retrieval-augmented generation (RAG) from first principles, then with production tooling.

## Stage 1a: keyword baseline (standard library only)

```
python3 ingest.py      # RSS feed -> data/posts.json      (20 posts, 16k words)
python3 chunk.py       # posts -> data/chunks.json        (76 chunks of ~300 words, 50 overlap)
python3 search.py "how did the trading bot do"            # BM25 top 5
python3 evaluate.py -v # score retrieval on data/eval.json
```

The eval set has 24 answerable questions with a known source post. Ten reuse the post's own words (`keyword`). Fourteen ask the same thing in different words (`paraphrase`). Two have no answer in the archive (`unanswerable`, used in later stages).

**BM25 results (chunk size 300):**

| style | n | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| keyword | 10 | 100% | 100% | 100% | 1.00 |
| paraphrase | 14 | 64% | 86% | 93% | 0.75 |
| all | 24 | 79% | 92% | 96% | 0.86 |

Keyword search is perfect when you use the author's words and drops sharply when you don't. That gap is what embeddings (Stage 1b) should close.

**Chunk size sweep (hit@5 / MRR, all questions):** 150 words: 88% / 0.83 · **300: 96% / 0.86** · 500: 92% / 0.86 · 1000: 96% / 0.80. Small chunks lose context. Large chunks mix topics, so the right post ranks lower.

**Known limits:** the RSS feed returns only the latest 20 posts. The eval questions were written after reading the posts, which flatters keyword search a little.
