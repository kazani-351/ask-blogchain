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

## Stage 1b: embeddings, hybrid search, cited answers

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "MESH_API_KEY=..." > .env          # Mesh API, OpenAI-compatible gateway
.venv/bin/python embed.py               # 76 chunks -> 1536-dim vectors (text-embedding-3-small)
.venv/bin/python evaluate.py hybrid     # bm25 | vector | hybrid
.venv/bin/python ask.py "Where did my trading bot finish?"
```

Vector search is plain numpy cosine similarity, the same exact search as FAISS `IndexFlatIP`. At 76 chunks a vector database adds nothing. Hybrid merges BM25 and vector rankings with reciprocal rank fusion.

| retriever | hit@1 keyword | hit@1 paraphrase | hit@1 all | hit@5 all | MRR |
|---|---|---|---|---|---|
| BM25 | 100% | 64% | 79% | 96% | 0.86 |
| vector | 90% | 79% | 83% | 92% | 0.86 |
| **hybrid** | **100%** | **79%** | **88%** | 92% | **0.89** |

Vectors fix paraphrase at the top rank (64% to 79%) but lose some exact-word matches, like "USDC on Chelsea's shirts". Hybrid keeps both strengths and has the best first-rank accuracy. With 24 questions, one question is about 4 points, so treat small gaps as noise.

`ask.py` sends the top 5 hybrid chunks to `gpt-4o-mini` with a rule to cite sources and to say "I couldn't find that in BlogChain" otherwise. Both unanswerable eval questions get that refusal. A typical answer costs about 1,700 input tokens, roughly $0.0003.

**Known limits:** the RSS feed returns only the latest 20 posts. The eval questions were written after reading the posts, which flatters keyword search a little.
