# Lessons

What I learned building Ask BlogChain, one numbered record per stage. Each record has what I built, what the numbers showed, the ideas worth keeping, and questions to check that I understand it.

**Why I'm learning this:** to add real AI engineering skills to my resume, starting with the skills AI engineering job posts ask for most: RAG, LangGraph, structured outputs, and tracing.

---

## 0001: Stage 1, RAG from first principles (2026-09-29)

### What I built

A question-answering tool over my own newsletter. It has five steps, and I built each one by hand without a framework:

| Step | File | What it does |
|---|---|---|
| Load | `ingest.py` | Reads 20 posts from the RSS feed and strips out the HTML |
| Chunk | `chunk.py` | Cuts the posts into 76 pieces of about 300 words, with 50 words of overlap |
| Embed | `embed.py` | Turns each piece into 1,536 numbers that capture its meaning |
| Search | `search.py` | Finds the pieces closest to the question: by words (BM25), by meaning (vector), or both (hybrid) |
| Answer | `ask.py` | Gives the top 5 pieces to an LLM, which must answer only from them and cite them |
| Measure | `evaluate.py` | Checks whether search found the right post, on 24 questions with known answers |

### What the numbers showed

| Search type | Right post first | Right post first, reworded questions |
|---|---|---|
| Keyword (BM25) | 79% | 64% |
| Meaning (vector) | 83% | 79% |
| Hybrid | **88%** | 79% |

Chunk size sweep (right post in top 5): 150 words 88%, **300 words 96%**, 500 words 92%, 1,000 words 96% but ranked lower.

### Ideas worth keeping

1. **RAG means "look it up, then answer."** The model doesn't know my posts. I find the relevant pieces first and paste them into the prompt. The model's job is only to read and summarize what I hand it.

2. **Measure search separately from answers.** If search misses, the best model in the world never sees the answer. So I score search first, with no LLM involved. That's also cheaper and faster to test.

3. **Keyword search matches words, not meaning.** "Where did my trading bot finish" failed because my post says "agent" and "finished around 48th". It's perfect when you use the author's words (100%) and weak when you don't (64%).

4. **Embeddings match meaning.** Similar ideas get similar numbers, even with no words in common. That fixed the reworded questions (64% to 79%).

5. **But embeddings lose exact matches.** "USDC on Chelsea's shirts" dropped to rank 4 with vector search. Names, numbers, and rare words are where keywords win.

6. **Hybrid takes the best of both.** Run both searches and merge them by rank (reciprocal rank fusion). A piece both methods like rises to the top. Rank is used, not score, because the two scores are on different scales. This is why real systems usually combine them.

7. **Chunk size is a real tradeoff.** Too small and a piece loses its context. Too big and a piece mixes topics, so it matches everything a little. 300 words won here, but the only way to know is to measure.

8. **Overlap stops answers from being cut in half.** Each chunk repeats the last 50 words of the previous one, so an answer that sits on a boundary still lands whole in one chunk.

9. **Tell the model how to say "I don't know."** The prompt gives an exact refusal line. Both questions with no answer in the archive got it, with no made-up answers.

10. **A vector database isn't needed at this size.** With 76 chunks, comparing the question against every chunk takes no time (numpy, about 5 lines). FAISS or Pinecone start to matter at hundreds of thousands of chunks. FAISS also doesn't install on macOS 12.

### Surprises

- I expected meaning search to beat keywords everywhere. It didn't. It traded some exact-word hits for better reworded hits.
- With only 24 questions, one question is worth about 4 points. The gap between vector and hybrid could be noise. A bigger eval set is the fix.
- I wrote the questions after reading the posts, which makes keyword search look a little better than it really is.

### Cost

Under $0.01 for the whole stage. Embedding all 76 chunks cost about $0.0005. Each answer uses about 1,700 tokens, roughly $0.0003.

### Check that I understand it

Answer these without looking. If I can't, I don't own it yet.

1. Why does the eval test search without an LLM?
2. Why can't BM25 match "bot" with "agent"?
3. Why would embeddings miss "USDC on Chelsea's shirts" when keywords get it?
4. Why does hybrid merge by rank instead of adding the scores?
5. What goes wrong with 1,000-word chunks?
6. What does the 50-word overlap protect against?
7. When would I actually need a vector database?

### Open for Stage 2

- Citations come at the end of the answer ("[1][4]"), not after each claim.
- The source list repeats the same post.
- Stage 2 (Pydantic plus structured outputs) returns the answer as checked data with one citation per claim.
