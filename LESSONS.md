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

---

## 0002: Stage 2, structured outputs with Pydantic (2026-10-02)

### What I built

The answer is no longer a paragraph. It's data with a fixed shape:

```json
{"found": true, "claims": [{"text": "My bot finished around 48th of 107.", "sources": [1]}]}
```

| File | What it does |
|---|---|
| `schemas.py` | Defines the answer shape with Pydantic, plus rules for checking it |
| `ask.py` | Asks for that shape, checks the reply, and retries with the error if it fails |
| `test_answers.py` | 11 tests that run offline with a fake model, to force failures on purpose |
| `evaluate_answers.py` | Scores the final answers, not just search |

### What the numbers showed

| Check | Result |
|---|---|
| Said "found" or "not found" correctly | 25 of 26 |
| Cited a correct post | 23 of 24 answerable |
| Share of cited posts that were correct | 98% |
| Answers that needed a retry | 0 |
| Cost for all 26 questions | about $0.008 |

### Ideas worth keeping

1. **Free text can't be checked by code. Data can.** In Stage 1, I had to read each answer to judge it. Now a script scores 26 answers in seconds, because `found` and `sources` are fields, not words in a sentence.

2. **Two layers, two jobs.** The JSON schema, sent to the API, controls the shape: which fields exist and their types. Pydantic, running in my code, controls the meaning: "source 7 doesn't exist, you only got 5 sources." A schema can't know how many sources I sent. My code does.

3. **The schema comes from the code.** I wrote the Pydantic class once, and `model_json_schema()` produced the JSON schema the API needs. One source of truth, so they can't drift apart.

4. **Strict mode needs every field required and no extra fields.** `extra="forbid"` in Pydantic produces `additionalProperties: false`, which strict mode needs.

5. **Retry with the error message, not just "try again."** When validation fails, the exact error text goes back to the model, like "source numbers [9] don't exist; valid numbers are 1 to 5". The model knows what to fix. It also stops after 3 tries, so it can't loop forever.

6. **Test the failure path with a fake model.** With strict mode on, the real model never produced a bad shape, so the retry loop never ran live. I'd never know it worked. The fake model returns scripted bad replies to force it, offline, for free.

7. **An honest "not found" can be a search failure in disguise.** The one wrong answer said "not found" for a question that the archive does answer. Search never surfaced the right post, so the model correctly refused rather than guess. That's the right behavior, and it points at the real fix: better search, not a better prompt.

8. **Format for people, store for machines.** The model cites chunks, like [1][2][3]. Three of those can be the same post. The printing step merges them into one post citation. The stored data keeps the detail, and the reader sees something clean.

### Bug I caught

In Python, the `e` in `except ValidationError as e` is deleted when the `except` block ends. My "gave up after 3 tries" message used `e` after the loop and would have crashed. I saved it to another variable first. The offline test for giving up is what proves the fix.

### Check that I understand it

1. What does the JSON schema guarantee that Pydantic doesn't, and the other way around?
2. Why can't the schema alone stop the model from citing source 7?
3. Why send the error text back instead of just asking again?
4. The retry count was 0. How do I know the retry loop works?
5. Why was the one "not found" answer actually good behavior?
6. Why merge chunk citations into post citations only when printing?

### Open for Stage 3

- One question still fails because search misses the right post. Stage 3 (LangGraph) adds a loop: grade the retrieved chunks, and if they're weak, rewrite the question and search again.

---

## 0003: Stage 3, agentic RAG with LangGraph (2026-10-03)

### What I built

Stage 2 was a straight line: search once, answer once. Stage 3 is a loop that checks its own work. This is called corrective RAG.

```
retrieve → grade ──some relevant──→ generate ──found──→ done
   ↑         │ none relevant           │ not found
   └─ rewrite ←────────────────────────┘   (at most 2 rewrites)
```

| File | What it does |
|---|---|
| `agent.py` | The graph: four steps (retrieve, grade, rewrite, generate) and the rules for which runs next |
| `ask.py` | Now has one shared `structured()` helper. The grader, the rewriter and the answer step all use it |
| `schemas.py` | Two new shapes: `Grades` (which sources are relevant) and `Rewrite` (a new query) |
| `test_agent.py` | 4 offline tests that check which path the graph takes |

### What the numbers showed

| | Stage 2 | Stage 3 |
|---|---|---|
| Found or not found, correct | 25 of 26 | **26 of 26** |
| Cited a correct post | 23 of 24 | **24 of 24** |
| Share of citations that were correct | 98% | **100%** |
| Tokens for 26 questions | 55,697 | 81,200 (46% more) |

### What LangGraph actually is

A way to write a program as a flowchart where an AI decides some of the arrows. It has four parts:

- **State**: one shared dictionary. Every step reads it and returns only the keys it changed.
- **Node**: a plain Python function. Mine are `retrieve`, `grade`, `rewrite`, `generate`.
- **Edge**: what runs next. A conditional edge is a small function that looks at the state and picks the next node. Mine are `after_grade` and `after_generate`.
- **Reducer**: how a node's output merges into the state. The default replaces the old value. `operator.add` appends instead. I used it for the list of queries tried, the step log, and the token count.

### Ideas worth keeping

1. **Pipeline vs agent.** A pipeline runs the same steps every time. An agent decides what to do next based on what just happened. Here the decision is "was this search good enough?" That one decision is the whole difference.

2. **The loop needs a judge.** Without the grade step, the program can't know search failed. Grading is a small, cheap LLM call that answers one question: which of these chunks actually answer it?

3. **Every loop needs an exit.** `MAX_REWRITES = 2` means at most 3 searches. Without a limit, an unanswerable question would loop forever and burn money. Cost limits are a design decision, not an afterthought.

4. **Two checks catch more than one.** The grader made a mistake on the rescued question: it kept an off-topic chunk. The answer step read it and said "not found", which sent it back to rewrite. If only the grader could trigger a rewrite, that question would still fail.

5. **Rewriting works because search is literal.** "Real-world results vs smarter model" missed. The rewritten "real-world data for AI performance versus model intelligence" found it. The model guessed words closer to the ones the post actually uses.

6. **Better answers cost more.** 46% more tokens, and more time per question. Most of it goes to unanswerable questions, which now try 3 queries before giving up. Whether that's worth it depends on the product. For a demo, yes. At a million questions a day, maybe only rewrite once.

7. **Skip calls you don't need.** If no chunk is relevant after the last rewrite, the answer is "not found" with no LLM call at all. Cheap wins like this add up.

8. **Test the paths, not just the pieces.** The routing tests use a fake search that only finds the answer when the query contains a magic word. That lets me force each path: found on the first try, rescued by a rewrite, gave up. A real model would take whichever path it felt like.

9. **One helper for every structured call.** Grading, rewriting, and answering all need "ask for JSON, validate, send the error back, retry." I moved that into one function, `ask.structured()`, instead of copying it three times.

10. **Draw the graph from the code.** `get_graph().draw_mermaid()` produces the diagram in the README from the real compiled graph. A hand-drawn diagram can drift from the code. This one can't.

### Install check

LangGraph pulled in about 40 packages. I checked each one on PyPI before installing. Three unfamiliar names turned up (`httpx2`, `httpcore2`, `httpx2-jsfetch`). The first two come from the real `pydantic` GitHub organization. The third only installs when Python runs inside a browser, so it never landed here. Checking unfamiliar names before installing guards against fake look-alike packages.

### Check that I understand it

1. What makes this an agent and not a pipeline?
2. What are State, Node, Edge, and Reducer, in one sentence each?
3. Why does `queries` use `operator.add` while `query` doesn't?
4. Why does the graph need `MAX_REWRITES`?
5. The grader made a mistake on the rescued question. What caught it?
6. Where did the extra 46% of tokens go?
7. How do the routing tests force the "rescued by rewrite" path without a real model?

### Open for Stage 4

- I can see each run's steps in a printed log, but not the timing, cost per step, or the exact prompt each step sent. Stage 4 (Langfuse) records all of that for every run.
