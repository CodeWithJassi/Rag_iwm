# CLAUDE.md

Guidance for Claude Code working in this repo. Read this before touching anything.

---

## What this is

An internal tool for the IWM Research equity desk. Analysts upload broker
research PDFs on Indian equities. Each PDF is summarised and indexed. Analysts
then ask questions about a specific report and get answers grounded **only** in
that report's text.

The system is built to **abstain rather than guess**. This desk acts on the
numbers it reads here. A confidently wrong target price is worse than "I don't
have enough information."

Read `README.md` for the setup steps and the pipeline diagram. This file covers
the things that will bite you if you change code without knowing them.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Existing house stack |
| DB + vectors | Postgres 16 + pgvector | One service holds documents, chunks, vectors and chat memory |
| LLM | Ollama (`llama3.1:8b`) with API failover | Local first, paid providers only when it fails |
| Embeddings | Ollama (3 models supported, see `_EMBED_REGISTRY` in config.py) | Same Ollama process. No new infra |
| Rerank | bge-reranker-v2-m3 via FlagEmbedding (local), or TEI container (remote) | Cross-encoder scores query+passage together |
| Frontend | One vanilla HTML file, served by FastAPI | No build step for an internal tool |

**There is no Redis.** Session memory is the `turns` table. Do not add Redis; the
EC2 box has run out of RAM before, and a research desk's query volume does not
need sub-millisecond memory reads.

**There is no graph DB.** The company → broker → report → chunk tree is foreign
keys. Reach for Neo4j only if a query needs edges between arbitrary nodes.

---

## Layout

```
backend/
  config.py       every tunable constant. Nothing else hardcodes one.
  prompts.py      every prompt. Iterate prompts without touching pipeline code.
  llm.py          provider failover chain. All LLM calls go through llm_complete().
  embeddings.py   embedding model via Ollama (see _EMBED_REGISTRY in config.py)
  db.py           schema, connection pool, query()/execute() helpers
  extract.py      tiered PDF text extraction, page-aware
  chunking.py     TWO chunkers. See below.
  summarize.py    map-reduce summary + extract_facts()
  ingest.py       upload orchestrator (background task)
  rerank.py       cross-encoder reranker (local or TEI remote)
  rerank.py       cross-encoder reranker (local or TEI remote)
  retrieval.py    the query pipeline. The heart of the system.
  judge.py        deep-search validation
  memory.py       sessions + turns
  main.py         routes, static mount
static/index.html the whole dashboard
```

---

## Five invariants. Breaking any of these breaks correctness silently.

### 1. The abstention gate reads raw cosine, never RRF score

`config.SIM_GATE` is compared against the **pre-fusion** max cosine similarity,
returned as the second element of `retrieve_for()`.

RRF is rank-based. The top hit of a completely irrelevant retrieval still gets
the best RRF score. Gate on RRF and the system will never abstain — it will
answer every question, confidently, from whatever chunks happened to rank first.

If you touch `rrf_fuse()`, preserve the `similarity` field. It carries the
**best** raw cosine seen for that chunk across all variant lists, not the last
one written.

### 2. Passages are numbered globally across sub-questions

In `retrieval.answer()`, `numbering` maps `chunk_id -> passage number`, assigned
in order of first appearance across **all** sub-questions.

If each sub-question numbered its own context from 1, the `[2]` in one
sub-answer and the `[2]` in another would collide at synthesis. Citations would
point at the wrong page and nothing would raise an error.

### 3. Each sub-question is answered from its own chunks only

`SUB_ANSWER_PROMPT` runs once per sub-question, seeing only that sub-question's
fused chunks. `SYNTHESIZE_PROMPT` then reads the **sub-answers**, never raw
chunks.

Do not "optimise" this into one generation call over pooled chunks. Pooling lets
evidence for one sub-question contaminate the answer to another.

### 4. The judge fails closed

`judge.validate()` scores an unparseable LLM response as `0.0`, which fails the
thresholds and triggers abstention. This is deliberate. In deep-search mode, an
answer that cannot be verified against its own sources is not an answer.

The judge grades against **all** retrieved chunks the generator could see, not
just the ones it cited — otherwise an uncited hallucination would score well.

### 5. Embedding models require the right task prefixes

Each embedding model in `_EMBED_REGISTRY` declares its own task prefixes.
nomic-embed-text REQUIRES `search_document: ` on ingestion and `search_query: `
on retrieval. mxbai-embed-large and qwen3-embedding work correctly without
prefixes. Prefixes are applied at embed time only and never stored in
`chunks.content`.

When you change `EMBED_MODEL`, the system auto-migrates the `chunks.embedding`
column to the new dimension on startup — but existing vectors are dropped (they
are useless in a different model's embedding space). You MUST re-ingest every
report. A dimension mismatch between Ollama's output and the pgvector column
would otherwise surface as a cryptic Postgres error; the startup check prevents
that.

---

## Two chunkers, on purpose

| | Function | Size | Job |
|---|---|---|---|
| Summarisation | `chunk_text()` / `map_chunks()` | 12000 chars | Few LLM calls, wide context |
| Retrieval | `retrieval_chunks()` | 1800 chars, 250 overlap | Precise vector matches |

They are not to be unified. One chunk size cannot serve both without being wrong
for one of them.

`retrieval_chunks()` never spans a page boundary. That costs a little recall at
seams and buys exact page citations, which analysts need to check a number
against the source PDF. Keep it that way.

Section tagging walks headings **forwards** as it scans lines. An earlier version
searched backwards from the chunk's start offset and silently returned `None`
for every chunk that opened with its own heading. Do not reintroduce that.

---

## Running it

```bash
docker compose up -d                                  # postgres + pgvector
ollama pull nomic-embed-text && ollama pull llama3.1:8b    # or mxbai-embed-large / qwen3-embedding
cp .env.example .env
pip install -r requirements.txt
cd backend && uvicorn main:app --reload                # http://localhost:8000
```

Schema is created on first boot by `init_db()`. The embedding column is
auto-migrated when you switch models. Reports must be re-ingested after a model
change.

Ingestion runs as a FastAPI `BackgroundTask`. The upload endpoint returns `202`
immediately and the UI polls `reports.status` until it flips `pending →
processing → ready | failed`. Errors land in `reports.error`, not in a traceback
the user never sees.

---

## Verifying changes

Success criteria, in order of cheapness:

```
1. python -m py_compile backend/*.py        -> verify: exits 0
2. node --check on the extracted <script>   -> verify: parses
3. chunker + RRF unit checks                -> verify: assertions pass
4. Upload one real broker PDF               -> verify: status reaches 'ready',
                                               n_chunks > 0, broker + rating populated
5. Ask one answerable + one unanswerable Q  -> verify: the second abstains
```

Steps 1–3 need no Postgres and no Ollama. Everything below step 3 does.

There is no test suite yet. If you add one, `chunking.py` and `rrf_fuse()` are
pure functions and should be tested first — they are where the subtle bugs live.

---

## Calibration is not optional

`SIM_GATE = 0.55`, `FAITHFULNESS_MIN = 0.7`, `RELEVANCY_MIN = 0.6` are
**placeholders**. Cosine distributions are model-specific; each embedding
model produces a different range of cosine similarities.

Before trusting output in front of the team: take ten real broker PDFs, write
five answerable and five unanswerable questions for each, and move the threshold
until the unanswerable ones abstain and the answerable ones don't.

Do not tune these by intuition, and do not lower them to make a demo look good.
Every point you lower `SIM_GATE` buys a confident answer built on chunks that do
not contain the answer.

---

## Deferred work — the seams are already cut

- **Tables and images.** `chunks.chunk_type` already carries the field and the
  pipeline is agnostic to it. Emit them as chunks (`table` / `image`) and
  retrieval, citation and the UI work unchanged. Stubs at the bottom of
  `extract.py`.

---

## House rules

Follow `~/.claude/skills/bestprac` (Karpathy's guidelines) — they apply here in
full. In particular, for this repo:

- **State assumptions before implementing.** If a change could break one of the
  five invariants above, say so and stop.
- **No speculative abstraction.** There is one report type, one vector store.
  Three embedding models are supported via the registry; don't build a provider
  interface for a fourth one.
- **Constants go in `config.py`, prompts go in `prompts.py`.** Never inline
  either.
- **Match the existing style**: minimal code, comments that explain *why* not
  *what*, individual files kept small.
- **Never widen the abstention path.** If a change makes the system answer where
  it previously abstained, that is a correctness regression until proven
  otherwise with calibration data.
