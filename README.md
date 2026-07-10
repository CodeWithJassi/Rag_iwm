# Report Desk

Summarise broker PDFs, then interrogate them. Grounded answers only — the system
abstains rather than guessing.

## Run

```bash
docker compose up -d                      # postgres + pgvector (the only new service)
ollama pull nomic-embed-text && ollama pull llama3.1:8b
cp .env.example .env
pip install -r requirements.txt
cd backend && uvicorn main:app --reload    # http://localhost:8000
```

Schema is created on first boot. Upload a PDF from the `+` in the left rail.

## Flow

**Ingestion** (background, on upload)

```
PDF -> extract_pages()        tiered: pymupdf4llm -> pymupdf -> pdfplumber
    -> summarize_report()     map-reduce, large chunks
    -> extract_facts()        broker, rating, CMP, target  (one LLM call)
    -> retrieval_chunks()     small overlapped chunks, page + section tagged
    -> embed_documents()      nomic-embed-text, "search_document: " prefix
    -> chunks + reports       one transaction; never half-written
```

**Query**

```
query + history
  -> contextualize()   follow-up becomes standalone
  -> decompose()       complex becomes N sub-questions (max 4)
  -> rephrase()        each becomes 3 search variants
  -> vector_search()   one ranked list per variant, scoped to one report
  -> SIMILARITY GATE   max raw cosine < 0.55  ->  abstain
  -> rrf_fuse()        variants collapse into one list per sub-question
  -> sub-answer        each sub-question answered from its own chunks only
  -> synthesize()      sub-answers combined
  -> [deep search]     judge faithfulness + relevancy -> abstain if below threshold
```

## Four things worth not breaking

**The gate reads raw cosine, never RRF score.** RRF is rank-based. The top hit of
a garbage retrieval still scores best under RRF, so gating on it would never
abstain. `retrieve_for()` returns the pre-fusion max cosine for exactly this.

**Passages are numbered globally across sub-questions.** If each sub-question
numbered its own context from 1, the `[2]` in one sub-answer and the `[2]` in
another would collide at synthesis, and citations would silently point at the
wrong page.

**Each sub-question is answered from its own chunks.** Pooling all retrieved
chunks into one generation call lets evidence for one sub-question contaminate
the answer to another. The synthesis step reads sub-*answers*, not raw chunks.

**The judge fails closed.** An unparseable judge response scores 0.0 and fails
the thresholds. In deep-search mode an answer that cannot be verified against
its own sources is not an answer.

## Two chunkers, on purpose

`SUMMARY_CHUNK_CHARS=12000` for map-reduce (few LLM calls, wide context).
`RETRIEVAL_CHUNK_CHARS=1800` with 250 overlap for vector search (precise matches).
One size cannot serve both without being wrong for one of them.

## Deferred, with the seams already cut

- **Tables and images.** `chunks.chunk_type` already carries the field and the
  pipeline is agnostic. Emit them as chunks (`table` / `image`), and retrieval,
  citation and the UI work unchanged. Stubs are at the bottom of `extract.py`.
- **Hybrid retrieval.** Add a Postgres `tsvector` full-text list as a fourth
  ranked input to the same `rrf_fuse()` call. Catches exact terms — tickers,
  "EBITDA margin", broker names — that dense retrieval sometimes misses. Zero
  new infra.
- **Cross-encoder reranker.** Slots in *after* fusion (fuse to top-20, rerank to
  top-6). Nothing else changes.

## Calibrate before you trust it

`SIM_GATE = 0.55` is a placeholder. Cosine distributions are model-specific.
Run ten real broker reports, ask five questions each — five answerable, five
not — and move the threshold until the unanswerable ones abstain and the
answerable ones don't. Same for `FAITHFULNESS_MIN` and `RELEVANCY_MIN`.
