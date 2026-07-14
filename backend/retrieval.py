"""The query pipeline.

    query + history
      -> contextualise (follow-up becomes standalone)
      -> decompose     (complex becomes N independent sub-questions)
      -> rephrase      (each sub-question becomes N search variants)
      -> vector search (one ranked list per variant, scoped to one report)
      -> text search   (full-text PostgreSQL tsvector — catches exact terms)
      -> SIMILARITY GATE on raw cosine  <-- abstention decided here
      -> RRF fusion    (variants + text collapse into one ranked list)
      -> cross-encoder rerank  (query+passage scored together by bge-reranker)
      -> sub-answer    (each sub-question answered from its own chunks only)
      -> synthesise    (sub-answers combined into the final answer)

Two things worth not breaking:

1. The gate reads raw cosine similarity, never RRF score. RRF is rank-based --
   the top hit of a garbage retrieval still gets the best RRF score, so gating
   on it would never abstain.

2. Chunks are numbered globally across all sub-questions before generation. If
   each sub-question numbered its own passages from 1, the [2] in one
   sub-answer and the [2] in another would collide when synthesised, and the
   citations would silently point at the wrong page.
"""
import logging
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from config import (ABSTAIN_MSG, EMBED_QUERY_PREFIX, MAX_SUB_QUESTIONS,
                    N_REPHRASINGS_DEEP, N_REPHRASINGS_NORMAL, RERANK_TOP_K,
                    RRF_K, SIM_GATE, TOP_K_PER_VARIANT, TOP_K_TEXT,
                    TOP_N_AFTER_FUSION)
from db import query
from embeddings import _embed_with_failover, embed_query


def embed_queries_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple query texts in a single API call — faster than N separate
    calls when ``retrieve_for`` has multiple rephrasing variants."""
    if not texts:
        return []
    prefixed = [EMBED_QUERY_PREFIX + t for t in texts]
    return _embed_with_failover(prefixed, timeout=30)
from llm import llm_complete, llm_json
from prompts import (CONTEXTUALIZE_PROMPT, DECOMPOSE_PROMPT, REPHRASE_PROMPT,
                     SUB_ANSWER_PROMPT, SYNTHESIZE_PROMPT)
from rerank import rerank

logger = logging.getLogger(__name__)
_CITE = re.compile(r"\[(\d+)\]")


def _is_empty(answer: str) -> bool:
    """True if the answer contains no extractable facts.

    Exact ``INSUFFICIENT`` is empty.  Beyond that, we check for citation
    markers like ``[1]`` — if the answer cites a passage, the model found and
    referenced real facts.  No citations = nothing extractable."""
    a = answer.strip()
    if a == "INSUFFICIENT":
        return True
    # Citation markers mean the model found and referenced real passages.
    return not bool(_CITE.search(a))


def _clean_answer(ans: str) -> str:
    """Strip a hedging INSUFFICIENT preamble when real content follows.

    Some models write ``INSUFFICIENT`` then immediately provide the facts they
    did find — the word is a hedge, not a true abstention.  Drop the preamble
    so the user sees the facts, not the hedge.  If nothing with citations
    remains, the answer is truly INSUFFICIENT."""
    a = ans.strip()
    lines = a.split("\n")
    # Strip leading lines that are exactly "INSUFFICIENT" (possibly separated
    # by blank lines).
    while lines and (not lines[0].strip() or lines[0].strip() == "INSUFFICIENT"):
        lines.pop(0)
    if not lines:
        return "INSUFFICIENT"
    # If the first remaining line starts with "INSUFFICIENT " followed by
    # real content, strip just that word.
    if lines[0].startswith("INSUFFICIENT ") and len(lines[0]) > 13:
        lines[0] = lines[0][13:].strip()
    result = "\n".join(lines).strip()
    # If nothing with a citation remains, the answer is truly empty.
    if not _CITE.search(result):
        return "INSUFFICIENT"
    return result


class CancelledError(Exception):
    """Raised when a user clicks Stop — caught by the endpoint to return a partial result."""
    pass


# ------------------------------------------------------------------ query prep

def contextualize(user_query: str, history: list[dict], model: str = "") -> str:
    """Follow-up -> standalone question. Passes through unchanged on turn one.

    History is formatted with recency markers so the LLM knows which messages are
    most likely to contain the referent. Only the most recent turns are included
    to keep the prompt focused."""
    if not history:
        return user_query
    # Keep only the most recent turns and add recency markers.
    MAX_HISTORY = 6  # last 3 exchanges
    recent = history[-MAX_HISTORY:]
    lines = []
    for i, t in enumerate(recent):
        tag = " [MOST RECENT]" if i == len(recent) - 1 else ""
        lines.append(f"{t['role']}: {t['content']}{tag}")
    transcript = "\n".join(lines)
    out = llm_complete(CONTEXTUALIZE_PROMPT.format(history=transcript, query=user_query),
                       max_tokens=150, label="contextualize", model=model)
    return out or user_query


def decompose(standalone: str, company: str, model: str = "") -> list[str]:
    """Complex question -> independent sub-questions. Simple questions return [self].

    Post-decomposition guards prevent scope drift: if the LLM produced a
    sub-question that dropped the company name or broadened to industry-level,
    we fold it back into the standalone question rather than letting bogus
    sub-questions pollute retrieval and confuse the judge."""
    subs = llm_json(DECOMPOSE_PROMPT.format(company=company, query=standalone,
                                            max_subs=MAX_SUB_QUESTIONS),
                    max_tokens=250, label="decompose", model=model)
    if not isinstance(subs, list) or not subs:
        return [standalone]
    clean: list[str] = []
    company_lower = company.lower()
    for s in subs:
        s = str(s).strip()
        if not s:
            continue
        # If the original question mentions a specific company but this
        # sub-question doesn't, the LLM broadened scope.  Skip it — the
        # standalone question already covers what the user asked.
        if company_lower not in s.lower():
            logger.info("decompose: dropping sub-question that lost company "
                        "scope — '%s'", s[:100])
            continue
        clean.append(s)
    if not clean:
        return [standalone]
    return clean[:MAX_SUB_QUESTIONS]


def rephrase(sub_question: str, n: int = N_REPHRASINGS_NORMAL,
              model: str = "") -> list[str]:
    """One sub-question -> n search variants. Original always included.

    When n <= 1 there is nothing to generate — the original query is the only
    variant we need. Skipping the LLM call saves ~1 s per sub-question."""
    if n <= 1:
        return [sub_question]
    variants = llm_json(REPHRASE_PROMPT.format(query=sub_question, n=n),
                        max_tokens=200, label="rephrase", model=model)
    if not isinstance(variants, list):
        return [sub_question]
    out = [str(v).strip() for v in variants if str(v).strip()][:n]
    if sub_question not in out:
        out = [sub_question] + out[:n - 1]
    return out


# ------------------------------------------------------------------ search

def vector_search(report_id: int, text: str = "", *,
                   vec: list[float] | None = None,
                   k: int = TOP_K_PER_VARIANT) -> list[dict]:
    """Cosine search over one report's chunks. `<=>` is cosine distance, so
    similarity is 1 - distance.

    Pass *vec* to skip the embedding call (batch path).  Pass *text* for the
    legacy one-at-a-time path."""
    if vec is None:
        vec = embed_query(text)
    return query(
        """SELECT id, page_no, section, chunk_type, content,
                  1 - (embedding <=> %s::vector) AS similarity,
                  COALESCE((metadata->>'importance')::int, 3) AS importance
             FROM chunks
            WHERE report_id = %s
         ORDER BY embedding <=> %s::vector
            LIMIT %s""",
        (vec, report_id, vec, k),
    )


def text_search(report_id: int, text: str, k: int = TOP_K_TEXT) -> list[dict]:
    """Full-text search over one report's chunks using PostgreSQL tsvector.
    Catches exact terms — tickers, broker names, "EBITDA margin" — that dense
    retrieval sometimes misses.

    Uses OR semantics: any word in the query can match.  With small chunks
    (800 chars), AND-ing all terms would rarely hit.  `plainto_tsquery` does
    proper stemming + stop-word removal; we convert its `&` tree to `|` so
    chunks matching *any* query term surface and ts_rank sorts them."""
    return query(
        """SELECT id, page_no, section, chunk_type, content,
                  ts_rank(fts, q) AS similarity,
                  COALESCE((metadata->>'importance')::int, 3) AS importance
             FROM chunks,
                  LATERAL (SELECT replace(plainto_tsquery('english', %s)::text,
                                          ' & ', ' | ')::tsquery AS q) tq
            WHERE report_id = %s AND fts @@ tq.q
         ORDER BY ts_rank(fts, tq.q) DESC
            LIMIT %s""",
        (text, report_id, k),
    )


def rrf_fuse(ranked_lists: list[list[dict]], top_n: int = TOP_N_AFTER_FUSION) -> list[dict]:
    """Reciprocal Rank Fusion.  score(chunk) = sum over lists of 1/(RRF_K + rank)

    A chunk that surfaces in several rephrasings' results accumulates score and
    floats up. That is precisely the signal multi-query expansion is producing,
    and it costs no model and no dependency to read it.
    """
    scores = defaultdict(float)
    chunks = {}
    best_sim = defaultdict(float)

    for ranked in ranked_lists:
        for rank, ch in enumerate(ranked, start=1):
            scores[ch["id"]] += 1.0 / (RRF_K + rank)
            chunks[ch["id"]] = ch
            best_sim[ch["id"]] = max(best_sim[ch["id"]], float(ch["similarity"]))

    fused = []
    for cid, score in sorted(scores.items(), key=lambda x: -x[1])[:top_n]:
        ch = dict(chunks[cid])
        ch["rrf_score"] = score
        ch["similarity"] = best_sim[cid]  # keep the raw score for the gate/UI
        fused.append(ch)
    return fused


def retrieve_for(sub_question: str, report_id: int, n_rephrasings: int = N_REPHRASINGS_NORMAL,
                  model: str = "") -> tuple[list[dict], float]:
    """Search a sub-question every way we know how, gate, fuse, rerank.

    Hybrid: vector search (n rephrased variants) + full-text search (1 raw query).
    All lists enter RRF fusion → cross-encoder rerank → top-K.

    The abstention gate reads raw cosine from the vector lists ONLY — ts_rank
    scores and reranker scores never influence it (invariant #1)."""
    variants = rephrase(sub_question, n=n_rephrasings, model=model)
    # Batch-embed all variants in one API call instead of N separate calls.
    variant_vecs = embed_queries_batch(variants)
    vec_lists = [vector_search(report_id, vec=v) for v in variant_vecs]
    text_list = text_search(report_id, sub_question)
    # Gate on vector cosine only. Text search ts_rank is not a cosine and must
    # not influence the abstention decision (invariant #1).
    max_sim = max((float(c["similarity"]) for lst in vec_lists for c in lst), default=0.0)

    # Fuse the ranked lists into one pool, then rerank with a cross-encoder that
    # reads query+passage together.  The fusion pool is wider (20) than the final
    # set fed to the generator (6) so the reranker has room to promote good
    # passages that RRF might have ranked lower.
    fused = rrf_fuse(vec_lists + [text_list], top_n=TOP_N_AFTER_FUSION)
    reranked = rerank(sub_question, fused, top_k=RERANK_TOP_K)
    # Importance acts as a tiebreaker after reranking.  The reranker's score is
    # the primary signal; importance only breaks near-ties — a chunk the ingestion
    # pipeline flagged as highly decision-relevant (cover page thesis, key
    # financials) surfaces before an equally-relevant disclaimer paragraph.
    reranked.sort(key=lambda c: (
        c.get("rerank_score", c.get("similarity", 0)),
        int(c.get("importance", 3))),
        reverse=True)
    return reranked, max_sim


# ------------------------------------------------------------------ generation

def _render(chunks: list[dict], numbering: dict[int, int]) -> str:
    """Context block, labelled with each chunk's *global* passage number.

    For table chunks, a ``[UNITS: ...]`` note is inserted between the header and
    the content when ``unit_context`` is present in metadata.  This puts the unit
    declaration directly above the table in the generator's context window —
    impossible to miss, even for smaller models.
    """
    blocks: list[str] = []
    for c in chunks:
        header = f"[{numbering[c['id']]}] (page {c['page_no']}"
        if c.get("section"):
            header += f", {c['section']}"
        header += ")"
        body = c["content"]
        if c.get("chunk_type") == "table":
            unit_ctx = (c.get("metadata") or {}).get("unit_context", "")
            if unit_ctx:
                header += f"\n[UNITS: {unit_ctx}]"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def answer(report: dict, user_query: str, history: list[dict],
           deep_search: bool = False, model: str = "",
           verbose: bool = False,
           check_cancel: Callable[[], None] | None = None) -> dict:
    """Run the full pipeline. Returns everything the UI and the judge need.

    Normal mode: contextualize → rephrase → search → answer. One pass, fast.
    Deep-search: also decomposes complex queries into sub-questions and runs
    more rephrasing variants per sub-question, then validates with the judge.

    When *verbose* is True the result includes a ``trace`` list — one entry per
    pipeline step with label, summary, optional detail, and elapsed ms.  The UI
    can render this to show exactly what the retriever did.

    *check_cancel*, when provided, is called between pipeline stages.  If the
    user clicked Stop it raises CancelledError, which the endpoint catches to
    return a partial result."""
    company = report["company"]
    report_id = report["id"]
    t0 = time.time()
    trace: list[dict] = []

    _trace_lock = threading.Lock()

    def _add(label: str, summary: str, detail: str = "") -> None:
        with _trace_lock:
            trace.append({"label": label, "summary": summary, "detail": detail,
                          "ms": round((time.time() - t0) * 1000)})

    def _check() -> None:
        if check_cancel:
            check_cancel()

    standalone = contextualize(user_query, history, model=model)
    _add("contextualize", "follow-up → standalone" if history else "already standalone",
         f"\"{user_query[:120]}\" → \"{standalone[:120]}\"" if history else standalone[:200])

    _check()

    # Decomposition only runs in deep-search mode. Normal mode treats the
    # standalone question as the single sub-question — faster and fewer LLM calls.
    if deep_search:
        sub_questions = decompose(standalone, company, model=model)
        n_rephrasings = N_REPHRASINGS_DEEP
        _add("decompose", f"1 → {len(sub_questions)} sub-question(s)",
             " · ".join(f"\"{q[:100]}\"" for q in sub_questions))
    else:
        sub_questions = [standalone]
        n_rephrasings = N_REPHRASINGS_NORMAL
        _add("decompose", "normal mode — single sub-question", standalone[:200])

    _check()

    # Retrieve per sub-question in parallel — each retrieval is an independent
    # I/O-bound operation (embedding + vector search + text search + rerank).
    retrieved: dict[str, list[dict]] = {}
    gated_out: list[str] = []
    _gate_lock = threading.Lock()

    def _retrieve_one(sq: str) -> tuple[str, list[dict] | None, float]:
        _check()
        chunks, max_sim = retrieve_for(sq, report_id, n_rephrasings=n_rephrasings,
                                       model=model)
        if max_sim < SIM_GATE:
            logger.info(f"gate: '{sq}' best sim {max_sim:.3f} < {SIM_GATE} -- skipping")
            with _gate_lock:
                gated_out.append(sq)
            _add("retrieve", f"GATED OUT — best cosine {max_sim:.3f} < {SIM_GATE}",
                 f"\"{sq[:120]}\" — no chunk in this report is close enough")
            return sq, None, max_sim
        chunk_lines = []
        for c in chunks[:6]:
            rr = f" → rerank {c.get('rerank_score', 0):.3f}" if c.get('rerank_score') else ""
            snip = c['content'][:80].replace('\n', ' ')
            chunk_lines.append(f"p{c['page_no']} cos={c['similarity']:.3f}{rr} | {snip}…")
        _add("retrieve", f"✓ {len(chunks)} chunks, best cosine {max_sim:.3f}",
             "\n".join(chunk_lines))
        return sq, chunks, max_sim

    workers = min(4, len(sub_questions))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_retrieve_one, sq): sq for sq in sub_questions}
        for fut in as_completed(futures):
            sq, chunks, _ = fut.result()
            if chunks is not None:
                retrieved[sq] = chunks

    if not retrieved:  # nothing in this report is close enough to any sub-question
        return {"answer": ABSTAIN_MSG, "abstained": True, "reason": "similarity_gate",
                "standalone": standalone, "sub_questions": sub_questions,
                "citations": [], "context": "", "chunks": [], "trace": trace}

    # Global passage numbering, assigned in order of first appearance. Keeps [n]
    # unambiguous once the sub-answers are synthesised together.
    numbering, all_chunks = {}, []
    for chunks in retrieved.values():
        for c in chunks:
            if c["id"] not in numbering:
                numbering[c["id"]] = len(numbering) + 1
                all_chunks.append(c)

    _add("fuse", f"{len(all_chunks)} unique chunks across {len(retrieved)} "
                 f"sub-question(s)",
         f"chunk IDs: {sorted(numbering.values())}")

    # Cap the context fed to the judge. When many chunks are retrieved the prompt
    # overflows the model's effective attention window — which makes it score
    # conservatively (failing closed) even on good answers.
    MAX_JUDGE_CHUNKS = 12
    judge_context = _render(all_chunks[:MAX_JUDGE_CHUNKS], numbering) \
        if len(all_chunks) > MAX_JUDGE_CHUNKS else None

    _check()

    # Answer each sub-question from its own chunks only, in parallel — each
    # sub-answer is an independent LLM call (I/O-bound).
    sub_answers: dict[str, str] = {}

    def _answer_one(sq: str, chunks: list[dict]) -> tuple[str, str]:
        _check()
        out = llm_complete(
            SUB_ANSWER_PROMPT.format(company=company, query=sq,
                                     context=_render(chunks, numbering)),
            max_tokens=400, label="sub_answer", model=model)
        ans = _clean_answer(out or "INSUFFICIENT")
        _add("sub_answer",
             "INSUFFICIENT" if _is_empty(ans) else "answered",
             ans[:500] + ("…" if len(ans) > 500 else ""))
        return sq, ans

    workers = min(4, max(1, len(retrieved)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_answer_one, sq, chunks): sq
                   for sq, chunks in retrieved.items()}
        for fut in as_completed(futures):
            sq, ans = fut.result()
            sub_answers[sq] = ans

    for sq in gated_out:
        sub_answers[sq] = "INSUFFICIENT"

    if all(_is_empty(a) for a in sub_answers.values()):
        return {"answer": ABSTAIN_MSG, "abstained": True,
                "reason": "no_grounded_sub_answer",
                "standalone": standalone, "sub_questions": sub_questions,
                "citations": [], "context": "", "chunks": [], "trace": trace}

    _check()

    # Single sub-question: the sub-answer *is* the answer. Skip a pointless LLM call.
    if len(sub_answers) == 1:
        final = next(iter(sub_answers.values()))
        _add("synthesize", "single sub-question — used as-is", "")
    else:
        # Separate real answers from INSUFFICIENT ones. Only the real answers go
        # into the synthesis block — the LLM should never see "INSUFFICIENT" as
        # raw material to paste into its output. Missing facets are noted in the
        # prompt so the LLM can acknowledge gaps honestly.
        good = {q: a for q, a in sub_answers.items()
                if not _is_empty(a)}
        missing = [q for q, a in sub_answers.items()
                   if _is_empty(a)]
        missing_note = ""
        if missing:
            missing_note = ("The following facets could NOT be answered from the "
                            "report:\n" + "\n".join(f"- {q}" for q in missing))
        block = "\n\n".join(f"SUB-QUESTION: {q}\nANSWER: {a}"
                            for q, a in good.items())
        final = llm_complete(
            SYNTHESIZE_PROMPT.format(company=company, query=standalone,
                                     sub_answers=block,
                                     missing_note=missing_note),
            max_tokens=600, label="synthesize", model=model) or ""
        _add("synthesize", f"{len(sub_answers)} sub-answers → 1",
             final[:500] + ("…" if len(final) > 500 else ""))

    if not final.strip() or _is_empty(final.strip()):
        return {"answer": ABSTAIN_MSG, "abstained": True,
                "reason": "generator_abstained",
                "standalone": standalone, "sub_questions": sub_questions,
                "citations": [], "context": "", "chunks": [], "trace": trace}

    # Resolve the [n] markers the generator actually used back to real chunks.
    cited_nums = {int(n) for n in _CITE.findall(final)}
    by_num = {v: k for k, v in numbering.items()}
    cited_ids = {by_num[n] for n in cited_nums if n in by_num}
    citations = [
        {"n": numbering[c["id"]], "chunk_id": c["id"], "page_no": c["page_no"],
         "section": c["section"], "chunk_type": c.get("chunk_type", "text"),
         "similarity": round(float(c["similarity"]), 3),
         "snippet": c["content"][:280]}
        for c in all_chunks if c["id"] in cited_ids
    ]

    # If the LLM ignored the citation instruction (common with smaller models),
    # auto-append a Sources reference block so the analyst always sees where the
    # information came from. Each source shows page number and a snippet preview.
    if not cited_nums and all_chunks:
        lines = ["\n\n---\n**Sources:**"]
        for c in all_chunks:
            pg = c["page_no"]
            sec = f", {c['section']}" if c.get("section") else ""
            snip = c["content"][:200].replace("\n", " ")
            n = numbering[c["id"]]
            lines.append(f"- [{n}] Page {pg}{sec} — {snip}…")
        final = final.strip() + "\n".join(lines)
        # Include all retrieved chunks as citations since the LLM didn't cite inline.
        citations = [
            {"n": numbering[c["id"]], "chunk_id": c["id"], "page_no": c["page_no"],
             "section": c["section"], "chunk_type": c.get("chunk_type", "text"),
             "similarity": round(float(c["similarity"]), 3),
             "snippet": c["content"][:280]}
            for c in all_chunks
        ]
        citations.sort(key=lambda c: c["n"])

    return {
        "answer": final.strip(),
        "abstained": False,
        "reason": None,
        "standalone": standalone,
        "sub_questions": sub_questions,
        "sub_answers": sub_answers,
        "citations": sorted(citations, key=lambda c: c["n"]),
        # Truncate context for the judge when many chunks are retrieved. The full
        # set is still available in result["chunks"] for the UI. Capping at ~12
        # chunks prevents the judge prompt from overflowing the model's attention
        # window, which made deep-search mode score conservatively on good answers.
        "context": judge_context or _render(all_chunks, numbering),
        "chunks": all_chunks,
        "trace": trace,
    }
