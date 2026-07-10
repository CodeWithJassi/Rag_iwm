"""The query pipeline.

    query + history
      -> contextualise (follow-up becomes standalone)
      -> decompose     (complex becomes N independent sub-questions)
      -> rephrase      (each sub-question becomes N search variants)
      -> vector search (one ranked list per variant, scoped to one report)
      -> SIMILARITY GATE on raw cosine  <-- abstention decided here
      -> RRF fusion    (variants collapse into one ranked list per sub-question)
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
from collections import defaultdict

from config import (ABSTAIN_MSG, MAX_SUB_QUESTIONS, N_REPHRASINGS_DEEP,
                    N_REPHRASINGS_NORMAL, RRF_K, SIM_GATE, TOP_K_PER_VARIANT,
                    TOP_K_TEXT, TOP_N_AFTER_FUSION)
from db import query
from embeddings import embed_query
from llm import llm_complete, llm_json
from prompts import (CONTEXTUALIZE_PROMPT, DECOMPOSE_PROMPT, REPHRASE_PROMPT,
                     SUB_ANSWER_PROMPT, SYNTHESIZE_PROMPT)

logger = logging.getLogger(__name__)
_CITE = re.compile(r"\[(\d+)\]")


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
    """Complex question -> independent sub-questions. Simple questions return [self]."""
    subs = llm_json(DECOMPOSE_PROMPT.format(company=company, query=standalone,
                                            max_subs=MAX_SUB_QUESTIONS),
                    max_tokens=250, label="decompose", model=model)
    if not isinstance(subs, list) or not subs:
        return [standalone]
    clean = [str(s).strip() for s in subs if str(s).strip()]
    return clean[:MAX_SUB_QUESTIONS] or [standalone]


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

def vector_search(report_id: int, text: str, k: int = TOP_K_PER_VARIANT) -> list[dict]:
    """Cosine search over one report's chunks. `<=>` is cosine distance, so
    similarity is 1 - distance."""
    vec = embed_query(text)
    return query(
        """SELECT id, page_no, section, content,
                  1 - (embedding <=> %s::vector) AS similarity
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
        """SELECT id, page_no, section, content,
                  ts_rank(fts, q) AS similarity
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
    """Search a sub-question every way we know how, fuse, and report the best raw
    cosine similarity seen -- the gate needs it.

    Hybrid: vector search (n rephrased variants) + full-text search (1 raw query).
    All lists enter RRF fusion. The abstention gate reads raw cosine from
    the vector lists ONLY — ts_rank scores never influence the gate."""
    variants = rephrase(sub_question, n=n_rephrasings, model=model)
    vec_lists = [vector_search(report_id, v) for v in variants]
    text_list = text_search(report_id, sub_question)
    # Gate on vector cosine only. Text search ts_rank is not a cosine and must
    # not influence the abstention decision (invariant #1).
    max_sim = max((float(c["similarity"]) for lst in vec_lists for c in lst), default=0.0)
    return rrf_fuse(vec_lists + [text_list]), max_sim


# ------------------------------------------------------------------ generation

def _render(chunks: list[dict], numbering: dict[int, int]) -> str:
    """Context block, labelled with each chunk's *global* passage number."""
    return "\n\n".join(
        f"[{numbering[c['id']]}] (page {c['page_no']}"
        + (f", {c['section']}" if c.get("section") else "") + f")\n{c['content']}"
        for c in chunks
    )


def answer(report: dict, user_query: str, history: list[dict],
           deep_search: bool = False, model: str = "") -> dict:
    """Run the full pipeline. Returns everything the UI and the judge need.

    Normal mode: contextualize → rephrase → search → answer. One pass, fast.
    Deep-search: also decomposes complex queries into sub-questions and runs
    more rephrasing variants per sub-question, then validates with the judge."""
    company = report["company"]
    report_id = report["id"]

    standalone = contextualize(user_query, history, model=model)

    # Decomposition only runs in deep-search mode. Normal mode treats the
    # standalone question as the single sub-question — faster and fewer LLM calls.
    if deep_search:
        sub_questions = decompose(standalone, company, model=model)
        n_rephrasings = N_REPHRASINGS_DEEP
    else:
        sub_questions = [standalone]
        n_rephrasings = N_REPHRASINGS_NORMAL
    logger.info(f"'{standalone}' -> {len(sub_questions)} sub-question(s) "
                f"(deep={deep_search}, rephrasings={n_rephrasings})")

    # Retrieve per sub-question, gating each on raw cosine similarity.
    retrieved: dict[str, list[dict]] = {}
    gated_out: list[str] = []
    for sq in sub_questions:
        chunks, max_sim = retrieve_for(sq, report_id, n_rephrasings=n_rephrasings,
                                       model=model)
        if max_sim < SIM_GATE:
            logger.info(f"gate: '{sq}' best sim {max_sim:.3f} < {SIM_GATE} -- skipping")
            gated_out.append(sq)
            continue
        retrieved[sq] = chunks

    if not retrieved:  # nothing in this report is close enough to any sub-question
        return {"answer": ABSTAIN_MSG, "abstained": True, "reason": "similarity_gate",
                "standalone": standalone, "sub_questions": sub_questions,
                "citations": [], "context": "", "chunks": []}

    # Global passage numbering, assigned in order of first appearance. Keeps [n]
    # unambiguous once the sub-answers are synthesised together.
    numbering, all_chunks = {}, []
    for chunks in retrieved.values():
        for c in chunks:
            if c["id"] not in numbering:
                numbering[c["id"]] = len(numbering) + 1
                all_chunks.append(c)

    # Cap the context fed to the judge. When many chunks are retrieved the prompt
    # overflows the model's effective attention window — which makes it score
    # conservatively (failing closed) even on good answers.
    MAX_JUDGE_CHUNKS = 12
    judge_context = _render(all_chunks[:MAX_JUDGE_CHUNKS], numbering) \
        if len(all_chunks) > MAX_JUDGE_CHUNKS else None

    # Answer each sub-question from its own chunks only. This is what stops
    # evidence for one sub-question from bleeding into the answer for another.
    sub_answers = {}
    for sq, chunks in retrieved.items():
        out = llm_complete(
            SUB_ANSWER_PROMPT.format(company=company, query=sq,
                                     context=_render(chunks, numbering)),
            max_tokens=600, label="sub_answer", model=model)
        sub_answers[sq] = (out or "INSUFFICIENT").strip()

    for sq in gated_out:
        sub_answers[sq] = "INSUFFICIENT"

    if all(a.startswith("INSUFFICIENT") for a in sub_answers.values()):
        return {"answer": ABSTAIN_MSG, "abstained": True, "reason": "no_grounded_sub_answer",
                "standalone": standalone, "sub_questions": sub_questions,
                "citations": [], "context": "", "chunks": []}

    # Single sub-question: the sub-answer *is* the answer. Skip a pointless LLM call.
    if len(sub_answers) == 1:
        final = next(iter(sub_answers.values()))
    else:
        block = "\n\n".join(f"SUB-QUESTION: {q}\nANSWER: {a}" for q, a in sub_answers.items())
        final = llm_complete(
            SYNTHESIZE_PROMPT.format(company=company, query=standalone, sub_answers=block),
            max_tokens=900, label="synthesize", model=model) or ""

    if not final.strip() or final.strip().startswith("INSUFFICIENT"):
        return {"answer": ABSTAIN_MSG, "abstained": True, "reason": "generator_abstained",
                "standalone": standalone, "sub_questions": sub_questions,
                "citations": [], "context": "", "chunks": []}

    # Resolve the [n] markers the generator actually used back to real chunks.
    cited_nums = {int(n) for n in _CITE.findall(final)}
    by_num = {v: k for k, v in numbering.items()}
    cited_ids = {by_num[n] for n in cited_nums if n in by_num}
    citations = [
        {"n": numbering[c["id"]], "chunk_id": c["id"], "page_no": c["page_no"],
         "section": c["section"], "similarity": round(float(c["similarity"]), 3),
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
             "section": c["section"], "similarity": round(float(c["similarity"]), 3),
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
    }
