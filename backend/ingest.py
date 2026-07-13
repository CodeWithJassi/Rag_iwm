"""Ingestion. Runs in the background on upload; the API returns immediately.

    extract -> (summarise | extract facts | retrieval-chunk) -> embed -> store

Summarisation, fact extraction and retrieval chunking read the same extracted
text but are independent of each other — they run in parallel to keep the
slowest step (summarisation, which makes LLM calls) from blocking the others.
"""
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from db import execute, pool
from embeddings import embed_documents
from extract import extract_pages, full_text
from chunking import retrieval_chunks
from summarize import extract_facts, summarize_report

logger = logging.getLogger(__name__)


def _prefix(chunk: dict, company: str) -> str:
    """Assemble a contextual prefix for embedding so the vector carries company
    and section identity, not just bare sentence text.  Never stored in the DB —
    ``chunks.content`` stays clean (invariant #5)."""
    parts = [company]
    if chunk.get("section"):
        parts.append(chunk["section"])
    return " — ".join(parts) + ": "


def _fail(report_id: int, msg: str) -> None:
    execute("UPDATE reports SET status='failed', error=%s WHERE id=%s", (msg[:500], report_id))
    logger.error(f"report {report_id} failed: {msg}")


def ingest_report(report_id: int, pdf_path: str, company: str) -> None:
    """Full ingestion for one report. Never raises -- failures land in reports.status."""
    try:
        execute("UPDATE reports SET status='processing', error=NULL WHERE id=%s", (report_id,))

        pages = extract_pages(pdf_path)
        if not pages:
            return _fail(report_id, "No text could be extracted from the PDF")
        text = full_text(pages)

        # 1–3) Run summarisation, fact extraction and chunking in parallel.
        #    Each is independent — all three read from `text`/`pages` but none
        #    modifies shared state.  The slowest step (summarisation, 30–60 s of
        #    LLM calls) dominates the critical path; the other two complete for
        #    free during that wait.
        with ThreadPoolExecutor(max_workers=3) as ex:
            fut_summary = ex.submit(summarize_report, text, company)
            fut_facts = ex.submit(extract_facts, text, company)
            fut_chunks = ex.submit(retrieval_chunks, pages)

            facts = fut_facts.result()
            chunks = fut_chunks.result()
            summary = fut_summary.result()

        # Fall back to a raw snippet if every LLM provider is down.
        if not summary:
            logger.warning(f"report {report_id}: falling back to non-LLM summary")
            summary = text[:1500].strip() + " …"

        if not chunks:
            return _fail(report_id, "Extraction produced no usable chunks")

        # Embedding: contextual prefix is prepended so the vector carries
        # company + section identity.  The DB stores the original content —
        # prefix is embed-time only (invariant #5).
        prefixed = [_prefix(c, company) + c["content"] for c in chunks]
        vectors = embed_documents(prefixed)

        # 4) one transaction: chunks and report metadata land together, so a
        #    report is never marked 'ready' with a half-written chunk table.
        with pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE report_id=%s", (report_id,))  # idempotent re-ingest
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (report_id, ord, page_no, section, chunk_type, content, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                    [(report_id, c["ord"], c["page_no"], c["section"],
                      c["chunk_type"], c["content"], v) for c, v in zip(chunks, vectors)],
                )
            conn.execute(
                """UPDATE reports SET summary=%s, broker=%s, recommendation=%s,
                          current_price=%s, target_price=%s, n_chunks=%s, status='ready'
                   WHERE id=%s""",
                (summary, facts["broker"], facts["recommendation"],
                 facts["current_price"], facts["target_price"], len(chunks), report_id),
            )

        logger.info(f"report {report_id} ready: {len(chunks)} chunks, broker={facts['broker']}")

    except Exception:
        _fail(report_id, traceback.format_exc())
