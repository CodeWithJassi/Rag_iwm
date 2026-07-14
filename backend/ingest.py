"""Ingestion. Runs in the background on upload; the API returns immediately.

    extract -> (summarise | extract facts | retrieval-chunk | table extract+enrich)
           -> embed -> store

Summarisation, fact extraction, retrieval chunking and table extraction all read
the same source text but are independent — they run in parallel so the slowest
step (summarisation, 30–60 s of LLM calls) hides the others.
"""
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from db import execute, pool
from embeddings import embed_documents
from extract import (caption_images, enrich_tables, extract_images,
                     extract_pages, extract_tables, full_text)
from chunking import image_chunks, retrieval_chunks, table_chunks
from summarize import extract_facts, summarize_report

logger = logging.getLogger(__name__)


def _embed_text(chunk: dict, company: str) -> str:
    """Build the text that gets embedded.  Prepends company, section and tags so
    the vector carries identity and topic signals.  For table chunks also prepends
    the enrichment summary, questions, and key metrics — this puts the embedded
    representation in the same semantic neighbourhood as analyst queries.

    Never stored in ``chunks.content`` — that stays clean (invariant #5).
    """
    parts = [company]
    if chunk.get("section"):
        parts.append(chunk["section"])
    meta = chunk.get("metadata") or {}
    tags = meta.get("tags", [])
    if tags:
        parts.append("[" + ", ".join(tags) + "]")
    prefix = " — ".join(parts)

    extra = ""
    if chunk.get("chunk_type") == "table":
        extra_parts = []
        unit_context = meta.get("unit_context", "")
        summary = meta.get("summary", "")
        questions = meta.get("questions", [])
        key_metrics = meta.get("key_metrics", [])
        if unit_context:
            extra_parts.append(f"UNITS: {unit_context}")
        if summary:
            extra_parts.append(f"SUMMARY: {summary}")
        if questions:
            extra_parts.append("QUESTIONS: " + " | ".join(questions))
        if key_metrics:
            extra_parts.append("KEY FACTS: " + " | ".join(key_metrics))
        if extra_parts:
            extra = " [TABLE] " + " ".join(extra_parts)
    elif chunk.get("chunk_type") == "image":
        extra = " [IMAGE]"

    return f"{prefix}:{extra}\n{chunk['content']}"


def _fts_text(chunk: dict) -> str:
    """Build the text for the full-text search tsvector column.  Includes content,
    tags and enrichment data so ``tsquery`` matches concepts, not just tokens.

    For example, a table tagged "revenue" will match a tsquery search for
    "revenue growth" even if the word "revenue" never appears in the markdown.
    """
    meta = chunk.get("metadata") or {}
    parts = [chunk["content"]]
    tags = meta.get("tags", [])
    if tags:
        parts.append(" ".join(tags))
    if chunk.get("chunk_type") == "table":
        for field in ("questions", "summary", "key_metrics"):
            val = meta.get(field, [])
            if isinstance(val, list):
                parts.append(" ".join(val))
            elif val:
                parts.append(val)
    return " ".join(parts)


def _fail(report_id: int, msg: str) -> None:
    execute("UPDATE reports SET status='failed', error=%s WHERE id=%s", (msg[:500], report_id))
    logger.error(f"report {report_id} failed: {msg}")


def ingest_report(report_id: int, pdf_path: str, company: str,
                  extract_tables_enabled: bool = True,
                  ocr_enabled: bool = True) -> None:
    """Full ingestion for one report. Never raises — failures land in reports.status.

    *extract_tables_enabled* — UI toggle: skip table pipeline when False.
    *ocr_enabled*            — UI toggle: skip VLM OCR fallback for scanned PDFs.
    """
    try:
        execute("UPDATE reports SET status='processing', error=NULL WHERE id=%s", (report_id,))

        pages = extract_pages(pdf_path, ocr_enabled=ocr_enabled)
        if not pages:
            return _fail(report_id, "No text could be extracted from the PDF")
        text = full_text(pages)

        # 1–4) Run summarisation, fact extraction, chunking, and (optionally)
        #      table extraction, and image extraction in parallel.  All read from
        #      text/pages but none modifies shared state.  The slowest step
        #      (summarisation, 30–60 s) dominates; the others run for free.
        tbl_chunks: list[dict] = []
        img_chunks: list[dict] = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            fut_summary = ex.submit(summarize_report, text, company)
            fut_facts = ex.submit(extract_facts, text, company)
            fut_chunks = ex.submit(retrieval_chunks, pages)
            # Table pipeline: extract → enrich → chunk.
            if extract_tables_enabled:
                fut_tables = ex.submit(
                    lambda: table_chunks(
                        enrich_tables(extract_tables(pdf_path, pages), company),
                        pages,
                    ))
            # Image pipeline: extract → caption → chunk.  The VLM calls
            # (caption_images) are the bottleneck here, but they overlap with
            # summarisation so wall-clock impact is near zero.
            fut_images = ex.submit(
                lambda: image_chunks(
                    caption_images(extract_images(pdf_path, report_id, pages),
                                   company),
                    pages,
                ))

            facts = fut_facts.result()
            chunks = fut_chunks.result()
            summary = fut_summary.result()
            if extract_tables_enabled:
                tbl_chunks = fut_tables.result()
            img_chunks = fut_images.result()

        # Fall back to a raw snippet if every LLM provider is down.
        if not summary:
            logger.warning(f"report {report_id}: falling back to non-LLM summary")
            summary = text[:1500].strip() + " …"

        # Merge table and image chunks into the main list, renumber ord globally.
        n_text = len(chunks)
        for extra in (tbl_chunks, img_chunks):
            if extra:
                for c in extra:
                    c["ord"] = len(chunks)
                    chunks.append(c)

        if not chunks:
            return _fail(report_id, "Extraction produced no usable chunks "
                         f"({n_text} text, {len(tbl_chunks)} table, "
                         f"{len(img_chunks)} image)")

        # Embedding: each chunk gets a contextual prefix so the vector carries
        # company, section, tags and (for tables) enrichment semantics.  The DB
        # stores the original content — prefix is embed-time only (invariant #5).
        embed_inputs = [_embed_text(c, company) for c in chunks]
        vectors = embed_documents(embed_inputs)

        # One transaction: chunks and report metadata land together, so a report
        # is never marked 'ready' with a half-written chunk table.
        with pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE report_id=%s", (report_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks
                       (report_id, ord, page_no, section, chunk_type, content,
                        embedding, fts, metadata)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,
                               to_tsvector('english', %s), %s)""",
                    [(report_id, c["ord"], c["page_no"], c["section"],
                      c["chunk_type"], c["content"], v,
                      _fts_text(c),
                      json.dumps(c.get("metadata")) if c.get("metadata") else None)
                     for c, v in zip(chunks, vectors)],
                )
            conn.execute(
                """UPDATE reports SET summary=%s, broker=%s, recommendation=%s,
                          current_price=%s, target_price=%s, n_chunks=%s, status='ready'
                   WHERE id=%s""",
                (summary, facts["broker"], facts["recommendation"],
                 facts["current_price"], facts["target_price"], len(chunks), report_id),
            )

        n_text = sum(1 for c in chunks if c["chunk_type"] == "text")
        n_table = sum(1 for c in chunks if c["chunk_type"] == "table")
        n_image = sum(1 for c in chunks if c["chunk_type"] == "image")
        logger.info(f"report {report_id} ready: {len(chunks)} chunks "
                    f"({n_text} text, {n_table} table, {n_image} image), "
                    f"broker={facts['broker']}")

    except Exception:
        _fail(report_id, traceback.format_exc())
