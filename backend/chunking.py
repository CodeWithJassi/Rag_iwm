"""Two chunkers, two jobs. Deliberately not unified.

Summarisation wants few, large chunks (fewer LLM calls, more context per call).
Retrieval wants many, small, overlapping chunks (precise vector matches).
One chunk size cannot serve both without being wrong for one of them.
"""
import logging
import re

from config import (MAX_MAP_CHUNKS, RETRIEVAL_CHUNK_CHARS,
                    RETRIEVAL_CHUNK_OVERLAP, SUMMARY_CHUNK_CHARS)

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^#{1,6}\s+(.+)$")  # markdown headings from pymupdf4llm


# ====================================================================
# Summarisation chunker -- unchanged from the existing script.
# ====================================================================

def chunk_text(text: str, max_chars: int = SUMMARY_CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries into <= max_chars chunks (never mid-word)."""
    try:
        paragraphs = text.split("\n\n")
        chunks, current = [], ""
        for para in paragraphs:
            if len(para) > max_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars])
                continue
            if len(current) + len(para) + 2 <= max_chars:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = para + "\n\n"
        if current.strip():
            chunks.append(current.strip())
        return [c for c in chunks if c.strip()] or [text[:max_chars]]
    except Exception as e:
        logger.error(f"chunk_text failed: {e}")
        return [text[:max_chars]]


def map_chunks(text: str) -> list[str]:
    """Split into <= MAX_MAP_CHUNKS chunks, scaling chunk size up for huge reports."""
    chunks = chunk_text(text, SUMMARY_CHUNK_CHARS)
    if len(chunks) <= MAX_MAP_CHUNKS:
        return chunks
    bigger = (len(text) // MAX_MAP_CHUNKS) + 1
    logger.info(f"{len(chunks)} chunks > cap {MAX_MAP_CHUNKS}; rechunking at ~{bigger} chars")
    return chunk_text(text, max(SUMMARY_CHUNK_CHARS, bigger))


# ====================================================================
# Retrieval chunker -- page-aware, overlapped, section-tagged.
# ====================================================================

def _split_page(text: str, size: int, overlap: int) -> list[tuple[str | None, str]]:
    """Sliding window over a page's lines. Returns (section, chunk_content).

    The section is whatever markdown heading was in force when the chunk opened,
    tracked forwards as we walk. Searching backwards from the chunk's offset
    would miss the common case: a chunk whose *first line* is its own heading.

    Keeps a 'Risks' chunk from being retrieved as if it were 'Investment Thesis'.
    """
    out: list[tuple[str | None, str]] = []
    buf, buf_sec, sec = "", None, None

    for line in (l for l in text.split("\n") if l.strip()):
        m = _HEADING.match(line)
        if m:
            sec = m.group(1).strip()
        # A single line longer than the chunk size (dense table rows do this)
        # gets hard-split, otherwise it would blow past the window on its own.
        pieces = ([line[i:i + size] for i in range(0, len(line), size)]
                  if len(line) > size else [line])

        for piece in pieces:
            if buf and len(buf) + len(piece) + 1 > size:
                out.append((buf_sec, buf.strip()))
                # Carry the tail forward so a fact straddling the seam stays
                # retrievable from at least one chunk.
                tail = buf[-overlap:] if overlap else ""
                buf = f"{tail}\n{piece}" if tail else piece
                buf_sec = sec
            else:
                if not buf:
                    buf_sec = sec
                buf += ("\n" if buf else "") + piece

    if buf.strip():
        out.append((buf_sec, buf.strip()))
    return out


def retrieval_chunks(pages: list[dict]) -> list[dict]:
    """Page list -> [{ord, page_no, section, chunk_type, content}].

    Chunks never span pages. That costs a little recall at page boundaries and
    buys exact page citations, which analysts need to check a number against
    the source PDF.
    """
    out, ord_ = [], 0
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue
        for section, content in _split_page(
                text, RETRIEVAL_CHUNK_CHARS, RETRIEVAL_CHUNK_OVERLAP):
            if len(content) < 50:  # drop page-number fragments and stray footers
                continue
            out.append({
                "ord": ord_,
                "page_no": page["page_no"],
                "section": section,
                "chunk_type": "text",
                "content": content,
            })
            ord_ += 1
    logger.info(f"built {len(out)} retrieval chunks from {len(pages)} pages")
    return out
