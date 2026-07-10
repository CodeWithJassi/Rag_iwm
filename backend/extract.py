"""Tiered PDF text extraction, same fallback ladder as the existing summariser.

One change: extraction is now page-aware. It returns a list of pages rather than
one blob, because retrieval chunks need a page number to cite. The summariser
just joins them back together.

Tables and images are not extracted yet -- see the stubs at the bottom.
"""
import logging

logger = logging.getLogger(__name__)

Page = dict  # {"page_no": int, "text": str}


def _via_pymupdf4llm(path: str) -> list[Page]:
    """Best quality: preserves headings as markdown, which we later use to tag
    each chunk with the section it came from."""
    import pymupdf4llm
    pages = pymupdf4llm.to_markdown(path, page_chunks=True)
    return [{"page_no": i + 1, "text": p.get("text", "")} for i, p in enumerate(pages)]


def _via_pymupdf(path: str) -> list[Page]:
    import pymupdf
    doc = pymupdf.open(path)
    return [{"page_no": i + 1, "text": p.get_text("text")} for i, p in enumerate(doc)]


def _via_pdfplumber(path: str) -> list[Page]:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return [{"page_no": i + 1, "text": p.extract_text() or ""}
                for i, p in enumerate(pdf.pages)]


def extract_pages(path: str) -> list[Page]:
    """Walk the ladder. First extractor that yields real text wins."""
    for name, fn in (("pymupdf4llm", _via_pymupdf4llm),
                     ("pymupdf", _via_pymupdf),
                     ("pdfplumber", _via_pdfplumber)):
        try:
            pages = [p for p in fn(path) if p["text"].strip()]
            if pages:
                logger.info(f"extracted {len(pages)} pages via {name}")
                return pages
        except Exception as e:
            logger.warning(f"{name} extraction failed: {e}")
    logger.error(f"all extractors failed for {path}")
    return []


def full_text(pages: list[Page]) -> str:
    """Flatten for the summariser, which does not care about page boundaries."""
    return "\n\n".join(p["text"] for p in pages).strip()


# --------------------------------------------------------------------------
# Deferred. When you add these, emit them as chunks with chunk_type='table' /
# 'image' and they flow through embedding, retrieval and citation unchanged --
# the schema and pipeline already carry the field.
#
#   tables: pdfplumber .extract_tables() or camelot, serialised to markdown
#   images: PyMuPDF image extraction, then a vision-LLM caption pass; the
#           caption becomes the chunk text, the image path goes in metadata.
# --------------------------------------------------------------------------
