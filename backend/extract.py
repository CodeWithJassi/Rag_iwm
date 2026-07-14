"""Tiered PDF text extraction, same fallback ladder as the existing summariser.

One change: extraction is now page-aware. It returns a list of pages rather than
one blob, because retrieval chunks need a page number to cite. The summariser
just joins them back together.

Tables are extracted via pdfplumber + pymupdf4llm markdown parsing and enriched
with an LLM pass that generates questions, summary, key metrics and topic tags.
Images are deferred — see the stubs at the bottom.
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (TABLE_ENRICH_CONCURRENCY, TABLE_ENRICH_ENABLED,
                    TABLE_ENRICH_MAX_TOKENS, TABLE_EXTRACTION_ENABLED,
                    TABLE_MIN_ROWS)

logger = logging.getLogger(__name__)

Page = dict  # {"page_no": int, "text": str}

# Matches a markdown table row: starts with |, ends with |, has at least one
# more | in between (so || is rejected — those are alignment separators).
_MD_TABLE_ROW = re.compile(r"^\|.+\|.*\|$")


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


def _via_ocr(path: str) -> list[Page]:
    """OCR fallback for scanned PDFs — renders pages as images, transcribes via VLM.

    Only invoked when all three text extractors returned empty.  Each page is
    rendered to a 200-DPI PNG via PyMuPDF, then sent to the vision model with a
    transcription prompt.  Pages run in parallel (capped at OCR_CONCURRENCY) so a
    40-page scanned report transcribes in ~40-60 seconds.

    Disabled when OCR_ENABLED is False — scanned PDFs will fail instead.
    """
    from io import BytesIO
    from pathlib import Path

    import pymupdf
    from PIL import Image

    from config import OCR_CONCURRENCY, OCR_DPI
    from llm import llm_vision
    from prompts import OCR_PROMPT

    doc = pymupdf.open(path)
    n_pages = len(doc)
    logger.info("OCR: transcribing %d page(s) via VLM", n_pages)

    def _ocr_page(i: int) -> dict | None:
        """Render page *i* (0-indexed) to PNG and transcribe."""
        try:
            page = doc[i]
            pix = page.get_pixmap(dpi=OCR_DPI)
            # Convert pixmap to PNG bytes via Pillow for smaller size.
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buf = BytesIO()
            img.save(buf, format="PNG")
            # Write temp file since llm_vision reads from disk.
            tmp = Path(f"/tmp/ocr_p{i}.png")
            tmp.write_bytes(buf.getvalue())
            prompt = OCR_PROMPT.format(company="the company")
            text = llm_vision(str(tmp), prompt, max_tokens=600,
                              label=f"ocr_p{i + 1}")
            tmp.unlink(missing_ok=True)
            if text and text.strip():
                return {"page_no": i + 1, "text": text.strip(), "extractor": "ocr"}
        except Exception as e:
            logger.warning("OCR page %d failed: %s", i + 1, e)
        return None

    results: list[dict | None]
    if n_pages == 1:
        results = [_ocr_page(0)]
    else:
        results = [None] * n_pages
        workers = min(OCR_CONCURRENCY, n_pages)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_i = {ex.submit(_ocr_page, i): i for i in range(n_pages)}
            for fut in as_completed(fut_to_i):
                i = fut_to_i[fut]
                try:
                    results[i] = fut.result()
                except Exception:
                    results[i] = None

    pages = [r for r in results if r and r["text"].strip()]
    logger.info("OCR transcribed %d/%d pages", len(pages), n_pages)
    return pages


def extract_pages(path: str, ocr_enabled: bool = True) -> list[Page]:
    """Walk the ladder. First extractor that yields real text wins.

    When all three text extractors return empty (scanned PDF, no embedded text
    layer) and *ocr_enabled* is True, falls back to VLM-based OCR via
    ``_via_ocr()``.  This handles scanned documents, handwritten notes, and
    misprinted values.

    Each page dict gets an ``extractor`` key so provenance flows through to
    chunk metadata — when a bad answer surfaces you can trace it to the exact
    PDF library that produced the source text."""
    for name, fn in (("pymupdf4llm", _via_pymupdf4llm),
                     ("pymupdf", _via_pymupdf),
                     ("pdfplumber", _via_pdfplumber)):
        try:
            pages = [p for p in fn(path) if p["text"].strip()]
            if pages:
                for p in pages:
                    p["extractor"] = name
                logger.info(f"extracted {len(pages)} pages via {name}")
                return pages
        except Exception as e:
            logger.warning(f"{name} extraction failed: {e}")

    # OCR fallback: scanned PDFs with no embedded text layer.
    if ocr_enabled:
        from config import OCR_ENABLED
        if OCR_ENABLED:
            try:
                pages = _via_ocr(path)
                if pages:
                    return pages
            except Exception as e:
                logger.warning(f"OCR extraction failed: {e}")

    logger.error(f"all extractors failed for {path}")
    return []


def full_text(pages: list[Page]) -> str:
    """Flatten for the summariser, which does not care about page boundaries."""
    return "\n\n".join(p["text"] for p in pages).strip()


# ====================================================================== tables
# Two sources, combined with deduplication:
#   1. pdfplumber .extract_tables() — more reliable cell alignment
#   2. pymupdf4llm markdown output — tables already rendered as | col | col |
# Each table is serialised to GitHub-flavoured markdown and enriched with an LLM
# call that generates questions, a one-line summary, key metrics, and topic tags.

def _markdown_table(rows: list[list[str | None]]) -> str:
    """Serialise a pdfplumber table (list of rows) to a markdown table string."""
    if not rows or not any(rows):
        return ""
    # Strip fully-empty trailing rows (common pdfplumber artifact).
    while rows and not any(cell for cell in rows[-1] if cell and str(cell).strip()):
        rows.pop()
    if not rows:
        return ""
    # Normalise: every row has the same number of columns.
    n_cols = max(len(r) for r in rows)
    padded: list[list[str]] = []
    for r in rows:
        cells = [str(c or "").replace("\n", " ").strip() for c in r]
        while len(cells) < n_cols:
            cells.append("")
        padded.append(cells)
    lines = ["| " + " | ".join(padded[0]) + " |",
             "|" + "|".join([" --- " for _ in range(n_cols)]) + "|"]
    for r in padded[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _tables_via_pdfplumber(path: str) -> list[dict]:
    """Extract tables using pdfplumber's built-in table detector."""
    import pdfplumber
    tables: list[dict] = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_tables()
                if not raw:
                    continue
                for raw_table in raw:
                    if not raw_table or len(raw_table) < TABLE_MIN_ROWS:
                        continue
                    md = _markdown_table(raw_table)
                    if md:
                        tables.append({
                            "page_no": i,
                            "markdown": md,
                            "source": "pdfplumber",
                            "rows": len(raw_table),
                            "cols": max(len(r) for r in raw_table),
                        })
        logger.info("pdfplumber: %d table(s) found", len(tables))
    except Exception as e:
        logger.warning("pdfplumber table extraction failed: %s", e)
    return tables


def _tables_via_markdown(pages: list[Page]) -> list[dict]:
    """Parse markdown tables that pymupdf4llm already rendered in the page text.

    A markdown table is a contiguous block of lines that all match the |...|
    pattern, with at least one alignment row (|---|...|).
    """
    tables: list[dict] = []
    for page in pages:
        lines = page["text"].split("\n")
        i = 0
        while i < len(lines):
            if not _MD_TABLE_ROW.match(lines[i]):
                i += 1
                continue
            # Collect the contiguous table block.
            start = i
            while i < len(lines) and _MD_TABLE_ROW.match(lines[i]):
                i += 1
            block = lines[start:i]
            # Must have at least a header, an alignment row, and one data row.
            if len(block) < 3:
                continue
            # Check for an alignment separator row (|---|...|).
            has_sep = any(re.match(r"^\|[\s\-:]+\|", ln) for ln in block)
            if not has_sep:
                continue
            tables.append({
                "page_no": page["page_no"],
                "markdown": "\n".join(block),
                "source": "pymupdf4llm",
                "rows": len(block),
                "cols": block[0].count("|") - 1,
            })
    logger.info("pymupdf4llm markdown: %d table(s) found", len(tables))
    return tables


def _deduplicate_tables(pdfplumber_tables: list[dict],
                        md_tables: list[dict]) -> list[dict]:
    """Merge overlapping tables from the two sources.

    A pdfplumber table and a markdown table on the same page whose content shares
    >50% of numeric cells are the same table.  Prefer pdfplumber (better cell
    alignment); discard the markdown duplicate.
    """
    if not md_tables:
        return pdfplumber_tables
    if not pdfplumber_tables:
        return md_tables

    # Group markdown tables by page for quick lookup.
    md_by_page: dict[int, list[dict]] = {}
    for t in md_tables:
        md_by_page.setdefault(t["page_no"], []).append(t)

    keep = list(pdfplumber_tables)
    for md_t in md_tables:
        candidates = [t for t in pdfplumber_tables
                      if t["page_no"] == md_t["page_no"]]
        if not candidates:
            keep.append(md_t)
            continue
        is_dup = False
        md_nums = set(re.findall(r"\d+\.?\d*", md_t["markdown"]))
        for pt in candidates:
            pt_nums = set(re.findall(r"\d+\.?\d*", pt["markdown"]))
            if not md_nums or not pt_nums:
                continue
            overlap = len(md_nums & pt_nums) / min(len(md_nums), len(pt_nums))
            if overlap > 0.5:
                is_dup = True
                break
        if not is_dup:
            keep.append(md_t)

    if len(keep) < len(pdfplumber_tables) + len(md_tables):
        logger.info("deduplicated: %d → %d tables",
                    len(pdfplumber_tables) + len(md_tables), len(keep))
    return keep


def extract_tables(path: str, pages: list[Page] | None = None) -> list[dict]:
    """Extract tables from a PDF.  Returns a list of dicts with keys:
    page_no, markdown, source, rows, cols.

    When *pages* are provided (from a prior pymupdf4llm extraction) the markdown
    parser runs; otherwise only pdfplumber is used.  Disabled entirely when
    ``TABLE_EXTRACTION_ENABLED`` is False.
    """
    if not TABLE_EXTRACTION_ENABLED:
        return []

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_pp = ex.submit(_tables_via_pdfplumber, path)
        fut_md = ex.submit(_tables_via_markdown, pages) if pages else None
        pp_tables = fut_pp.result()
        md_tables = fut_md.result() if fut_md else []

    return _deduplicate_tables(pp_tables, md_tables)


# ====================================================================== enrichment

def _enrich_one(table: dict, company: str, model: str = "") -> dict | None:
    """Run the enrichment LLM call for a single table.  Returns the parsed JSON
    response merged with the table dict, or None on failure."""
    from llm import llm_json
    from prompts import TABLE_ENRICH_PROMPT

    prompt = TABLE_ENRICH_PROMPT.format(
        company=company,
        markdown=table["markdown"][:3000],  # cap — very wide tables get cut
    )
    result = llm_json(prompt, max_tokens=TABLE_ENRICH_MAX_TOKENS,
                      label=f"table_enrich_p{table['page_no']}", model=model)
    if not isinstance(result, dict):
        logger.warning("table enrichment failed for page %d — not valid JSON",
                       table["page_no"])
        return None
    enriched = dict(table)
    enriched["questions"] = (result.get("questions") or [])[:5]
    enriched["summary"] = (result.get("summary") or "").strip()
    enriched["key_metrics"] = (result.get("key_metrics") or [])[:8]
    enriched["tags"] = [t.lower().strip() for t in (result.get("tags") or [])
                        if t and t.strip()]
    # Importance: 1–5, default to 3 (neutral) if missing or out of range.
    imp = result.get("importance", 3)
    enriched["importance"] = max(1, min(5, int(imp) if isinstance(imp, (int, float)) else 3))
    # Unit context: helps the generator attach units to cited figures.
    enriched["unit_context"] = (result.get("unit_context") or "").strip()
    return enriched


def enrich_tables(tables: list[dict], company: str, model: str = "") -> list[dict]:
    """Run enrichment LLM calls in parallel for every extracted table.

    Returns the same list with ``questions``, ``summary``, ``key_metrics`` and
    ``tags`` fields added.  Tables whose enrichment fails are returned unchanged
    (with empty enrichment fields).
    """
    if not tables or not TABLE_ENRICH_ENABLED:
        for t in tables:
            t.setdefault("questions", [])
            t.setdefault("summary", "")
            t.setdefault("key_metrics", [])
            t.setdefault("tags", [])
            t.setdefault("importance", 3)
            t.setdefault("unit_context", "")
        return tables

    if len(tables) == 1:
        enriched = _enrich_one(tables[0], company, model=model)
        return [enriched] if enriched else [dict(tables[0],
                   questions=[], summary="", key_metrics=[], tags=[],
                   importance=3, unit_context="")]

    out: list[dict] = [None for _ in tables]  # type: ignore[assignment]
    workers = min(TABLE_ENRICH_CONCURRENCY, len(tables))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_idx = {ex.submit(_enrich_one, t, company, model): i
                      for i, t in enumerate(tables)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            enriched = fut.result()
            if enriched:
                out[idx] = enriched
            else:
                t = tables[idx]
                out[idx] = dict(t, questions=[], summary="",
                                key_metrics=[], tags=[], importance=3,
                                unit_context="")
    logger.info("enriched %d/%d tables", sum(1 for o in out if o["summary"]),
                len(out))
    return out


# ====================================================================== images

def _count_text_chars(page_text: str) -> int:
    """Count characters that are actual prose, not markdown table lines."""
    return len(re.sub(r"^\|.+\|.*\|$", "", page_text, flags=re.MULTILINE).strip())


def extract_images(path: str, report_id: int,
                   pages: list[Page] | None = None) -> list[dict]:
    """Extract images from a PDF.  Two sources:

    1. **Embedded raster images** — extracted via PyMuPDF's ``get_images()`` on
       each page.  Filtered by minimum pixel size (logos, icons, spacers skipped).

    2. **Vector chart pages** — pages whose text extraction yielded very little
       prose (< 200 chars) but which are not blank.  These typically contain
       charts, graphs, or diagrams drawn as vector paths — invisible to
       ``get_images()``.  Rendered to a 200-DPI PNG via ``get_pixmap()``.

    Returns a list of dicts with keys: page_no, image_path, width, height,
    is_page_render.  Disabled when ``VISION_ENABLED`` is False.
    """
    from pathlib import Path

    from config import IMAGE_MIN_SIZE, VISION_ENABLED
    if not VISION_ENABLED:
        return []

    import pymupdf

    out_dir = Path("data/images") / str(report_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    seen_pages: set[int] = set()  # track pages that already have raster images
    doc = pymupdf.open(path)

    # --- Source 1: embedded raster images ---
    for i, page in enumerate(doc, start=1):
        for j, img in enumerate(page.get_images(full=True)):
            width, height = img[2], img[3]
            if width < IMAGE_MIN_SIZE or height < IMAGE_MIN_SIZE:
                continue
            try:
                base = doc.extract_image(img[0])
                ext = base["ext"]
                img_path = out_dir / f"p{i}_{j}.{ext}"
                img_path.write_bytes(base["image"])
                images.append({
                    "page_no": i,
                    "image_path": str(img_path),
                    "width": width,
                    "height": height,
                    "is_page_render": False,
                })
                seen_pages.add(i)
            except Exception as e:
                logger.warning("failed to extract image on page %d: %s", i, e)

    # --- Source 2: vector chart detection ---
    # Pages with very little text but not blank → likely a vector chart.
    if pages:
        for p in pages:
            pn = p["page_no"]
            if pn in seen_pages:
                continue
            text = p.get("text", "").strip()
            if not text:
                continue  # truly blank page
            if _count_text_chars(text) < 200:
                try:
                    page = doc[pn - 1]  # PyMuPDF is 0-indexed
                    pix = page.get_pixmap(dpi=200)
                    img_path = out_dir / f"p{pn}_vector.png"
                    pix.save(str(img_path))
                    images.append({
                        "page_no": pn,
                        "image_path": str(img_path),
                        "width": pix.width,
                        "height": pix.height,
                        "is_page_render": True,
                    })
                except Exception as e:
                    logger.warning("vector chart render failed page %d: %s", pn, e)

    logger.info("extracted %d images (%d raster, %d vector) from %s",
                len(images), len(seen_pages), len(images) - len(seen_pages), path)
    return images


def caption_images(images: list[dict], company: str, model: str = "") -> list[dict]:
    """Run the vision LLM on every extracted image in parallel.

    Returns the same list with a ``caption`` key added.  Images whose captioning
    fails get ``caption=""`` and are still returned — the caller can decide
    whether to drop them.
    """
    if not images:
        return images

    from config import IMAGE_CAPTION_CONCURRENCY, IMAGE_CAPTION_MAX_TOKENS
    from llm import llm_vision
    from prompts import IMAGE_CAPTION_PROMPT

    prompt = IMAGE_CAPTION_PROMPT.format(company=company)

    def _caption_one(img: dict) -> dict:
        cap = llm_vision(
            img["image_path"], prompt,
            max_tokens=IMAGE_CAPTION_MAX_TOKENS,
            model=model,
            label=f"caption_p{img['page_no']}",
        )
        out = dict(img)
        out["caption"] = cap.strip() if cap else ""
        return out

    if len(images) == 1:
        return [_caption_one(images[0])]

    results: list[dict] = [None for _ in images]
    workers = min(IMAGE_CAPTION_CONCURRENCY, len(images))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_idx = {ex.submit(_caption_one, img): i
                      for i, img in enumerate(images)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                logger.warning("caption failed for image on page %d: %s",
                               images[idx]["page_no"], e)
                results[idx] = dict(images[idx], caption="")

    succeeded = sum(1 for r in results if r and r.get("caption"))
    logger.info("captioned %d/%d images", succeeded, len(results))
    return results
