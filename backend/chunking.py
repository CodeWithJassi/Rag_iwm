"""Two chunkers, two jobs. Deliberately not unified.

Summarisation wants few, large chunks (fewer LLM calls, more context per call).
Retrieval wants many, small, overlapping chunks (precise vector matches).
One chunk size cannot serve both without being wrong for one of them.

Retrieval chunking has three selectable strategies (config.CHUNKING_STRATEGY):
  "char"     — fixed-size sliding window + overlap
  "heading"  — splits at markdown heading boundaries
  "semantic" — LLM detects topic shifts, splits at context boundaries
"""
import logging
import re

from config import (CHUNK_TAG_RULES, CHUNKING_STRATEGY, MAX_MAP_CHUNKS,
                    RETRIEVAL_CHUNK_CHARS, RETRIEVAL_CHUNK_OVERLAP,
                    SECTION_TAG_RULES, SEMANTIC_CHUNK_BLOCK,
                    STRUCTURE_CEILING_CHARS, STRUCTURE_FLOOR_CHARS,
                    SUMMARY_CHUNK_CHARS, TABLE_MAX_CHARS)

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^#{1,6}\s+(.+)$")  # markdown headings from pymupdf4llm
# Bold "Label:" lead-ins from pymupdf4llm.  The colon is inside the bold markers
# — pymupdf4llm renders \"bold Label:\" as **Label:** not **Label**:
_BOLD_LABEL = re.compile(r"^\*\*([^*]+?):\*\*\s")


# ====================================================================
# Semantic tagger.  Rule-based, zero LLM cost.  Tags are stored in chunk metadata
# and included in the full-text search index so tsquery can match concepts (e.g.
# searching "revenue" finds a chunk about "top line" because we tagged it).
# ====================================================================

def _tag_chunk(content: str, section: str | None) -> list[str]:
    """Derive topic tags from chunk content and its parent section heading.

    Two sources, merged:
    1. Section heading → tags via SECTION_TAG_RULES (heading text substring match)
    2. Content words → tags via CHUNK_TAG_RULES (keyword presence match)

    Tags are lowercase, deduplicated, and sorted for stability.  Returns an empty
    list when nothing matches — most chunks carry 2–5 tags.
    """
    tags: set[str] = set()
    content_lower = content.lower()

    # Source 1: section heading → tags.
    if section:
        section_lower = section.lower()
        for tag, phrases in SECTION_TAG_RULES.items():
            for phrase in phrases:
                if phrase in section_lower:
                    tags.add(tag)
                    break  # one match per tag is enough

    # Source 2: content keyword scanning.
    for tag, keywords in CHUNK_TAG_RULES.items():
        for kw in keywords:
            if kw in content_lower:
                tags.add(tag)
                break

    return sorted(tags)


# ====================================================================
# Importance scoring for text chunks.  Heuristic — zero LLM cost.  Tables get
# an LLM-assigned importance score via the enrichment prompt instead.
# ====================================================================

def _score_importance(content: str, section: str | None, page_no: int,
                      total_pages: int) -> int:
    """Rate a text chunk 1–5 based on structural and content signals.

    Broker reports have a natural importance gradient: the cover page and
    investment thesis matter more than the disclaimer on the last page.
    These heuristics approximate that gradient without an LLM call.
    """
    score = 3  # neutral baseline
    content_lower = content.lower()

    # --- Section-based signals ---
    if section:
        sec = section.lower()
        # High-importance sections.
        if any(w in sec for w in ("investment thesis", "recommendation",
                                   "target price", "valuation", "key catalyst",
                                   "outlook", "guidance", "result", "financial",
                                   "executive summary")):
            score += 1
        # Low-importance sections.
        if any(w in sec for w in ("disclaimer", "appendix", "glossary",
                                   "annexure", "general disclosure",
                                   "rating definition", "terms of use")):
            score -= 2

    # --- Page-position signals ---
    # First 15% of the report: cover, thesis, key numbers — higher importance.
    if total_pages > 3 and page_no <= max(2, total_pages * 0.15):
        score += 1
    # Last 10%: disclaimers, annexures — lower importance.
    if total_pages > 5 and page_no >= total_pages * 0.9:
        score -= 1

    # --- Content signals ---
    has_numbers = bool(re.search(r"\d+\.?\d*\s*(cr|crore|lakh|%|bps|₹|rs|inr)",
                                  content_lower))
    has_target = bool(re.search(r"target\s+price|price\s+target|cmp|current\s+price",
                                 content_lower))
    if has_target:
        score += 1
    elif has_numbers:
        score += 0  # numbers alone don't raise importance, but their absence lowers it
    # Very short chunks are often page artifacts.
    if len(content) < 100:
        score -= 1

    return max(1, min(5, score))


# ====================================================================
# Section breadcrumbs.  Tracks the full heading hierarchy (e.g. "Investment
# Thesis > Key Catalysts > Demand Drivers") instead of just the nearest heading.
# Pure string processing — zero LLM cost.
# ====================================================================

def _build_heading_tree(pages: list[dict]) -> list[tuple[int, str, int]]:
    """Scan all pages and return a flat list of (level, text, page_no) for every
    markdown heading.  Level is the number of leading ``#`` characters.

    Used by ``_resolve_path`` to reconstruct the ancestor chain for any chunk.
    """
    headings: list[tuple[int, str, int]] = []
    for page in pages:
        for line in page["text"].split("\n"):
            m = _HEADING.match(line.strip())
            if m:
                level = m.group(0).index(" ")  # position of first space = # count
                headings.append((level, m.group(1).strip(), page["page_no"]))
    return headings


def _resolve_path(heading_tree: list[tuple[int, str, int]],
                  leaf: str | None, page_no: int) -> list[str]:
    """Rebuild the heading stack up to *leaf* on *page_no*.

    Walks the flattened heading tree in document order, maintaining a stack that
    pops deeper/equal headings when a higher-level heading appears.  Returns the
    full path from the document root to the leaf.  If *leaf* is None or not found,
    returns the path at the end of the given page.
    """
    if not leaf:
        return []
    stack: list[tuple[int, str]] = []
    found = False
    for level, text, pn in heading_tree:
        if pn > page_no:
            break
        # Pop headings at this level or deeper before pushing.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        if pn == page_no and text == leaf:
            found = True
            break
    if not found:
        # Leaf not in tree — return whatever the stack had at end of page.
        pass
    return [text for _, text in stack]


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
# Retrieval chunkers — three strategies, one dispatcher.
# ====================================================================

# --- strategy 1: character limit + overlap (default) -----------------

def _split_page_char(text: str, size: int, overlap: int) -> list[tuple[str | None, str]]:
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


# --- strategy 2: heading-aware ---------------------------------------

def _split_page_heading(text: str, size: int) -> list[tuple[str | None, str]]:
    """Split at heading boundaries. Sections stay together; long sections fall
    back to character-level splitting internally.

    A heading is a natural topic boundary — the analyst writing the report
    already told us where one idea ends and the next begins.  Using those
    boundaries keeps chunks thematically coherent."""
    out: list[tuple[str | None, str]] = []
    buf, buf_sec, sec = "", None, None

    lines = [l for l in text.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        is_heading = bool(m)

        # A new heading means the topic changed — emit what we have and
        # start fresh.  If the buffer is tiny (just a stray line) fold it in.
        if is_heading and buf.strip() and len(buf) > 120:
            out.append((buf_sec, buf.strip()))
            buf, buf_sec = "", None

        if m:
            sec = m.group(1).strip()

        # Handle an overlong single line (dense table rows).
        pieces = ([line[i:i + size] for i in range(0, len(line), size)]
                  if len(line) > size else [line])

        for piece in pieces:
            if buf and len(buf) + len(piece) + 1 > size:
                out.append((buf_sec, buf.strip()))
                buf, buf_sec = piece, sec
            else:
                if not buf:
                    buf_sec = sec
                buf += ("\n" if buf else "") + piece

    if buf.strip():
        out.append((buf_sec, buf.strip()))
    return out


# --- strategy 3: semantic (LLM detects topic shifts) ------------------

def _split_page_semantic(text: str, size: int) -> list[tuple[str | None, str]]:
    """Use the LLM to find where the topic shifts, then split there.

    Paragraphs are grouped into blocks of SEMANTIC_CHUNK_BLOCK chars.  For each
    block the LLM returns paragraph indices where a new chunk should start.
    Those breakpoints are used to build chunks; overlong chunks fall back to
    character-level splitting."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    # Find the current section heading (if any) for tagging.
    sec: str | None = None
    for p in paragraphs:
        m = _HEADING.match(p.split("\n")[0])
        if m:
            sec = m.group(1).strip()
            break

    # Gather split-point indices from the LLM.  We process the page in blocks
    # to keep each LLM call focused and fast.
    split_indices: set[int] = set()
    block_start = 0
    while block_start < len(paragraphs):
        # Build a block of paragraphs up to SEMANTIC_CHUNK_BLOCK chars.
        block_paras: list[str] = []
        chars = 0
        idx = block_start
        while idx < len(paragraphs) and chars < SEMANTIC_CHUNK_BLOCK:
            block_paras.append(paragraphs[idx])
            chars += len(paragraphs[idx])
            idx += 1

        if len(block_paras) >= 3:
            # Number paragraphs for the LLM so it can refer to indices.
            numbered = "\n\n".join(
                f"[{i}] {p}" for i, p in enumerate(block_paras, start=block_start))
            try:
                from llm import llm_json
                from prompts import SEMANTIC_CHUNK_PROMPT
                result = llm_json(
                    SEMANTIC_CHUNK_PROMPT.format(paragraphs=numbered),
                    max_tokens=150, label="semantic_chunk")
                if isinstance(result, list):
                    for n in result:
                        if isinstance(n, int) and 0 < n < len(paragraphs):
                            split_indices.add(n)
            except Exception as e:
                logger.warning("semantic chunking LLM call failed: %s — "
                               "falling back to char-splitting for this block", e)

        block_start = idx

    # Build chunks between split points.
    if not split_indices:
        # No topic shifts found — treat the whole page as one section and
        # fall back to char-splitting.
        return _split_page_char(text, size, RETRIEVAL_CHUNK_OVERLAP)

    split_indices.add(0)
    splits = sorted(split_indices)

    out: list[tuple[str | None, str]] = []
    for i, start in enumerate(splits):
        end = splits[i + 1] if i + 1 < len(splits) else len(paragraphs)
        chunk_text_blob = "\n\n".join(paragraphs[start:end]).strip()
        if not chunk_text_blob:
            continue
        if len(chunk_text_blob) <= size:
            out.append((sec, chunk_text_blob))
        else:
            # Section is longer than one chunk — fall back to char-splitting.
            for s, c in _split_page_char(chunk_text_blob, size, 0):
                out.append((s or sec, c))

    return out


# --- strategy 4: structure-aware (bold Label: lead-ins) ----------------

def _split_page_structure(text: str, floor: int = STRUCTURE_FLOOR_CHARS,
                          ceiling: int = STRUCTURE_CEILING_CHARS
                          ) -> list[tuple[str | None, str]]:
    """Split at bold ``**Label:**`` lead-ins from pymupdf4llm markdown.

    Each labeled block becomes its own chunk regardless of length — the
    analyst who wrote the report already told us where topics begin and end.
    Blocks shorter than *floor* chars are merged into the previous chunk;
    blocks longer than *ceiling* chars fall back to character-level splitting
    (preserving the section label on each sub-chunk).

    Pages with no bold labels fall back to char-splitting so disclaimer pages,
    tables, and unstructured sections still chunk correctly.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return []

    # ---- pass 1: split into labeled blocks ----
    blocks: list[tuple[str | None, list[str]]] = []
    current_label: str | None = None
    current_lines: list[str] = []

    for line in lines:
        m = _BOLD_LABEL.match(line)
        if m:
            if current_lines:
                blocks.append((current_label, current_lines))
            current_label = m.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append((current_label, current_lines))

    # If no bold labels found anywhere, fall back to char-splitting.
    if len(blocks) == 1 and blocks[0][0] is None:
        return _split_page_char(text, ceiling, 0)

    # ---- pass 2: merge blocks below floor ----
    merged: list[tuple[str | None, str]] = []
    for label, block_lines in blocks:
        content = "\n".join(block_lines).strip()
        if len(content) < floor and merged:
            # Merge into the previous block — a tiny labeled block
            # (e.g. a one-line "Risks: None.") doesn't stand alone.
            prev_label, prev_content = merged[-1]
            merged[-1] = (prev_label, prev_content + "\n" + content)
        else:
            merged.append((label, content))

    # ---- pass 3: split blocks above ceiling ----
    out: list[tuple[str | None, str]] = []
    for label, content in merged:
        if len(content) <= ceiling:
            out.append((label, content))
        else:
            # Long block — fall back to char-splitting, carrying the label.
            for sub_sec, sub_content in _split_page_char(content, ceiling, 0):
                out.append((label or sub_sec, sub_content))

    return out


# --- dispatcher -------------------------------------------------------

def retrieval_chunks(pages: list[dict]) -> list[dict]:
    """Page list -> [{ord, page_no, section, chunk_type, content, metadata}].

    Chunks never span pages. That costs a little recall at page boundaries and
    buys exact page citations, which analysts need to check a number against
    the source PDF.

    The strategy is controlled by config.CHUNKING_STRATEGY.

    Metadata includes:
      - tags: auto-derived topic labels for text search
      - section_path: full heading hierarchy [root, …, leaf]
      - provenance: {extractor, strategy} — traces a bad answer to its source
    """
    strategy = CHUNKING_STRATEGY
    if strategy not in ("char", "heading", "semantic", "structure"):
        logger.warning("unknown CHUNKING_STRATEGY '%s' — falling back to 'char'", strategy)
        strategy = "char"

    size = RETRIEVAL_CHUNK_CHARS
    # The extractor name is the same for all pages (only one extractor ladder
    # wins).  Grab it from the first page that has it.
    extractor = next((p.get("extractor", "") for p in pages if p.get("extractor")), "")
    heading_tree = _build_heading_tree(pages)

    out, ord_ = [], 0
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue

        if strategy == "char":
            pairs = _split_page_char(text, size, RETRIEVAL_CHUNK_OVERLAP)
        elif strategy == "heading":
            pairs = _split_page_heading(text, size)
        elif strategy == "structure":
            pairs = _split_page_structure(text)
        else:  # semantic
            pairs = _split_page_semantic(text, size)

        for section, content in pairs:
            if len(content) < 50:  # drop page-number fragments and stray footers
                continue
            tags = _tag_chunk(content, section)
            section_path = _resolve_path(heading_tree, section, page["page_no"])
            importance = _score_importance(content, section, page["page_no"],
                                          len(pages))
            meta: dict = {
                "provenance": {"extractor": extractor, "strategy": strategy},
                "importance": importance,
            }
            if tags:
                meta["tags"] = tags
            if section_path:
                meta["section_path"] = section_path
            out.append({
                "ord": ord_,
                "page_no": page["page_no"],
                "section": section,
                "chunk_type": "text",
                "content": content,
                "metadata": meta,
            })
            ord_ += 1

    logger.info("built %d retrieval chunks from %d pages (strategy=%s)",
                len(out), len(pages), strategy)
    return out


# ====================================================================
# Table chunker — converts enriched table dicts into retrieval-ready chunks.
# ====================================================================

def _find_nearest_section(pages: list[dict], page_no: int) -> str | None:
    """Find the nearest markdown heading on or before the given page.

    Used only by ``table_chunks`` — text chunks get their section from the split
    functions directly, and section_path from the heading tree.
    """
    for p in sorted(pages, key=lambda p: p["page_no"], reverse=True):
        if p["page_no"] > page_no:
            continue
        m = _HEADING.search(p["text"])
        if m:
            return m.group(1).strip()
    return None


def _guess_units(markdown: str) -> str:
    """Scan a markdown table for unit indicators — zero LLM cost fallback.

    Broker reports consistently use a small set of unit labels in column headers
    and row labels.  When the LLM enrichment is disabled or fails, this heuristic
    provides a reasonable unit_context so the generator still attaches units.
    """
    patterns = [
        (r"\(Rs\s*Mn\)", "Monetary values in Rs millions (Rs Mn)"),
        (r"\(Rs\s*Cr\)", "Monetary values in Rs crores (Rs Cr)"),
        (r"\(₹\s*Cr?\)", "Monetary values in ₹ crores"),
        (r"\(₹\s*Mn?\)", "Monetary values in ₹ millions"),
        (r"\(\$\s*[Mm]n?\)", "Monetary values in USD millions"),
        (r"\(\$\s*[Bb]n?\)", "Monetary values in USD billions"),
        (r"Rs\s+Lakh", "Monetary values in Rs lakhs"),
        (r"Rs\s+Cr", "Monetary values in Rs crores"),
        (r"in\s+[Ll]akhs?", "Monetary values in lakhs"),
        (r"in\s+[Mm]illions?", "Monetary values in millions"),
        (r"in\s+[Cc]rores?", "Monetary values in crores"),
        (r"\(\s*%\s*\)", "Percentages and margins in %"),
        (r"\(\s*bps?\s*\)", "Basis points (bps)"),
        (r"\(\s*x\s*\)", "Multiples (x) — P/E, EV/EBITDA, etc."),
    ]
    found: list[str] = []
    for pattern, label in patterns:
        if re.search(pattern, markdown, re.IGNORECASE):
            found.append(label)
    return "; ".join(found) if found else ""


def _make_table_meta(t: dict, section: str | None,
                     heading_tree: list[tuple[int, str, int]],
                     source: str) -> dict:
    """Build the metadata dict for a table chunk.  Called once per table, reused
    across sub-chunks when a large table is split."""
    tags = t.get("tags", [])
    section_path = _resolve_path(heading_tree, section, t["page_no"])
    # Unit context: prefer the LLM-enriched value, fall back to heuristic.
    unit_context = t.get("unit_context", "").strip()
    if not unit_context:
        unit_context = _guess_units(t.get("markdown", ""))
    meta: dict = {
        "provenance": {"extractor": t.get("source", source), "strategy": "table"},
        "importance": t.get("importance", 3),
        "unit_context": unit_context,
        "tags": tags,
        "questions": t.get("questions", []),
        "summary": t.get("summary", ""),
        "key_metrics": t.get("key_metrics", []),
        "rows": t.get("rows", 0),
        "cols": t.get("cols", 0),
        "table_source": t.get("source", ""),
    }
    if section_path:
        meta["section_path"] = section_path
    return meta


def table_chunks(tables: list[dict], pages: list[dict],
                 start_ord: int = 0) -> list[dict]:
    """Convert enriched table dicts into retrieval chunks.

    Each chunk carries ``chunk_type: "table"``, the clean markdown table in
    ``content``, and enrichment data in ``metadata``.  Large tables are split to
    stay under TABLE_MAX_CHARS, with the header row repeated on each sub-table.

    *start_ord* lets the caller continue numbering after text chunks.
    """
    heading_tree = _build_heading_tree(pages)
    # Table source provenance: the extractor name from the first page, or "table".
    table_extractor = next((p.get("extractor", "") for p in pages if p.get("extractor")), "table")

    out: list[dict] = []
    ord_ = start_ord
    for t in tables:
        markdown = t["markdown"]
        section = _find_nearest_section(pages, t["page_no"])
        meta = _make_table_meta(t, section, heading_tree, table_extractor)

        if len(markdown) <= TABLE_MAX_CHARS:
            out.append({
                "ord": ord_,
                "page_no": t["page_no"],
                "section": section,
                "chunk_type": "table",
                "content": markdown,
                "metadata": meta,
            })
            ord_ += 1
            continue

        # Split large tables: keep the header row, chunk body rows.
        lines = markdown.split("\n")
        if len(lines) < 3:
            continue
        header = lines[:2]
        body = lines[2:]
        buf = list(header)
        for row in body:
            candidate = buf + [row]
            if len("\n".join(candidate)) > TABLE_MAX_CHARS and len(buf) > 2:
                out.append({
                    "ord": ord_, "page_no": t["page_no"],
                    "section": section, "chunk_type": "table",
                    "content": "\n".join(buf), "metadata": meta,
                })
                ord_ += 1
                buf = list(header) + [row]
            else:
                buf.append(row)
        if len(buf) > 2:
            out.append({
                "ord": ord_, "page_no": t["page_no"],
                "section": section, "chunk_type": "table",
                "content": "\n".join(buf), "metadata": meta,
            })
            ord_ += 1

    logger.info("built %d table chunks from %d tables", ord_ - start_ord,
                len(tables))
    return out


# ====================================================================
# Image chunker — converts captioned images into retrieval-ready chunks.
# ====================================================================

def image_chunks(images: list[dict], pages: list[dict],
                 start_ord: int = 0) -> list[dict]:
    """Convert captioned image dicts into retrieval chunks.

    Only images with a non-empty caption are kept — uncaptioned images (VLM
    failures, logos, blank pages) are silently dropped.  Each chunk carries
    ``chunk_type: "image"`` with the caption in ``content`` and the image path,
    dimensions, and page number in ``metadata``.

    Tags are derived from the caption text via the same rule-based tagger used
    for text chunks.  Importance is scored heuristically: early-page images score
    higher; very small images score lower.
    """
    heading_tree = _build_heading_tree(pages)
    extractor = next((p.get("extractor", "") for p in pages if p.get("extractor")), "")
    total_pages = len(pages)

    out: list[dict] = []
    ord_ = start_ord
    for img in images:
        caption = img.get("caption", "").strip()
        if len(caption) < 30:  # too short to be useful — logo, blank, or failed
            continue

        page_no = img["page_no"]
        section = _find_nearest_section(pages, page_no)
        section_path = _resolve_path(heading_tree, section, page_no)
        tags = _tag_chunk(caption, section)
        importance = _score_importance(caption, section, page_no, total_pages)

        # Very small embedded images (logos, icons that passed the pixel filter
        # but aren't meaningful charts) get a slight importance penalty.
        if not img.get("is_page_render") and img.get("width", 999) < 300:
            importance = max(1, importance - 1)

        meta: dict = {
            "provenance": {"extractor": extractor, "strategy": "image"},
            "importance": importance,
            "image_path": img["image_path"],
            "page_no": page_no,
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "is_page_render": img.get("is_page_render", False),
        }
        if tags:
            meta["tags"] = tags
        if section_path:
            meta["section_path"] = section_path

        out.append({
            "ord": ord_,
            "page_no": page_no,
            "section": section,
            "chunk_type": "image",
            "content": caption,
            "metadata": meta,
        })
        ord_ += 1

    logger.info("built %d image chunks from %d extracted images",
                len(out), len(images))
    return out
