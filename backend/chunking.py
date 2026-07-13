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

from config import (CHUNKING_STRATEGY, MAX_MAP_CHUNKS, RETRIEVAL_CHUNK_CHARS,
                    RETRIEVAL_CHUNK_OVERLAP, SEMANTIC_CHUNK_BLOCK,
                    STRUCTURE_CEILING_CHARS, STRUCTURE_FLOOR_CHARS,
                    SUMMARY_CHUNK_CHARS)

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^#{1,6}\s+(.+)$")  # markdown headings from pymupdf4llm
# Bold "Label:" lead-ins from pymupdf4llm.  The colon is inside the bold markers
# — pymupdf4llm renders \"bold Label:\" as **Label:** not **Label**:
_BOLD_LABEL = re.compile(r"^\*\*([^*]+?):\*\*\s")


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
    """Page list -> [{ord, page_no, section, chunk_type, content}].

    Chunks never span pages. That costs a little recall at page boundaries and
    buys exact page citations, which analysts need to check a number against
    the source PDF.

    The strategy is controlled by config.CHUNKING_STRATEGY.
    """
    strategy = CHUNKING_STRATEGY
    if strategy not in ("char", "heading", "semantic", "structure"):
        logger.warning("unknown CHUNKING_STRATEGY '%s' — falling back to 'char'", strategy)
        strategy = "char"

    size = RETRIEVAL_CHUNK_CHARS

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
            out.append({
                "ord": ord_,
                "page_no": page["page_no"],
                "section": section,
                "chunk_type": "text",
                "content": content,
            })
            ord_ += 1

    logger.info("built %d retrieval chunks from %d pages (strategy=%s)",
                len(out), len(pages), strategy)
    return out
