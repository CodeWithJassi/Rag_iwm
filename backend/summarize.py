"""Map-reduce summarisation and fact extraction with validation.

The map phase runs chunks in parallel (up to SUMMARY_CONCURRENCY) so the remote
LLM GPU isn't waiting for individual HTTP round-trips.  After generation, a
lightweight validation pass cross-references extracted facts against the source
text to catch hallucinations before they reach the dashboard.
"""
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from chunking import map_chunks
from config import MAX_REDUCE_DEPTH, SUMMARY_CHUNK_CHARS, SUMMARY_CONCURRENCY
from llm import llm_complete, llm_json
from prompts import CHUNK_PROMPT, FACTS_PROMPT, REDUCE_PROMPT

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ validation

def _find_in_text(number: float | int, text: str) -> bool:
    """Check whether *number* (or a close variant) appears in *text*.

    Handles common Indian number formats: 450, 4,500, 4 500, 450.0, 4.5 (hundreds
    vs crores vs billions — we check all plausible representations).  Returns True
    if any variant is found.
    """
    if number is None:
        return False
    variants = {
        str(int(number)),                      # "450"
        str(number).replace(".0", ""),          # "450.0" → "450"
        f"{number:,.0f}",                       # "4,500"
        f"{number:,.1f}".rstrip("0").rstrip("."),  # "4,500" or "4,500.5"
    }
    # Also try the raw value as-is.
    variants.add(str(number))
    # Indian-style: "4 500" and "4,500"
    variants.add(f"{number:,.0f}".replace(",", " "))
    text_clean = text.replace(",", "").replace(" ", "")
    for v in variants:
        if v.replace(",", "").replace(" ", "") in text_clean:
            return True
    return False


def _validate_facts(facts: dict, text: str) -> dict:
    """Cross-reference extracted facts against the source text.

    Returns a copy of *facts* with any unverifiable numeric values replaced by
    None.  Also flags the "NOT FOUND" sentinel from the LLM prompt.
    """
    validated = dict(facts)
    source = (text or "").lower()

    # Broker: check if the name actually appears in the report text.
    broker = (facts.get("broker") or "").strip()
    if broker and broker != "NOT FOUND":
        # Check if at least 50% of the broker name words appear in the source.
        words = [w for w in broker.lower().split() if len(w) > 2]
        if words:
            found = sum(1 for w in words if w in source)
            if found < max(1, len(words) * 0.5):
                logger.warning("broker '%s' not found in source text — discarding",
                               broker)
                validated["broker"] = None
    if broker == "NOT FOUND":
        validated["broker"] = None

    # Prices: check if the number (or a close variant) appears in the source.
    for field in ("current_price", "target_price"):
        val = facts.get(field)
        if val is not None and isinstance(val, (int, float)):
            if not _find_in_text(val, text):
                logger.warning("%s=%s not found in source text — discarding",
                               field, val)
                validated[field] = None

    # Recommendation: already validated downstream (must be Buy/Hold/Sell).
    rec = (facts.get("recommendation") or "").strip().title()
    if rec == "Not Found":
        validated["recommendation"] = None

    return validated


def _clean_summary(summary: str, source_text: str) -> str:
    """Post-process a summary to flag numbers that can't be verified.

    Does NOT remove unverifiable numbers — it appends a confidence note when
    the summary contains figures absent from the source.  The analyst can then
    decide whether to trust the LLM or check the original report.
    """
    if not summary or not source_text:
        return summary or ""

    # Extract all numbers from the summary.
    nums_in_summary = set()
    for m in re.finditer(r"(\d[\d,.]*(?:\s*(?:Cr|Mn|Lakh|cr|crore|lakh|mn|%|bps|₹|Rs|USD)\b)?)",
                          summary, re.IGNORECASE):
        nums_in_summary.add(m.group(1).strip())

    # Check each against the source text.
    unverified: list[str] = []
    for n in nums_in_summary:
        # Skip percentages and small integers (likely section numbers).
        if n.endswith("%") or n.isdigit() and int(n) < 10:
            continue
        clean_n = re.sub(r"[₹Rs,\s]", "", n.lower())
        clean_n = re.sub(r"(cr|crore|mn|million|lakh|bps|usd)", "", clean_n)
        if clean_n not in source_text.replace(",", "").replace(" ", "").lower():
            unverified.append(n)

    if unverified:
        note = ("\n\n---\n⚠ Some figures in this summary could not be verified "
                "against the report text: " + ", ".join(unverified[:8]) + ". "
                "Cross-check with the original report before acting on these.")
        return summary + note

    return summary


# ------------------------------------------------------------------ summarisation

def summarize_report(text: str, company: str, _depth: int = 0) -> str | None:
    """Map-reduce summary of an arbitrarily large report. None if the LLM fully fails."""
    text = (text or "").strip()
    if not text:
        return None

    if len(text) <= SUMMARY_CHUNK_CHARS:  # small enough for a single pass
        raw = llm_complete(CHUNK_PROMPT.format(company=company, text=text),
                           max_tokens=700, label="summary")
        return _clean_summary(raw, text) if raw else None

    chunks = map_chunks(text)
    logger.info(f"summarising in {len(chunks)} chunk(s) [depth={_depth}]")

    if len(chunks) == 1 or SUMMARY_CONCURRENCY == 1:
        partials = []
        for i, ch in enumerate(chunks, 1):
            s = llm_complete(CHUNK_PROMPT.format(company=company, text=ch),
                             max_tokens=500, label=f"map[{i}]")
            if s:
                partials.append(s)
            time.sleep(0.3)
    else:
        partials: list[str] = []
        workers = min(SUMMARY_CONCURRENCY, len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_i = {
                ex.submit(llm_complete, CHUNK_PROMPT.format(company=company, text=ch),
                          500, f"map[{i}]"): i
                for i, ch in enumerate(chunks, 1)
            }
            for fut in as_completed(fut_to_i):
                s = fut.result()
                if s:
                    partials.append(s)

    if not partials:
        return None

    combined = "\n\n".join(f"- {p}" for p in partials)

    if len(combined) > SUMMARY_CHUNK_CHARS and _depth < MAX_REDUCE_DEPTH:
        return summarize_report(combined, company, _depth + 1)

    raw = llm_complete(REDUCE_PROMPT.format(company=company, text=combined),
                       max_tokens=1200, label="reduce") or combined
    return _clean_summary(raw, text)


# ------------------------------------------------------------------ fact extraction

def _num(v):
    """Coerce an LLM's idea of a number into a float, or None.

    Returns None for "NOT FOUND" sentinel values — the LLM is now instructed to
    output "NOT FOUND" (string) rather than null (JSON null) so we can distinguish
    "I searched and didn't find it" from "I forgot to output this field."
    """
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("not found", "--", "n/a", "na", ""):
        return None
    try:
        return float(s.replace(",", "").replace("₹", "").replace("Rs", "").strip())
    except ValueError:
        return None


def extract_facts(text: str, company: str) -> dict:
    """Broker, recommendation, CMP and target price — only if stated in the report.

    Reads the first 6000 chars: brokers put all four on the cover page.  After
    extraction, validates each field against the source text to catch hallucinated
    broker names and phantom target prices.
    """
    blank = {"broker": None, "recommendation": None,
             "current_price": None, "target_price": None}
    if not (text or "").strip():
        return blank

    data = llm_json(FACTS_PROMPT.format(company=company, text=text[:6000]),
                    max_tokens=120, label="facts")
    if not isinstance(data, dict):
        return blank

    rec = (data.get("recommendation") or "").strip().title() or None
    raw = {
        "broker": (data.get("broker") or "").strip() or None,
        "recommendation": rec if rec in ("Buy", "Hold", "Sell") else None,
        "current_price": _num(data.get("current_price")),
        "target_price": _num(data.get("target_price")),
    }
    # Validate against the cover-page text before returning.
    return _validate_facts(raw, text[:6000])
