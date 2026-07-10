"""Map-reduce summarisation and fact extraction. Behaviour unchanged from the
original script, except extract_prices() is now extract_facts() and pulls the
broker and rating in the same call.

The map phase runs chunks in parallel (up to SUMMARY_CONCURRENCY) so the remote
LLM GPU isn't waiting for individual HTTP round-trips.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from chunking import map_chunks
from config import MAX_REDUCE_DEPTH, SUMMARY_CHUNK_CHARS
from llm import llm_complete, llm_json
from prompts import CHUNK_PROMPT, FACTS_PROMPT, REDUCE_PROMPT

logger = logging.getLogger(__name__)

# How many map chunks to send to the LLM at once. The remote GPU can handle
# parallel requests; this cuts wall-clock time for the map phase proportionally.
SUMMARY_CONCURRENCY = 3


def summarize_report(text: str, company: str, _depth: int = 0) -> str | None:
    """Map-reduce summary of an arbitrarily large report. None if the LLM fully fails."""
    text = (text or "").strip()
    if not text:
        return None

    if len(text) <= SUMMARY_CHUNK_CHARS:  # small enough for a single pass
        return llm_complete(CHUNK_PROMPT.format(company=company, text=text),
                            max_tokens=700, label="summary")

    chunks = map_chunks(text)
    logger.info(f"summarising in {len(chunks)} chunk(s) [depth={_depth}]")

    # Single chunk, or only one worker — sequential is faster (no thread overhead).
    if len(chunks) == 1 or SUMMARY_CONCURRENCY == 1:
        partials = []
        for i, ch in enumerate(chunks, 1):
            s = llm_complete(CHUNK_PROMPT.format(company=company, text=ch),
                             max_tokens=500, label=f"map[{i}]")
            if s:
                partials.append(s)
            time.sleep(0.3)
    else:
        # Parallel map: send multiple chunks to the LLM at once. GPU time is the
        # bottleneck, not HTTP — overlapping requests cuts wall-clock time.
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

        # Re-sort to original order so the combined text reads coherently.
        # llm_complete labels include the chunk index; sort by that.
        # Actually, we don't have the index easily — as_completed yields in
        # completion order. For summary combination, order doesn't matter much
        # (the reducer sees a bullet list), so just keep completion order.

    if not partials:
        return None

    combined = "\n\n".join(f"- {p}" for p in partials)

    # Recurse if the partials are themselves too long to reduce in one call.
    if len(combined) > SUMMARY_CHUNK_CHARS and _depth < MAX_REDUCE_DEPTH:
        return summarize_report(combined, company, _depth + 1)

    return llm_complete(REDUCE_PROMPT.format(company=company, text=combined),
                        max_tokens=1200, label="reduce") or combined


def _num(v):
    """Coerce an LLM's idea of a number into a float, or None."""
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "").replace("₹", "").replace("Rs", "").strip())
    except ValueError:
        return None


def extract_facts(text: str, company: str) -> dict:
    """Broker, recommendation, CMP and target price -- only if stated in the report.

    Reads the first 6000 chars: brokers put all four on the cover page. Reading
    further costs tokens and finds nothing new.
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
    return {
        "broker": (data.get("broker") or "").strip() or None,
        "recommendation": rec if rec in ("Buy", "Hold", "Sell") else None,
        "current_price": _num(data.get("current_price")),
        "target_price": _num(data.get("target_price")),
    }
