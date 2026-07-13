"""nomic-embed-text via the Ollama instance you already run. No new infra.

Task prefixes are mandatory for this model -- it was trained with them, and
retrieval quality drops without them. Documents and queries get *different*
prefixes, which is the whole point: it pushes a passage and the question it
answers closer together in embedding space than a naive symmetric encoder would.

Parallel batch requests keep the GPU busy: while one batch is embedding, the
next one is already queued so the GPU doesn't idle between Python's HTTP
round-trips. The concurrency cap stops Ollama from swapping.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config import (EMBED_BASE_URL, EMBED_BATCH, EMBED_CONCURRENCY, EMBED_DIM,
                    EMBED_DOC_PREFIX, EMBED_LOCAL_URL, EMBED_MODEL,
                    EMBED_QUERY_PREFIX)

logger = logging.getLogger(__name__)


def _embed_raw(inputs: list[str], base_url: str = "",
               timeout: int = 60) -> list[list[float]]:
    """POST to an Ollama-compatible endpoint.  /api/embed is the batch endpoint;
    older builds only have /api/embeddings, which is one-at-a-time."""
    url = base_url or EMBED_BASE_URL
    try:
        r = requests.post(f"{url}/api/embed",
                          json={"model": EMBED_MODEL, "input": inputs},
                          timeout=timeout)
        r.raise_for_status()
        vecs = r.json()["embeddings"]
    except Exception as e:
        logger.warning(f"/api/embed at {url} failed ({e}); falling back to /api/embeddings")
        vecs = []
        for text in inputs:
            r = requests.post(f"{url}/api/embeddings",
                              json={"model": EMBED_MODEL, "prompt": text},
                              timeout=timeout)
            r.raise_for_status()
            vecs.append(r.json()["embedding"])

    for v in vecs:
        if len(v) != EMBED_DIM:
            raise ValueError(f"Expected {EMBED_DIM}-dim vectors, got {len(v)}. "
                             f"Check EMBED_MODEL and the pgvector column width.")
    return vecs


def _embed_endpoints() -> list[str]:
    """Deduplicated failover chain: primary first, then local fallback."""
    urls, seen = [], set()
    for u in [EMBED_BASE_URL, EMBED_LOCAL_URL]:
        u = u.rstrip("/")
        if u and u not in seen:
            urls.append(u)
            seen.add(u)
    return urls


def _embed_with_failover(inputs: list[str], timeout: int = 60) -> list[list[float]]:
    """Try each embedding endpoint in order. First success wins."""
    endpoints = _embed_endpoints()
    last_err = None
    for i, url in enumerate(endpoints):
        try:
            label = "primary" if i == 0 else "fallback"
            vecs = _embed_raw(inputs, base_url=url, timeout=timeout)
            logger.info(f"embedded {len(inputs)} texts via {label} ({url})")
            return vecs
        except Exception as e:
            last_err = e
            logger.warning(f"embedding endpoint {url} failed: {e}")
    raise last_err or Exception("All embedding endpoints failed")


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks for storage. Parallel batches to keep the GPU fed — while
    one batch is on the GPU the next is already in flight."""
    if not texts:
        return []

    batches = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        batches.append([EMBED_DOC_PREFIX + t for t in batch])

    # Batch ingestion may need more time — pass a longer timeout.
    batch_timeout = 120 if len(batches) > 1 or len(texts) > 16 else 60

    if len(batches) == 1:
        return _embed_with_failover(batches[0], timeout=batch_timeout)

    results: list[list[float]] = [b'' for _ in batches]  # type: ignore[assignment]
    workers = min(EMBED_CONCURRENCY, len(batches))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_idx = {ex.submit(_embed_with_failover, b, batch_timeout): idx
                      for idx, b in enumerate(batches)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            results[idx] = fut.result()
            logger.info(f"embedded batch {idx + 1}/{len(batches)} "
                        f"({len(results[idx])} vectors)")

    # Flatten in original order so chunk->vector zip stays correct.
    return [v for batch in results for v in batch]


def embed_query(text: str) -> list[float]:
    """Embed a single search query.  Uses the failover chain with a short timeout
    — a single embedding should never take more than 30 seconds."""
    return _embed_with_failover([EMBED_QUERY_PREFIX + text], timeout=30)[0]
