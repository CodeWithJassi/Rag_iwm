"""Cross-encoder reranker. One job: given a query and a list of candidate
passages, return the top_k most relevant passages according to a model that
reads query+passage *together* — not as independently-embedded vectors.

Two modes, dispatched on config.RERANK_BASE_URL:

  LOCAL  (default) — loads bge-reranker-v2-m3 in-process via FlagEmbedding.
          No extra container.  First call pays the model-load cost (~5 s);
          subsequent calls are fast (~100 ms for 30 passages).

  REMOTE — calls a TEI container over HTTP.  Set RERANK_BASE_URL to the
           container's /rerank endpoint.

When neither mode is available (no FlagEmbedding installed and no TEI
container), degrades to simple top-k RRF truncation — better than failing
the query.
"""
import logging

import requests

from config import RERANK_BASE_URL, RERANK_LOCAL_MODEL, RERANK_TOP_K

logger = logging.getLogger(__name__)

_local_reranker = None  # FlagReranker instance, loaded lazily


def _rerank_local(query: str, chunks: list[dict], top_k: int,
                  model_name: str) -> list[dict]:
    """Rerank with a local FlagEmbedding model loaded in-process."""
    global _local_reranker

    if _local_reranker is None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed — run: pip install FlagEmbedding\n"
                "Falling back to RRF-order truncation.")
            for c in chunks[:top_k]:
                c.setdefault("rerank_score", None)
            return chunks[:top_k]

        logger.info("loading local reranker model '%s' (this is one-time) ...",
                    model_name)
        _local_reranker = FlagReranker(model_name, use_fp16=True)
        logger.info("local reranker ready")

    pairs = [[query, c["content"]] for c in chunks]
    scores = _local_reranker.compute_score(pairs, normalize=True)

    # compute_score returns a float for a single pair, list for multiple.
    if not isinstance(scores, list):
        scores = [scores]

    for i, c in enumerate(chunks):
        c["rerank_score"] = round(float(scores[i]), 4)

    ranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0.0),
                    reverse=True)
    kept = ranked[:top_k]

    logger.info("local-rerank: %d candidates -> top %d (best score %.3f)",
                len(chunks), len(kept),
                kept[0]["rerank_score"] if kept else 0.0)
    return kept


def _rerank_remote(query: str, chunks: list[dict], top_k: int,
                   base_url: str) -> list[dict]:
    """Rerank via a TEI container's /rerank endpoint."""
    texts = [c["content"] for c in chunks]
    try:
        resp = requests.post(
            f"{base_url}/rerank",
            json={"query": query, "texts": texts, "truncate": True},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        scores = resp.json()  # TEI returns [{"index": 0, "score": 0.95}, ...]
    except Exception as e:
        logger.warning("remote reranker call failed (%s) — "
                       "falling back to RRF order", e)
        for c in chunks[:top_k]:
            c.setdefault("rerank_score", None)
        return chunks[:top_k]

    indexed = {item["index"]: item["score"] for item in scores}
    for i, c in enumerate(chunks):
        c["rerank_score"] = round(indexed.get(i, 0.0), 4)

    ranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0.0),
                    reverse=True)
    kept = ranked[:top_k]

    logger.info("remote-rerank: %d candidates -> top %d (best score %.3f)",
                len(chunks), len(kept),
                kept[0]["rerank_score"] if kept else 0.0)
    return kept


def preload() -> None:
    """Eagerly load the local reranker model so the first query doesn't pay a
    ~5 s cold-start cost.  Safe to call at import time or during startup."""
    if RERANK_BASE_URL and not RERANK_BASE_URL.startswith("local://"):
        return  # remote TEI — nothing to load locally
    global _local_reranker
    if _local_reranker is not None:
        return  # already loaded
    try:
        from FlagEmbedding import FlagReranker
        logger.info("preloading reranker model '%s' ...", RERANK_LOCAL_MODEL)
        _local_reranker = FlagReranker(RERANK_LOCAL_MODEL, use_fp16=True)
        logger.info("reranker model ready")
    except ImportError:
        logger.info("FlagEmbedding not installed — reranker will load lazily")
    except Exception as e:
        logger.warning("reranker preload failed (%s) — will retry on first query", e)


def rerank(query: str, chunks: list[dict],
           top_k: int = RERANK_TOP_K) -> list[dict]:
    """Re-score *chunks* against *query* with a cross-encoder and keep the top_k.

    Each chunk keeps its original ``similarity`` field (raw cosine from vector
    search) so the abstention gate is unaffected.  The reranker's score is stored
    in ``rerank_score``.

    Dispatches to local (FlagEmbedding) or remote (TEI container) based on
    RERANK_BASE_URL.  Degrades gracefully to RRF truncation when neither is
    available.
    """
    if not chunks:
        return []

    # Nothing to rerank if we already have fewer chunks than the target.
    if len(chunks) <= top_k:
        for c in chunks:
            c.setdefault("rerank_score", None)
        return chunks

    if RERANK_BASE_URL and not RERANK_BASE_URL.startswith("local://"):
        return _rerank_remote(query, chunks, top_k, RERANK_BASE_URL)
    else:
        return _rerank_local(query, chunks, top_k, RERANK_LOCAL_MODEL)
