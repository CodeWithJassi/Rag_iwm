"""Deep search: grade the answer before showing it, and withhold it if it fails.

These are reference-free metrics -- they need no ground truth, only the answer
and the passages it was built from. That is what makes them usable in production
on reports nobody has annotated.

  faithfulness : is every claim in the answer entailed by the retrieved context?
  relevancy    : do the retrieved passages actually address the question?

Faithfulness is the hallucination check. Relevancy catches the failure where the
model answers fluently from passages that are about the right company and the
wrong topic.

Cost: one extra LLM call per query. That is why it is a toggle, not a default.
"""
import logging

from config import FAITHFULNESS_MIN, JUDGE_MAX_TOKENS, RELEVANCY_MIN
from llm import llm_json
from prompts import JUDGE_PROMPT

logger = logging.getLogger(__name__)


def _clamp(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def validate(question: str, context: str, answer_text: str, model: str = "") -> dict:
    """Score an answer. Returns {faithfulness, relevancy, unsupported[], passed}.

    A failed judge call scores 0.0 and therefore fails the thresholds. That is
    deliberate: in deep search mode, an unverifiable answer is not an answer.
    """
    data = llm_json(
        JUDGE_PROMPT.format(query=question, context=context, answer=answer_text),
        max_tokens=JUDGE_MAX_TOKENS, label="judge", model=model)

    if not isinstance(data, dict):
        logger.warning("judge returned no parseable score; failing closed")
        data = {}

    faith = _clamp(data.get("faithfulness"))
    rel = _clamp(data.get("relevancy"))
    unsupported = data.get("unsupported") or []
    if not isinstance(unsupported, list):
        unsupported = []

    passed = faith >= FAITHFULNESS_MIN and rel >= RELEVANCY_MIN
    logger.info(f"judge: faithfulness={faith:.2f} relevancy={rel:.2f} passed={passed}")

    return {
        "faithfulness": round(faith, 3),
        "relevancy": round(rel, 3),
        "unsupported": [str(u) for u in unsupported][:5],
        "passed": passed,
        "thresholds": {"faithfulness": FAITHFULNESS_MIN, "relevancy": RELEVANCY_MIN},
    }
