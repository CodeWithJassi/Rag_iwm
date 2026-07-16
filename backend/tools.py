"""Tool implementations for the Agentic RAG mode.

Every tool is bounded to the report only — no web search, no external data.
Each tool is a standalone function whose signature is described in the planner
prompt.  Tools return plain Python types that get serialised to JSON for the
SSE stream and stored as JSONB in reasoning_traces.
"""
import ast
import logging
import operator
from datetime import date, datetime
from typing import Any

from config import TOP_N_AFTER_FUSION
from db import query
from embeddings import embed_query
from retrieval import rrf_fuse, text_search, vector_search
from rerank import rerank

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ retrieve

def retrieve(report_id: int, query_text: str, top_k: int = 8) -> list[dict]:
    """Search the report using hybrid vector + full-text retrieval.

    No rephrasing — the planner handles search strategy.  Returns chunks
    sorted by reranker score, each with:
      id, page_no, section, chunk_type, content, similarity, rerank_score
    """
    if not query_text.strip():
        return []

    # Wider initial sweep than normal mode — the planner makes fewer calls
    # so each call needs broader coverage.
    vec = embed_query(query_text)
    vec_results = vector_search(report_id, vec=vec, k=25)
    txt_results = text_search(report_id, query_text, k=25)

    if not vec_results and not txt_results:
        return []

    fused = rrf_fuse([vec_results, txt_results], top_n=TOP_N_AFTER_FUSION)
    # Return up to top_k chunks, capped at a reasonable max so the
    # planner prompt doesn't overflow.
    reranked = rerank(query_text, fused, top_k=min(top_k, 12))

    return [
        {
            "id": c["id"],
            "page_no": c["page_no"],
            "section": c.get("section"),
            "chunk_type": c.get("chunk_type", "text"),
            "content": c["content"],
            "similarity": round(float(c["similarity"]), 3),
            "rerank_score": c.get("rerank_score"),
        }
        for c in reranked
    ]


# ------------------------------------------------------------------ calculator

# Whitelist of AST node types and operators.  Everything else is rejected.
_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _eval_ast(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted AST.  Raises ValueError on anything
    outside the safe set — no imports, calls, attribute access, or builtins."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_ast(node.operand))
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_ast(node.left), _eval_ast(node.right))
    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Only arithmetic: + - * / ** % and parentheses.  No function calls,
    no variable names, no attribute access.  Returns the result rounded
    to 4 decimal places, or an error string.
    """
    if not expression or not expression.strip():
        return "Error: empty expression"
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_ast(tree.body)
        # Round intelligently — integers stay whole, floats get 4 places
        if result == int(result):
            return str(int(result))
        return f"{result:.4f}".rstrip("0").rstrip(".")
    except (SyntaxError, ValueError, ZeroDivisionError) as e:
        return f"Error: {e}"


# ------------------------------------------------------------------ date tools

def today() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return date.today().isoformat()


def date_diff(date1: str, date2: str) -> int:
    """Return the number of calendar days between two dates (absolute value).
    Both dates in YYYY-MM-DD format."""
    d1 = datetime.strptime(date1.strip(), "%Y-%m-%d").date()
    d2 = datetime.strptime(date2.strip(), "%Y-%m-%d").date()
    return abs((d2 - d1).days)


# ------------------------------------------------------------------ report metadata

def list_sections(report_id: int) -> list[str]:
    """Return all unique section headings in the report, in order of first
    appearance.  Acts as a table-of-contents scan for the planner."""
    rows = query(
        """SELECT DISTINCT section
             FROM chunks
            WHERE report_id = %s
              AND section IS NOT NULL
              AND section != ''
         ORDER BY MIN(ord)""",
        (report_id,),
    )
    return [r["section"] for r in rows]


def get_report_date(report_id: int) -> str | None:
    """Return the publication date of the report in YYYY-MM-DD, or None if
    it was never extracted."""
    r = query(
        "SELECT report_date FROM reports WHERE id = %s",
        (report_id,), one=True,
    )
    if r and r["report_date"]:
        return r["report_date"].isoformat()
    return None


# ------------------------------------------------------------------ previous reasoning

def lookup_previous_reasoning(turn_id: int) -> list[dict]:
    """Return the reasoning traces for a previous turn, ordered by step.
    Each entry shows what tool was called, its inputs and outputs, and the
    planner's reasoning at that step.  Used by the planner to understand how
    a prior answer was constructed."""
    traces = query(
        """SELECT step, tool, input_data, output_data, plan_text, reflection
             FROM reasoning_traces
            WHERE turn_id = %s
         ORDER BY step""",
        (turn_id,),
    )
    return [
        {
            "step": t["step"],
            "tool": t["tool"],
            "input": t["input_data"],
            "output": t["output_data"],
            "plan": t["plan_text"],
            "reflection": t["reflection"],
        }
        for t in traces
    ]
