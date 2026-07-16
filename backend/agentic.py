"""Agentic RAG mode — planner-driven multi-step loop with tools.

Replaces the decompose → rephrase → search → sub-answer pipeline with a
ReAct-style loop: the LLM plans, calls tools, reflects on results, and
iterates until it has enough information to answer.

Yields SSE event strings via an async generator so the frontend can render
each step as it happens.  Uses ``asyncio.to_thread()`` to run sync LLM and
DB calls from the event loop without blocking.
"""
import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from config import ABSTAIN_MSG, AGENTIC_MAX_STEPS
from judge import validate
from llm import llm_complete, llm_json
from prompts import (AGENT_STEP_PROMPT, AGENT_SYNTHESIZE_PROMPT,
                     CONTEXTUALIZE_PROMPT, HISTORY_RELEVANCE_PROMPT,
                     SUB_ANSWER_PROMPT)
from retrieval import _CITE, _clean_answer, _is_empty, _render, contextualize

import tools

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ SSE helpers

def _sse(event_type: str, **kwargs) -> str:
    """Format a Server-Sent Event data line.  The caller yields these."""
    payload = {"type": event_type, **kwargs}
    return f"data: {json.dumps(payload, default=str)}\n\n"


# ------------------------------------------------------------------ history gate

async def _history_relevant(history: list[dict], query: str,
                            model: str = "") -> bool:
    """Check whether the chat history is about the same topic as the new query.

    Fails OPEN (returns True) on LLM error — better to include irrelevant
    context than to drop a genuine follow-up reference.
    """
    if not history:
        return False
    transcript = "\n".join(
        f"{t['role']}: {t['content']}" for t in history[-6:]
    )
    out = await asyncio.to_thread(
        llm_complete,
        HISTORY_RELEVANCE_PROMPT.format(history=transcript, query=query),
        10, "history_relevance", model,
    )
    if not out:
        return True  # fail open
    return out.strip().lower().startswith("yes")


# ------------------------------------------------------------------ tool descriptions

def _tool_descriptions() -> str:
    """Return the tool descriptions block injected into the planner prompt."""
    return """- retrieve(query: str, top_k: int = 8) → list[dict]
    Search the broker report for passages matching the query using hybrid
    vector + full-text retrieval.  Returns up to top_k chunks (max 12),
    each with: id, page_no, section, chunk_type, content, similarity.
    Use this for ALL factual questions — it is your only source of company
    data.  Try different query phrasings if the first attempt returns
    irrelevant results.

- calculator(expression: str) → str
    Evaluate a mathematical expression safely.  Supports + - * / ** % and
    parentheses.  Example: calculator("(5248 - 4800) / 4800 * 100").
    Use for growth rates, margins, ratios, and any arithmetic on numbers
    retrieved from the report.

- today() → str
    Returns the current date in YYYY-MM-DD format.  Use to assess recency
    — how old is the report relative to today.

- date_diff(date1: str, date2: str) → int
    Returns the absolute number of calendar days between two dates.
    Both dates must be in YYYY-MM-DD format.
    Example: date_diff("2025-03-31", "2026-07-14") returns the days between
    the report date and today.

- list_sections() → list[str]
    Returns all unique section headings found in the report, in order of
    appearance.  Use this early to understand what topics the report covers
    before asking detailed questions — like scanning a table of contents.

- get_report_date() → str
    Returns the publication date of the report in YYYY-MM-DD format.
    Returns "Report date not available" if the date was not extracted.

- lookup_previous_reasoning(turn_id: int) → list[dict]
    Returns the step-by-step reasoning traces from a previous answer in
    this conversation.  Each trace shows what tool was called, its inputs
    and outputs, and the planner's reasoning.  Use this when the user asks
    about how a previous answer was arrived at."""


# ------------------------------------------------------------------ tool dispatch

def _execute_tool(name: str, params: dict, report_id: int,
                  ) -> tuple[Any, str | None]:
    """Dispatch a tool call.  Returns (result, error_string)."""
    try:
        if name == "retrieve":
            query_text = str(params.get("query", ""))
            top_k = int(params.get("top_k", 6))
            return tools.retrieve(report_id, query_text, top_k=top_k), None

        elif name == "calculator":
            expr = str(params.get("expression", ""))
            return tools.calculator(expr), None

        elif name == "today":
            return tools.today(), None

        elif name == "date_diff":
            d1 = str(params.get("date1", ""))
            d2 = str(params.get("date2", ""))
            return str(tools.date_diff(d1, d2)), None

        elif name == "list_sections":
            return tools.list_sections(report_id), None

        elif name == "get_report_date":
            result = tools.get_report_date(report_id)
            return result or "Report date not available", None

        elif name == "lookup_previous_reasoning":
            turn_id = int(params.get("turn_id", 0))
            return tools.lookup_previous_reasoning(turn_id), None

        else:
            return None, f"Unknown tool: {name}"

    except Exception as e:
        logger.warning("tool '%s' failed: %s", name, e)
        return None, str(e)


# ------------------------------------------------------------------ output formatting

def _summarize_tool_output(name: str, output: Any) -> str:
    """Create a concise summary of tool output for the planner prompt.
    Keeps the prompt under token limits while preserving key information."""
    if name == "retrieve":
        if not output:
            return "No matching passages found in the report."
        chunks = output if isinstance(output, list) else []
        lines = []
        for c in chunks[:5]:
            lines.append(
                f"[chunk_id={c['id']}] p{c['page_no']} "
                f"(cos={c['similarity']:.3f}) "
                f"{c['content'][:300].replace(chr(10), ' ')}"
            )
        if len(chunks) > 5:
            lines.append(f"... and {len(chunks) - 5} more passages")
        return "\n".join(lines)

    elif name == "list_sections":
        sections = output if isinstance(output, list) else []
        return ", ".join(sections[:20]) if sections else "No sections found"

    elif name == "lookup_previous_reasoning":
        traces = output if isinstance(output, list) else []
        if not traces:
            return "No previous reasoning traces found"
        lines = []
        for t in traces[:10]:
            lines.append(f"Step {t['step']}: {t['tool']} → {str(t.get('output', ''))[:120]}")
        return "\n".join(lines)

    return str(output)[:500]


def _format_tool_detail(name: str, output: Any) -> str:
    """Full output detail for the SSE event shown to the user."""
    if name == "retrieve" and isinstance(output, list):
        lines = []
        for c in output:
            lines.append(
                f"[chunk {c.get('id', '?')}] p{c['page_no']} "
                f"cos={c.get('similarity', '?')} "
                f"| {c['content'][:120].replace(chr(10), ' ')}"
            )
        return "\n".join(lines)
    if isinstance(output, (dict, list)):
        return json.dumps(output, indent=2, default=str)[:2000]
    return str(output)[:2000]


def _format_steps(steps: list[dict]) -> str:
    """Format the accumulated steps log for inclusion in the planner prompt."""
    if not steps:
        return "(no steps taken yet)"
    lines = []
    for s in steps[-12:]:  # keep only recent steps to avoid token overflow
        lines.append(
            f"Step {s['step']}: Called {s['tool']}({json.dumps(s.get('params', {}))})\n"
            f"  Thought: {s.get('thought', '')}\n"
            f"  Result: {str(s.get('result', ''))[:300]}"
        )
    return "\n\n".join(lines)


# ------------------------------------------------------------------ final result

def _build_final_result(
    standalone_query: str,
    answer_text: str,
    all_chunks: dict[int, dict],
    steps_log: list[dict],
    t0: float,
) -> dict:
    """Assemble the final result dict matching the existing answer() interface
    so the endpoint and frontend can consume it identically."""
    # Clean the answer — remove hedging INSUFFICIENT preambles, etc.
    cleaned = _clean_answer(answer_text or "")
    if not cleaned.strip() or _is_empty(cleaned):
        return {
            "answer": ABSTAIN_MSG,
            "abstained": True,
            "reason": "generator_abstained",
            "standalone": standalone_query,
            "sub_questions": [standalone_query],
            "citations": [],
            "context": "",
            "chunks": [],
            "trace": _build_trace(steps_log, t0),
            "reasoning_steps": steps_log,
        }

    # Assign global passage numbers and resolve citation markers.
    chunk_list = list(all_chunks.values())
    numbering = {c["id"]: i + 1 for i, c in enumerate(chunk_list)}
    by_num = {v: k for k, v in numbering.items()}

    cited_nums = {int(n) for n in _CITE.findall(cleaned)}
    cited_ids = {by_num[n] for n in cited_nums if n in by_num}

    citations = [
        {
            "n": numbering[c["id"]],
            "chunk_id": c["id"],
            "page_no": c["page_no"],
            "section": c.get("section"),
            "chunk_type": c.get("chunk_type", "text"),
            "similarity": round(float(c.get("similarity", 0)), 3),
            "snippet": c["content"][:280],
        }
        for c in chunk_list if c["id"] in cited_ids
    ]

    # Build context for the judge.
    context = "\n\n".join(
        f"[{numbering[c['id']]}] (page {c['page_no']}"
        f"{', ' + c['section'] if c.get('section') else ''})\n{c['content']}"
        for c in chunk_list
    )

    return {
        "answer": cleaned.strip(),
        "abstained": False,
        "reason": None,
        "standalone": standalone_query,
        "sub_questions": [standalone_query],
        "citations": sorted(citations, key=lambda c: c["n"]),
        "context": context,
        "chunks": chunk_list,
        "trace": _build_trace(steps_log, t0),
        "reasoning_steps": steps_log,
    }


def _build_trace(steps: list[dict], t0: float) -> list[dict]:
    """Build a trace list compatible with the existing verbose UI."""
    trace = []
    for s in steps:
        trace.append({
            "label": f"agent:{s.get('tool', '?')}",
            "summary": str(s.get("result", ""))[:120],
            "detail": json.dumps(s.get("params", {}))[:500],
            "ms": round((time.time() - t0) * 1000),
        })
    return trace


# ================================================================== main loop

async def answer(
    report: dict,
    user_query: str,
    history: list[dict],
    *,
    model: str = "",
    verbose: bool = False,
    cancel_event: asyncio.Event | None = None,
) -> AsyncIterator[str]:
    """Run the agentic ReAct loop.  Yields SSE-formatted event strings.

    The FastAPI endpoint wraps this in a ``StreamingResponse`` with
    ``media_type="text/event-stream"``.  Each ``yield`` sends one event
    to the browser.
    """
    company: str = report["company"]
    report_id: int = report["id"]
    t0 = time.time()

    # ---- Step 0: history relevance gate ----
    history_relevant = await _history_relevant(history, user_query, model=model)
    yield _sse("check", label="history_relevance",
               summary="history relevant" if history_relevant else "history not relevant — skipping",
               detail="", ms=round((time.time() - t0) * 1000))

    if not history_relevant:
        history = []

    # ---- Step 1: contextualize ----
    standalone_query = await asyncio.to_thread(
        contextualize, user_query, history, model,
    )
    yield _sse("check", label="contextualize",
               summary=f"\"{user_query[:100]}\" → \"{standalone_query[:100]}\"",
               detail=standalone_query[:300] if verbose else "",
               ms=round((time.time() - t0) * 1000))

    # ---- Step 1.5: initial retrieval (always runs) ----
    # Before the planner loop, we always retrieve with the standalone
    # query.  This guarantees we have chunks to synthesize from even if
    # the planner LLM fails to produce valid JSON on every step.  Normal
    # mode does the same — it always retrieves before generating.
    steps_log: list[dict] = []
    all_chunks: dict[int, dict] = {}

    yield _sse("plan", step=0, thought="Initial search with the standalone query",
               tool="retrieve", params={"query": standalone_query},
               ms=round((time.time() - t0) * 1000))

    init_chunks, init_err = await asyncio.to_thread(
        _execute_tool, "retrieve", {"query": standalone_query}, report_id,
    )
    if init_err:
        yield _sse("tool_result", step=0, tool="retrieve",
                   status="error", error=init_err,
                   ms=round((time.time() - t0) * 1000))
    else:
        init_summary = _summarize_tool_output("retrieve", init_chunks)
        init_detail = _format_tool_detail("retrieve", init_chunks) if verbose else ""
        yield _sse("tool_result", step=0, tool="retrieve",
                   status="ok", summary=init_summary, detail=init_detail,
                   ms=round((time.time() - t0) * 1000))
        steps_log.append({
            "step": 0, "tool": "retrieve",
            "params": {"query": standalone_query},
            "result": init_summary,
            "thought": "Initial search with the standalone query",
        })
        if isinstance(init_chunks, list):
            for c in init_chunks:
                if c["id"] not in all_chunks:
                    all_chunks[c["id"]] = dict(c)

    # ---- Step 2..N: planner loop ----
    tool_descriptions = _tool_descriptions()

    for step_num in range(1, AGENTIC_MAX_STEPS + 1):
        # Check cancellation between steps.
        if cancel_event and cancel_event.is_set():
            yield _sse("error", step=step_num,
                       message="Cancelled by user")
            return

        # Build the planner prompt with accumulated context.
        steps_text = _format_steps(steps_log)

        history_context = ""
        if history:
            transcript = "\n".join(
                f"{t['role']}: {t['content'][:200]}"
                for t in history[-4:]
            )
            history_context = f"RELEVANT CONVERSATION HISTORY:\n{transcript}\n"

        prompt = AGENT_STEP_PROMPT.format(
            company=company,
            query=standalone_query,
            tool_descriptions=tool_descriptions,
            steps=steps_text,
            history_context=history_context,
        )

        # Ask the LLM what to do next.
        result = await asyncio.to_thread(
            llm_json, prompt, 800, f"agent_step_{step_num}", model,
        )

        if not result or not isinstance(result, dict):
            yield _sse("error", step=step_num,
                       message="Planner returned invalid or empty JSON — "
                               "may retry with fallback synthesis")
            break

        thought = str(result.get("thought", ""))
        action = str(result.get("action", ""))

        if action == "call":
            tool_name = str(result.get("tool", ""))
            params = result.get("params", {}) or {}

            # Tell the frontend what the planner decided.
            yield _sse("plan", step=step_num, thought=thought,
                       tool=tool_name, params=params,
                       ms=round((time.time() - t0) * 1000))

            # Execute the tool (blocking call offloaded to thread).
            tool_result, error = await asyncio.to_thread(
                _execute_tool, tool_name, params, report_id,
            )

            if error:
                yield _sse("tool_result", step=step_num, tool=tool_name,
                           status="error", error=error,
                           ms=round((time.time() - t0) * 1000))
                steps_log.append({
                    "step": step_num, "tool": tool_name,
                    "params": params, "result": f"[Error] {error}",
                    "thought": thought,
                })
                continue

            # Summarize for the prompt, send full detail to the UI.
            summary = _summarize_tool_output(tool_name, tool_result)
            detail = _format_tool_detail(tool_name, tool_result) if verbose else ""

            yield _sse("tool_result", step=step_num, tool=tool_name,
                       status="ok", summary=summary, detail=detail,
                       ms=round((time.time() - t0) * 1000))

            steps_log.append({
                "step": step_num, "tool": tool_name,
                "params": params, "result": summary,
                "thought": thought,
            })

            # Accumulate retrieved chunks for citation resolution.
            if tool_name == "retrieve" and isinstance(tool_result, list):
                for c in tool_result:
                    if c["id"] not in all_chunks:
                        all_chunks[c["id"]] = dict(c)

        elif action == "final":
            yield _sse("finalizing", step=step_num, thought=thought,
                       ms=round((time.time() - t0) * 1000))

            # ---- Generate the answer from FULL chunk context ----
            # The LLM's in-loop answer is based on truncated 150-char
            # summaries.  Instead, we re-render every accumulated chunk
            # with global numbering and feed them into the same
            # SUB_ANSWER_PROMPT that normal/deep mode uses.  This way
            # the generator sees the full report text, not just snippets.
            chunk_list = list(all_chunks.values())
            numbering = {c["id"]: i + 1 for i, c in enumerate(chunk_list)}
            context_block = _render(chunk_list, numbering) if chunk_list else ""

            if context_block:
                answer_text = await asyncio.to_thread(
                    llm_complete,
                    SUB_ANSWER_PROMPT.format(
                        company=company, query=standalone_query,
                        context=context_block,
                    ),
                    600, "agent_answer", model,
                )
                if not answer_text:
                    answer_text = str(result.get("answer", ""))
            else:
                answer_text = str(result.get("answer", ""))

            # Build the final result.
            final_result = _build_final_result(
                standalone_query, answer_text or "", all_chunks,
                steps_log, t0,
            )

            # Run the judge (same as deep-search mode).
            scores = None
            if not final_result["abstained"] and chunk_list:
                max_judge = 12
                judge_context = _render(chunk_list[:max_judge], numbering)
                scores = await asyncio.to_thread(
                    validate, standalone_query, judge_context,
                    final_result["answer"], model,
                )

            final_result["scores"] = scores

            yield _sse("final",
                       answer=final_result["answer"],
                       abstained=final_result["abstained"],
                       citations=final_result["citations"],
                       scores=scores,
                       trace=final_result["trace"],
                       reasoning_steps=steps_log,
                       ms=round((time.time() - t0) * 1000))
            return

        else:
            yield _sse("error", step=step_num,
                       message=f"Unknown action '{action}' — planner must "
                               "respond with 'call' or 'final'")
            break

    # ---- Fallback: max steps reached without a final answer ----
    yield _sse("finalizing", step=AGENTIC_MAX_STEPS,
               thought="Maximum steps reached. Synthesising from "
                       "accumulated research.",
               ms=round((time.time() - t0) * 1000))

    chunk_list = list(all_chunks.values())
    numbering = {c["id"]: i + 1 for i, c in enumerate(chunk_list)}
    context_block = _render(chunk_list, numbering) if chunk_list else ""

    if context_block:
        # Prefer full chunk context over step-note summaries.
        fallback_answer = await asyncio.to_thread(
            llm_complete,
            SUB_ANSWER_PROMPT.format(
                company=company, query=standalone_query,
                context=context_block,
            ),
            600, "agent_synthesize", model,
        )
    else:
        notes = "\n".join(
            f"Step {s['step']} [{s.get('tool', '?')}]: {str(s.get('result', ''))[:500]}"
            for s in steps_log if s.get("result")
        )
        fallback_prompt = AGENT_SYNTHESIZE_PROMPT.format(
            company=company, notes=notes or "(no data gathered)",
            query=standalone_query,
        )
        fallback_answer = await asyncio.to_thread(
            llm_complete, fallback_prompt, 600, "agent_synthesize", model,
        )

    final_result = _build_final_result(
        standalone_query, fallback_answer or ABSTAIN_MSG,
        all_chunks, steps_log, t0,
    )

    scores = None
    if not final_result["abstained"] and all_chunks:
        chunk_list = list(all_chunks.values())
        numbering = {c["id"]: i + 1 for i, c in enumerate(chunk_list)}
        judge_context = _render(chunk_list[:12], numbering)
        scores = await asyncio.to_thread(
            validate, standalone_query, judge_context,
            final_result["answer"], model,
        )
    final_result["scores"] = scores

    yield _sse("final",
               answer=final_result["answer"],
               abstained=final_result["abstained"],
               citations=final_result["citations"],
               scores=scores,
               trace=final_result["trace"],
               reasoning_steps=steps_log,
               ms=round((time.time() - t0) * 1000))
