"""Conversation memory. A rolling window of turns, read by the contextualiser.

Stored in Postgres rather than Redis. The window is small and read once per
query, so the persistence layer is not the bottleneck -- and keeping turns
durable means you can go back and audit which chunks a bad answer came from.

Abstentions are written to history too, but excluded from the window fed to the
contextualiser: "I don't have enough information" is not context, and letting it
into the transcript teaches the rewriter to keep rephrasing a dead question.
"""
import json

from config import MEMORY_TURNS
from db import execute, query


def create_session(report_id: int, title: str = "New chat") -> int:
    return execute("INSERT INTO sessions (report_id, title) VALUES (%s,%s) RETURNING id",
                   (report_id, title))


def list_sessions(report_id: int) -> list[dict]:
    return query(
        """SELECT s.id, s.title, s.created_at,
                  (SELECT count(*) FROM turns t WHERE t.session_id = s.id) AS n_turns
             FROM sessions s WHERE s.report_id = %s ORDER BY s.created_at DESC""",
        (report_id,))


def delete_session(session_id: int) -> None:
    execute("DELETE FROM sessions WHERE id=%s", (session_id,))


def get_turns(session_id: int) -> list[dict]:
    """Full transcript, for rendering the chat panel."""
    return query(
        """SELECT id, role, content, standalone_query, sub_questions, citations,
                  scores, abstained, created_at
             FROM turns WHERE session_id=%s ORDER BY created_at, id""",
        (session_id,))


def get_history(session_id: int) -> list[dict]:
    """The window handed to contextualize(). Grounded turns only, oldest first."""
    rows = query(
        """SELECT role, content FROM turns
            WHERE session_id=%s AND abstained = FALSE
         ORDER BY created_at DESC, id DESC LIMIT %s""",
        (session_id, MEMORY_TURNS))
    return list(reversed(rows))


def add_turn(session_id: int, role: str, content: str, *, standalone=None,
             sub_questions=None, citations=None, scores=None, abstained=False) -> int:
    return execute(
        """INSERT INTO turns (session_id, role, content, standalone_query,
                              sub_questions, citations, scores, abstained)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (session_id, role, content, standalone,
         json.dumps(sub_questions) if sub_questions else None,
         json.dumps(citations) if citations else None,
         json.dumps(scores) if scores else None,
         abstained))


def autotitle(session_id: int, first_query: str) -> None:
    """Name the session after its opening question. Cheap, no LLM call."""
    title = (first_query[:60] + "…") if len(first_query) > 60 else first_query
    execute("UPDATE sessions SET title=%s WHERE id=%s AND title='New chat'",
            (title, session_id))
