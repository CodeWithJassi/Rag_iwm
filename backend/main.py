"""FastAPI app. Serves the dashboard and the six endpoints behind it."""
import logging
import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import judge
import memory
import retrieval
from config import ABSTAIN_MSG, AVAILABLE_MODELS, LLM_MODEL, STATIC_DIR, UPLOAD_DIR
from db import close_db, execute, init_db, query
from ingest import ingest_report

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

REPORT_COLS = ("id, company, broker, file_name, uploaded_at, summary, recommendation, "
               "current_price, target_price, n_chunks, status, error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield
    close_db()


app = FastAPI(title="IWM Research — Report Desk", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str
    deep_search: bool = False
    model: str = ""  # empty -> use LLM_MODEL default


def _get_report(report_id: int) -> dict:
    r = query(f"SELECT {REPORT_COLS} FROM reports WHERE id=%s", (report_id,), one=True)
    if not r:
        raise HTTPException(404, "Report not found")
    return r


# ------------------------------------------------------------------ reports

@app.get("/api/reports")
def list_reports():
    return query(f"SELECT {REPORT_COLS} FROM reports ORDER BY uploaded_at DESC")


@app.get("/api/reports/{report_id}")
def get_report(report_id: int):
    return _get_report(report_id)


@app.post("/api/reports", status_code=202)
def upload_report(bg: BackgroundTasks, company: str = Form(...), file: UploadFile = File(...)):
    """Save the PDF, register the row, return immediately. Ingestion runs behind it.

    Embedding a 40-page report takes tens of seconds on CPU; the browser should
    not wait for it. The UI polls `status` until it flips to 'ready'.
    """
    company = company.strip()
    if not company:
        raise HTTPException(400, "Company name is required")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    path = UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
    with path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    report_id = execute(
        "INSERT INTO reports (company, file_name, file_path) VALUES (%s,%s,%s) RETURNING id",
        (company, file.filename, str(path)))

    bg.add_task(ingest_report, report_id, str(path), company)
    return {"id": report_id, "status": "pending"}


@app.delete("/api/reports/{report_id}", status_code=204)
def delete_report(report_id: int):
    r = _get_report(report_id)
    execute("DELETE FROM reports WHERE id=%s", (report_id,))  # cascades to chunks/sessions
    from pathlib import Path
    Path(r["file_path"]).unlink(missing_ok=True)


@app.get("/api/reports/{report_id}/pdf")
def get_pdf(report_id: int):
    r = query("SELECT file_path, file_name FROM reports WHERE id=%s", (report_id,), one=True)
    if not r:
        raise HTTPException(404, "Report not found")
    return FileResponse(r["file_path"], media_type="application/pdf", filename=r["file_name"])


# ------------------------------------------------------------------ sessions

@app.get("/api/reports/{report_id}/sessions")
def get_sessions(report_id: int):
    _get_report(report_id)
    return memory.list_sessions(report_id)


@app.post("/api/reports/{report_id}/sessions", status_code=201)
def new_session(report_id: int):
    _get_report(report_id)
    return {"id": memory.create_session(report_id)}


@app.delete("/api/sessions/{session_id}", status_code=204)
def drop_session(session_id: int):
    memory.delete_session(session_id)


@app.get("/api/sessions/{session_id}/turns")
def session_turns(session_id: int):
    return memory.get_turns(session_id)


# ------------------------------------------------------------------ models

@app.get("/api/models")
def list_models():
    """Return the models the user can choose from in the UI."""
    return [{"id": mid, "label": label, "default": mid == LLM_MODEL}
            for mid, label in AVAILABLE_MODELS]


# ------------------------------------------------------------------ chat

@app.post("/api/sessions/{session_id}/chat")
def chat(session_id: int, req: ChatRequest):
    sess = query("SELECT report_id FROM sessions WHERE id=%s", (session_id,), one=True)
    if not sess:
        raise HTTPException(404, "Session not found")
    report = _get_report(sess["report_id"])
    if report["status"] != "ready":
        raise HTTPException(409, f"Report is {report['status']} — not ready to query")

    user_query = req.query.strip()
    if not user_query:
        raise HTTPException(400, "Query is required")

    history = memory.get_history(session_id)
    memory.add_turn(session_id, "user", user_query)
    memory.autotitle(session_id, user_query)

    result = retrieval.answer(report, user_query, history,
                              deep_search=req.deep_search, model=req.model)

    # Deep search: grade the answer, and withhold it if it fails. An answer that
    # cannot be verified against its own sources is worse than no answer here --
    # this desk acts on these numbers.
    scores = None
    if req.deep_search and not result["abstained"]:
        scores = judge.validate(result["standalone"], result["context"],
                                result["answer"], model=req.model)
        if not scores["passed"]:
            result.update(answer=ABSTAIN_MSG, abstained=True,
                          reason="judge_failed", citations=[])

    memory.add_turn(session_id, "assistant", result["answer"],
                    standalone=result["standalone"],
                    sub_questions=result["sub_questions"],
                    citations=result["citations"],
                    scores=scores, abstained=result["abstained"])

    return {"answer": result["answer"], "abstained": result["abstained"],
            "reason": result["reason"], "standalone": result["standalone"],
            "sub_questions": result["sub_questions"], "citations": result["citations"],
            "scores": scores}


@app.get("/api/chunks/{chunk_id}")
def get_chunk(chunk_id: int):
    """Full chunk text behind a citation pill."""
    c = query("SELECT id, page_no, section, content FROM chunks WHERE id=%s",
              (chunk_id,), one=True)
    if not c:
        raise HTTPException(404, "Chunk not found")
    return c


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
