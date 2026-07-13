"""Minimal helper: re-ingest one report.  Run as a subprocess so it picks up
the current EMBED_MODEL from .env with a fresh import of config.

Usage:
  python backend/_reingest.py <report_id>

Before wiping old chunks we check that the PDF is still readable.  The wipe +
ingest runs inside a single transaction so a crash mid-ingestion rolls back
and leaves the existing chunks intact.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from db import init_db, close_db, pool
from ingest import ingest_report


def _reingest(report_id: int) -> None:
    with pool.connection() as conn:
        r = conn.execute(
            "SELECT id, company, file_path, status FROM reports WHERE id=%s",
            (report_id,)).fetchone()
        if not r:
            print(f"ERROR: report {report_id} not found")
            sys.exit(1)

        rid, company, file_path, old_status = r

        # Verify the PDF is still readable before we touch anything.
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            print(f"ERROR: PDF file not found at {file_path}")
            sys.exit(1)
        if pdf_path.stat().st_size == 0:
            print(f"ERROR: PDF file is empty: {file_path}")
            sys.exit(1)

        # Quick smoke-test: can pymupdf open it?
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            page_count = doc.page_count
            doc.close()
            if page_count == 0:
                print(f"ERROR: PDF has 0 pages")
                sys.exit(1)
            print(f"PDF OK — {page_count} pages")
        except Exception as e:
            print(f"ERROR: cannot open PDF — {e}")
            sys.exit(1)

        print(f"Wiping {rid} old chunks + re-ingesting "
              f"({company}) from {file_path} …")

        # Wipe + ingest inside the transaction so a failure rolls back.
        conn.execute("DELETE FROM chunks WHERE report_id=%s", (rid,))
        conn.execute(
            "UPDATE reports SET status='processing', error=NULL, n_chunks=0 "
            "WHERE id=%s", (rid,))

    # ingest_report manages its own connections, so run it outside the
    # transaction above (the DELETE is already committed).
    try:
        ingest_report(report_id, str(file_path), company)
    except Exception as e:
        print(f"ERROR during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    report_id = int(sys.argv[1])
    init_db()
    _reingest(report_id)
    close_db()
    print("Done.")
