"""Postgres + pgvector. One database holds documents, chunks, vectors and sessions.

No Redis: session memory lives in `turns`. One fewer resident process on a box
that has run out of RAM before, and a research desk's query volume does not need
sub-millisecond memory reads.
"""
import logging

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from config import EMBED_DIM, PG_DSN

logger = logging.getLogger(__name__)

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS reports (
    id              BIGSERIAL PRIMARY KEY,
    company         TEXT NOT NULL,
    broker          TEXT,
    file_name       TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    report_date     DATE,
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    summary         TEXT,
    recommendation  TEXT,                      -- Buy | Hold | Sell | NULL
    current_price   NUMERIC,
    target_price    NUMERIC,
    n_chunks        INT NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|ready|failed
    error           TEXT
);

-- The tree: company -> broker -> report -> chunks. Modelled as a foreign key,
-- not a graph. Reach for Neo4j only when a query needs edges between arbitrary
-- nodes; parent-child does not.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    report_id   BIGINT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    ord         INT NOT NULL,               -- position in the document
    page_no     INT,
    section     TEXT,                       -- nearest markdown heading, if any
    chunk_type  TEXT NOT NULL DEFAULT 'text',  -- text | table | image (stub)
    content     TEXT NOT NULL,
    embedding   vector({EMBED_DIM})
);

CREATE TABLE IF NOT EXISTS sessions (
    id          BIGSERIAL PRIMARY KEY,
    report_id   BIGINT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT 'New chat',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS turns (
    id                BIGSERIAL PRIMARY KEY,
    session_id        BIGINT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role              TEXT NOT NULL,        -- user | assistant
    content           TEXT NOT NULL,
    standalone_query  TEXT,                 -- what the retriever actually searched
    sub_questions     JSONB,
    citations         JSONB,                -- [{{chunk_id, page_no, snippet}}]
    scores            JSONB,                -- judge output when deep_search=true
    abstained         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_report_idx  ON chunks (report_id);
CREATE INDEX IF NOT EXISTS turns_session_idx  ON turns (session_id, created_at);
CREATE INDEX IF NOT EXISTS sessions_report_idx ON sessions (report_id, created_at DESC);

-- HNSW over cosine distance. Built once the table has rows; cheap to create empty.
CREATE INDEX IF NOT EXISTS chunks_embed_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

# Migrations: changes that can't live inside CREATE TABLE IF NOT EXISTS because
# the table already exists in deployments that predate the change.
MIGRATIONS = [
    # Hybrid retrieval — full-text search alongside vector. Catches exact
    # terms (tickers, broker names, "EBITDA margin") that dense retrieval
    # sometimes misses. Zero new infra.
    """ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fts tsvector
       GENERATED ALWAYS AS (to_tsvector('english', content)) STORED""",
    "CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING gin (fts)",
]

pool = ConnectionPool(PG_DSN, min_size=1, max_size=8, open=False,
                      configure=register_vector)


def init_db() -> None:
    # Create the pgvector extension FIRST on a raw connection — the pool's
    # configure=register_vector callback needs the vector type to exist before
    # it can register it on each pooled connection.
    # If the app user lacks superuser privileges the extension must already
    # exist (created by the DBA / setup step).
    import psycopg
    try:
        with psycopg.connect(PG_DSN, autocommit=True) as raw:
            raw.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg.errors.InsufficientPrivilege:
        logger.info("cannot create vector extension — assuming it already exists")
    pool.open()
    with pool.connection() as conn:
        conn.execute(SCHEMA)
        for m in MIGRATIONS:
            conn.execute(m)
    logger.info("schema ready")


def close_db() -> None:
    pool.close()


def query(sql: str, params: tuple = (), *, one: bool = False):
    """SELECT helper. Returns list[dict], or a single dict when one=True."""
    with pool.connection() as conn:
        cur = conn.execute(sql, params)
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return (rows[0] if rows else None) if one else rows


def execute(sql: str, params: tuple = ()):
    """INSERT/UPDATE/DELETE helper. Returns the first column of RETURNING, if any."""
    with pool.connection() as conn:
        cur = conn.execute(sql, params)
        if cur.description:
            row = cur.fetchone()
            return row[0] if row else None
    return None
