"""Postgres + pgvector. One database holds documents, chunks, vectors and sessions.

No Redis: session memory lives in `turns`. One fewer resident process on a box
that has run out of RAM before, and a research desk's query volume does not need
sub-millisecond memory reads.
"""
import logging

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from config import EMBED_DIM, EMBED_MODEL, PG_DSN

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
    #
    # NOTE: the fts column starts as GENERATED for fresh installs (see SCHEMA)
    # but is converted to a regular column by _migrate_fts_to_regular() below
    # so tags and enrichment text can be included in the search index.
    """ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fts tsvector
       GENERATED ALWAYS AS (to_tsvector('english', content)) STORED""",
    "CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING gin (fts)",
    # JSONB metadata — tags, enrichment data, image paths, and future extensions.
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata JSONB",
    # Agentic RAG reasoning traces.  One row per planner step, linked to the
    # turn that produced it.  CASCADE delete so removing a chat cleans up.
    """CREATE TABLE IF NOT EXISTS reasoning_traces (
        id          BIGSERIAL PRIMARY KEY,
        turn_id     BIGINT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
        step        INT NOT NULL,
        tool        TEXT,
        input_data  JSONB,
        output_data JSONB,
        plan_text   TEXT,
        reflection  TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
    "CREATE INDEX IF NOT EXISTS traces_turn_idx ON reasoning_traces (turn_id, step)",
]

# Post-SCHEMA migrations that need procedural logic (conditionals, loops).
# These run AFTER the SCHEMA and MIGRATIONS SQL has executed.
_POST_MIGRATIONS = [
    "_migrate_fts_to_regular",
]

pool = ConnectionPool(PG_DSN, min_size=1, max_size=8, open=False,
                      configure=register_vector)


def _check_embedding_dimension() -> None:
    """Verify chunks.embedding column matches EMBED_DIM. Auto-migrate on mismatch.

    Embedding models produce different dimension vectors.  If the pgvector column
    is the wrong width, INSERT will fail with a cryptic Postgres error.  This
    check catches the mismatch at startup and auto-migrates — dropping the old
    vectors (they are useless in the new model's space anyway) and resizing the
    column.  The caller MUST re-ingest every report afterwards.
    """
    with pool.connection() as conn:
        exists = conn.execute(
            "SELECT EXISTS ("
            "  SELECT FROM information_schema.tables WHERE table_name = 'chunks'"
            ")"
        ).fetchone()[0]
        if not exists:
            return  # table not created yet — SCHEMA will use the right dim

        result = conn.execute(
            "SELECT format_type(atttypid, atttypmod) "
            "FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass "
            "  AND attname = 'embedding' AND attnum > 0"
        ).fetchone()

        if not result:
            return  # column doesn't exist yet — shouldn't happen

        col_type: str = result[0]  # e.g. "vector(768)"
        if not col_type.startswith("vector("):
            logger.warning("unexpected embedding column type '%s' — "
                           "skipping dimension check", col_type)
            return

        actual_dim = int(col_type[7:-1])
        if actual_dim == EMBED_DIM:
            return  # all good

        logger.warning(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "EMBEDDING DIMENSION MISMATCH\n"
            "  DB column:      vector(%d)\n"
            "  Config expects: vector(%d)  (model '%s')\n"
            "\n"
            "Auto-migrating now.  Existing vectors will be dropped — they\n"
            "were produced by a different model and cannot be used for\n"
            "retrieval in the new embedding space.\n"
            "\n"
            "You MUST re-ingest every report.  Until then, queries will\n"
            "fail because chunks lack valid embeddings.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            actual_dim, EMBED_DIM, EMBED_MODEL)

        conn.execute("DROP INDEX IF EXISTS chunks_embed_idx")
        conn.execute("ALTER TABLE chunks DROP COLUMN embedding")
        conn.execute(f"ALTER TABLE chunks ADD COLUMN embedding vector({EMBED_DIM})")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embed_idx "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        logger.info("migrated chunks.embedding from vector(%d) to vector(%d). "
                    "All reports must be re-ingested.", actual_dim, EMBED_DIM)


def _migrate_fts_to_regular() -> None:
    """Convert chunks.fts from a GENERATED column to a regular column.

    The generated column can only index ``chunks.content``.  We need tags and
    enrichment text included — so we drop the generated column, re-add it as a
    regular tsvector, and repopulate from content.  Future INSERTs will compute
    fts in ingest.py with content + tags concatenated.

    Idempotent: checks pg_attribute.attgenerated; skips if already regular.
    """
    with pool.connection() as conn:
        is_generated = conn.execute(
            "SELECT attgenerated FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'fts' "
            "  AND attnum > 0"
        ).fetchone()
        if not is_generated or is_generated[0] != 's':
            return  # already regular, or column doesn't exist
        logger.info("converting chunks.fts from GENERATED to regular column")
        conn.execute("DROP INDEX IF EXISTS chunks_fts_idx")
        conn.execute("ALTER TABLE chunks DROP COLUMN fts")
        conn.execute("ALTER TABLE chunks ADD COLUMN fts tsvector")
        conn.execute("UPDATE chunks SET fts = to_tsvector('english', content)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING gin (fts)")
        logger.info("fts migration complete — tags can now be included in search")


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

    # Run procedural post-migrations.
    for fn_name in _POST_MIGRATIONS:
        fn = globals().get(fn_name)
        if fn:
            fn()

    # After the schema is confirmed to exist, verify the embedding column width
    # matches the configured model.  Auto-migrates with a warning on mismatch.
    _check_embedding_dimension()


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
