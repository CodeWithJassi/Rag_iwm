"""Every tunable knob lives here. Nothing else in the codebase hardcodes a constant."""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
STATIC_DIR = ROOT / "static"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- database
PG_DSN = os.getenv("PG_DSN", "postgresql://iwm:iwm@localhost:5432/iwm_rag")

# ---------------------------------------------------------------- llm chain
# Failover order is fixed: Ollama first (free, local), paid APIs after.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_CYCLES = 1  # full passes through the provider list before giving up

# Available models exposed in the UI dropdown. Each entry is (model_id, label).
# The first model is the default. Models must be pulled in Ollama before use.
AVAILABLE_MODELS = [
    ("llama3.1:8b", "Llama 3.1 8B"),
    ("gemma4:26b", "Gemma 4 26B"),
    ("MichelRosselli/GLM-4.5-Air:Q4_K_M", "GLM-4.5 Q4"),
    ("MichelRosselli/GLM-4.5-Air:IQ1_M", "GLM-4.5 Q1")
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ---------------------------------------------------------------- embeddings
# Failover chain: EMBED_BASE_URL is tried first (defaults to LLM_BASE_URL, which
# may point to a cluster).  EMBED_LOCAL_URL is the fallback — a local Ollama
# instance that runs when the cluster is unreachable.
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", LLM_BASE_URL)
EMBED_LOCAL_URL = os.getenv("EMBED_LOCAL_URL", "http://localhost:11434")

# Embedding model registry — every supported model declares its dimension and
# task prefixes.  Pick one via the EMBED_MODEL env var.  Unknown models raise
# an error at import time; add new entries here before using a new model.
#
# nomic-embed-text REQUIRES its task prefixes — retrieval quality drops
# measurably without them.  mxbai and qwen3 were trained without mandatory
# prefixes and work correctly with empty strings.
_EMBED_REGISTRY: dict[str, tuple[int, str, str]] = {
    "nomic-embed-text:latest":  (768,  "search_document: ", "search_query: "),
    "mxbai-embed-large:latest": (1024, "",                  ""),
    "qwen3-embedding:4b":       (2560, "",                  ""),
}

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
if EMBED_MODEL not in _EMBED_REGISTRY:
    raise ValueError(
        f"Unknown EMBED_MODEL '{EMBED_MODEL}'. "
        f"Known models: {list(_EMBED_REGISTRY.keys())}. "
        f"Add the new model to _EMBED_REGISTRY in config.py with its dimension "
        f"and task prefixes before using it."
    )
EMBED_DIM, EMBED_DOC_PREFIX, EMBED_QUERY_PREFIX = _EMBED_REGISTRY[EMBED_MODEL]

EMBED_BATCH = int(os.getenv("EMBED_BATCH", "128"))
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "8"))
# Parallel map-phase summarisation chunks sent to the LLM at once.
SUMMARY_CONCURRENCY = int(os.getenv("SUMMARY_CONCURRENCY", "6"))

# ---------------------------------------------------------------- chunking
# Two different chunkers for two different jobs. Do not merge them.
SUMMARY_CHUNK_CHARS = 12000   # map-reduce summarisation: big chunks, few LLM calls
MAX_MAP_CHUNKS = 12
MAX_REDUCE_DEPTH = 2

RETRIEVAL_CHUNK_CHARS = 2100  # max chars per retrieval chunk (all strategies)
RETRIEVAL_CHUNK_OVERLAP = 250 # overlap for sliding-window strategy only

# Retrieval chunking strategy. Pick one:
#   "char"     — fixed-size sliding window + overlap (fast, no model needed)
#   "heading"  — splits at markdown heading boundaries; long sections fall back
#                to char-splitting. Keeps thematic sections together.
#   "semantic" — uses the LLM to detect topic shifts and split at natural
#                context boundaries. Best coherence, slowest ingestion.
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "structure")

# Semantic chunking: max chars per LLM call when finding split points.
# Larger = fewer LLM calls but coarser splits.
SEMANTIC_CHUNK_BLOCK = 4000

# Structure-aware chunking (strategy "structure"): split at bold Label: lead-ins
# found in earnings-call sections.  Each labeled block is its own chunk; only
# merge when below the floor (~30 tokens) and split when above the ceiling
# (~400 tokens).  Char counts are proxy for token counts.
STRUCTURE_FLOOR_CHARS = 120     # merge smaller chunks into previous neighbor
STRUCTURE_CEILING_CHARS = 1600  # split larger chunks with char-level fallback

# ---------------------------------------------------------------- retrieval
MAX_SUB_QUESTIONS = 3     # cap on complex-query decomposition (deep-search only)
N_REPHRASINGS_NORMAL = 1  # search variants in normal mode (faster, fewer LLM calls)
N_REPHRASINGS_DEEP = 3    # search variants in deep-search mode (thorough)
TOP_K_PER_VARIANT = 20    # vector hits fetched per rephrasing
TOP_K_TEXT = 20            # full-text hits fetched per sub-question
TOP_N_AFTER_FUSION = 35    # chunks after RRF fusion — pool fed to the reranker

RRF_K = 60                # standard reciprocal-rank-fusion constant

# Cross-encoder reranker. Sits between fusion and generation: the reranker reads
# the query and every passage *together* (not as independent vectors), so its
# relevance scores are much sharper than cosine or RRF.  bge-reranker-v2-m3 is
# the standard choice.
#
# Two modes, controlled by RERANK_BASE_URL:
#   LOCAL (default) — when RERANK_BASE_URL is empty or starts with "local://".
#        Loads the model in-process via FlagEmbedding.  No extra container needed.
#        First query pays a ~5 s model-load cost; subsequent queries are fast.
#   REMOTE — set RERANK_BASE_URL to a TEI container endpoint.
#        docker run -d -p 8080:80 --gpus all \
#            ghcr.io/huggingface/text-embeddings-inference:latest \
#            --model-id BAAI/bge-reranker-v2-m3
RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "")
RERANK_LOCAL_MODEL = os.getenv("RERANK_LOCAL_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K = 6          # chunks handed to the generator after reranking

# ABSTENTION GATE. Must run on raw cosine similarity, never on RRF scores --
# RRF is rank-based and its top result scores well even on garbage retrieval.
SIM_GATE = 0.40          # calibrate on your own reports; nomic-specific

# ---------------------------------------------------------------- deep search
# LLM-as-judge thresholds. Only applied when deep_search=true on a query.
FAITHFULNESS_MIN = 0.7    # is the answer entailed by the retrieved chunks
RELEVANCY_MIN = 0.6       # do the chunks actually address the question
JUDGE_MAX_TOKENS = 200

ABSTAIN_MSG = "I don't have enough information in this report to answer that."

# ---------------------------------------------------------------- memory
MEMORY_TURNS = 3         # turns of history fed to the contextualiser
