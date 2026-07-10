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
LLM_CYCLES = 2  # full passes through the provider list before giving up

# Available models exposed in the UI dropdown. Each entry is (model_id, label).
# The first model is the default. Models must be pulled in Ollama before use.
AVAILABLE_MODELS = [
    ("llama3.1:8b", "Llama 3.1 8B"),
    ("gemma4:26b", "Gemma 4 26B"),
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ---------------------------------------------------------------- embeddings
# Separate endpoint so embeddings can run locally while the LLM is remote.
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", LLM_BASE_URL)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
EMBED_DIM = 768  # nomic-embed-text output dim. Must match the pgvector column.
EMBED_BATCH = 64        # GPU can handle much larger batches than CPU
EMBED_CONCURRENCY = 2  # parallel batch requests — keeps the GPU fed while one
                        # batch is in flight. Increase if you have a big GPU
                        # (A100-class) and Ollama is tuned for it.
# nomic-embed-text is trained with task prefixes. Retrieval quality drops
# measurably without them. Applied at embed time only; never stored.
EMBED_DOC_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "

# ---------------------------------------------------------------- chunking
# Two different chunkers for two different jobs. Do not merge them.
SUMMARY_CHUNK_CHARS = 12000   # map-reduce summarisation: big chunks, few LLM calls
MAX_MAP_CHUNKS = 12
MAX_REDUCE_DEPTH = 2

RETRIEVAL_CHUNK_CHARS = 1200 #1800  # ~450 tokens. Dense chunks for vector search.
RETRIEVAL_CHUNK_OVERLAP = 250 #250

# ---------------------------------------------------------------- retrieval
MAX_SUB_QUESTIONS = 3     # cap on complex-query decomposition (deep-search only)
N_REPHRASINGS_NORMAL = 1  # search variants in normal mode (faster, fewer LLM calls)
N_REPHRASINGS_DEEP = 3    # search variants in deep-search mode (thorough)
TOP_K_PER_VARIANT = 20    # vector hits fetched per rephrasing
TOP_K_TEXT = 20            # full-text hits fetched per sub-question
TOP_N_AFTER_FUSION = 10    # chunks handed to the generator per sub-question

RRF_K = 60                # standard reciprocal-rank-fusion constant

# ABSTENTION GATE. Must run on raw cosine similarity, never on RRF scores --
# RRF is rank-based and its top result scores well even on garbage retrieval.
SIM_GATE = 0.4           # calibrate on your own reports; nomic-specific

# ---------------------------------------------------------------- deep search
# LLM-as-judge thresholds. Only applied when deep_search=true on a query.
FAITHFULNESS_MIN = 0.5    # is the answer entailed by the retrieved chunks
RELEVANCY_MIN = 0.3       # do the chunks actually address the question
JUDGE_MAX_TOKENS = 200

ABSTAIN_MSG = "I don't have enough information in this report to answer that."

# ---------------------------------------------------------------- memory
MEMORY_TURNS = 3         # turns of history fed to the contextualiser
