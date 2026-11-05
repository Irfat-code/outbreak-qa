"""Project settings. Change things here, not inside the code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
RAW_DIR = ROOT / "data" / "raw"          # downloaded docs (git-ignored)
INDEX_DIR = ROOT / "data" / "index"      # chunks + embeddings (git-ignored)
SOURCES_CSV = ROOT / "data" / "sources.csv"

# --- LLM -------------------------------------------------------------------
# Pin the exact model name you see in Google AI Studio (a stable Flash model,
# not a "preview"). Write the same name in the README.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")      # "gemini" or "anthropic"
LLM_MODEL = os.getenv("LLM_MODEL", "PIN-ME-gemini-flash-model-name")
LLM_TEMPERATURE = 0.0                                   # deterministic-ish for eval
LLM_MAX_RETRIES = 8                                     # ~5 min of patience for 503 busy spikes
# Gap between calls. Raise it if you keep hitting per-minute (TPM/RPM) limits.
LLM_MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("LLM_MIN_SECONDS_BETWEEN_CALLS", "4"))
# For reasoning models (e.g. gpt-oss): "low" keeps answers fast and cheap on tokens.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")

# --- Retrieval backend ------------------------------------------------------
# "fastembed" (dense, no PyTorch), "bm25" (keyword, pure Python),
# "hybrid" (fastembed + bm25 combined) or "st" (PyTorch).
RETRIEVER = os.getenv("RETRIEVER", "fastembed")
# How many candidates each method contributes before hybrid fusion.
HYBRID_CANDIDATES = 50
# MiniLM reads only the first 256 tokens of a chunk; BAAI/bge-small-en-v1.5 reads 512.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Prefix each chunk with its document title/publisher for SEARCH only (not shown to the model
# as passage text). Helps questions that name the source, e.g. "NCDC's guideline".
CHUNK_HEADERS = os.getenv("CHUNK_HEADERS", "1") == "1"
BM25_K1 = 1.5
BM25_B = 0.75

# --- Chunking / retrieval --------------------------------------------------
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
TOP_K = int(os.getenv("TOP_K", "5"))
# If the best chunk scores below this, refuse without calling the LLM.
# Scales differ: cosine (dense) is 0-1, BM25 is unbounded. Tune on Day 6.
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE",
                                      "4.0" if RETRIEVER == "bm25" else "0.30"))
# For "hybrid", the threshold is applied to the embedding (cosine) score.

REFUSAL_TEXT = "Not in these documents."
