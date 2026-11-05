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
LLM_MAX_RETRIES = 5
LLM_MIN_SECONDS_BETWEEN_CALLS = 4.0                     # stay under free-tier per-minute caps

# --- Embeddings (run locally, no API key needed) ---------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking / retrieval --------------------------------------------------
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
TOP_K = 5
# If the best chunk scores below this, refuse without calling the LLM.
# Tune it on Day 6 using your eval results.
MIN_RETRIEVAL_SCORE = 0.30

REFUSAL_TEXT = "Not in these documents."
