"""Find the chunks most relevant to a question, using the backend in config."""
import json
from functools import lru_cache

import numpy as np

from src.config import INDEX_DIR, RETRIEVER, TOP_K


@lru_cache(maxsize=1)
def _load_index():
    built_with = (INDEX_DIR / "retriever.txt").read_text().strip() \
        if (INDEX_DIR / "retriever.txt").exists() else "st"
    if built_with != RETRIEVER:
        raise SystemExit(f"Index was built with '{built_with}' but RETRIEVER='{RETRIEVER}'. "
                         "Re-run: python -m src.ingest")
    with open(INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    if RETRIEVER == "bm25":
        from src.backends import bm25_build
        state = bm25_build([c["text"] for c in chunks])
    else:
        state = np.load(INDEX_DIR / "embeddings.npy")
    return state, chunks


def _scores(state, question: str) -> np.ndarray:
    if RETRIEVER == "bm25":
        from src.backends import bm25_score
        return bm25_score(state, question)
    from src.backends import dense_embed
    return state @ dense_embed(RETRIEVER, [question])[0]


def search(question: str, k: int = TOP_K, vectors=None, chunks=None, query_vec=None) -> list[dict]:
    """Return the top-k chunks with a 'score' field. Arguments after k are for tests."""
    if vectors is not None and chunks is not None:
        scores = vectors @ query_vec
    else:
        state, chunks = _load_index()
        scores = _scores(state, question)
    top = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]
