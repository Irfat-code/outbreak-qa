"""Find the chunks most similar to a question (cosine similarity, in memory)."""
import json
from functools import lru_cache

import numpy as np

from src.config import EMBED_MODEL, INDEX_DIR, TOP_K


@lru_cache(maxsize=1)
def _load_index():
    vectors = np.load(INDEX_DIR / "embeddings.npy")
    with open(INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return vectors, chunks


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def search(question: str, k: int = TOP_K, vectors=None, chunks=None, query_vec=None) -> list[dict]:
    """Return the top-k chunks with a 'score' field. Arguments after k are for tests."""
    if vectors is None or chunks is None:
        vectors, chunks = _load_index()
    if query_vec is None:
        query_vec = _model().encode([question], normalize_embeddings=True)[0]
    scores = vectors @ query_vec                 # vectors are normalised -> cosine
    top = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]
