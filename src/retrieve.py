"""Find the chunks most relevant to a question, using the backend in config.

"hybrid" runs embedding search and BM25 keyword search separately, then merges
the two rankings with reciprocal rank fusion (RRF): a chunk scores
1/(60 + rank) in each list, summed. Embeddings catch paraphrases; BM25 catches
exact terms such as '246', 'INRB' or 'life-course' that embeddings blur.
"""
import json
from functools import lru_cache

import numpy as np

from src.config import HYBRID_CANDIDATES, INDEX_DIR, RETRIEVER, TOP_K

RRF_K = 60
# Which index each retriever can use. An index embedded with fastembed works for hybrid too.
COMPATIBLE = {"hybrid": {"hybrid", "fastembed"}, "fastembed": {"fastembed", "hybrid"}}


def check_index_settings() -> None:
    """Questions must be embedded with the same model the chunks were embedded with."""
    from src.config import EMBED_MODEL
    path = INDEX_DIR / "settings.json"
    built = json.loads(path.read_text()).get("embed_model") if path.exists() \
        else "sentence-transformers/all-MiniLM-L6-v2"
    if built != EMBED_MODEL:
        raise SystemExit(f"Index was embedded with '{built}' but EMBED_MODEL='{EMBED_MODEL}'. "
                         "Re-run: python -m src.ingest")


@lru_cache(maxsize=1)
def _load_index():
    built_with = (INDEX_DIR / "retriever.txt").read_text().strip() \
        if (INDEX_DIR / "retriever.txt").exists() else "st"
    if built_with not in COMPATIBLE.get(RETRIEVER, {RETRIEVER}):
        raise SystemExit(f"Index was built with '{built_with}' but RETRIEVER='{RETRIEVER}'. "
                         "Re-run: python -m src.ingest")
    with open(INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    state = {}
    if RETRIEVER in ("bm25", "hybrid"):
        from src.backends import bm25_build, search_text
        state["bm25"] = bm25_build([search_text(c) for c in chunks])
    if RETRIEVER != "bm25":
        check_index_settings()
        state["vectors"] = np.load(INDEX_DIR / "embeddings.npy")
    return state, chunks


def _dense_scores(state, question: str) -> np.ndarray:
    from src.backends import dense_embed
    kind = "fastembed" if RETRIEVER == "hybrid" else RETRIEVER
    return state["vectors"] @ dense_embed(kind, [question])[0]


def rrf_merge(rankings: list[np.ndarray], k: int, n_candidates: int = HYBRID_CANDIDATES) -> list[int]:
    """Merge several score arrays into one ranking of chunk indices (highest first)."""
    fused: dict[int, float] = {}
    for scores in rankings:
        for rank, i in enumerate(np.argsort(-scores)[:n_candidates]):
            if scores[i] <= 0 and rank > 0:      # BM25 gives 0 to chunks sharing no words
                break
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)[:k]


def search(question: str, k: int = TOP_K, vectors=None, chunks=None, query_vec=None) -> list[dict]:
    """Return the top-k chunks. 'score' is the embedding (cosine) score where one exists,
    so the refusal threshold means the same thing for 'fastembed' and 'hybrid'.
    Arguments after k are for tests."""
    if vectors is not None and chunks is not None:
        scores = vectors @ query_vec
        top = np.argsort(-scores)[:k]
        return [{**chunks[i], "score": float(scores[i])} for i in top]

    state, chunks = _load_index()
    if RETRIEVER == "bm25":
        from src.backends import bm25_score
        scores = bm25_score(state["bm25"], question)
        top = np.argsort(-scores)[:k]
        return [{**chunks[i], "score": float(scores[i])} for i in top]

    dense = _dense_scores(state, question)
    if RETRIEVER == "hybrid":
        from src.backends import bm25_score
        top = rrf_merge([dense, bm25_score(state["bm25"], question)], k)
    else:
        top = np.argsort(-dense)[:k]
    return [{**chunks[i], "score": float(dense[i])} for i in top]
