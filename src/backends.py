"""Retrieval backends. Pick one with RETRIEVER in config / .env.

  "fastembed" - dense embeddings via ONNX Runtime (no PyTorch). Same MiniLM model.
  "bm25"      - keyword ranking in pure Python/NumPy. No model, no extra DLLs.
  "st"        - sentence-transformers (needs PyTorch).

All backends expose build(texts) -> state  and  score(state, query) -> np.ndarray.
"""
import math
import re
from collections import Counter

import numpy as np

from src.config import BM25_B, BM25_K1, EMBED_MODEL

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def search_text(chunk: dict) -> str:
    """What retrieval searches over: optional document header + the passage."""
    header = chunk.get("header") or ""
    return f"{header}\n{chunk['text']}" if header else chunk["text"]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# --- BM25 -------------------------------------------------------------------
def bm25_build(texts: list[str]) -> dict:
    docs = [tokenize(t) for t in texts]
    df = Counter(tok for d in docs for tok in set(d))
    n = len(docs)
    return {
        "tf": [Counter(d) for d in docs],
        "len": np.array([len(d) for d in docs], dtype="float32"),
        "idf": {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()},
    }


def bm25_score(state: dict, query: str) -> np.ndarray:
    avg = float(state["len"].mean()) or 1.0
    scores = np.zeros(len(state["tf"]), dtype="float32")
    for tok in set(tokenize(query)):
        idf = state["idf"].get(tok)
        if idf is None:
            continue
        for i, tf in enumerate(state["tf"]):
            f = tf.get(tok)
            if f:
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * state["len"][i] / avg)
                scores[i] += idf * f * (BM25_K1 + 1) / denom
    return scores


# --- Dense (fastembed / sentence-transformers) -----------------------------
_models = {}


def _encoder(kind: str):
    if kind not in _models:
        if kind == "fastembed":
            from fastembed import TextEmbedding
            m = TextEmbedding(EMBED_MODEL)
            _models[kind] = lambda texts: np.array(list(m.embed(texts)), dtype="float32")
        else:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(EMBED_MODEL)
            _models[kind] = lambda texts: m.encode(texts, batch_size=32).astype("float32")
    return _models[kind]


def dense_embed(kind: str, texts: list[str]) -> np.ndarray:
    v = _encoder(kind)(texts)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
