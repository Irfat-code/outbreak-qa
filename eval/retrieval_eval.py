"""Retrieval-only evaluation: no model calls, no API quota used.

For each answerable question, checks whether a passage that actually contains
the expected facts (the plain keyword groups in questions.csv) is among the
top-k retrieved passages, and at what rank. Compares embedding, keyword (BM25)
and hybrid retrieval at several k values in one go.

If the answer-bearing passage isn't retrieved, the model can't answer, so this
is the number to improve before spending API calls.

Usage:  python -m eval.retrieval_eval [--ks 5,8,10] [--label NAME]
Writes eval/retrieval_results.md
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eval.run_eval import groups, kw_match
from src.config import INDEX_DIR

HERE = Path(__file__).resolve().parent


def passage_keywords(row: dict) -> list[str]:
    """Keyword groups that should appear in the source passage. Regex groups
    (e.g. 'the answer must contain a date' or 'no/not') describe the answer's
    wording, not the passage, so they're left out."""
    return [g for g in groups(row["expected_keywords"]) if not g.strip().startswith("re:")]


def first_hit_rank(order, chunks: list[dict], kws: list[str], max_k: int):
    for rank, i in enumerate(order[:max_k], 1):
        if all(kw_match(g, chunks[i]["text"]) for g in kws):
            return rank
    return None


def rankings(question: str, vectors, bm25_state, embed) -> dict:
    from src.backends import bm25_score
    from src.retrieve import rrf_merge
    dense = vectors @ embed(question)
    bm25 = bm25_score(bm25_state, question)
    n = len(dense)
    return {
        "embeddings": list(np.argsort(-dense)),
        "bm25": list(np.argsort(-bm25)),
        "hybrid": rrf_merge([dense, bm25], k=n),
    }


def evaluate(rows, chunks, vectors, bm25_state, embed, ks):
    max_k = max(ks)
    per_q = []
    for row in rows:
        kws = passage_keywords(row)
        if row["should_refuse"] == "yes" or not kws:
            continue
        ranks = {m: first_hit_rank(order, chunks, kws, max_k)
                 for m, order in rankings(row["question"], vectors, bm25_state, embed).items()}
        per_q.append({"qid": row["qid"], "question": row["question"], "keywords": ";".join(kws), **ranks})
    return per_q


def report(per_q: list[dict], ks: list[int], settings: dict | None = None, label: str | None = None) -> str:
    methods = ["embeddings", "bm25", "hybrid"]
    n = len(per_q)
    s = settings or {}
    lines = ["# Retrieval evaluation (no model calls)", "",
             (f"Label: `{label}`  |  " if label else "")
             + f"Embedding model: `{s.get('embed_model', 'unknown')}`  |  "
             f"chunk headers: {s.get('chunk_headers', 'unknown')}  |  "
             f"chunk size: {s.get('chunk_words', '?')} words", "",
             f"Answerable questions with checkable keywords: {n}", "",
             "Share of questions where a passage containing the expected facts is in the top k:", "",
             "| Method | " + " | ".join(f"top {k}" for k in ks) + " |",
             "|---|" + "---|" * len(ks)]
    for m in methods:
        cells = [f"{sum(1 for q in per_q if q[m] and q[m] <= k)}/{n}" for k in ks]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Rank of first answer-bearing passage (blank = not in top "
              f"{max(ks)})", "", "| qid | embeddings | bm25 | hybrid | keywords |", "|---|---|---|---|---|"]
    for q in per_q:
        lines.append(f"| {q['qid']} | {q['embeddings'] or ''} | {q['bm25'] or ''} | {q['hybrid'] or ''} | {q['keywords']} |")
    lines += ["", "_Keyword matching is approximate: a passage can contain the words without "
              "answering the question. Spot-check misses with scripts/find_text.py._"]
    return "\n".join(lines) + "\n"


def main() -> None:
    from src.backends import bm25_build, dense_embed, search_text
    from src.retrieve import check_index_settings
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="5,8,10")
    ap.add_argument("--label", default=None, help="also save under eval/runs/retrieval-<label>.md")
    args = ap.parse_args()
    ks = sorted(int(k) for k in args.ks.split(","))

    with open(HERE / "questions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    vectors = np.load(INDEX_DIR / "embeddings.npy")
    bm25_state = bm25_build([search_text(c) for c in chunks])
    check_index_settings()
    settings_path = INDEX_DIR / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}

    per_q = evaluate(rows, chunks, vectors, bm25_state,
                     embed=lambda q: dense_embed("fastembed", [q])[0], ks=ks)
    text = report(per_q, ks, settings, args.label)
    (HERE / "retrieval_results.md").write_text(text, encoding="utf-8")
    if args.label:
        runs = HERE / "runs"
        runs.mkdir(exist_ok=True)
        (runs / f"retrieval-{args.label}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
