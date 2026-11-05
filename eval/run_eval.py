"""Run every question in eval/questions.csv and score the answers.

Automatic checks (a first pass - read the answers yourself too):
  refusal_correct  - refused when it should, answered when it should
  keywords_hit     - every expected keyword appears in the answer
  citation_correct - at least one cited chunk comes from an expected doc

Writes eval/results.csv (row per question) and eval/results.md (summary).
Usage:  python -m eval.run_eval [--limit N]
"""
import argparse
import csv
import datetime as dt
from pathlib import Path

from src.config import LLM_MODEL, LLM_PROVIDER, MIN_RETRIEVAL_SCORE, TOP_K

HERE = Path(__file__).resolve().parent


def score(row: dict, result: dict) -> dict:
    should_refuse = row["should_refuse"].strip().lower() == "yes"
    refusal_correct = result["refused"] == should_refuse

    keywords = [k for k in row["expected_keywords"].split("|") if k]
    keywords_hit = (not should_refuse and not result["refused"]
                    and all(k.lower() in result["answer"].lower() for k in keywords))

    expected_docs = {d for d in row["expected_doc_ids"].split("|") if d}
    cited_docs = {c["doc_id"] for c in result["citations"]}
    citation_correct = bool(expected_docs & cited_docs) if expected_docs else None
    retrieval_hit = bool(expected_docs & {c["doc_id"] for c in result["retrieved"]}) if expected_docs else None

    return {
        "qid": row["qid"], "type": row["type"], "question": row["question"],
        "answer": result["answer"].replace("\n", " "),
        "refused": result["refused"], "refusal_stage": result["refusal_stage"],
        "top_score": round(result["top_score"], 3),
        "cited_docs": "|".join(sorted(cited_docs)),
        "refusal_correct": refusal_correct,
        "keywords_hit": keywords_hit if not should_refuse else None,
        "citation_correct": citation_correct,
        "retrieval_hit": retrieval_hit,
        "manual_correct": "",           # fill in by hand: yes / no / partial
    }


def pct(values) -> str:
    vals = [v for v in values if v is not None]
    return f"{100 * sum(vals) / len(vals):.0f}% ({sum(vals)}/{len(vals)})" if vals else "n/a"


def main() -> None:
    from src.answer import answer

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(HERE / "questions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.limit]

    results = []
    for row in rows:
        print(f"{row['qid']}: {row['question'][:70]}")
        results.append(score(row, answer(row["question"])))

    with open(HERE / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    answerable = [r for r in results if r["keywords_hit"] is not None]
    lines = [
        "# Evaluation results", "",
        f"- Run: {dt.datetime.now():%Y-%m-%d %H:%M}",
        f"- Model: `{LLM_PROVIDER}/{LLM_MODEL}`  |  top_k={TOP_K}  |  min_score={MIN_RETRIEVAL_SCORE}",
        f"- Questions: {len(results)}", "",
        "| Metric | Score |", "|---|---|",
        f"| Answer keywords hit (answerable qs) | {pct(r['keywords_hit'] for r in answerable)} |",
        f"| Citation from expected doc | {pct(r['citation_correct'] for r in results)} |",
        f"| Expected doc retrieved in top-k | {pct(r['retrieval_hit'] for r in results)} |",
        f"| Refusal behaviour correct | {pct(r['refusal_correct'] for r in results)} |",
        "", "## By question type", "", "| Type | Refusal correct | Keywords hit |", "|---|---|---|",
    ]
    for t in sorted({r["type"] for r in results}):
        grp = [r for r in results if r["type"] == t]
        lines.append(f"| {t} | {pct(r['refusal_correct'] for r in grp)} | {pct(r['keywords_hit'] for r in grp)} |")
    lines += ["", "## Failures", ""]
    for r in results:
        if not r["refusal_correct"] or r["keywords_hit"] is False or r["citation_correct"] is False:
            lines.append(f"- **{r['qid']}** ({r['type']}): {r['question']}  \n  Answer: {r['answer'][:300]}")
    lines += ["", "_Automatic scores are a first pass. Check `manual_correct` in results.csv by hand._"]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {HERE / 'results.md'}")


if __name__ == "__main__":
    main()
