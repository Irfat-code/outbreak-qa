"""Run every question in eval/questions.csv and score the answers.

Automatic checks (a first pass - read the answers yourself too):
  refusal_correct  - refused when it should, answered when it should
  keywords_hit     - every expected keyword group matches and no forbidden one does
                     (';' separates required groups, '/' gives alternatives,
                      're:' starts a regex)
  citation_correct - at least one cited chunk comes from an expected doc

Writes eval/results.csv (row per question) and eval/results.md (summary).
Usage:  python -m eval.run_eval [--limit N] [--only q1,q2] [--resume] [--rescore] [--label NAME]
        --resume  keeps answers already saved and only re-runs missing/failed ones.
        --rescore re-scores saved answers without calling the model.
        --label   also saves the run under eval/runs/ so you can compare runs.
"""
import argparse
import csv
import datetime as dt
import re
import shutil
import unicodedata
from pathlib import Path

from src.config import LLM_MODEL, LLM_PROVIDER, MIN_RETRIEVAL_SCORE, TOP_K

HERE = Path(__file__).resolve().parent
OUT = "results"            # file stem; subset (--only) runs use "results_only"


def normalize(text: str) -> str:
    """Make formatting differences irrelevant to matching.

    Models write '1,048', '1 048', '25.5 %' with non-breaking spaces, or
    'case‑fatality' with a non-breaking hyphen. None of that changes the answer,
    so it shouldn't change the score.
    """
    t = unicodedata.normalize("NFKC", text)               # odd spaces -> normal space
    t = re.sub(r"[‐-―−]", "-", t)          # all dash/hyphen variants -> '-'
    t = re.sub(r"(?<=\d)[, ](?=\d{3}\b)", "", t)          # thousands separators: 1,048 / 1 048 -> 1048
    t = re.sub(r"[*_`]", "", t)                           # markdown bold/italics
    return re.sub(r"\s+", " ", t).lower()


def kw_match(group: str, text: str) -> bool:
    """One keyword group. 're:...' = regex; otherwise 'a/b/c' = any of these substrings."""
    group = group.strip()
    text = normalize(text)
    if group.startswith("re:"):
        return re.search(group[3:], text, re.IGNORECASE) is not None
    return any(normalize(alt).strip() in text for alt in group.split("/") if alt.strip())


def groups(cell: str) -> list[str]:
    """Groups are separated by ';' and ALL must match."""
    return [g for g in (cell or "").split(";") if g.strip()]


def score(row: dict, result: dict) -> dict:
    should_refuse = row["should_refuse"].strip().lower() == "yes"
    refusal_correct = result["refused"] == should_refuse

    text = result["answer"]
    keywords_hit = (not should_refuse and not result["refused"]
                    and all(kw_match(g, text) for g in groups(row["expected_keywords"]))
                    and not any(kw_match(g, text) for g in groups(row.get("forbidden_keywords", ""))))

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
        # in retrieval order, so --rescore can map [n] back to a document later
        "retrieved_docs": "|".join(c["doc_id"] for c in result["retrieved"]),
        "refusal_correct": refusal_correct,
        "keywords_hit": keywords_hit if not should_refuse else None,
        "citation_correct": citation_correct,
        "retrieval_hit": retrieval_hit,
        "manual_correct": "",           # fill in by hand: yes / no / partial
    }


def pct(values) -> str:
    vals = [v for v in values if v is not None]
    return f"{100 * sum(vals) / len(vals):.0f}% ({sum(vals)}/{len(vals)})" if vals else "n/a"


FIELDS = ["qid", "type", "question", "answer", "refused", "refusal_stage", "top_score",
          "cited_docs", "retrieved_docs", "refusal_correct", "keywords_hit", "citation_correct",
          "retrieval_hit", "manual_correct", "error"]
BOOL_FIELDS = ("refused", "refusal_correct", "keywords_hit", "citation_correct", "retrieval_hit")


def _from_csv(r: dict) -> dict:
    """Turn 'True'/'False'/'' strings from results.csv back into True/False/None."""
    return {**r, **{k: {"True": True, "False": False}.get(r.get(k, ""), None) for k in BOOL_FIELDS}}


def load_previous() -> dict:
    path = HERE / f"{OUT}.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["qid"]: _from_csv(r) for r in csv.DictReader(f) if not r.get("error")}


def save(results: list[dict]) -> None:
    with open(HERE / f"{OUT}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


def rescore(rows: list[dict]) -> list[dict]:
    """Re-score saved answers with the current scorer and questions - no API calls."""
    from src.answer import parse_citations

    path = HERE / f"{OUT}.csv"
    if not path.exists():
        raise SystemExit("No results.csv to rescore - run the evaluation first.")
    with open(path, newline="", encoding="utf-8") as f:
        saved = {r["qid"]: r for r in csv.DictReader(f)}
    out = []
    for row in rows:
        old = saved.get(row["qid"])
        if old is None or old.get("error"):
            continue
        if "retrieved_docs" not in old:
            raise SystemExit("results.csv is from an older version - do one full run first.")
        retrieved = [{"doc_id": d} for d in old["retrieved_docs"].split("|") if d]
        refused = old["refused"] == "True"
        result = {"answer": old["answer"], "refused": refused, "refusal_stage": old["refusal_stage"] or None,
                  "top_score": float(old["top_score"] or 0),
                  "citations": [] if refused else parse_citations(old["answer"], retrieved),
                  "retrieved": retrieved}
        out.append({**score(row, result), "manual_correct": old.get("manual_correct", ""), "error": ""})
    return out


def failure_reasons(r: dict) -> list[str]:
    reasons = []
    if not r["refusal_correct"]:
        reasons.append("refused but answerable" if r["refused"] else "answered but should refuse")
    if r["keywords_hit"] is False and not r["refused"]:
        reasons.append("expected facts missing")
    if r["citation_correct"] is False and not r["refused"]:
        reasons.append("cited wrong/no document")
    return reasons


def write_report(results: list[dict], errors: list[dict], label: str | None) -> None:
    answerable = [r for r in results if r["keywords_hit"] is not None]
    wrongly_refused = [r for r in answerable if r["refused"]]
    lines = [
        "# Evaluation results", "",
        f"- Run: {dt.datetime.now():%Y-%m-%d %H:%M}" + (f"  |  label: `{label}`" if label else ""),
        f"- Model: `{LLM_PROVIDER}/{LLM_MODEL}`  |  top_k={TOP_K}  |  min_score={MIN_RETRIEVAL_SCORE}",
        f"- Questions scored: {len(results)}"
        + (f"  |  **not run (API errors): {len(errors)}** - re-run with `--resume`" if errors else ""), "",
        "| Metric | Score |", "|---|---|",
        f"| Answer keywords hit (answerable qs) | {pct(r['keywords_hit'] for r in answerable)} |",
        f"| Citation from expected doc | {pct(r['citation_correct'] for r in results)} |",
        f"| Expected doc retrieved in top-k | {pct(r['retrieval_hit'] for r in results)} |",
        f"| Refusal behaviour correct | {pct(r['refusal_correct'] for r in results)} |",
        f"| Answerable questions wrongly refused | {len(wrongly_refused)}/{len(answerable)} |",
        "", "## By question type", "", "| Type | Refusal correct | Keywords hit |", "|---|---|---|",
    ]
    for t in sorted({r["type"] for r in results}):
        grp = [r for r in results if r["type"] == t]
        lines.append(f"| {t} | {pct(r['refusal_correct'] for r in grp)} | {pct(r['keywords_hit'] for r in grp)} |")
    lines += ["", "## Failures", ""]
    for r in results:
        reasons = failure_reasons(r)
        if reasons:
            lines.append(f"- **{r['qid']}** ({r['type']}; {', '.join(reasons)}): {r['question']}  \n"
                         f"  Answer: {r['answer'][:300]}")
    lines += ["", "_Automatic scores are a first pass. Check `manual_correct` in results.csv by hand._"]
    text = "\n".join(lines) + "\n"
    (HERE / f"{OUT}.md").write_text(text, encoding="utf-8")
    print(f"\nWrote {HERE / (OUT + '.md')}")
    if label:
        runs = HERE / "runs"
        runs.mkdir(exist_ok=True)
        stamp = f"{dt.datetime.now():%Y%m%d-%H%M}-{re.sub(r'[^A-Za-z0-9_-]+', '-', label)}"
        (runs / f"{stamp}.md").write_text(text, encoding="utf-8")
        shutil.copy(HERE / f"{OUT}.csv", runs / f"{stamp}.csv")
        print(f"Saved a copy as eval/runs/{stamp}.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="keep answers already in results.csv; only re-run missing or failed questions")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score saved answers with the current scorer and questions (no API calls)")
    ap.add_argument("--only", default=None,
                    help="comma-separated qids to run, e.g. --only b05,b08,c04 (saves tokens while iterating)")
    ap.add_argument("--label", default=None,
                    help="also save this run under eval/runs/, e.g. --label baseline")
    args = ap.parse_args()

    with open(HERE / "questions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.limit]
    global OUT
    if args.only:
        OUT = "results_only"           # never overwrite the full run's results.csv
        wanted = {q.strip() for q in args.only.split(",") if q.strip()}
        unknown = wanted - {r["qid"] for r in rows}
        if unknown:
            raise SystemExit(f"Unknown qid(s): {', '.join(sorted(unknown))}")
        rows = [r for r in rows if r["qid"] in wanted]

    if args.rescore:
        results = rescore(rows)
        save(results)
        write_report(results, [], args.label)
        return

    from src.answer import answer
    from src.llm import QuotaExhausted

    previous = load_previous() if args.resume else {}
    results = []
    for row in rows:
        if row["qid"] in previous:
            results.append(previous[row["qid"]])
            continue
        print(f"{row['qid']}: {row['question'][:70]}")
        try:
            results.append({**score(row, answer(row["question"])), "error": ""})
        except QuotaExhausted as e:
            save(results)
            wait = re.search(r"try again in ([\dhms.]+)", str(e))
            raise SystemExit(
                f"\nDaily API quota used up at {row['qid']}. {len(results)} answers are saved.\n"
                f"Provider said: {str(e)[:300]}\n"
                + (f"Suggested wait: {wait.group(1)}\n" if wait else "")
                + "Continue later with:  python -m eval.run_eval --resume")
        except Exception as e:               # keep going: one failed call must not lose the run
            print(f"   !! failed after retries: {type(e).__name__}: {str(e)[:100]}")
            results.append({"qid": row["qid"], "type": row["type"], "question": row["question"],
                            "answer": "", "error": f"{type(e).__name__}: {str(e)[:200]}"})
        save(results)                        # save after every question

    errors = [r for r in results if r.get("error")]
    results = [r for r in results if not r.get("error")]
    if not results:
        raise SystemExit("Every question failed - check the API with scripts/check_llm.py")
    write_report(results, errors, args.label)
    if errors:
        print(f"{len(errors)} question(s) hit API errors: {', '.join(r['qid'] for r in errors)}.\n"
              "Wait a few minutes, then run:  python -m eval.run_eval --resume")


if __name__ == "__main__":
    main()
