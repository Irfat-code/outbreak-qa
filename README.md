# Outbreak Q&A — a cited, testable RAG prototype for public-health documents

Ask a question about the 2026 Bundibugyo Ebola outbreak (DRC & Uganda), Nigeria's 2014 Ebola response, or COVID-19. The tool answers **only** from a fixed set of WHO, Africa CDC and NCDC documents, cites the passage it used with the document's date, and replies *"Not in these documents."* when the answer isn't there.

> Status: work in progress (Nov 2026). Results below are filled in after the final evaluation run.

## Why this is hard (and what the evaluation tests)

Outbreak documents are a bad fit for a naive chatbot:

- **Figures change week to week.** "How many cases?" has a different answer in every situation report. Answers must say which date a number refers to.
- **Species get confused.** Bundibugyo virus, Zaire ebolavirus and Sudan virus are different diseases for vaccines and treatment. A fact about one must not leak into another.
- **Sources disagree.** For example, WHO and a Nigerian field-epidemiology case study give different contact-tracing totals for 2014.
- **Some questions have no answer in the corpus**, and the right behaviour is to refuse.

## How it works

```
sources.csv → download.py → data/raw/ → ingest.py (parse, chunk, embed locally)
                                              ↓
question → retrieve.py (top-k cosine) → score below threshold? → refuse
                                              ↓
                       answer.py (numbered passages + rules) → llm.py → answer + [n] citations
```

- **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers, run locally (no API).
- **Generation:** `<PINNED MODEL NAME>` via Google AI Studio, temperature 0. All calls go through one wrapper (`src/llm.py`), so the provider can be swapped by config.
- **Two refusal layers:** a retrieval-score threshold (no LLM call if nothing relevant is found) and an instruction to the model to refuse when the passages don't contain the answer.
- **Metadata on every chunk:** document, publisher, publication date and page.

## Corpus

17 public documents, listed with links and dates in [`data/sources.csv`](data/sources.csv). Documents are downloaded by script and not committed.

## Evaluation

`eval/questions.csv` holds N questions in five types: lookup, date trap, species trap, conflicting-sources trap, and unanswerable.

| Metric | Score |
|---|---|
| Answer correct (manual check) | _tbd_ |
| Citation from expected document | _tbd_ |
| Correct refusal / non-refusal | _tbd_ |

Full output: [`eval/results.md`](eval/results.md).

### Where it fails

_Fill in honestly after Day 6–7: the failure patterns, examples, and what you changed._

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your Gemini key and pinned model name
python scripts/download.py
python -m src.ingest
python -m src.answer "What did WHO say about Nigeria's Ebola outbreak in October 2014?"
python -m eval.run_eval
pytest                       # offline tests, no key needed
```

## Limitations

A prototype, not a clinical tool. It knows only what is in 17 documents, the documents themselves go out of date, and automatic scoring is checked by hand.
