"""Offline tests: no API key, no embedding model download needed."""
import numpy as np

from src import answer as ans
from src.config import REFUSAL_TEXT
from src.ingest import chunk_words, clean
from src.retrieve import search


def fake_chunks():
    return [
        {"chunk_id": "a#p1#0", "doc_id": "bdbv_sitrep06", "title": "Sitrep 06", "publisher": "WHO AFRO",
         "pub_date": "2026-06-21", "topic": "bundibugyo_2026", "page": 1, "text": "1048 confirmed cases"},
        {"chunk_id": "b#p1#1", "doc_id": "nigeria2014_who", "title": "Nigeria statement", "publisher": "WHO",
         "pub_date": "2014-10-20", "topic": "ebola_nigeria_2014", "page": 1, "text": "19 people, 7 died"},
    ]


def test_chunking_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_words(text, size=200, overlap=50)
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]
    assert chunks[-1].split()[-1] == "w499"


def test_clean_rejoins_hyphens():
    assert clean("contact trac-\ning   works") == "contact tracing works"


def test_search_ranks_by_cosine():
    vecs = np.array([[1, 0], [0, 1]], dtype="float32")
    res = search("q", k=2, vectors=vecs, chunks=fake_chunks(), query_vec=np.array([0, 1], dtype="float32"))
    assert res[0]["doc_id"] == "nigeria2014_who" and res[0]["score"] == 1.0


def test_low_score_refuses_without_calling_llm():
    low = [dict(c, score=0.05) for c in fake_chunks()]
    called = []
    r = ans.answer("Marburg in Rwanda?", retrieve=lambda q: low, generate=lambda *a, **k: called.append(1))
    assert r["refused"] and r["refusal_stage"] == "retrieval" and not called


def test_citations_parsed_and_out_of_range_ignored():
    high = [dict(c, score=0.8) for c in fake_chunks()]
    r = ans.answer("Nigeria?", retrieve=lambda q: high,
                   generate=lambda p, system="": "19 infected and 7 died [2]. Extra [9].")
    assert not r["refused"]
    assert [c["doc_id"] for c in r["citations"]] == ["nigeria2014_who"]


def test_llm_refusal_detected():
    high = [dict(c, score=0.8) for c in fake_chunks()]
    r = ans.answer("x", retrieve=lambda q: high, generate=lambda p, system="": REFUSAL_TEXT)
    assert r["refused"] and r["refusal_stage"] == "llm" and r["citations"] == []


def test_prompt_contains_dates_and_publishers():
    p = ans.build_prompt("q", fake_chunks())
    assert "2026-06-21" in p and "WHO AFRO" in p and "[2]" in p


def test_bm25_ranks_keyword_match_first():
    from src.backends import bm25_build, bm25_score
    texts = ["Nigeria contact tracing 894 contacts in Lagos",
             "COVID-19 vaccine priority groups older adults",
             "Bundibugyo virus disease confirmed cases DRC"]
    scores = bm25_score(bm25_build(texts), "How many contacts were traced in Nigeria?")
    assert scores.argmax() == 0 and scores[1] == 0


def test_tokenizer_keeps_numbers_and_hyphens():
    from src.backends import tokenize
    assert tokenize("COVID-19 cases: 1048, CFR 25.5%") == ["covid-19", "cases", "1048", "cfr", "25.5"]


def test_eval_keyword_groups_or_regex_and_forbidden():
    from eval.run_eval import score
    row = {"qid": "t", "type": "x", "question": "q", "should_refuse": "no",
           "expected_keywords": "Lagos;20 July/July 20",
           "forbidden_keywords": "re:licensed vaccines? for bundibugyo",
           "expected_doc_ids": "nigeria2014_who"}
    res = {"answer": "He arrived in Lagos on July 20 [1].", "refused": False, "refusal_stage": None,
           "top_score": 0.7, "citations": [{"doc_id": "nigeria2014_who"}],
           "retrieved": [{"doc_id": "nigeria2014_who"}]}
    assert score(row, res)["keywords_hit"] is True
    assert score(row, dict(res, answer="He arrived in Abuja on July 20."))["keywords_hit"] is False
    assert score(row, dict(res, answer="Lagos, 20 July. Ervebo is a licensed vaccine for Bundibugyo."))["keywords_hit"] is False


def test_eval_date_regex():
    from eval.run_eval import kw_match
    rx = r"re:(\d{1,2}(st|nd|rd|th)? (may|june|july|august|september)|2026-0[5-9])"
    assert kw_match(rx, "6100 confirmed cases as of 30 August 2026 [1]")
    assert kw_match(rx, "As of 2026-08-30, 6100 cases")
    assert not kw_match(rx, "There are 6100 confirmed cases [1]")


def test_questions_csv_well_formed():
    import csv
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "eval" / "questions.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert rows, "no questions"
    for r in rows:
        assert None not in r, f"{r['qid']}: extra column - quote fields that contain commas"
        assert r["should_refuse"] in ("yes", "no"), r["qid"]
    assert len({r["qid"] for r in rows}) == len(rows), "duplicate qid"


def test_strip_page_furniture_drops_related_publications():
    from src.ingest import strip_page_furniture
    body = "Event description " + "real report text " * 20
    page = body + "Download from IRIS Share Related Publications Situation Report 23, Data as of 19 October 2026"
    out = strip_page_furniture(page)
    assert "19 October" not in out and out.startswith("Event description")
    assert strip_page_furniture("Related publications at the very top " + body).startswith("Related")


def test_eval_survives_api_errors_and_resumes(tmp_path, monkeypatch):
    import csv
    import sys
    import eval.run_eval as ev
    import src.answer

    qs = tmp_path / "questions.csv"
    with open(qs, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "type", "question", "expected_answer", "expected_keywords",
                    "forbidden_keywords", "expected_doc_ids", "should_refuse", "verified"])
        w.writerow(["a1", "lookup", "first", "", "19", "", "d", "no", "yes"])
        w.writerow(["a2", "lookup", "second", "", "7", "", "d", "no", "yes"])
    monkeypatch.setattr(ev, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval"])

    calls = []

    def flaky(q):
        calls.append(q)
        if q == "second" and len(calls) < 3:
            raise RuntimeError("503 UNAVAILABLE")
        return {"answer": "19 and 7 [1]", "refused": False, "refusal_stage": None, "top_score": 0.8,
                "citations": [{"doc_id": "d"}], "retrieved": [{"doc_id": "d"}]}

    monkeypatch.setattr(src.answer, "answer", flaky)
    ev.main()                                   # a2 fails, run still finishes
    rows = list(csv.DictReader(open(tmp_path / "results.csv", encoding="utf-8")))
    assert [r["qid"] for r in rows] == ["a1", "a2"] and rows[1]["error"]
    assert "not run (API errors): 1" in (tmp_path / "results.md").read_text(encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["run_eval", "--resume"])
    ev.main()                                   # only a2 re-run
    assert calls == ["first", "second", "second"]
    rows = list(csv.DictReader(open(tmp_path / "results.csv", encoding="utf-8")))
    assert all(not r["error"] for r in rows)
    assert "100% (2/2)" in (tmp_path / "results.md").read_text(encoding="utf-8")


def test_daily_quota_detected_and_retry_delay_parsed():
    from src.llm import is_daily_quota, retry_delay
    daily = ("429 RESOURCE_EXHAUSTED. {'error': {'details': [{'violations': [{'quotaId': "
             "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'retryDelay': '37s'}]}}")
    minute = "429 RESOURCE_EXHAUSTED. quotaId: 'GenerateRequestsPerMinutePerProjectPerModel', 'retryDelay': '12s'"
    assert is_daily_quota(daily) and not is_daily_quota(minute)
    assert retry_delay(minute, 1) == 13
    assert retry_delay("503 UNAVAILABLE", 2) == 8


def test_eval_stops_cleanly_on_daily_quota(tmp_path, monkeypatch):
    import csv
    import sys
    import pytest
    import eval.run_eval as ev
    import src.answer
    from src.llm import QuotaExhausted

    qs = tmp_path / "questions.csv"
    with open(qs, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "type", "question", "expected_answer", "expected_keywords",
                    "forbidden_keywords", "expected_doc_ids", "should_refuse", "verified"])
        for q in ("a1", "a2", "a3"):
            w.writerow([q, "lookup", q, "", "19", "", "d", "no", "yes"])
    monkeypatch.setattr(ev, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval"])
    seen = []

    def fake(q):
        seen.append(q)
        if q == "a2":
            raise QuotaExhausted("PerDay")
        return {"answer": "19 [1]", "refused": False, "refusal_stage": None, "top_score": 0.8,
                "citations": [{"doc_id": "d"}], "retrieved": [{"doc_id": "d"}]}

    monkeypatch.setattr(src.answer, "answer", fake)
    with pytest.raises(SystemExit):
        ev.main()
    assert seen == ["a1", "a2"]                 # stopped, did not grind through a3
    rows = list(csv.DictReader(open(tmp_path / "results.csv", encoding="utf-8")))
    assert [r["qid"] for r in rows] == ["a1"]   # a1 kept for --resume


def test_citations_in_other_model_styles():
    chunks = [{"doc_id": f"d{i}"} for i in range(1, 6)]
    text = "A【2】 B【1†L1-L3】 C [3, 5] D [4]."
    assert [c["doc_id"] for c in ans.parse_citations(text, chunks)] == ["d1", "d2", "d3", "d4", "d5"]


def test_normalize_formatting_differences():
    from eval.run_eval import kw_match
    text = "**1,048 laboratory‑confirmed cases** and 267 deaths"
    assert kw_match("1048", text) and kw_match("laboratory-confirmed", text)
    assert kw_match("1048", "1 048 cases")
    assert kw_match("25.5", "CFR 25.5 %")
    assert kw_match(r"re:\d{1,2} (june|july)", "as of 21 June 2026")
    assert not kw_match("1048", "104 cases")


def test_rescore_keeps_manual_marks_and_needs_no_api(tmp_path, monkeypatch):
    import csv
    import sys
    import eval.run_eval as ev
    with open(tmp_path / "questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "type", "question", "expected_answer", "expected_keywords",
                    "forbidden_keywords", "expected_doc_ids", "should_refuse", "verified"])
        w.writerow(["a1", "lookup", "q", "", "1048", "", "d2", "no", "yes"])
    with open(tmp_path / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ev.FIELDS)
        w.writeheader()
        w.writerow({"qid": "a1", "type": "lookup", "question": "q", "answer": "1,048 cases【2】",
                    "refused": "False", "refusal_stage": "", "top_score": "0.8",
                    "retrieved_docs": "d1|d2", "manual_correct": "yes", "error": ""})
    monkeypatch.setattr(ev, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--rescore", "--label", "test run"])
    ev.main()
    row = next(csv.DictReader(open(tmp_path / "results.csv", encoding="utf-8")))
    assert row["keywords_hit"] == "True" and row["citation_correct"] == "True"
    assert row["manual_correct"] == "yes"
    assert list((tmp_path / "runs").glob("*-test-run.md"))


def test_rrf_merge_rewards_agreement_and_keeps_keyword_only_hits():
    from src.retrieve import rrf_merge
    dense = np.array([0.9, 0.8, 0.1, 0.2], dtype="float32")   # likes chunk 0, then 1
    bm25 = np.array([0.0, 5.0, 7.0, 0.0], dtype="float32")    # exact-term hits: chunk 2, then 1
    order = rrf_merge([dense, bm25], k=3)
    assert order[0] == 1          # ranked well by both
    assert 2 in order             # keyword-only match still makes the cut


def test_hybrid_search_finds_exact_number_dense_misses(monkeypatch):
    import src.retrieve as rt
    from src.backends import bm25_build
    chunks = [dict(fake_chunks()[0], text="General overview of outbreak response and partners"),
              dict(fake_chunks()[0], doc_id="bdbv_don602",
                   text="As of 15 May, 246 suspected cases including 80 deaths were reported"),
              dict(fake_chunks()[1], text="Nigeria Ebola statement 19 people 7 died")]
    vectors = np.array([[1, 0], [0.2, 0.98], [0, 1]], dtype="float32")
    state = {"vectors": vectors, "bm25": bm25_build([c["text"] for c in chunks])}
    monkeypatch.setattr(rt, "RETRIEVER", "hybrid")
    monkeypatch.setattr(rt, "_load_index", lambda: (state, chunks))
    # stand-in embedding that prefers the vague overview chunk
    monkeypatch.setattr(rt, "_dense_scores", lambda st, q: st["vectors"] @ np.array([1, 0], dtype="float32"))
    res = rt.search("How many suspected cases (246?) and deaths were reported?", k=2)
    assert "bdbv_don602" in [c["doc_id"] for c in res]
    assert all(-1 <= c["score"] <= 1 for c in res)   # cosine kept for the refusal threshold


def test_only_runs_do_not_overwrite_full_results(tmp_path, monkeypatch):
    import csv
    import sys
    import eval.run_eval as ev
    import src.answer
    with open(tmp_path / "questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "type", "question", "expected_answer", "expected_keywords",
                    "forbidden_keywords", "expected_doc_ids", "should_refuse", "verified"])
        for q in ("a1", "a2"):
            w.writerow([q, "lookup", q, "", "19", "", "d", "no", "yes"])
    (tmp_path / "results.csv").write_text("full run - keep me", encoding="utf-8")
    monkeypatch.setattr(ev, "HERE", tmp_path)
    monkeypatch.setattr(ev, "OUT", "results")
    monkeypatch.setattr(sys, "argv", ["run_eval", "--only", "a2"])
    monkeypatch.setattr(src.answer, "answer", lambda q: {
        "answer": "19 [1]", "refused": False, "refusal_stage": None, "top_score": 0.8,
        "citations": [{"doc_id": "d"}], "retrieved": [{"doc_id": "d"}]})
    ev.main()
    assert (tmp_path / "results.csv").read_text(encoding="utf-8") == "full run - keep me"
    rows = list(csv.DictReader(open(tmp_path / "results_only.csv", encoding="utf-8")))
    assert [r["qid"] for r in rows] == ["a2"]


def test_retrieval_eval_finds_ranks_without_api():
    from eval.retrieval_eval import evaluate, passage_keywords, report
    from src.backends import bm25_build
    chunks = [{"text": "General outbreak response overview with partners"},
              {"text": "As of 15 May, 246 suspected cases including 80 deaths were reported"},
              {"text": "Nigeria statement: 19 people infected, 7 died"}]
    rows = [{"qid": "b08", "question": "How many suspected cases (246) and deaths?", "should_refuse": "no",
             "expected_keywords": "246;80"},
            {"qid": "b06", "question": "q", "should_refuse": "no", "expected_keywords": r"re:\d+ june"},
            {"qid": "u01", "question": "q", "should_refuse": "yes", "expected_keywords": ""}]
    vectors = np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype="float32")
    per_q = evaluate(rows, chunks, vectors, bm25_build([c["text"] for c in chunks]),
                     embed=lambda q: np.array([1, 0], dtype="float32"), ks=[1, 2])
    assert [q["qid"] for q in per_q] == ["b08"]          # regex-only and refusal rows skipped
    assert per_q[0]["embeddings"] is None                 # embeddings rank it 3rd, outside top 2
    assert per_q[0]["bm25"] == 1 and per_q[0]["hybrid"] <= 2
    assert passage_keywords({"expected_keywords": r"antibiotic;re:\b(no|not)\b"}) == ["antibiotic"]
    assert "| bm25 | 1/1 | 1/1 |" in report(per_q, [1, 2], {"embed_model": "m"}, "x")


def test_search_text_uses_header_but_passage_stays_clean():
    from src.backends import search_text
    c = {"header": "National Interim Guidelines for Clinical Management of COVID-19 (NCDC)",
         "text": "Cases may present as mild or severe."}
    assert search_text(c).startswith("National Interim") and c["text"].startswith("Cases")
    assert search_text({"text": "only text"}) == "only text"
    p = ans.build_prompt("q", [dict(fake_chunks()[0], header="HEADER-ONLY")])
    assert "HEADER-ONLY" not in p       # header is for search, not shown as passage text
