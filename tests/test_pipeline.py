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
