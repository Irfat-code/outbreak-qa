"""Parse downloaded docs, chunk them, embed locally, and save the index.

Every chunk keeps doc_id, title, publisher, pub_date and page so answers can
cite their source and say how old the information is.

Usage:  python -m src.ingest
"""
import csv
import json
import re

import numpy as np

from src.config import (CHUNK_OVERLAP_WORDS, CHUNK_WORDS, INDEX_DIR, RAW_DIR,
                        RETRIEVER, SOURCES_CSV)


def load_sources() -> dict:
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def read_pdf(path) -> list[tuple[int, str]]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def read_html(path) -> list[tuple[int, str]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return [(1, main.get_text(" "))]


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(sources: dict) -> list[dict]:
    chunks = []
    for path in sorted(RAW_DIR.glob("*")):
        doc_id = path.stem
        meta = sources.get(doc_id)
        if meta is None:
            print(f"skip {path.name}: doc_id not in sources.csv")
            continue
        pages = read_pdf(path) if path.suffix == ".pdf" else read_html(path)
        n_before = len(chunks)
        for page_no, page_text in pages:
            for text in chunk_words(clean(page_text)):
                if len(text.split()) < 25:       # drop headers/footers/fragments
                    continue
                chunks.append({
                    "chunk_id": f"{doc_id}#p{page_no}#{len(chunks)}",
                    "doc_id": doc_id,
                    "title": meta["title"],
                    "publisher": meta["publisher"],
                    "pub_date": meta["pub_date"],
                    "topic": meta["topic"],
                    "page": page_no,
                    "text": text,
                })
        print(f"{doc_id:<22} {len(chunks) - n_before:>4} chunks")
    return chunks


def main() -> None:
    chunks = build_chunks(load_sources())
    if not chunks:
        raise SystemExit("No chunks - run scripts/download.py first.")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if RETRIEVER in ("fastembed", "st"):
        from src.backends import dense_embed
        print(f"Embedding {len(chunks)} chunks with {RETRIEVER} (first run downloads the model)...")
        np.save(INDEX_DIR / "embeddings.npy", dense_embed(RETRIEVER, [c["text"] for c in chunks]))
    (INDEX_DIR / "retriever.txt").write_text(RETRIEVER)
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(chunks)} chunks to {INDEX_DIR} (retriever: {RETRIEVER})")


if __name__ == "__main__":
    main()
