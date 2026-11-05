"""Download every document listed in data/sources.csv into data/raw/.

Uses download_url when filled in (direct PDF), otherwise saves page_url as HTML.
WHO publication pages are landing pages: open them, copy the PDF link from the
Download button, and paste it into the download_url column.

Usage:  python scripts/download.py
"""
import csv
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DIR, SOURCES_CSV  # noqa: E402

HEADERS = {"User-Agent": "outbreak-qa research prototype (python-requests)"}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    failed = []
    for row in rows:
        url = row["download_url"].strip() or row["page_url"].strip()
        if not url:
            failed.append((row["doc_id"], "no URL"))
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            failed.append((row["doc_id"], str(e)))
            continue

        is_pdf = "pdf" in r.headers.get("Content-Type", "").lower() or r.content[:4] == b"%PDF"
        out = RAW_DIR / f"{row['doc_id']}.{'pdf' if is_pdf else 'html'}"
        out.write_bytes(r.content)
        print(f"ok   {row['doc_id']:<22} -> {out.name} ({len(r.content)//1024} KB)")
        time.sleep(1)

    if failed:
        print("\nFailed (fix the URL or download by hand into data/raw/ using the doc_id as filename):")
        for doc_id, why in failed:
            print(f"  {doc_id}: {why}")


if __name__ == "__main__":
    main()
