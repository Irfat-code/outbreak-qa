"""Search your indexed documents for a word or number, to verify expected answers.

Usage:
    python scripts/find_text.py "1048"
    python scripts/find_text.py "Port Harcourt" nigeria2014_contacts
    python scripts/find_text.py "life-course" covid_pheic_end

Prints each matching chunk's document, page and the text around the match.
Run python -m src.ingest first.
"""
import json
import sys
from pathlib import Path

INDEX = Path(__file__).resolve().parent.parent / "data" / "index" / "chunks.jsonl"
WINDOW = 160  # characters shown either side of the match


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    needle = sys.argv[1].lower()
    only_doc = sys.argv[2] if len(sys.argv) > 2 else None

    hits = 0
    with open(INDEX, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if only_doc and c["doc_id"] != only_doc:
                continue
            text = c["text"]
            pos = text.lower().find(needle)
            if pos == -1:
                continue
            hits += 1
            start, end = max(0, pos - WINDOW), pos + len(needle) + WINDOW
            snippet = text[start:end].replace("\n", " ")
            print(f"\n[{c['doc_id']} p.{c['page']} | {c['pub_date'] or 'no date'}]")
            print(f"  ...{snippet}...")

    print(f"\n{hits} matching chunk(s)" + (f" in {only_doc}" if only_doc else ""))


if __name__ == "__main__":
    main()
