"""Answer a question from retrieved chunks, with citations and a refusal path.

Usage:  python -m src.answer "How many confirmed cases did DRC report as of 21 June 2026?"
"""
import re
import sys

from src.config import MIN_RETRIEVAL_SCORE, REFUSAL_TEXT

SYSTEM = f"""You answer questions about Ebola and COVID-19 using ONLY the numbered
source passages provided. Rules:
1. Every factual sentence ends with the passage number(s) it came from, like [2] or [1][3].
2. If the passages do not contain the answer, reply exactly: {REFUSAL_TEXT}
3. Do not use outside knowledge, even if you are confident.
4. Figures change over time. When you give a number, say the date it refers to
   (from the passage or its publication date).
5. Keep virus species apart: Bundibugyo virus, Zaire ebolavirus (Ebola virus) and
   Sudan virus are different. Do not apply a fact about one to another.
6. If passages disagree, say so and give both, with their sources.
Answer in at most 5 sentences."""


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        date = c.get("pub_date") or "date unknown"
        blocks.append(f"[{i}] {c['title']} ({c['publisher']}, {date}, p.{c['page']})\n{c['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[dict]) -> str:
    return f"Source passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"


def parse_citations(text: str, chunks: list[dict]) -> list[dict]:
    """Map citation markers back to chunks; ignore out-of-range numbers.

    Accepts [2], [1][3], [1, 3] and the styles some models use instead,
    such as 【2】 and 【1†L1-L3】.
    """
    nums = set()
    for group in CITATION_RE.findall(text):
        nums.update(int(n) for n in re.findall(r"\d+", group))
    return [chunks[n - 1] for n in sorted(nums) if 1 <= n <= len(chunks)]


CITATION_RE = re.compile(r"[\[【]\s*(\d+(?:\s*,\s*\d+)*)\s*(?:†[^\]】]*)?[\]】]")


def answer(question: str, retrieve=None, generate=None) -> dict:
    """retrieve/generate can be injected for testing."""
    if retrieve is None:
        from src.retrieve import search as retrieve
    if generate is None:
        from src.llm import generate

    chunks = retrieve(question)
    top_score = max((c["score"] for c in chunks), default=0.0)
    if top_score < MIN_RETRIEVAL_SCORE:
        return {"question": question, "answer": REFUSAL_TEXT, "refused": True,
                "refusal_stage": "retrieval", "top_score": top_score,
                "citations": [], "retrieved": chunks}

    text = generate(build_prompt(question, chunks), system=SYSTEM)
    refused = REFUSAL_TEXT.lower().rstrip(".") in text.lower()
    return {"question": question, "answer": text, "refused": refused,
            "refusal_stage": "llm" if refused else None, "top_score": top_score,
            "citations": [] if refused else parse_citations(text, chunks),
            "retrieved": chunks}


def main() -> None:
    q = " ".join(sys.argv[1:]) or input("Question: ")
    r = answer(q)
    print("\n" + r["answer"] + "\n")
    for i, c in enumerate(r["citations"], 1):
        print(f"  source: {c['title']} ({c['publisher']}, {c['pub_date'] or 'n.d.'}), p.{c['page']}")
    print(f"\n(top retrieval score {r['top_score']:.2f})")


if __name__ == "__main__":
    main()
