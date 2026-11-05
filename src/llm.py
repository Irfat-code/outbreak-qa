"""One wrapper for 'send prompt, get answer'.

Switching provider = change LLM_PROVIDER / LLM_MODEL in config or .env.
Keys are read from the environment (GEMINI_API_KEY or ANTHROPIC_API_KEY),
never written in code.
"""
import os
import time

from src.config import (LLM_MAX_RETRIES, LLM_MIN_SECONDS_BETWEEN_CALLS,
                        LLM_MODEL, LLM_PROVIDER, LLM_TEMPERATURE)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_last_call = 0.0


def _throttle() -> None:
    global _last_call
    wait = LLM_MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _gemini(prompt: str, system: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system,
                                           temperature=LLM_TEMPERATURE),
    )
    return resp.text or ""


def _anthropic(prompt: str, system: str) -> str:
    import anthropic
    client = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=LLM_MODEL, max_tokens=800, temperature=LLM_TEMPERATURE,
        system=system, messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


PROVIDERS = {"gemini": _gemini, "anthropic": _anthropic}


def generate(prompt: str, system: str = "") -> str:
    """Call the configured model, retrying with backoff on rate limits/errors."""
    call = PROVIDERS[LLM_PROVIDER]
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        _throttle()
        try:
            return call(prompt, system).strip()
        except Exception as e:                   # SDK error classes differ per provider
            if attempt == LLM_MAX_RETRIES:
                raise
            delay = min(60, 2 ** attempt * 2)
            print(f"[llm] attempt {attempt} failed ({type(e).__name__}: {e}); retrying in {delay}s")
            time.sleep(delay)
    raise RuntimeError("unreachable")
