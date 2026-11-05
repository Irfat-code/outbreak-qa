"""One wrapper for 'send prompt, get answer'.

Switching provider = change LLM_PROVIDER / LLM_MODEL in .env.
Keys are read from the environment (GEMINI_API_KEY, GROQ_API_KEY or
ANTHROPIC_API_KEY), never written in code.
"""
import os
import re
import time

from src.config import (LLM_MAX_RETRIES, LLM_MIN_SECONDS_BETWEEN_CALLS,
                        LLM_MODEL, LLM_PROVIDER, LLM_REASONING_EFFORT,
                        LLM_TEMPERATURE)

_last_call = 0.0


class QuotaExhausted(RuntimeError):
    """Daily quota used up: retrying today is pointless, so callers should stop."""


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


def _groq(prompt: str, system: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    extra = {}
    if "gpt-oss" in LLM_MODEL:
        # Reasoning model: its hidden reasoning also uses the token budget, so allow
        # more tokens and keep reasoning light, or the visible answer can come back empty.
        extra = {"reasoning_effort": LLM_REASONING_EFFORT, "max_tokens": 3000}
    resp = client.chat.completions.create(model=LLM_MODEL, messages=messages,
                                          temperature=LLM_TEMPERATURE,
                                          **{"max_tokens": 800, **extra})
    return resp.choices[0].message.content or ""


def _anthropic(prompt: str, system: str) -> str:
    import anthropic
    client = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=LLM_MODEL, max_tokens=800, temperature=LLM_TEMPERATURE,
        system=system, messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


PROVIDERS = {"gemini": _gemini, "groq": _groq, "anthropic": _anthropic}


def is_daily_quota(msg: str) -> bool:
    m = msg.lower()
    return ("429" in m or "resource_exhausted" in m) and ("perday" in m or "per day" in m or "daily" in m)


def retry_delay(msg: str, attempt: int) -> float:
    """Use the server's suggested wait (e.g. retryDelay: '37s') when it gives one."""
    found = re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", msg, re.IGNORECASE) \
        or re.search(r"retry in (\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if found:
        return min(120.0, float(found.group(1)) + 1)
    return float(min(60, 2 ** attempt * 2))


def generate(prompt: str, system: str = "") -> str:
    """Call the configured model, retrying with backoff on temporary errors."""
    call = PROVIDERS[LLM_PROVIDER]
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        _throttle()
        try:
            return call(prompt, system).strip()
        except Exception as e:                   # SDK error classes differ per provider
            msg = str(e)
            if is_daily_quota(msg):
                raise QuotaExhausted(msg) from e
            permanent = any(code in msg for code in ("400 ", "401 ", "403 ", "404 "))
            if permanent or attempt == LLM_MAX_RETRIES:
                raise
            delay = retry_delay(msg, attempt)
            print(f"[llm] attempt {attempt} failed ({type(e).__name__}: {msg[:120]}); retrying in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError("unreachable")
