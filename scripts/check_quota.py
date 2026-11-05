"""Make one API call and print the FULL error, so you can see which quota you hit.

Usage:  python scripts/check_quota.py

Look for 'quotaId' / 'quotaMetric' in the output:
  ...PerMinute...  -> per-minute limit: wait a minute, slow down calls
  ...PerDay...     -> daily limit: used up until it resets
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import LLM_MODEL, LLM_PROVIDER  # noqa: E402  (also loads .env)

if LLM_PROVIDER != "gemini":
    sys.exit(f"This check is for Gemini; LLM_PROVIDER is {LLM_PROVIDER}.")

from google import genai  # noqa: E402

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
try:
    r = client.models.generate_content(model=LLM_MODEL, contents="Reply with the single word: working")
    print(f"OK - {LLM_MODEL} answered {r.text!r}. No quota problem right now.")
except Exception as e:
    msg = str(e)
    print(msg)
    low = msg.lower()
    print("\n---")
    if "perday" in low or "per day" in low:
        print("DAILY quota used up for this model. It resets once a day (midnight US Pacific time).")
    elif "perminute" in low or "per minute" in low:
        print("PER-MINUTE quota hit. Wait a minute; the eval's pacing should then be enough.")
    else:
        print("Not clearly a quota error - paste this output to get help.")
