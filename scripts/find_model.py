"""Try each Gemini Flash model your key can see, once, and report which respond.

Usage:  python scripts/find_model.py
Then put a working name in .env as LLM_MODEL=... (without the 'models/' prefix).
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

names = [m.name.removeprefix("models/") for m in client.models.list()
         if "flash" in m.name and "generateContent" in (m.supported_actions or [])]

print(f"Found {len(names)} Flash models. Testing each once...\n")
working = []
for name in names:
    try:
        r = client.models.generate_content(model=name, contents="Reply with the single word: working")
        print(f"  OK      {name:<40} -> {(r.text or '').strip()[:20]!r}")
        working.append(name)
    except Exception as e:
        print(f"  FAILED  {name:<40} -> {str(e)[:70]}")

print("\nWorking:", ", ".join(working) or "none right now - try again in 15 minutes")
print("Prefer one without 'preview', 'exp' or 'lite' in the name.")
