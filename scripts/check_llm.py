"""Quick check that your LLM key and model work.

Usage (from the project folder, venv active):
    python scripts/check_llm.py

Prints the provider, model and a safe preview of the key, then makes one call.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    print("python-dotenv not installed: run  python -m pip install python-dotenv")
    sys.exit(1)

from src.config import LLM_MODEL, LLM_PROVIDER  # noqa: E402  (read after .env loads)

KEY_VARS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def main() -> None:
    print(f"Provider : {LLM_PROVIDER}")
    print(f"Model    : {LLM_MODEL}")

    key_var = KEY_VARS.get(LLM_PROVIDER)
    if key_var is None:
        print(f"\nUnknown LLM_PROVIDER '{LLM_PROVIDER}'. Use one of: {', '.join(KEY_VARS)}")
        sys.exit(1)

    key = os.getenv(key_var, "")
    print(f"Key      : {key_var} = {key[:4]!r}... ({len(key)} chars)")

    if not key or key.startswith(("your", "paste")):
        print(f"\nNo real key found. Put your key in .env as  {key_var}=...  and save the file.")
        sys.exit(1)
    if key != key.strip() or key[0] in "\"'":
        print("\nThe key has spaces or quote marks around it. Remove them in .env.")
        sys.exit(1)
    if LLM_PROVIDER == "gemini" and key.startswith("AQ."):
        print("\nWarning: AQ. keys are currently rejected by the Gemini API for many accounts.")
    if "PIN-ME" in LLM_MODEL or LLM_MODEL.startswith(("paste", "the-exact")):
        print("\nSet LLM_MODEL in .env to the exact model name from your provider's model list.")
        sys.exit(1)

    print("\nCalling the model...")
    from src.llm import generate
    try:
        reply = generate("Reply with the single word: working")
    except Exception as e:  # show the provider's real message
        print(f"\nFAILED: {type(e).__name__}: {e}")
        msg = str(e)
        if "401" in msg or "UNAUTHENTICATED" in msg:
            print("-> The key was rejected. Create a new key and check you copied all of it.")
        elif "404" in msg or "not found" in msg.lower():
            print("-> The model name is wrong or not available to your account. Copy it exactly from the model list.")
        elif "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
            print("-> Rate limit or quota hit. Wait a minute and try again.")
        sys.exit(1)

    print(f"\nModel replied: {reply!r}")
    print("SUCCESS - the API is working." if "working" in reply.lower()
          else "Got a reply, so the connection works (the wording just differs).")


if __name__ == "__main__":
    main()
