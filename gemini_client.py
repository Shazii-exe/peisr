# gemini_client.py

import os
import json
from dotenv import load_dotenv

try:
    # New SDK: google-genai
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    _SDK = "google-genai"
except Exception:
    # Fallback: older SDK (google-generativeai)
    import google.generativeai as genai  # type: ignore
    types = None
    _SDK = "google-generativeai"

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env (local) or Streamlit Secrets (cloud).")

if _SDK == "google-genai":
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    genai.configure(api_key=GEMINI_API_KEY)
    client = None

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"


def _strip_code_fences(s: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences if the model wraps JSON."""
    s = (s or "").strip()
    if s.startswith("```"):
        # Remove first fence line
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        # Remove trailing fence
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].strip()
    return s.strip()


def _extract_json_object(s: str) -> str:
    """
    Best-effort extraction of the first JSON object/array from a string.
    Handles cases where the model adds extra text before/after JSON.
    """
    s = _strip_code_fences(s)
    if not s:
        return ""

    # If it's already clean JSON, return as-is
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        return s

    # Try to find a JSON object/array inside text
    obj_start = s.find("{")
    arr_start = s.find("[")
    start_candidates = [i for i in [obj_start, arr_start] if i != -1]
    if not start_candidates:
        return s  # nothing to extract

    start = min(start_candidates)

    # Find matching end by scanning braces/brackets
    stack = []
    for idx in range(start, len(s)):
        ch = s[idx]
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                # mismatched, ignore
                continue
            if not stack:
                return s[start : idx + 1].strip()

    # Fallback: return substring from first brace/bracket
    return s[start:].strip()


def _safe_json_load(txt: str) -> dict:
    """
    Never crash the app due to invalid/empty JSON.
    Returns {"error": "...", "raw_text": "..."} on failure.
    """
    raw = (txt or "").strip()
    if not raw:
        return {"error": "Empty response from Gemini (no JSON returned).", "raw_text": ""}

    candidate = _extract_json_object(raw)

    try:
        parsed = json.loads(candidate)
        # Ensure dict output for downstream code
        if isinstance(parsed, dict):
            return parsed
        return {"data": parsed}
    except json.JSONDecodeError as e:
        return {
            "error": f"Invalid JSON from Gemini: {e}",
            "raw_text": raw[:4000],
        }


def generate_text(system: str, user: str, temperature: float = 0.2) -> str:
    if _SDK == "google-genai":
        resp = client.models.generate_content(
            model=MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
            ),
        )
        return (resp.text or "").strip()

    # google-generativeai fallback
    resp = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=system,
    ).generate_content(
        user,
        generation_config={"temperature": temperature},
    )
    return (getattr(resp, "text", "") or "").strip()


def generate_json(system: str, user: str, temperature: float = 0.0) -> dict:
    """
    Best-effort JSON output from Gemini.
    - Uses response_mime_type on google-genai.
    - Still safely parses and never throws JSONDecodeError.
    """
    txt = ""

    if _SDK == "google-genai":
        resp = client.models.generate_content(
            model=MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        txt = (resp.text or "").strip()

    else:
        # google-generativeai: ask for JSON in prompt (best-effort)
        # (Keeping your original behavior, but safe-parse)
        resp = genai.GenerativeModel(
            model_name=MODEL,
            system_instruction=system,
        ).generate_content(
            user,
            generation_config={"temperature": temperature},
        )
        txt = (getattr(resp, "text", "") or "").strip()

    return _safe_json_load(txt)
