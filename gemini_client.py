# gemini_client.py

import os
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
if _SDK == "google-genai":
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
else:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    client = None

MODEL = "gemini-2.0-flash"

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
    Forces JSON output from Gemini.
    """
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
        # google-generativeai: ask for JSON in the prompt (best-effort)
        resp = genai.GenerativeModel(
            model_name=MODEL,
            system_instruction=system,
        ).generate_content(
            user,
            generation_config={"temperature": temperature},
        )
        txt = (getattr(resp, "text", "") or "").strip()
    # Gemini returns valid JSON text in resp.text when response_mime_type is set.
    import json
    return json.loads(txt)
