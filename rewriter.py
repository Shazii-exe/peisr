from typing import Dict
from dataclasses import dataclass

from gemini_client import generate_text, generate_json
from prompts import CRITIQUE_SYSTEM, REVISE_SYSTEM, REWRITE_SYSTEM_FULL, REWRITE_SYSTEM_LIGHT
from intent_classifier import classify_intent


@dataclass
class CritiqueResult:
    scores: Dict[str, int]
    weakest: str
    edit: str
    reason: str


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _normalize_critique(data: dict) -> CritiqueResult:
    """
    Accepts whatever the model returned (possibly empty/partial)
    and returns a valid CritiqueResult with safe defaults.
    """
    if not isinstance(data, dict):
        data = {}

    raw_scores = data.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    # Normalize score values to ints
    scores: Dict[str, int] = {}
    for k, v in raw_scores.items():
        if isinstance(k, str):
            scores[k] = _safe_int(v, 0)

    # If scores missing, give a sane default schema
    if not scores:
        scores = {
            "clarity": 0,
            "specificity": 0,
            "constraints": 0,
            "context": 0,
            "format": 0,
        }

    weakest = data.get("weakest", "")
    if not isinstance(weakest, str):
        weakest = ""

    edit = data.get("edit", "")
    if not isinstance(edit, str) or not edit.strip():
        edit = "Make the prompt clearer and better structured while preserving intent."

    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = ""

    # If weakest missing, infer from scores (lowest dimension)
    if not weakest.strip() and scores:
        weakest = min(scores, key=scores.get)

    return CritiqueResult(
        scores=scores,
        weakest=weakest.strip(),
        edit=edit.strip(),
        reason=reason.strip(),
    )


def critique_prompt(prompt: str) -> CritiqueResult:
    user = f"Original prompt:\n{prompt}\n"

    data = generate_json(
        system=CRITIQUE_SYSTEM,
        user=user,
        temperature=0.0,
    )

    # ✅ Never crash even if Gemini returns {} / partial JSON / text-wrapped JSON
    return _normalize_critique(data)


def rewrite_once(original_prompt: str, critique: CritiqueResult) -> str:
    # Safety guard
    if "json" in (critique.edit or "").lower():
        edit = "Make the prompt clearer and better structured while preserving intent."
    else:
        edit = critique.edit or "Make the prompt clearer and better structured while preserving intent."

    user = (
        f"Original prompt:\n{original_prompt}\n\n"
        f"Critic suggestion:\n{edit}\n"
    )

    return generate_text(
        system=REVISE_SYSTEM,
        user=user,
        temperature=0.2,
    )


def rewrite_prompt(original_prompt: str, mode: str = "full") -> str:
    """Option B (conditional editor prompt): rewrite with explicit SOCIAL passthrough."""
    route = classify_intent(original_prompt, allow_llm=False).route
    if route == "SOCIAL":
        return original_prompt.strip()

    system = REWRITE_SYSTEM_FULL if mode == "full" else REWRITE_SYSTEM_LIGHT
    return generate_text(system=system, user=original_prompt, temperature=0.2).strip()


def self_refine_rewrite(
    original_prompt: str,
    *,
    rewrite_threshold: int = 15,
    max_rounds: int = 2,
    mode: str = "full",
):
    """
    Self-refine loop:

    1) Critique current prompt
    2) If score < threshold, rewrite once based on critique
    3) Repeat up to max_rounds
    """
    trace = []
    current = original_prompt.strip()

    for i in range(max_rounds):
        crit = critique_prompt(current)
        total = sum((crit.scores or {}).values())

        trace.append({
            "round": i + 1,
            "prompt": current,
            "scores": crit.scores,
            "total": total,
            "weakest": crit.weakest,
            "edit": crit.edit,
            "reason": crit.reason,
        })

        if total >= rewrite_threshold:
            break

        # Option B: use conditional rewrite system first
        candidate = rewrite_prompt(current, mode=mode)

        # Apply critic edit via REVISE_SYSTEM
        current = rewrite_once(candidate, crit).strip()

    return current, trace
