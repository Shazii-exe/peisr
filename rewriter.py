from typing import Dict, Any, Optional, Tuple
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


def critique_prompt(prompt: str) -> CritiqueResult:
    user = f"Original prompt:\n{prompt}\n"

    data = generate_json(
        system=CRITIQUE_SYSTEM,
        user=user,
        temperature=0.0,
    )

    return CritiqueResult(
        scores=data["scores"],
        weakest=data["weakest"],
        edit=data["edit"],
        reason=data["reason"],
    )


def rewrite_once(original_prompt: str, critique: CritiqueResult) -> str:
    # Safety guard
    if "json" in critique.edit.lower():
        edit = "Make the prompt clearer and better structured while preserving intent."
    else:
        edit = critique.edit

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
    """Self-refine loop used by older scripts/tests.

    1) Critique current prompt
    2) If score < threshold, rewrite once based on critique
    3) Repeat up to max_rounds
    """
    trace = []
    current = original_prompt.strip()
    for i in range(max_rounds):
        crit = critique_prompt(current)
        total = sum(crit.scores.values())
        trace.append({"round": i + 1, "prompt": current, "scores": crit.scores, "total": total, "weakest": crit.weakest, "edit": crit.edit, "reason": crit.reason})
        if total >= rewrite_threshold:
            break
        # Option B: use conditional rewrite system first, then apply critique edit
        # (prevents SOCIAL over-rewrite and reduces big tone shifts)
        candidate = rewrite_prompt(current, mode=mode)
        # If still low, apply critic edit via REVISE_SYSTEM
        current = rewrite_once(candidate, crit).strip()
    return current, trace
