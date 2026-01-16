from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from gemini_client import generate_text
from intent_classifier import IntentResult, classify_intent, choose_temperature, choose_rewrite_threshold
from prompts import ANSWER_SYSTEM_BY_ROUTE
from rewriter import critique_prompt, self_refine_rewrite


@dataclass
class PipelineOutput:
    route: str
    enhance_mode: str  # NONE|LIGHT|FULL
    temperature_used: float
    original_prompt: str
    enhanced_prompt: str
    answer: str
    # dataclasses require non-default fields before default fields
    rewrite_threshold_used: Optional[int] = None
    critique_original: Optional[Dict[str, Any]] = None
    critique_final: Optional[Dict[str, Any]] = None
    trace: Optional[List[Dict[str, Any]]] = None


def baseline_answer(query: str, *, temperature: float = 0.4) -> str:
    """Baseline: no rewrite, single generic system prompt."""
    system = """You are a helpful assistant.
Answer the user's request as best as possible.
If critical information is missing, ask minimal clarifying questions or state brief assumptions.
Keep the response directly useful and not overly long."""
    return generate_text(system=system, user=query, temperature=temperature)


def run_pipeline(
    query: str,
    *,
    variant: str = "ABC",
    temperature: Optional[float] = None,
    temp_mode: str = "auto",  # auto|fixed
    rewrite_threshold: Optional[int] = None,
    max_rounds: int = 2,
) -> PipelineOutput:
    """Run the prompt-enhancement pipeline.

    Variants:
      - BASELINE: no routing, no rewrite (use baseline_answer)
      - A: intent gate only (route + decide rewrite NONE/LIGHT/FULL)
      - B: conditional editor prompt (SOCIAL passthrough) + critique loop
      - C: multi-prompt answering by route
      - ABC: A+B+C combined (recommended)
    """

    q = (query or "").strip()
    intent: IntentResult = classify_intent(q, allow_llm=True)
    route = intent.route

    # Temperature selection
    if temp_mode == "auto":
        temperature_used = choose_temperature(route)
    else:
        temperature_used = float(temperature if temperature is not None else 0.4)

    # Rewrite threshold selection (auto if None)
    threshold_used = choose_rewrite_threshold(intent) if rewrite_threshold is None else int(rewrite_threshold)
    if variant.upper() == "BASELINE":
        ans = baseline_answer(q, temperature=temperature_used)
        return PipelineOutput(
            route="QA",
            enhance_mode="NONE",
            temperature_used=temperature_used,
            rewrite_threshold_used=None,
            original_prompt=q,
            enhanced_prompt=q,
            answer=ans,
        )

    # Option A: intent-aware gate → decide if we should rewrite
    # SOCIAL: always bypass
    if route == "SOCIAL":
        enhance_mode = "NONE"
    else:
        # Light rewrite for QA; Full for TASK/TECH/CREATIVE
        enhance_mode = "LIGHT" if route in {"QA"} else "FULL"

    # If variant doesn't include A, treat as always FULL (except SOCIAL)
    if "A" not in variant.upper() and route != "SOCIAL":
        enhance_mode = "FULL"

    # Option B: conditional editor + self-refine loop
    critique_orig = critique_prompt(q)
    final_prompt = q
    trace = None

    if enhance_mode != "NONE":
        mode = "light" if enhance_mode == "LIGHT" else "full"
        final_prompt, trace = self_refine_rewrite(
            q,
            rewrite_threshold=threshold_used,
            max_rounds=max_rounds,
            mode=mode,
        )

    critique_final = critique_prompt(final_prompt)

    # Option C: route-specific answerer
    if "C" in variant.upper():
        system = ANSWER_SYSTEM_BY_ROUTE[route]
    else:
        system = """You are a helpful assistant.
Answer the user's request as best as possible.
If critical information is missing, ask minimal clarifying questions.
Keep it concise."""

    answer = generate_text(system=system, user=final_prompt, temperature=temperature_used)

    return PipelineOutput(
        route=route,
        enhance_mode=enhance_mode,
        temperature_used=temperature_used,
        rewrite_threshold_used=threshold_used,
        original_prompt=q,
        enhanced_prompt=final_prompt,
        answer=answer,
        critique_original={"scores": critique_orig.scores, "total": sum(critique_orig.scores.values()), "weakest": critique_orig.weakest, "edit": critique_orig.edit, "reason": critique_orig.reason},
        critique_final={"scores": critique_final.scores, "total": sum(critique_final.scores.values()), "weakest": critique_final.weakest, "edit": critique_final.edit, "reason": critique_final.reason},
        trace=trace,
    )


def refined_answer(
    query: str,
    *,
    task_tag: str = "auto",
    max_rounds: int = 2,
    rewrite_threshold: Optional[int] = None,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Backward compatible API for interactive_ab scripts.

    Returns: (refined_prompt, answer, trace)
    """
    # If rewrite_threshold is None -> auto threshold selection inside run_pipeline.
    out = run_pipeline(
        query,
        variant="ABC",
        temp_mode="auto",
        rewrite_threshold=rewrite_threshold,
        max_rounds=max_rounds,
    )
    return out.enhanced_prompt, out.answer, (out.trace or [])


def gated_answer(
    messy_query: str,
    temperature: float = 0.4,
    rewrite_threshold: Optional[int] = None,
):
    """Streamlit app compatibility.

    Keeps the old return shape but now uses routing + option A/B/C.
    """
    # Original (no rewrite) answer
    base = run_pipeline(messy_query, variant="BASELINE", temp_mode="fixed", temperature=temperature)

    # Enhanced answer (ABC)
    enhanced = run_pipeline(
        messy_query,
        variant="ABC",
        temp_mode="fixed",  # keep app slider meaningful
        temperature=temperature,
        rewrite_threshold=rewrite_threshold,
        max_rounds=2,
    )

    rewritten = enhanced.enhanced_prompt.strip() != messy_query.strip()

    orig_scores = (enhanced.critique_original or {}).get("scores", {})
    rew_scores = (enhanced.critique_final or {}).get("scores", {})

    return (
        base.answer if rewritten else None,
        enhanced.answer,
        orig_scores,
        rew_scores if rewritten else None,
        rewritten,
        enhanced.enhanced_prompt if rewritten else None,
        # For UI details we return "critique-like" objects
        type("Crit", (), {
            "scores": orig_scores,
            "weakest": (enhanced.critique_original or {}).get("weakest", ""),
            "edit": (enhanced.critique_original or {}).get("edit", ""),
            "reason": (enhanced.critique_original or {}).get("reason", ""),
        })(),
        type("Crit", (), {
            "scores": rew_scores,
            "weakest": (enhanced.critique_final or {}).get("weakest", ""),
            "edit": (enhanced.critique_final or {}).get("edit", ""),
            "reason": (enhanced.critique_final or {}).get("reason", ""),
        })() if rewritten else None,
    )
