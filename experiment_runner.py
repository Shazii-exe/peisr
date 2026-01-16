from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from answerer import run_pipeline
from excel_logger import append_run
from judge import judge_pair, total_score


DEFAULT_VARIANTS = ["BASELINE", "A", "B", "C", "ABC"]


def run_and_log(
    user_input: str,
    *,
    test_id: str = "adhoc",
    variants: Optional[List[str]] = None,
    temp_mode: str = "auto",
    fixed_temperature: float = 0.4,
    rewrite_threshold: int | None = None,
    max_rounds: int = 2,
    xlsx_path: str = "peisr_runs.xlsx",
):
    """Runs variants, judges each vs BASELINE, and appends rows to an Excel sheet.

    This implements your Step-2 evaluation loop:
      - computer rating (LLM judge)
      - stored in Excel for later tally/pivots

    Notes:
      - We store one row per (variant). The judge score fields correspond to the
        VARIANT response (Y) when compared to BASELINE (X).
      - BASELINE itself is also logged (without judge fields).
    """

    run_id = uuid.uuid4().hex[:10]
    variants = variants or DEFAULT_VARIANTS

    # Always compute baseline first (so the comparison is stable)
    baseline = run_pipeline(
        user_input,
        variant="BASELINE",
        temp_mode=("fixed" if temp_mode == "fixed" else "auto"),
        temperature=(fixed_temperature if temp_mode == "fixed" else None),
        rewrite_threshold=rewrite_threshold,
        max_rounds=max_rounds,
    )

    append_run(
        path=xlsx_path,
        run_id=run_id,
        test_id=test_id,
        variant="BASELINE",
        user_input=user_input,
        route_predicted=baseline.route,
        enhance_mode=baseline.enhance_mode,
        temperature_used=baseline.temperature_used,
        rewrite_threshold_used=baseline.rewrite_threshold_used,
        original_prompt_total=None,
        final_prompt_total=None,
        enhanced_prompt=baseline.enhanced_prompt,
        response_text=baseline.answer,
        judge=None,
    )

    results: Dict[str, Dict] = {"BASELINE": {"prompt": baseline.enhanced_prompt, "answer": baseline.answer}}

    for v in variants:
        if v.upper() == "BASELINE":
            continue

        out = run_pipeline(
            user_input,
            variant=v,
            temp_mode=("fixed" if temp_mode == "fixed" else "auto"),
            temperature=(fixed_temperature if temp_mode == "fixed" else None),
            rewrite_threshold=rewrite_threshold,
            max_rounds=max_rounds,
        )

        # Blind judge: X=baseline, Y=variant
        jr = judge_pair(user_input, baseline.answer, out.answer)
        score_y = total_score(jr.Y)
        judge_block = {
            "overall": score_y,
            "intent": int(jr.Y["intent"]),
            "clarity": int(jr.Y["clarity"]),
            "structure": int(jr.Y["structure"]),
            "safety": int(jr.Y["safety"]),
            "notes": jr.Y.get("notes", ""),
            "winner": jr.winner,
            "reason": jr.reason,
        }

        append_run(
            path=xlsx_path,
            run_id=run_id,
            test_id=test_id,
            variant=v,
            user_input=user_input,
            route_predicted=out.route,
            enhance_mode=out.enhance_mode,
            temperature_used=out.temperature_used,
            rewrite_threshold_used=out.rewrite_threshold_used,
            original_prompt_total=(out.critique_original or {}).get("total"),
            final_prompt_total=(out.critique_final or {}).get("total"),
            enhanced_prompt=out.enhanced_prompt,
            response_text=out.answer,
            judge=judge_block,
        )

        results[v] = {
            "prompt": out.enhanced_prompt,
            "answer": out.answer,
            "judge": judge_block,
        }

    return run_id, results


if __name__ == "__main__":
    # Quick CLI smoke test
    q = input("User prompt: ").strip()
    rid, res = run_and_log(q, test_id="cli")
    print("Logged run_id:", rid)
    for k, v in res.items():
        if k == "BASELINE":
            continue
        print(f"\n[{k}] judge_overall={v['judge']['overall']} winner={v['judge']['winner']} reason={v['judge']['reason']}")