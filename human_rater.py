from __future__ import annotations

from typing import List

from experiment_runner import run_and_log
from excel_logger import set_human_rating


def main():
    print("PEISR Human Rating (writes into Excel)")
    q = input("User prompt: ").strip()
    if not q:
        print("Empty prompt. Exiting.")
        return

    name = input("Your name (rater id): ").strip() or "human"

    run_id, results = run_and_log(q, test_id="human_cli")
    print(f"\nRun logged: {run_id}")

    variants: List[str] = [v for v in results.keys() if v != "BASELINE"]
    print("\nRate each variant 1-10 (overall). Optionally add notes.")

    for v in variants:
        print("\n" + "=" * 70)
        print("Variant:", v)
        print("Enhanced prompt:\n", results[v]["prompt"])
        print("Response:\n", results[v]["answer"])
        score = input("Score 1-10 (blank to skip): ").strip()
        if not score:
            continue
        notes = input("Notes (optional): ").strip()
        set_human_rating(
            run_id=run_id,
            variant=v,
            human_rater=name,
            human_score_overall=int(score),
            human_pick=v,
            human_notes=notes,
        )

    print("\nDone. Ratings saved into peisr_runs.xlsx")


if __name__ == "__main__":
    main()
