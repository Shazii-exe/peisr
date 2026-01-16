# interactive_ab_judged.py
from answerer import baseline_answer, refined_answer
from judge import judge_pair, total_score

def guess_task_tag(q: str) -> str:
    ql = q.lower()
    if "summarize" in ql or "summary" in ql:
        return "summary"
    if "sql" in ql or "join" in ql or "select" in ql:
        return "coding_sql"
    if "code" in ql or "python" in ql or "java" in ql or "error" in ql:
        return "coding"
    return "explain"

def main():
    print("PEISR Interactive A/B + Judge")
    print("Type a messy query. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        tag = guess_task_tag(q)

        # A: baseline
        a = baseline_answer(q)

        # B: refined
        refined_prompt, b, _trace = refined_answer(q, task_tag=tag, max_rounds=2)

        print("\n--- BASELINE ANSWER (A) ---\n", a)
        print("\n--- REFINED PROMPT ---\n", refined_prompt)
        print("\n--- REFINED ANSWER (B) ---\n", b)

        # Judge (blind): X=a, Y=b
        jr = judge_pair(q, a, b)
        sx, sy = total_score(jr.X), total_score(jr.Y)

        print("\n====== RUBRIC SCORES (1–5 each) ======")
        print(f"X (Baseline A) total={sx} | intent={jr.X['intent']} clarity={jr.X['clarity']} structure={jr.X['structure']} safety={jr.X['safety']}")
        print(f"Y (Refined  B) total={sy} | intent={jr.Y['intent']} clarity={jr.Y['clarity']} structure={jr.Y['structure']} safety={jr.Y['safety']}")
        print("Winner:", jr.winner, "| Reason:", jr.reason)
        print("Notes X:", jr.X.get("notes", ""))
        print("Notes Y:", jr.Y.get("notes", ""))

        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
