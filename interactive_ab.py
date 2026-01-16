# interactive_ab.py
from answerer import baseline_answer, refined_answer

def guess_task_tag(q: str) -> str:
    ql = q.lower()
    if "summarize" in ql or "summary" in ql:
        return "summary"
    if "sql" in ql or "query" in ql or "join" in ql or "select" in ql:
        return "coding_sql"
    if "code" in ql or "python" in ql or "java" in ql or "error" in ql:
        return "coding"
    return "explain"

def main():
    print("PEISR Interactive A/B Tester")
    print("Type a messy query. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        tag = guess_task_tag(q)

        print("\n[1] Baseline (A): Raw query → LLM")
        a = baseline_answer(q)
        print(a)

        print("\n[2] Proposed (B): Self-Refine → LLM")
        refined_prompt, b, trace = refined_answer(q, task_tag=tag, max_rounds=2)

        print("\nRefined Prompt:")
        print(refined_prompt)

        print("\nAnswer (B):")
        print(b)

        # Optional: uncomment if you want to see the full trace each time
        # print("\nTrace:", trace)

        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
