from answerer import baseline_answer, refined_answer

q = "moon explain 5 yr kid fast"

a = baseline_answer(q)
rp, b, trace = refined_answer(q, task_tag="explain")

print("\n--- BASELINE ANSWER (A) ---\n", a)
print("\n--- REFINED PROMPT ---\n", rp)
print("\n--- REFINED ANSWER (B) ---\n", b)
print("\n--- TRACE ---\n", trace)
