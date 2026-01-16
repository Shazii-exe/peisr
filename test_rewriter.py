from rewriter import self_refine_rewrite

q = "moon explain 5 yr kid fast"
final_prompt, trace = self_refine_rewrite(q, task_tag="explain", max_rounds=2)

print("FINAL PROMPT:\n", final_prompt)
print("\nTRACE:\n", trace)
