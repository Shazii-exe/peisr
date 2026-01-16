# PEISR – Prompt Enhancement & Intelligent Self-Refinement

PEISR is a research-oriented Streamlit application designed to study and evaluate
**prompt enhancement techniques** using rewriting, judging, and human-in-the-loop feedback.

The system enables controlled A/B comparisons between original and enhanced prompts,
while logging model behavior and human ratings for analysis.

---

## ✨ Features

- Prompt rewriting using LLMs (Gemini)
- Automated judging of responses
- Human rating interface (blind to judge output)
- Admin-only judge JSON visibility
- A/B testing support
- SQLite logging for experiments
- Streamlit-based interactive UI

---

## 🧠 Research Motivation

Prompt engineering often lacks structured evaluation.
PEISR introduces:
- Separation between **public raters** and **system judges**
- Controlled visibility to prevent bias
- Persistent logging for experimental analysis

This makes PEISR suitable for:
- Academic research
- Prompt evaluation studies
- Early-stage benchmarking of LLM behaviors

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- Google Gemini API
- SQLite
- Prompt engineering & evaluation logic

---

## 📂 Project Structure

```text
peisr/
├── app.py                  # Streamlit app entry point
├── rewriter.py             # Prompt rewriting logic
├── judge.py                # Automated judging logic
├── prompts.py              # Prompt templates
├── gemini_client.py        # Gemini API wrapper
├── db.py                   # SQLite logging
├── experiment_runner.py    # Offline experiments
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
