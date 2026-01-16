import uuid
import time
import hashlib
import streamlit as st
import pandas as pd

import os

from answerer import run_pipeline
from judge import judge_pair, heuristic_judge_pair, heuristic_prompt_critique
from db import init_db, save_comparison, fetch_comparisons, DB_PATH
from intent_classifier import classify_intent, choose_temperature, choose_rewrite_threshold


def _toggle(label: str, value: bool = False, key: str = "") -> bool:
    """Use st.toggle when available, otherwise fall back to st.checkbox."""
    if hasattr(st, "toggle"):
        return st.toggle(label, value=value, key=key)
    return st.checkbox(label, value=value, key=key)


# ------------------ PAGE SETUP ------------------

st.set_page_config(page_title="PEISR — Prompt Gating", layout="wide")
st.title("PEISR — Prompt Critique & Gated Rewrite")

# Initialize SQLite (our logging store)
init_db()

# Per-browser session id (stored in Streamlit session_state)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Session identifiers
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "saved_ids" not in st.session_state:
    st.session_state["saved_ids"] = set()
if "last_submit_ts" not in st.session_state:
    st.session_state["last_submit_ts"] = 0.0

# ------------------ SIDEBAR SETTINGS ------------------
with st.sidebar:
    st.header("Settings")

    rater_name = st.text_input(
        "Your name (for logging)",
        value="",
        placeholder="e.g., Mahek",
        key="rater_name",
    )
    user_tag = st.text_input(
        "User tag (optional)",
        value="",
        placeholder="e.g., Tuba-UK",
        key="user_tag",
    )

    # Admin mode so judge JSON doesn't influence public raters
    admin_secret = os.getenv("ADMIN_KEY", "").strip()

# Only show the input if an ADMIN_KEY exists in the environment
admin_key = ""
if admin_secret:
    admin_key = st.text_input("Admin key (optional)", value="", type="password", key="admin_key")

is_admin = bool(admin_secret) and (admin_key == admin_secret)

show_judge_json = False
if is_admin:
    show_judge_json = _toggle("Show judge JSON (admin)", value=False, key="show_judge_json")
    st.caption("✅ Admin mode enabled")
else:
    st.caption("Judge JSON hidden (public mode)")


    auto_temp = _toggle("Auto temperature (by intent)", value=True, key="auto_temp")
    temperature = st.slider(
        "Answer temperature (used if auto OFF)",
        0.0,
        1.0,
        0.4,
        0.05,
        disabled=auto_temp,
    )

    auto_threshold = _toggle("Auto rewrite threshold (by intent)", value=True, key="auto_threshold")
    rewrite_threshold = st.slider(
        "Rewrite threshold (prompt total)",
        4, 20, 15,
        disabled=auto_threshold
    )

    st.divider()
    st.caption("SQLite logging")
    st.write(f"DB: `{DB_PATH}`")
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            st.download_button(
                "Download SQLite DB",
                data=f.read(),
                file_name=DB_PATH,
                mime="application/x-sqlite3",
                use_container_width=True,
            )
    else:
        st.info("No DB file yet. Submit a rating once to create it.")

    # Quick view of latest ratings
    st.caption("Latest ratings (preview)")
    try:
        rows = fetch_comparisons(limit=10)
        if rows:
            df = pd.DataFrame(
                rows,
                columns=[
                    "comparison_id",
                    "ts",
                    "rater",
                    "route",
                    "temp",
                    "threshold",
                    "rewritten",
                    "score_A",
                    "score_B",
                    "pick",
                    "user_input",
                ],
            )
            st.dataframe(df, use_container_width=True, height=240)
        else:
            st.caption("(no ratings yet)")
    except Exception:
        st.caption("(preview unavailable)")

# ------------------ INPUT ------------------

st.subheader("Enter a prompt")
query = st.text_area(
    "User prompt",
    height=140,
    placeholder="Type a messy or unclear prompt...",
)

run = st.button("Run", type="primary", use_container_width=True)

# ------------------ RUN PIPELINE ------------------
if run:
    q = (query or "").strip()
    if not q:
        st.warning("Please enter a prompt.")
        st.stop()

    # Intent for preview line
    intent = classify_intent(q, allow_llm=False)
    route = intent.route

    # Temperature
    if auto_temp:
        temperature_used = choose_temperature(route)
    else:
        temperature_used = float(temperature)

    # Threshold
    threshold_used = choose_rewrite_threshold(intent) if auto_threshold else int(rewrite_threshold)

    # Compute ORIGINAL side (baseline) and ENHANCED side (ABC)
    base = run_pipeline(q, variant="BASELINE", temp_mode="fixed", temperature=temperature_used)
    enhanced = run_pipeline(
        q,
        variant="ABC",
        temp_mode="fixed",
        temperature=temperature_used,
        rewrite_threshold=None if auto_threshold else threshold_used,
        max_rounds=2,
    )

    rewritten = enhanced.enhanced_prompt.strip() != q

    # --- Computer judges (LLM + heuristic) ---
    # Prompt judge: LLM critique is already produced by pipeline (critique_original / critique_final)
    llm_prompt_original = enhanced.critique_original or {}
    llm_prompt_enhanced = enhanced.critique_final or llm_prompt_original or {}

    heur_prompt_original = heuristic_prompt_critique(q)
    heur_prompt_enhanced = heuristic_prompt_critique(enhanced.enhanced_prompt)

    # Response judge: LLM pairwise judge + heuristic pairwise judge
    try:
        llm_resp_judge = judge_pair(q, base.answer, enhanced.answer)
        llm_resp_judge_json = {"X": llm_resp_judge.X, "Y": llm_resp_judge.Y, "winner": llm_resp_judge.winner, "reason": llm_resp_judge.reason, "judge_type": "llm"}
    except Exception as e:
        llm_resp_judge_json = {"error": str(e), "judge_type": "llm"}

    heur_resp_judge_json = heuristic_judge_pair(q, base.answer, enhanced.answer)

    # Stable run_id for dedupe (same run -> same id)
    run_id_src = "|".join([
        q,
        enhanced.route,
        f"{enhanced.temperature_used:.3f}",
        str(enhanced.rewrite_threshold_used),
        enhanced.enhanced_prompt,
        base.answer,
        enhanced.answer,
    ])
    run_id = hashlib.sha256(run_id_src.encode("utf-8")).hexdigest()

    # Store everything in session for rating submit
    st.session_state["last_result"] = {
        "comparison_id": str(uuid.uuid4()),
        "run_id": run_id,
        "session_id": st.session_state["session_id"],
        "user_tag": (st.session_state.get("user_tag", "") or "").strip(),
        "variant": "ABC",
        "temp_mode": "auto" if auto_temp else "fixed",
        "threshold_mode": "auto" if auto_threshold else "fixed",
        "model_mode": "gemini" if bool(os.getenv("GEMINI_API_KEY")) else "no_key",
        "user_input": q,
        "route": enhanced.route,
        "temperature_used": enhanced.temperature_used,
        "threshold_used": enhanced.rewrite_threshold_used,
        "rewritten": rewritten,
        "original_prompt": q,
        "original_response": base.answer,
        "original_critique": llm_prompt_original,
        "original_heur": heur_prompt_original,
        "enhanced_prompt": enhanced.enhanced_prompt,
        "enhanced_response": enhanced.answer,
        "enhanced_critique": llm_prompt_enhanced,
        "enhanced_heur": heur_prompt_enhanced,
        "resp_llm_judge": llm_resp_judge_json,
        "resp_heur_judge": heur_resp_judge_json,
    }

# ------------------ DISPLAY (if we have last result) ------------------
res = st.session_state.get("last_result")
if res:
    st.caption(
        f"Route: **{res['route']}** | Temp used: **{res['temperature_used']:.2f}** | "
        f"Threshold used: **{res['threshold_used']}** | Rewritten: **{res['rewritten']}**"
    )

    # -------- Two-column comparison --------
    st.markdown("## Side-by-side comparison")
    colL, colR = st.columns(2, gap="large")

    with colL:
        st.markdown("### A) Original prompt")
        st.markdown("**Prompt**")
        st.code(res["original_prompt"])
        st.markdown("**Response**")
        st.write(res["original_response"])

        # Judge JSON is intentionally hidden from public raters to avoid bias.

    with colR:
        st.markdown("### B) Rewritten/enhanced prompt")
        st.markdown("**Prompt**")
        st.code(res["enhanced_prompt"])
        st.markdown("**Response**")
        st.write(res["enhanced_response"])

        # Judge JSON is intentionally hidden from public raters to avoid bias.

    # Admin-only: show machine judge JSON (prompt + response)
    if show_judge_json:
        with st.expander("Judge details (admin)", expanded=False):
            tabs = st.tabs(["Prompt judge", "Response judge"])
            with tabs[0]:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original prompt — LLM critique")
                    st.json(res["original_critique"])
                    st.caption("Original prompt — Heuristic critique")
                    st.json(res["original_heur"])
                with c2:
                    st.caption("Enhanced prompt — LLM critique")
                    st.json(res["enhanced_critique"])
                    st.caption("Enhanced prompt — Heuristic critique")
                    st.json(res["enhanced_heur"])
            with tabs[1]:
                st.caption("LLM response judge (pairwise, X=Original, Y=Enhanced)")
                st.json(res["resp_llm_judge"])
                st.caption("Heuristic response judge (pairwise, X=Original, Y=Enhanced)")
                st.json(res["resp_heur_judge"])

    st.divider()

    # -------- Human rating --------
    st.markdown("## Human rating")
    st.caption("Rate both responses, then choose which one is better. This will be logged to SQLite.")

    rcol1, rcol2 = st.columns(2, gap="large")
    with rcol1:
        score_original = st.radio(
            "Score for **Original** response",
            options=[1, 2, 3, 4, 5],
            horizontal=True,
            index=2,
            key="score_original",
        )
    with rcol2:
        score_enhanced = st.radio(
            "Score for **Enhanced** response",
            options=[1, 2, 3, 4, 5],
            horizontal=True,
            index=2,
            key="score_enhanced",
        )

    pick = st.radio(
        "Which response is better overall?",
        options=["ORIGINAL", "ENHANCED", "TIE"],
        horizontal=True,
        key="pick",
    )
    notes = st.text_area("Optional notes (why?)", height=80, key="notes")

    submit = st.button("Submit rating (log to SQLite)", use_container_width=True)

    if submit:
        # basic anti-spam cooldown
        now = time.time()
        if now - float(st.session_state.get("last_submit_ts", 0.0)) < 3.0:
            st.warning("Please wait a couple seconds before submitting again.")
            st.stop()
        if res["comparison_id"] in st.session_state.get("saved_ids", set()):
            st.info("This rating was already saved for the current run.")
            st.stop()

        human_rater = st.session_state.get("rater_name", "").strip() or "anonymous"

        try:
            save_comparison(
                comparison_id=res["comparison_id"],
                run_id=res["run_id"],
                session_id=res["session_id"],
                human_rater=human_rater,
                user_tag=res.get("user_tag", ""),
                variant=res.get("variant", "ABC"),
                temp_mode=res.get("temp_mode", "fixed"),
                threshold_mode=res.get("threshold_mode", "fixed"),
                model_mode=res.get("model_mode", "gemini"),
                user_input=res["user_input"],
                route_predicted=res["route"],
                temperature_used=res["temperature_used"],
                rewrite_threshold_used=res["threshold_used"],
                rewritten=bool(res["rewritten"]),
                original_prompt=res["original_prompt"],
                original_response=res["original_response"],
                original_prompt_critique=res["original_critique"],
                original_prompt_heuristic=res["original_heur"],
                enhanced_prompt=res["enhanced_prompt"],
                enhanced_response=res["enhanced_response"],
                enhanced_prompt_critique=res["enhanced_critique"],
                enhanced_prompt_heuristic=res["enhanced_heur"],
                response_llm_judge=res["resp_llm_judge"],
                response_heuristic_judge=res["resp_heur_judge"],
                human_score_original=int(score_original),
                human_score_enhanced=int(score_enhanced),
                human_pick=pick,
                human_notes=notes,
            )
            st.session_state["last_submit_ts"] = now
            st.session_state["saved_ids"].add(res["comparison_id"])
            st.success("Saved ✅ (logged to SQLite)")
        except Exception as e:
            st.error(f"Could not write to SQLite: {e}")
