import sqlite3
from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Tuple

# Single SQLite file for the whole app
DB_PATH = "peisr_runs.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,

        task_tag TEXT,
        query TEXT,
        refined_prompt TEXT,

        answer_a TEXT,
        answer_b TEXT,

        -- Per-metric scores (Baseline A)
        intent_a INTEGER,
        clarity_a INTEGER,
        structure_a INTEGER,
        safety_a INTEGER,

        -- Per-metric scores (Refined B)
        intent_b INTEGER,
        clarity_b INTEGER,
        structure_b INTEGER,
        safety_b INTEGER,

        score_a INTEGER,
        score_b INTEGER,

        winner TEXT,
        judge_json TEXT
    )
    """)

    # New: human-rated comparisons (replaces Excel logging)
    # NOTE: SQLite has no native JSON type -> we store JSON blobs as TEXT.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS comparisons (
        comparison_id TEXT PRIMARY KEY,
        run_id TEXT,
        ts TEXT,

        session_id TEXT,
        human_rater TEXT,
        user_tag TEXT,

        variant TEXT,
        temp_mode TEXT,
        threshold_mode TEXT,
        model_mode TEXT,

        user_input TEXT,
        route_predicted TEXT,
        temperature_used REAL,
        rewrite_threshold_used INTEGER,
        rewritten INTEGER,

        original_prompt TEXT,
        original_response TEXT,
        original_prompt_critique_json TEXT,
        original_prompt_heuristic_json TEXT,

        enhanced_prompt TEXT,
        enhanced_response TEXT,
        enhanced_prompt_critique_json TEXT,
        enhanced_prompt_heuristic_json TEXT,

        response_llm_judge_json TEXT,
        response_heuristic_judge_json TEXT,

        human_score_original INTEGER,
        human_score_enhanced INTEGER,
        human_pick TEXT,
        human_notes TEXT
    )
    """)

    # --- lightweight migration for older DBs ---
    cur.execute("PRAGMA table_info(comparisons)")
    existing_cols = {row[1] for row in cur.fetchall()}
    desired_cols = {
        "run_id": "TEXT",
        "session_id": "TEXT",
        "user_tag": "TEXT",
        "variant": "TEXT",
        "temp_mode": "TEXT",
        "threshold_mode": "TEXT",
        "model_mode": "TEXT",
        "original_prompt_heuristic_json": "TEXT",
        "enhanced_prompt_heuristic_json": "TEXT",
        "response_llm_judge_json": "TEXT",
        "response_heuristic_judge_json": "TEXT",
    }
    for col, col_type in desired_cols.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE comparisons ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()


def save_comparison(
    *,
    comparison_id: str,
    run_id: str,
    session_id: str,
    human_rater: str,
    user_tag: str,
    variant: str,
    temp_mode: str,
    threshold_mode: str,
    model_mode: str,
    user_input: str,
    route_predicted: str,
    temperature_used: float,
    rewrite_threshold_used: int,
    rewritten: bool,
    original_prompt: str,
    original_response: str,
    original_prompt_critique: Dict[str, Any],
    original_prompt_heuristic: Dict[str, Any],
    enhanced_prompt: str,
    enhanced_response: str,
    enhanced_prompt_critique: Dict[str, Any],
    enhanced_prompt_heuristic: Dict[str, Any],
    response_llm_judge: Dict[str, Any],
    response_heuristic_judge: Dict[str, Any],
    human_score_original: int,
    human_score_enhanced: int,
    human_pick: str,
    human_notes: str,
):
    """Persist one human-rated A/B comparison to SQLite."""

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO comparisons (
            comparison_id,
            run_id,
            ts,

            session_id,
            human_rater,
            user_tag,

            variant,
            temp_mode,
            threshold_mode,
            model_mode,

            user_input,
            route_predicted,
            temperature_used,
            rewrite_threshold_used,
            rewritten,

            original_prompt,
            original_response,
            original_prompt_critique_json,
            original_prompt_heuristic_json,

            enhanced_prompt,
            enhanced_response,
            enhanced_prompt_critique_json,
            enhanced_prompt_heuristic_json,

            response_llm_judge_json,
            response_heuristic_judge_json,

            human_score_original,
            human_score_enhanced,
            human_pick,
            human_notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            comparison_id,
            run_id,
            datetime.now().isoformat(timespec="seconds"),
            session_id,
            human_rater,
            user_tag,
            variant,
            temp_mode,
            threshold_mode,
            model_mode,
            user_input,
            route_predicted,
            float(temperature_used),
            int(rewrite_threshold_used),
            1 if rewritten else 0,
            original_prompt,
            original_response,
            json.dumps(original_prompt_critique or {}, ensure_ascii=False),
            json.dumps(original_prompt_heuristic or {}, ensure_ascii=False),
            enhanced_prompt,
            enhanced_response,
            json.dumps(enhanced_prompt_critique or {}, ensure_ascii=False),
            json.dumps(enhanced_prompt_heuristic or {}, ensure_ascii=False),
            json.dumps(response_llm_judge or {}, ensure_ascii=False),
            json.dumps(response_heuristic_judge or {}, ensure_ascii=False),
            int(human_score_original),
            int(human_score_enhanced),
            human_pick,
            human_notes or "",
        ),
    )

    conn.commit()
    conn.close()


def fetch_comparisons(limit: int = 200) -> List[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            comparison_id,
            ts,
            human_rater,
            route_predicted,
            temperature_used,
            rewrite_threshold_used,
            rewritten,
            human_score_original,
            human_score_enhanced,
            human_pick,
            user_input
        FROM comparisons
        ORDER BY ts DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()
    return rows


def save_run(
    task_tag: str,
    query: str,
    refined_prompt: str,
    answer_a: str,
    answer_b: str,
    judge_json: str,
    winner: str,
    score_a: int,
    score_b: int,
    intent_a: int,
    clarity_a: int,
    structure_a: int,
    safety_a: int,
    intent_b: int,
    clarity_b: int,
    structure_b: int,
    safety_b: int,
):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO runs (
        ts,
        task_tag,
        query,
        refined_prompt,
        answer_a,
        answer_b,
        intent_a,
        clarity_a,
        structure_a,
        safety_a,
        intent_b,
        clarity_b,
        structure_b,
        safety_b,
        score_a,
        score_b,
        winner,
        judge_json
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        task_tag,
        query,
        refined_prompt,
        answer_a,
        answer_b,
        intent_a,
        clarity_a,
        structure_a,
        safety_a,
        intent_b,
        clarity_b,
        structure_b,
        safety_b,
        score_a,
        score_b,
        winner,
        judge_json
    ))

    conn.commit()
    conn.close()


def fetch_runs(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    SELECT
        id,
        ts,
        task_tag,
        query,
        winner,
        score_a,
        score_b,
        intent_a,
        clarity_a,
        structure_a,
        safety_a,
        intent_b,
        clarity_b,
        structure_b,
        safety_b
    FROM runs
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    conn.close()
    return rows