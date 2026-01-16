import os
import time
import json
import sqlite3
from typing import Any, Dict, List, Tuple, Optional

DB_PATH = "peisr_runs.db"
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps({"_repr": repr(value)}, ensure_ascii=False)


def _is_postgres() -> bool:
    return bool(DATABASE_URL)


def _connect_sqlite():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def _connect_postgres():
    import psycopg2  # from psycopg2-binary
    return psycopg2.connect(DATABASE_URL)


def init_db() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS comparisons (
        comparison_id TEXT PRIMARY KEY,
        run_id TEXT,
        session_id TEXT,
        ts DOUBLE PRECISION,
        human_rater TEXT,
        user_tag TEXT,
        variant TEXT,
        temp_mode TEXT,
        threshold_mode TEXT,
        model_mode TEXT,
        user_input TEXT,
        route_predicted TEXT,
        temperature_used DOUBLE PRECISION,
        rewrite_threshold_used INTEGER,
        rewritten INTEGER,
        original_prompt TEXT,
        original_response TEXT,
        original_prompt_critique TEXT,
        original_prompt_heuristic TEXT,
        enhanced_prompt TEXT,
        enhanced_response TEXT,
        enhanced_prompt_critique TEXT,
        enhanced_prompt_heuristic TEXT,
        response_llm_judge TEXT,
        response_heuristic_judge TEXT,
        human_score_original INTEGER,
        human_score_enhanced INTEGER,
        human_pick TEXT,
        human_notes TEXT
    );
    """

    if _is_postgres():
        conn = _connect_postgres()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(ddl)
        finally:
            conn.close()
        return

    conn = _connect_sqlite()
    try:
        cur = conn.cursor()
        cur.execute(ddl)
        conn.commit()
    finally:
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
) -> None:
    ts = time.time()

    row = (
        comparison_id,
        run_id,
        session_id,
        ts,
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
        _to_json(original_prompt_critique),
        _to_json(original_prompt_heuristic),
        enhanced_prompt,
        enhanced_response,
        _to_json(enhanced_prompt_critique),
        _to_json(enhanced_prompt_heuristic),
        _to_json(response_llm_judge),
        _to_json(response_heuristic_judge),
        int(human_score_original),
        int(human_score_enhanced),
        human_pick,
        human_notes,
    )

    placeholders = ",".join(["%s"] * len(row)) if _is_postgres() else ",".join(["?"] * len(row))

    sql = f"""
    INSERT INTO comparisons (
        comparison_id, run_id, session_id, ts, human_rater, user_tag, variant,
        temp_mode, threshold_mode, model_mode, user_input, route_predicted,
        temperature_used, rewrite_threshold_used, rewritten,
        original_prompt, original_response, original_prompt_critique, original_prompt_heuristic,
        enhanced_prompt, enhanced_response, enhanced_prompt_critique, enhanced_prompt_heuristic,
        response_llm_judge, response_heuristic_judge,
        human_score_original, human_score_enhanced, human_pick, human_notes
    ) VALUES ({placeholders});
    """

    if _is_postgres():
        conn = _connect_postgres()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, row)
        finally:
            conn.close()
        return

    conn = _connect_sqlite()
    try:
        cur = conn.cursor()
        cur.execute(sql, row)
        conn.commit()
    finally:
        conn.close()


def fetch_comparisons(limit: int = 10) -> List[Tuple[Any, ...]]:
    sql = """
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
    LIMIT %s;
    """

    if _is_postgres():
        conn = _connect_postgres()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (int(limit),))
                return cur.fetchall()
        finally:
            conn.close()

    # sqlite uses ? not %s
    conn = _connect_sqlite()
    try:
        cur = conn.cursor()
        cur.execute(sql.replace("%s", "?"), (int(limit),))
        return cur.fetchall()
    finally:
        conn.close()
