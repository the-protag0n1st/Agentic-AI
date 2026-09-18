"""Database Schema, Checkpointing, and Logging for Experiment 3.

Maintains strict separation from frozen Experiment 1 and 2 databases.
Creates and manages the Experiment 3 tables:
  1. experiment3_runs
  2. experiment3_rounds
  3. agentic_log
  4. llm_call_log (with rate limit headers)
  5. experiment3_checkpoints (for safe resume)
"""

import io
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional
import torch


def init_experiment3_db(db_path: str = "fl_metrics_experiment3.db"):
    """Initialize dedicated Experiment 3 tables and safely migrate columns."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # 1. Runs table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiment3_runs (
            run_id TEXT PRIMARY KEY,
            method TEXT,
            seed INTEGER,
            model TEXT,
            provider TEXT,
            config_json TEXT,
            created_at REAL
        )
    """)

    # 2. Rounds table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiment3_rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            round INTEGER,
            alpha REAL,
            attack_active INTEGER,
            attacker_ids TEXT,
            selected_clients TEXT,
            aggregation_method TEXT,
            train_loss REAL,
            val_accuracy REAL,
            val_loss REAL,
            val_f1 REAL,
            val_auc REAL,
            test_accuracy REAL,
            test_loss REAL,
            test_f1 REAL,
            test_auc REAL,
            oracle_accuracy REAL,
            oracle_candidate_id TEXT,
            regret REAL,
            cumulative_regret REAL,
            round_time_s REAL
        )
    """)

    # 3. Agentic decision log
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agentic_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            round INTEGER,
            telemetry_json TEXT,
            history_json TEXT,
            analyst_output_json TEXT,
            proposals_json TEXT,
            critic_output_json TEXT,
            final_decision_json TEXT,
            validation_result TEXT,
            repair_status TEXT,
            repair_reason TEXT,
            oracle_decision_json TEXT,
            selected_score REAL,
            oracle_score REAL,
            regret REAL,
            cumulative_regret REAL
        )
    """)

    # 4. Detailed LLM call log with rate-limit headers
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_call_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            round INTEGER,
            stage TEXT,
            model TEXT,
            prompt_version TEXT,
            prompt TEXT,
            completion TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            latency_ms REAL,
            retries INTEGER,
            status TEXT,
            error TEXT,
            request_id TEXT,
            requests_remaining INTEGER,
            requests_limit INTEGER,
            requests_reset_time TEXT,
            tokens_remaining INTEGER,
            tokens_limit INTEGER,
            tokens_reset_time TEXT
        )
    """)

    # Safe migration for llm_call_log if previously created without rate-limit columns
    existing_cols = [r[1] for r in cur.execute("PRAGMA table_info(llm_call_log)").fetchall()]
    new_cols = [
        ("requests_remaining", "INTEGER"),
        ("requests_limit", "INTEGER"),
        ("requests_reset_time", "TEXT"),
        ("tokens_remaining", "INTEGER"),
        ("tokens_limit", "INTEGER"),
        ("tokens_reset_time", "TEXT"),
    ]
    for col_name, col_type in new_cols:
        if col_name not in existing_cols:
            cur.execute(f"ALTER TABLE llm_call_log ADD COLUMN {col_name} {col_type}")

    # Safe migration for agentic_log if previously created without agentic columns
    existing_agentic_cols = [r[1] for r in cur.execute("PRAGMA table_info(agentic_log)").fetchall()]
    new_agentic_cols = [
        ("iterations_used", "INTEGER"),
        ("reflection_summary", "TEXT"),
        ("assessed_regime", "TEXT"),
        ("tool_calls_made", "TEXT"),
        ("llm_calls_used", "INTEGER"),
    ]
    for col_name, col_type in new_agentic_cols:
        if col_name not in existing_agentic_cols:
            cur.execute(f"ALTER TABLE agentic_log ADD COLUMN {col_name} {col_type}")

    # 5. Checkpoints table for safe, deterministic resume
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiment3_checkpoints (
            method TEXT,
            seed INTEGER,
            round INTEGER,
            model_weights BLOB,
            history_json TEXT,
            cumulative_regret REAL,
            updated_at REAL,
            PRIMARY KEY (method, seed, round)
        )
    """)

    con.commit()
    con.close()


def insert_run_record(
    run_id: str,
    method: str,
    seed: int,
    model: str,
    provider: str,
    config: Dict[str, Any],
    db_path: str = "fl_metrics_experiment3.db",
):
    con = sqlite3.connect(db_path)
    con.execute(
        """INSERT OR REPLACE INTO experiment3_runs
        (run_id, method, seed, model, provider, config_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (run_id, method, seed, model, provider, json.dumps(config, default=str), time.time()),
    )
    con.commit()
    con.close()


def insert_round_record(
    run_id: str,
    round_num: int,
    alpha: float,
    attack_active: bool,
    attacker_ids: List[int],
    selected_clients: List[int],
    aggregation_method: str,
    train_loss: Optional[float] = None,
    val_metrics: Optional[Dict[str, float]] = None,
    test_metrics: Optional[Dict[str, float]] = None,
    oracle_accuracy: Optional[float] = None,
    oracle_candidate_id: Optional[str] = None,
    regret: Optional[float] = None,
    cumulative_regret: Optional[float] = None,
    round_time_s: Optional[float] = None,
    db_path: str = "fl_metrics_experiment3.db",
):
    vm = val_metrics or {}
    tm = test_metrics or {}
    con = sqlite3.connect(db_path)
    con.execute(
        """INSERT INTO experiment3_rounds
        (run_id, round, alpha, attack_active, attacker_ids, selected_clients,
         aggregation_method, train_loss, val_accuracy, val_loss, val_f1, val_auc,
         test_accuracy, test_loss, test_f1, test_auc, oracle_accuracy, oracle_candidate_id,
         regret, cumulative_regret, round_time_s)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            round_num,
            alpha,
            int(attack_active),
            json.dumps(attacker_ids),
            json.dumps(selected_clients),
            aggregation_method,
            train_loss or 0.0,
            vm.get("accuracy"),
            vm.get("loss"),
            vm.get("f1"),
            vm.get("auc"),
            tm.get("accuracy"),
            tm.get("loss"),
            tm.get("f1"),
            tm.get("auc"),
            oracle_accuracy,
            oracle_candidate_id,
            regret,
            cumulative_regret,
            round_time_s,
        ),
    )
    con.commit()
    con.close()


def insert_agentic_log(
    run_id: str,
    round_num: int,
    telemetry: Optional[Any] = None,
    history: Optional[Any] = None,
    analyst_output: Optional[Any] = None,
    proposals: Optional[Any] = None,
    critic_output: Optional[Any] = None,
    final_decision: Optional[Any] = None,
    validation_result: Optional[str] = None,
    repair_status: Optional[str] = None,
    repair_reason: Optional[str] = None,
    oracle_decision: Optional[Any] = None,
    selected_score: Optional[float] = None,
    oracle_score: Optional[float] = None,
    regret: Optional[float] = None,
    cumulative_regret: Optional[float] = None,
    db_path: str = "fl_metrics_experiment3.db",
):
    def _to_json(obj):
        if obj is None:
            return None
        if hasattr(obj, "model_dump_json"):
            return obj.model_dump_json()
        if hasattr(obj, "dict"):
            return json.dumps(obj.dict(), default=str)
        return json.dumps(obj, default=str)

    con = sqlite3.connect(db_path)
    con.execute(
        """INSERT INTO agentic_log
        (run_id, round, telemetry_json, history_json, analyst_output_json,
         proposals_json, critic_output_json, final_decision_json, validation_result,
         repair_status, repair_reason, oracle_decision_json, selected_score,
         oracle_score, regret, cumulative_regret)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            round_num,
            _to_json(telemetry),
            _to_json(history),
            _to_json(analyst_output),
            _to_json(proposals),
            _to_json(critic_output),
            _to_json(final_decision),
            validation_result,
            repair_status,
            repair_reason,
            _to_json(oracle_decision),
            selected_score,
            oracle_score,
            regret,
            cumulative_regret,
        ),
    )
    con.commit()
    con.close()


def get_completed_rounds(db_path: str, method: str, seed: int) -> List[int]:
    """Inspect fl_metrics_experiment3.db for completed rounds for the exact (method, seed)."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    # Find all run_ids corresponding to this method and seed
    cur.execute(
        """SELECT r.round FROM experiment3_rounds r
           JOIN experiment3_runs u ON r.run_id = u.run_id
           WHERE u.method = ? AND u.seed = ?
           ORDER BY r.round ASC""",
        (method, seed),
    )
    rows = cur.fetchall()
    con.close()
    completed = sorted(list(set([r[0] for r in rows])))
    return completed


def load_latest_checkpoint(db_path: str, method: str, seed: int) -> Optional[Dict[str, Any]]:
    """Load the latest checkpoint state for a (method, seed) pair."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """SELECT round, model_weights, history_json, cumulative_regret
           FROM experiment3_checkpoints
           WHERE method = ? AND seed = ?
           ORDER BY round DESC LIMIT 1""",
        (method, seed),
    )
    row = cur.fetchone()
    con.close()

    if not row:
        return None

    round_num, weights_blob, hist_json, cum_regret = row
    buf = io.BytesIO(weights_blob)
    weights = torch.load(buf, weights_only=True)

    return {
        "round": round_num,
        "model_weights": weights,
        "history_json": json.loads(hist_json) if hist_json else [],
        "cumulative_regret": cum_regret,
    }


def commit_round_transaction(
    db_path: str,
    run_id: str,
    method: str,
    seed: int,
    round_num: int,
    round_record: Dict[str, Any],
    agentic_log: Optional[Dict[str, Any]] = None,
    checkpoint_state: Optional[Dict[str, Any]] = None,
):
    """Atomically commit a completed round, agentic log, and checkpoint in a single transaction.

    If an interruption or error occurs, the transaction rolls back completely,
    ensuring an incomplete round cannot leave a misleading "completed" state.
    """
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()

        # 1. Insert into experiment3_rounds
        vm = round_record.get("val_metrics") or {}
        tm = round_record.get("test_metrics") or {}
        cur.execute(
            """INSERT INTO experiment3_rounds
            (run_id, round, alpha, attack_active, attacker_ids, selected_clients,
             aggregation_method, train_loss, val_accuracy, val_loss, val_f1, val_auc,
             test_accuracy, test_loss, test_f1, test_auc, oracle_accuracy, oracle_candidate_id,
             regret, cumulative_regret, round_time_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                round_num,
                round_record.get("alpha"),
                int(round_record.get("attack_active", False)),
                json.dumps(round_record.get("attacker_ids", [])),
                json.dumps(round_record.get("selected_clients", [])),
                round_record.get("aggregation_method"),
                round_record.get("train_loss", 0.0),
                vm.get("accuracy"),
                vm.get("loss"),
                vm.get("f1"),
                vm.get("auc"),
                tm.get("accuracy"),
                tm.get("loss"),
                tm.get("f1"),
                tm.get("auc"),
                round_record.get("oracle_accuracy"),
                round_record.get("oracle_candidate_id"),
                round_record.get("regret"),
                round_record.get("cumulative_regret"),
                round_record.get("round_time_s"),
            ),
        )

        # 2. Insert into agentic_log if present
        if agentic_log:
            def _to_json(obj):
                if obj is None:
                    return None
                if isinstance(obj, list):
                    return json.dumps([
                        x.model_dump() if hasattr(x, "model_dump") else (x.dict() if hasattr(x, "dict") else x)
                        for x in obj
                    ], default=str)
                if hasattr(obj, "model_dump_json"):
                    return obj.model_dump_json()
                if hasattr(obj, "dict"):
                    return json.dumps(obj.dict(), default=str)
                return json.dumps(obj, default=str)

            cur.execute(
                """INSERT INTO agentic_log
                (run_id, round, telemetry_json, history_json, analyst_output_json,
                 proposals_json, critic_output_json, final_decision_json, validation_result,
                 repair_status, repair_reason, oracle_decision_json, selected_score,
                 oracle_score, regret, cumulative_regret, iterations_used, reflection_summary,
                 assessed_regime, tool_calls_made, llm_calls_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    round_num,
                    _to_json(agentic_log.get("telemetry")),
                    _to_json(agentic_log.get("history")),
                    _to_json(agentic_log.get("analyst_output")),
                    _to_json(agentic_log.get("proposals")),
                    _to_json(agentic_log.get("critic_output")),
                    _to_json(agentic_log.get("final_decision")),
                    agentic_log.get("validation_result"),
                    agentic_log.get("repair_status"),
                    agentic_log.get("repair_reason"),
                    _to_json(agentic_log.get("oracle_decision")),
                    agentic_log.get("selected_score"),
                    agentic_log.get("oracle_score"),
                    agentic_log.get("regret"),
                    agentic_log.get("cumulative_regret"),
                    agentic_log.get("iterations_used", 1),
                    agentic_log.get("reflection_summary"),
                    agentic_log.get("assessed_regime"),
                    json.dumps(agentic_log.get("tool_calls_made", [])),
                    agentic_log.get("llm_calls_used", 0),
                ),
            )

        # 3. Insert checkpoint
        if checkpoint_state:
            buf = io.BytesIO()
            torch.save(checkpoint_state["model_weights"], buf)
            weights_blob = buf.getvalue()
            hist_json = json.dumps(checkpoint_state.get("history_json", []), default=str)
            cur.execute(
                """INSERT OR REPLACE INTO experiment3_checkpoints
                (method, seed, round, model_weights, history_json, cumulative_regret, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    method,
                    seed,
                    round_num,
                    weights_blob,
                    hist_json,
                    checkpoint_state.get("cumulative_regret", 0.0),
                    time.time(),
                ),
            )

        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def insert_llm_call_log(
    run_id: str,
    round_num: int,
    stage: str,
    model: str,
    prompt_version: str,
    prompt: str,
    completion: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    latency_ms: Optional[float] = None,
    retries: int = 0,
    status: str = "success",
    error: Optional[str] = None,
    request_id: Optional[str] = None,
    requests_remaining: Optional[int] = None,
    requests_limit: Optional[int] = None,
    requests_reset_time: Optional[str] = None,
    tokens_remaining: Optional[int] = None,
    tokens_limit: Optional[int] = None,
    tokens_reset_time: Optional[str] = None,
    db_path: str = "fl_metrics_experiment3.db",
):
    con = sqlite3.connect(db_path)
    con.execute(
        """INSERT INTO llm_call_log
        (run_id, round, stage, model, prompt_version, prompt, completion,
         input_tokens, output_tokens, total_tokens, latency_ms, retries,
         status, error, request_id, requests_remaining, requests_limit,
         requests_reset_time, tokens_remaining, tokens_limit, tokens_reset_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            round_num,
            stage,
            model,
            prompt_version,
            prompt,
            completion,
            input_tokens,
            output_tokens,
            total_tokens,
            latency_ms,
            retries,
            status,
            error,
            request_id,
            requests_remaining,
            requests_limit,
            requests_reset_time,
            tokens_remaining,
            tokens_limit,
            tokens_reset_time,
        ),
    )
    con.commit()
    con.close()
