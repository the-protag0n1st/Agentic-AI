import sqlite3
import json

DB_PATH = "fl_metrics.db"


def _ensure_column(con, table, column_name, column_type):
    try:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")
        con.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


def init_db(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS client_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, round INTEGER, alpha REAL, client_id INTEGER,
            norm REAL, cosine_sim REAL, weight_var REAL, flagged INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS round_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, round INTEGER, alpha REAL, method TEXT,
            accuracy REAL, loss REAL, f1 REAL, auc REAL,
            n_anomalous INTEGER, clean_ids TEXT,
            selected_method TEXT, selected_combination TEXT, agent_reasoning TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, round INTEGER, alpha REAL,
            method_chosen TEXT, reason TEXT, source TEXT,
            selected_combination TEXT,
            raw_response TEXT, prompt_version TEXT, latency_ms REAL,
            valid INTEGER, fallback_reason TEXT, confidence REAL
        )
    """)
    con.commit()

    _ensure_column(con, "round_metrics", "selected_method", "TEXT")
    _ensure_column(con, "round_metrics", "selected_combination", "TEXT")
    _ensure_column(con, "round_metrics", "agent_reasoning", "TEXT")
    _ensure_column(con, "agent_log", "selected_combination", "TEXT")
    _ensure_column(con, "agent_log", "raw_response", "TEXT")
    _ensure_column(con, "agent_log", "prompt_version", "TEXT")
    _ensure_column(con, "agent_log", "latency_ms", "REAL")
    _ensure_column(con, "agent_log", "valid", "INTEGER")
    _ensure_column(con, "agent_log", "fallback_reason", "TEXT")
    _ensure_column(con, "agent_log", "confidence", "REAL")

    _ensure_column(con, "client_metrics", "seed", "INTEGER")
    _ensure_column(con, "round_metrics", "seed", "INTEGER")
    _ensure_column(con, "agent_log", "seed", "INTEGER")

    con.close()


def insert_client_metrics(run_id, round_num, alpha, client_id,
                           norm, cosine_sim, weight_var, flagged,
                           seed=None, db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    con.execute("""INSERT INTO client_metrics
        (run_id, round, alpha, client_id, norm, cosine_sim, weight_var, flagged, seed)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (run_id, round_num, alpha, client_id, norm, cosine_sim, weight_var, int(flagged), seed))
    con.commit()
    con.close()


def insert_round_metrics(run_id, round_num, alpha, method, accuracy, loss,
                          f1, auc, n_anomalous, clean_ids,
                          selected_method=None, selected_combination=None,
                          agent_reasoning=None, seed=None, db_path=DB_PATH):
    if selected_method is None:
        selected_method = method
    con = sqlite3.connect(db_path)
    con.execute("""INSERT INTO round_metrics
        (run_id, round, alpha, method, accuracy, loss, f1, auc, n_anomalous, clean_ids,
         selected_method, selected_combination, agent_reasoning, seed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, round_num, alpha, method, accuracy, loss, f1, auc,
         n_anomalous, json.dumps(clean_ids),
         selected_method,
         json.dumps(selected_combination) if selected_combination is not None else None,
         agent_reasoning, seed))
    con.commit()
    con.close()


def insert_agent_log(run_id, round_num, alpha, method_chosen, reason, source,
                      selected_combination=None, raw_response=None,
                      prompt_version=None, latency_ms=None, valid=None,
                      fallback_reason=None, confidence=None, seed=None, db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    con.execute("""INSERT INTO agent_log
        (run_id, round, alpha, method_chosen, reason, source, selected_combination,
         raw_response, prompt_version, latency_ms, valid, fallback_reason, confidence, seed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, round_num, alpha, method_chosen, reason, source,
         json.dumps(selected_combination) if selected_combination is not None else None,
         raw_response, prompt_version, latency_ms, valid, fallback_reason, confidence, seed))
    con.commit()
    con.close()


def get_round_history(run_id, alpha, last_n=5, db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    rows = con.execute("""
        SELECT round, method, accuracy, loss, f1, auc, n_anomalous,
               selected_method, selected_combination, agent_reasoning
        FROM round_metrics
        WHERE run_id=? AND alpha=? ORDER BY round DESC LIMIT ?
    """, (run_id, alpha, last_n)).fetchall()
    con.close()

    history = []
    for (rnd, method, acc, loss, f1, auc, n_anom,
         sel_method, sel_combo, reasoning) in reversed(rows):
        history.append({
            "round": rnd,
            "method": method,
            "accuracy": acc,
            "loss": loss,
            "f1": f1,
            "auc": auc,
            "n_anomalous": n_anom,
            "selected_method": sel_method,
            "selected_combination": json.loads(sel_combo) if sel_combo else None,
            "reason": reasoning,
        })
    return history


def get_previous_metrics(run_id, alpha, db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    row = con.execute("""
        SELECT round, accuracy, loss, f1, auc, n_anomalous
        FROM round_metrics WHERE run_id=? AND alpha=?
        ORDER BY round DESC LIMIT 1
    """, (run_id, alpha)).fetchone()
    con.close()
    if not row:
        return {}
    rnd, acc, loss, f1, auc, n_anom = row
    return {"round": rnd, "accuracy": acc, "loss": loss,
            "f1": f1, "auc": auc, "n_anomalous": n_anom}


def fetch_dataframe(query, params=(), db_path=DB_PATH):
    import pandas as pd
    con = sqlite3.connect(db_path)
    df = pd.read_sql(query, con, params=params)
    con.close()
    return df
