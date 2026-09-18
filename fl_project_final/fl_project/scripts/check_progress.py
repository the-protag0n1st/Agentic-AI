import sqlite3
import os
from datetime import datetime

PROJECT_DIR = r"c:\Users\akhil\Downloads\fl_project_final (2)\fl_project_final\fl_project"
DB_PATH = os.path.join(PROJECT_DIR, "fl_metrics_experiment3.db")

if not os.path.exists(DB_PATH):
    print("Database not found:", DB_PATH)
    exit(1)

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

methods = [
    "AgenticAI", "FixedFedAvg", "FixedKrum", "FixedMedian", 
    "FixedTrimmedMean", "ReflectiveAgent", "RuleBased", "SingleShotLLM"
]
seeds = [1, 2, 3]

now = datetime.now().strftime("%H:%M:%S")
print(f"\n=== DB STATUS @ {now} ===")
print(f"{'Method':<20} | {'Seed 1':<17} | {'Seed 2':<17} | {'Seed 3':<17}")
print("-" * 78)

total_completed = 0
total_rounds = 0

for m in methods:
    row_str = f"{m:<20} | "
    for s in seeds:
        rounds = cur.execute("""
            SELECT count(distinct rd.round), max(rd.round), max(rd.test_accuracy)
            FROM experiment3_rounds rd
            JOIN experiment3_runs r ON rd.run_id = r.run_id
            WHERE r.method = ? AND r.seed = ?
        """, (m, s)).fetchone()
        
        cnt = rounds[0] or 0
        max_r = rounds[1] or 0
        acc = rounds[2] or 0.0
        total_rounds += cnt
        
        if cnt >= 20:
            total_completed += 1
            # Check for any LLM errors
            err_cnt = cur.execute("""
                SELECT count(*)
                FROM llm_call_log l
                JOIN experiment3_runs r ON l.run_id = r.run_id
                WHERE r.method = ? AND r.seed = ? AND l.status != 'success'
            """, (m, s)).fetchone()[0]
            tag = "DONE (100% LLM)" if err_cnt == 0 and m in ["AgenticAI", "SingleShotLLM", "ReflectiveAgent"] else "DONE"
            status = f"{tag:<17}"
        elif cnt > 0:
            status = f"R{max_r}/20 ({acc:.1f}%)"
            status = f"{status:<17}"
        else:
            status = f"{'--/20':<17}"
        row_str += status + " | "
    print(row_str[:-3])

print("-" * 78)
print(f"Complete: {total_completed}/24 runs  |  Rounds in DB: {total_rounds}/480\n")

# Check LLM call log
print("=== LLM CALL VERIFICATION (0 = No fake/fallback calls) ===")
for m in ["AgenticAI", "SingleShotLLM", "ReflectiveAgent"]:
    for s in seeds:
        succ = cur.execute("""
            SELECT count(*) FROM llm_call_log l
            JOIN experiment3_runs r ON l.run_id = r.run_id
            WHERE r.method = ? AND r.seed = ? AND l.status = 'success'
        """, (m, s)).fetchone()[0]
        errs = cur.execute("""
            SELECT count(*) FROM llm_call_log l
            JOIN experiment3_runs r ON l.run_id = r.run_id
            WHERE r.method = ? AND r.seed = ? AND l.status != 'success'
        """, (m, s)).fetchone()[0]
        if succ + errs > 0:
            print(f"  {m} Seed {s}: {succ} real LLM calls, {errs} fallbacks/errors")

con.close()
