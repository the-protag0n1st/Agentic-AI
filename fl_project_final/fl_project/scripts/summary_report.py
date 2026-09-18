import sqlite3
import numpy as np

db = r"c:\Users\akhil\Downloads\fl_project_final (2)\fl_project_final\fl_project\fl_metrics_experiment3.db"
con = sqlite3.connect(db)
cur = con.cursor()

methods = [
    "FixedFedAvg", "FixedMedian", "FixedTrimmedMean", "FixedKrum",
    "RuleBased", "SingleShotLLM", "ReflectiveAgent", "AgenticAI"
]

print("=" * 80)
print(f"{'Method':<18} | {'Final Acc (R20)':<18} | {'Cumulative Regret':<20} | {'LLM Calls'}")
print("-" * 80)

for m in methods:
    accs = []
    regrets = []
    for s in [1, 2, 3]:
        row = cur.execute("""
            SELECT rd.test_accuracy, rd.cumulative_regret
            FROM experiment3_rounds rd JOIN experiment3_runs r ON rd.run_id = r.run_id
            WHERE r.method = ? AND r.seed = ? AND rd.round = 20
        """, (m, s)).fetchone()
        if row:
            accs.append(row[0])
            regrets.append(row[1])
    
    llm_cnt = cur.execute("""
        SELECT count(*)
        FROM llm_call_log l JOIN experiment3_runs r ON l.run_id = r.run_id
        WHERE r.method = ? AND l.status = 'success'
    """, (m,)).fetchone()[0]

    mean_acc = np.mean(accs) if accs else 0
    std_acc = np.std(accs) if accs else 0
    mean_reg = np.mean(regrets) if regrets else 0
    std_reg = np.std(regrets) if regrets else 0
    print(f"{m:<18} | {mean_acc:5.2f}% +/- {std_acc:4.2f}% | {mean_reg:6.2f}% +/- {std_reg:4.2f}% | {llm_cnt}")

print("=" * 80)
con.close()
