import sqlite3
import json
import numpy as np

db = r"c:\Users\akhil\Downloads\fl_project_final (2)\fl_project_final\fl_project\fl_metrics_experiment3.db"
con = sqlite3.connect(db)
cur = con.cursor()

print("=" * 80)
print("1. PER-SEED ROUND 20 ACCURACY & CUMULATIVE REGRET")
print("=" * 80)

methods = [
    "FixedFedAvg", "FixedMedian", "FixedTrimmedMean", "FixedKrum",
    "RuleBased", "SingleShotLLM", "ReflectiveAgent", "AgenticAI"
]

print(f"{'Method':<18} | {'Seed 1 Acc':<11} | {'Seed 2 Acc':<11} | {'Seed 3 Acc':<11} | {'Seed 1 Reg':<11} | {'Seed 2 Reg':<11} | {'Seed 3 Reg':<11}")
print("-" * 80)

for m in methods:
    accs = {}
    regs = {}
    for s in [1, 2, 3]:
        row = cur.execute("""
            SELECT rd.test_accuracy, rd.cumulative_regret
            FROM experiment3_rounds rd JOIN experiment3_runs r ON rd.run_id = r.run_id
            WHERE r.method = ? AND r.seed = ? AND rd.round = 20
        """, (m, s)).fetchone()
        if row:
            accs[s] = row[0]
            regs[s] = row[1]
        else:
            accs[s] = None
            regs[s] = None
    print(f"{m:<18} | {str(accs.get(1))+'%':<11} | {str(accs.get(2))+'%':<11} | {str(accs.get(3))+'%':<11} | {str(regs.get(1))+'%':<11} | {str(regs.get(2))+'%':<11} | {str(regs.get(3))+'%':<11}")

print("\n" + "=" * 80)
print("2. LLM CALL COUNT AUDIT BY RUN AND SEED")
print("=" * 80)

for m in ["AgenticAI", "SingleShotLLM", "ReflectiveAgent"]:
    print(f"\n--- {m} ---")
    runs = cur.execute("""
        SELECT r.run_id, r.seed, count(distinct rd.round)
        FROM experiment3_runs r
        LEFT JOIN experiment3_rounds rd ON r.run_id = rd.run_id
        WHERE r.method = ?
        GROUP BY r.run_id
        ORDER BY r.seed, r.rowid
    """, (m,)).fetchall()
    for run_id, seed, rd_cnt in runs:
        n_llm = cur.execute("SELECT count(*) FROM llm_call_log WHERE run_id = ?", (run_id,)).fetchone()[0]
        n_succ = cur.execute("SELECT count(*) FROM llm_call_log WHERE run_id = ? AND status = 'success'", (run_id,)).fetchone()[0]
        n_err = cur.execute("SELECT count(*) FROM llm_call_log WHERE run_id = ? AND status != 'success'", (run_id,)).fetchone()[0]
        print(f"  Seed {seed} | run={run_id} | rounds={rd_cnt} | total_calls={n_llm} (success={n_succ}, error={n_err})")

print("\n" + "=" * 80)
print("3. ATTACK ROUNDS (16-20): CLIENT SELECTION & ATTACKER REJECTION")
print("=" * 80)

focus_methods = ["FixedTrimmedMean", "SingleShotLLM", "ReflectiveAgent", "AgenticAI"]

for m in focus_methods:
    print(f"\n==================== METHOD: {m} ====================")
    for s in [1, 2, 3]:
        print(f"\n--- Seed {s} ---")
        rows = cur.execute("""
            SELECT rd.round, rd.attacker_ids, rd.selected_clients, rd.aggregation_method, rd.test_accuracy, rd.oracle_accuracy, rd.regret
            FROM experiment3_rounds rd JOIN experiment3_runs r ON rd.run_id = r.run_id
            WHERE r.method = ? AND r.seed = ? AND rd.round >= 15
            ORDER BY rd.round
        """, (m, s)).fetchall()
        for r in rows:
            rnd = r[0]
            att_ids = json.loads(r[1]) if isinstance(r[1], str) else r[1]
            sel_clients = json.loads(r[2]) if isinstance(r[2], str) else r[2]
            agg = r[3]
            acc = r[4]
            ora = r[5]
            reg = r[6]
            
            # Check attacker inclusion
            if att_ids:
                included_attackers = [cid for cid in att_ids if cid in sel_clients]
                clean_selected = [cid for cid in sel_clients if cid not in att_ids]
                print(f"  R{rnd:<2} | {agg:<11} | Acc: {acc:5.2f}% (Ora: {ora:5.2f}%, Reg: {reg:4.2f}%) | Attackers in pool: {att_ids} | Attackers INCLUDED: {included_attackers} | Clean selected: {len(clean_selected)}")
            else:
                print(f"  R{rnd:<2} | {agg:<11} | Acc: {acc:5.2f}% (Ora: {ora:5.2f}%, Reg: {reg:4.2f}%) | Clean round (no attack)")

con.close()
