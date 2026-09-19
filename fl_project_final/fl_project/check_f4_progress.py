import sqlite3
import os

db_path = r"c:\Users\akhil\Downloads\fl_project_final (2)\fl_project_final\fl_project\fl_metrics_byzantine_stress_f4.db"

if not os.path.exists(db_path):
    print(f"Database {db_path} not found.")
    exit(0)

con = sqlite3.connect(db_path)
cur = con.cursor()

print("=" * 80)
print("EXPERIMENT 3 PROGRESS (f=4 Attackers, 40% Infiltration, Seed 1)")
print("=" * 80)

# Summary of all 3 methods
summary = cur.execute("""
    SELECT runs.method, COUNT(r.id) as n_rounds, 
           MAX(r.round) as latest, 
           ROUND(MAX(r.test_accuracy), 2) as max_acc,
           ROUND(MAX(CASE WHEN r.round = 20 THEN r.test_accuracy END), 2) as r20_acc
    FROM experiment3_rounds r
    JOIN experiment3_runs runs ON r.run_id = runs.run_id
    GROUP BY runs.method
    ORDER BY runs.method
""").fetchall()

print(f"{'Method':<20} | {'Status':<14} | {'Latest Round':<12} | {'Max Acc':<10} | R20 Final")
print("-" * 75)
for m, n, latest, max_acc, r20 in summary:
    status = "COMPLETE" if n == 20 else f"RUNNING ({n}/20)"
    r20_str = f"{r20:.2f}%" if r20 is not None else "--"
    print(f"{m:<20} | {status:<14} | R{latest:<11} | {max_acc:>6.2f}%    | {r20_str}")

# Detail for AgenticAI rounds
print("\n" + "=" * 80)
print("AGENTIC AI ROUND-BY-ROUND PROGRESS:")
print("=" * 80)
agent_rounds = cur.execute("""
    SELECT r.round, r.alpha, r.attack_active, r.test_accuracy, 
           r.aggregation_method, r.selected_clients, r.round_time_s
    FROM experiment3_rounds r
    JOIN experiment3_runs runs ON r.run_id = runs.run_id
    WHERE runs.method = 'AgenticAI'
    ORDER BY r.round ASC
""").fetchall()

if not agent_rounds:
    print("AgenticAI round 1 is currently training...")
else:
    print(f"{'Round':<6} | {'Regime':<12} | {'Attack':<8} | {'Test Acc':<10} | {'Operator':<12} | Selected Clients")
    print("-" * 80)
    for rnd, alpha, atk, acc, agg, clients, t in agent_rounds:
        regime = "Near-IID" if rnd <= 5 else ("Mod-NonIID" if rnd <= 10 else ("Sev-NonIID" if rnd <= 15 else "4-ATTACKERS"))
        atk_str = "YES" if atk else "no"
        print(f"R{rnd:<5} | {regime:<12} | {atk_str:<8} | {acc:>8.2f}% | {agg:<12} | {clients}")

con.close()
