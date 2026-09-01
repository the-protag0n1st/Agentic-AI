import pandas as pd
import sqlite3
import ast
import numpy as np

def run_analysis():
    print("=== PRIORITY 1: MISSING RUNS & VERIFICATION ===")
    results_df = pd.read_csv("experiment2_label_flip_results.csv")
    expected_runs = 6 * 2 * 5 # 60
    completed_runs = len(results_df)
    print(f"Total expected: {expected_runs}")
    print(f"Total completed: {completed_runs}")
    
    # Find missing
    methods = ["FedAvg", "Krum", "Median", "Trimmed Mean", "RuleBased", "Agent"]
    alphas = [0.3, 1.0]
    seeds = [1, 2, 3, 4, 5]
    
    missing = []
    for m in methods:
        for a in alphas:
            for s in seeds:
                if len(results_df[(results_df['method']==m) & (results_df['alpha']==a) & (results_df['seed']==s)]) == 0:
                    missing.append(f"{m}, alpha={a}, seed={s}")
    print("Missing runs (PyTorch NaN crashes during Krum aggregation on highly skewed data):")
    for msg in missing:
        print(f"  - {msg}")

    print("\n=== PRIORITY 2: DETECTOR METRICS (THE SMOKING GUN) ===")
    det_df = pd.read_csv("experiment2_label_flip_detection.csv")
    # Overall averages
    mean_prec = det_df['precision'].mean()
    mean_rec = det_df['recall'].mean()
    
    # Calculate False Positive Rate (FP / (FP + TN))
    # Calculate False Negative Rate (FN / (TP + FN))
    det_df['FPR'] = det_df['FP'] / (det_df['FP'] + det_df['TN']).replace(0, 1)
    det_df['FNR'] = det_df['FN'] / (det_df['TP'] + det_df['FN']).replace(0, 1)
    
    print(f"Average Precision: {mean_prec:.2f}")
    print(f"Average Recall:    {mean_rec:.2f}")
    print(f"Average FPR:       {det_df['FPR'].mean():.2f}")
    print(f"Average FNR:       {det_df['FNR'].mean():.2f}")
    
    print("\nDetection over rounds (Does it degrade?):")
    round_group = det_df.groupby('round')[['recall', 'precision']].mean()
    print(round_group)

    print("\n=== PRIORITY 3: AGENT VS FEDAVG (HEAD TO HEAD) ===")
    # Pair Agent and FedAvg
    agent_df = results_df[results_df['method'] == 'Agent']
    fedavg_df = results_df[results_df['method'] == 'FedAvg']
    
    merged = pd.merge(agent_df, fedavg_df, on=['alpha', 'seed'], suffixes=('_Agent', '_FedAvg'))
    merged['Diff'] = merged['final_accuracy_Agent'] - merged['final_accuracy_FedAvg']
    
    for alpha in alphas:
        sub = merged[merged['alpha'] == alpha]
        if len(sub) == 0: continue
        print(f"\nAlpha = {alpha}:")
        for _, row in sub.iterrows():
            print(f"  Seed {row['seed']}: Agent {row['final_accuracy_Agent']:.2f}% | FedAvg {row['final_accuracy_FedAvg']:.2f}% | Diff: {row['Diff']:+.2f}%")
        print(f"  MEAN DIFF (Alpha={alpha}): {sub['Diff'].mean():+.2f}% (Std: {sub['Diff'].std():.2f})")

    print("\n=== PRIORITY 4: AGENT DECISIONS & ATTACKER INCLUSION ===")
    # Connect to DB to get agent choices and attacker inclusion
    con = sqlite3.connect("fl_metrics_experiment2_label_flip.db")
    
    # From det_df we can get actual_malicious_client (it's a list string like '[4]')
    # Let's map run_id to its malicious client
    mal_map = {}
    for _, row in det_df.iterrows():
        run_id = f"exp2_mAgent_a{row['alpha']}_s{row['seed']}"
        mal_map[run_id] = ast.literal_eval(row['actual_malicious_client'])[0]
        
    cur = con.cursor()
    # Only look at completed Agent runs
    agent_run_ids = agent_df['run_id'].tolist()
    
    total_decisions = 0
    method_counts = {}
    attacker_included_count = 0
    
    for rid in agent_run_ids:
        mal_id = mal_map[rid]
        rows = cur.execute("SELECT selected_method, selected_combination FROM round_metrics WHERE run_id=?", (rid,)).fetchall()
        for r_method, r_combo in rows:
            total_decisions += 1
            method_counts[r_method] = method_counts.get(r_method, 0) + 1
            
            combo_list = json.loads(r_combo) if r_combo else []
            if mal_id in combo_list:
                attacker_included_count += 1
                
    print(f"Total Agent Decisions Analyzed: {total_decisions}")
    print("Method Selection Frequencies:")
    for m, c in method_counts.items():
        print(f"  - {m}: {c} ({c/total_decisions*100:.1f}%)")
        
    print(f"Attacker included in aggregation: {attacker_included_count} times ({attacker_included_count/total_decisions*100:.1f}%)")
    
    con.close()

if __name__ == "__main__":
    import json
    run_analysis()
