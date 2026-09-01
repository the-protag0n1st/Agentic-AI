import sys
sys.stdout.reconfigure(line_buffering=True)
import os
import time
import copy
import json
import sqlite3
import statistics
import traceback
import csv

import torch

from models import Net, train_local, evaluate_model, device
from anomaly import (compute_update_norms, compute_cosine_similarities,
                      compute_weight_variations, detect_anomalies)
from combinations import generate_client_combinations
from aggregation import aggregate
from agent import agent_select
from database import (init_db, insert_client_metrics, insert_round_metrics,
                       get_round_history, get_previous_metrics)
from dataset import get_datasets, make_clients
from attacks import apply_attack, select_malicious_clients

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ALPHAS = [0.3, 1.0]
SEEDS = [1, 2, 3, 4, 5]
METHODS = ["FedAvg", "Krum", "Median", "Trimmed Mean", "RuleBased", "Agent"]
ROUNDS = 8
NUM_CLIENTS = 5
NUM_MALICIOUS = 1
ATTACK_TYPE = "label_flip"
BATCH_SIZE = 32
AGENT_MODEL = "ollama:phi3"
DB_PATH = "fl_metrics_experiment2_label_flip.db"

# ==============================================================================
# DB UTILS FOR RESUMABILITY
# ==============================================================================
def check_run_status(run_id, db_path):
    """Returns the number of completed rounds for a given run_id."""
    if not os.path.exists(db_path):
        return 0
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT COUNT(round_num) FROM round_metrics WHERE run_id=?", (run_id,))
        count = cur.fetchone()[0]
        con.close()
        return count
    except Exception:
        return 0

def clear_incomplete_run(run_id, db_path):
    """If a run didn't finish all rounds, clear its data to restart cleanly."""
    if not os.path.exists(db_path):
        return
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("DELETE FROM client_metrics WHERE run_id=?", (run_id,))
        cur.execute("DELETE FROM round_metrics WHERE run_id=?", (run_id,))
        cur.execute("DELETE FROM agent_log WHERE run_id=?", (run_id,))
        con.commit()
        con.close()
    except Exception as e:
        print(f"Error clearing incomplete run {run_id}: {e}")

# ==============================================================================
# DETECTION METRICS
# ==============================================================================
def compute_detection_metrics(anomalous_ids, malicious_ids, num_clients):
    malicious_set = set(malicious_ids)
    anomalous_set = set(anomalous_ids)
    all_ids = set(range(num_clients))

    tp = len(anomalous_set & malicious_set)
    fp = len(anomalous_set - malicious_set)
    fn = len(malicious_set - anomalous_set)
    tn = len((all_ids - anomalous_set) & (all_ids - malicious_set))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp, fp, tn, fn, precision, recall, f1

# ==============================================================================
# EXPORT METRICS
# ==============================================================================
def export_results(db_path, run_stats, detection_stats):
    print("\nExporting results...")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1. Round-level CSV
    rounds = cur.execute("SELECT * FROM round_metrics ORDER BY run_id, round_num").fetchall()
    if rounds:
        with open("experiment2_label_flip_round_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rounds[0].keys())
            for r in rounds:
                writer.writerow(tuple(r))

    # 2. Run-level CSV
    with open("experiment2_label_flip_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "alpha", "seed", "run_id", "final_accuracy", "final_f1", "final_auc", 
                         "average_llm_latency", "total_llm_calls", "fallback_count", "total_runtime"])
        for run in run_stats:
            writer.writerow([
                run["method"], run["alpha"], run["seed"], run["run_id"],
                f"{run['final_accuracy']:.2f}", f"{run['final_f1']:.4f}", f"{run['final_auc']:.4f}",
                f"{run['avg_llm_latency']:.2f}", run["total_llm_calls"], run["fallback_count"],
                f"{run['total_runtime']:.2f}"
            ])

    # 3. Detection CSV
    with open("experiment2_label_flip_detection.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "alpha", "seed", "round", "actual_malicious_client", "detected_anomalous_clients", 
                         "TP", "FP", "TN", "FN", "precision", "recall", "F1"])
        for det in detection_stats:
            writer.writerow([
                det["method"], det["alpha"], det["seed"], det["round"],
                str(det["malicious_ids"]), str(det["anomalous_ids"]),
                det["tp"], det["fp"], det["tn"], det["fn"],
                f"{det['precision']:.2f}", f"{det['recall']:.2f}", f"{det['f1']:.2f}"
            ])

    # 4. Summary JSON
    summary = {}
    for alpha in ALPHAS:
        summary[str(alpha)] = {}
        for method in METHODS:
            method_runs = [r for r in run_stats if r["method"] == method and r["alpha"] == alpha]
            if not method_runs:
                continue
            accs = [r["final_accuracy"] for r in method_runs]
            f1s = [r["final_f1"] for r in method_runs]
            aucs = [r["final_auc"] for r in method_runs]
            
            summary[str(alpha)][method] = {
                "accuracy_mean": statistics.mean(accs),
                "accuracy_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "f1_mean": statistics.mean(f1s),
                "f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
                "auc_mean": statistics.mean(aucs),
                "auc_std": statistics.stdev(aucs) if len(aucs) > 1 else 0.0,
                "runs_completed": len(method_runs)
            }
    
    with open("experiment2_label_flip_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    con.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("==================================================")
    print("EXPERIMENT 2 — LABEL FLIPPING")
    print("=============================")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Malicious clients: {NUM_MALICIOUS}")
    print(f"Attack: {ATTACK_TYPE}")
    print(f"Alphas: {ALPHAS}")
    print(f"Seeds: {SEEDS}")
    print(f"Methods: {len(METHODS)} ({METHODS})")
    print(f"Rounds: {ROUNDS}")
    print(f"Total runs: {len(METHODS) * len(ALPHAS) * len(SEEDS)}")
    print(f"Total rounds: {len(METHODS) * len(ALPHAS) * len(SEEDS) * ROUNDS}")
    print("LLM caching: DISABLED")
    print("Fresh LLM call every Agent round: YES")
    print("=====================================")

    init_db(DB_PATH)
    
    train_data, test_data = get_datasets(root="./data", download=True)
    
    run_stats = []
    detection_stats = []
    
    total_runs = len(METHODS) * len(ALPHAS) * len(SEEDS)
    current_run = 0
    completed_rounds = 0
    failed_runs = 0
    experiment_start_time = time.time()

    for method in METHODS:
        for alpha in ALPHAS:
            for seed in SEEDS:
                current_run += 1
                run_id = f"exp2_m{method.replace(' ','')}_a{alpha}_s{seed}"
                
                # Check resumability
                status = check_run_status(run_id, DB_PATH)
                if status == ROUNDS:
                    print(f"[{current_run}/{total_runs}] SKIP (Already Complete) - Method: {method} | Alpha: {alpha} | Seed: {seed}")
                    # Note: To safely populate run_stats for skipped runs, we'd query DB. 
                    # For simplicity, if we skip, we still query the final row.
                    con = sqlite3.connect(DB_PATH)
                    cur = con.cursor()
                    last_metrics = cur.execute("SELECT accuracy, f1, auc FROM round_metrics WHERE run_id=? ORDER BY round_num DESC LIMIT 1", (run_id,)).fetchone()
                    agent_logs = cur.execute("SELECT latency_ms, fallback_reason FROM agent_log WHERE run_id=?", (run_id,)).fetchall()
                    con.close()
                    
                    # Estimate stats for skipped runs
                    latencies = [l[0]/1000 for l in agent_logs if l[0] is not None]
                    fallbacks = sum(1 for l in agent_logs if l[1] is not None)
                    run_stats.append({
                        "method": method, "alpha": alpha, "seed": seed, "run_id": run_id,
                        "final_accuracy": last_metrics[0] if last_metrics else 0.0,
                        "final_f1": last_metrics[1] if last_metrics else 0.0,
                        "final_auc": last_metrics[2] if last_metrics else 0.0,
                        "avg_llm_latency": sum(latencies)/len(latencies) if latencies else 0.0,
                        "total_llm_calls": len(agent_logs),
                        "fallback_count": fallbacks,
                        "total_runtime": 0.0 # Unknown if skipped, handled fine.
                    })
                    completed_rounds += ROUNDS
                    continue
                elif status > 0:
                    print(f"[{current_run}/{total_runs}] CLEANING INCOMPLETE RUN - Method: {method} | Alpha: {alpha} | Seed: {seed} (Found {status}/{ROUNDS} rounds)")
                    clear_incomplete_run(run_id, DB_PATH)

                print(f"\n[{current_run}/{total_runs}]")
                print(f"Method: {method}\nAlpha: {alpha}\nSeed: {seed}")
                
                run_start = time.time()
                try:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                        
                    # Deterministic malicious selection
                    malicious_ids = select_malicious_clients(NUM_CLIENTS, NUM_MALICIOUS, seed)
                    
                    clients_clean = make_clients(train_data, num_clients=NUM_CLIENTS, alpha=alpha, seed=seed)
                    clients_attacked = apply_attack(clients_clean, malicious_ids, attack_type=ATTACK_TYPE)
                    
                    gm = Net().to(device)
                    all_client_ids = list(range(NUM_CLIENTS))
                    
                    total_llm_calls = 0
                    fallback_count = 0
                    llm_latencies = []
                    
                    final_metrics = None
                    
                    for r in range(1, ROUNDS + 1):
                        print(f"Round: {r}/{ROUNDS}")
                        round_start = time.time()
                        
                        global_w = {k: v.cpu() for k, v in gm.state_dict().items()}
                        
                        # 1. Local Training
                        t0 = time.time()
                        local_w = [train_local(copy.deepcopy(gm), c, epochs=2, lr=0.01, batch_size=BATCH_SIZE)
                                   for c in clients_attacked]
                        t_train = time.time() - t0
                        
                        # 2. Anomaly Detection (Agent MUST NOT know ground truth)
                        t0 = time.time()
                        norms = compute_update_norms(global_w, local_w)
                        cos_sims = compute_cosine_similarities(global_w, local_w)
                        weight_vars = compute_weight_variations(local_w)
                        anomalous_ids, clean_ids, _ = detect_anomalies(norms, cos_sims, weight_vars)
                        t_anomaly = time.time() - t0
                        
                        # Calculate and store detection metrics manually
                        tp, fp, tn, fn, prec, rec, f1 = compute_detection_metrics(anomalous_ids, malicious_ids, NUM_CLIENTS)
                        detection_stats.append({
                            "method": method, "alpha": alpha, "seed": seed, "round": r,
                            "malicious_ids": malicious_ids, "anomalous_ids": anomalous_ids,
                            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                            "precision": prec, "recall": rec, "f1": f1
                        })
                        
                        # Log client metrics
                        for i in range(NUM_CLIENTS):
                            insert_client_metrics(run_id, r, alpha, i, norms[i], cos_sims[i],
                                                   weight_vars[i], i in anomalous_ids, seed=seed, db_path=DB_PATH)
                                                   
                        # 3. Agent / Rule Decision
                        usable_ids = clean_ids if clean_ids else all_client_ids
                        raw_combos = generate_client_combinations(usable_ids, local_w)
                        candidate_combinations = [c["client_ids"] for c in raw_combos]
                        
                        history = get_round_history(run_id, alpha, last_n=5, db_path=DB_PATH)
                        prev_metrics = get_previous_metrics(run_id, alpha, db_path=DB_PATH)
                        
                        t_agent = 0.0
                        if method in ["FedAvg", "Krum", "Median", "Trimmed Mean"]:
                            selected_method = method
                            selected_combo = clean_ids if clean_ids else all_client_ids
                            reason = "Fixed baseline"
                            confidence = 1.0
                        else:
                            effective_agent_model = "RuleBased" if method == "RuleBased" else AGENT_MODEL
                            
                            t0 = time.time()
                            decision = agent_select(
                                round_num=r,
                                anomalous_clients=anomalous_ids,
                                candidate_combinations=candidate_combinations,
                                history=history,
                                metrics=prev_metrics,
                                all_client_ids=all_client_ids,
                                alpha=alpha,
                                run_id=run_id,
                                agent_model=effective_agent_model,
                                seed=seed,
                                db_path=DB_PATH,
                            )
                            t_agent = time.time() - t0
                            
                            selected_combo = decision["selected_combination"]
                            selected_method = decision["selected_method"]
                            reason = decision["reason"]
                            confidence = decision.get("confidence", 1.0)
                            
                            if method == "Agent":
                                total_llm_calls += 1
                                llm_latencies.append(t_agent)
                                if decision.get("source", "") == "rule_based":
                                    fallback_count += 1
                                    
                        # 4. Aggregation
                        t0 = time.time()
                        selected_weights = [local_w[i] for i in selected_combo]
                        new_w = aggregate(selected_method, selected_weights, global_weights=global_w)
                        gm.load_state_dict(new_w)
                        t_agg = time.time() - t0
                        
                        # 5. Evaluation
                        t0 = time.time()
                        metrics = evaluate_model(gm, test_data)
                        t_eval = time.time() - t0
                        final_metrics = metrics
                        
                        # 6. Logging
                        insert_round_metrics(run_id, r, alpha, method,
                                              metrics["accuracy"], metrics["loss"],
                                              metrics["f1"], metrics["auc"],
                                              len(anomalous_ids), clean_ids,
                                              selected_method=selected_method,
                                              selected_combination=selected_combo,
                                              agent_reasoning=reason,
                                              seed=seed, db_path=DB_PATH)
                                              
                        round_time = time.time() - round_start
                        completed_rounds += 1
                        
                        print(f"[ROUND COMPLETE]")
                        print(f"Method: {method}")
                        print(f"Alpha: {alpha}")
                        print(f"Seed: {seed}")
                        print(f"Round: {r}/{ROUNDS}")
                        print(f"Runtime: {round_time:.2f}s")
                        print(f"Accuracy: {metrics['accuracy']:.2f}%")
                        print(f"F1: {metrics['f1']:.4f}")
                        print(f"AUC: {metrics['auc']:.4f}\n")

                    run_time = time.time() - run_start
                    run_stats.append({
                        "method": method, "alpha": alpha, "seed": seed, "run_id": run_id,
                        "final_accuracy": final_metrics["accuracy"] if final_metrics else 0.0,
                        "final_f1": final_metrics["f1"] if final_metrics else 0.0,
                        "final_auc": final_metrics["auc"] if final_metrics else 0.0,
                        "avg_llm_latency": sum(llm_latencies)/len(llm_latencies) if llm_latencies else 0.0,
                        "total_llm_calls": total_llm_calls,
                        "fallback_count": fallback_count,
                        "total_runtime": run_time
                    })
                    
                    print(f"[RUN COMPLETE]")
                    print(f"Run: {current_run}/{total_runs}")
                    print(f"Method: {method}")
                    print(f"Alpha: {alpha}")
                    print(f"Seed: {seed}")
                    print(f"Runtime: {run_time:.2f}s")
                    print(f"Final Accuracy: {final_metrics['accuracy']:.2f}%")
                    print(f"Final F1: {final_metrics['f1']:.4f}")
                    print(f"Final AUC: {final_metrics['auc']:.4f}\n")
                    
                except Exception as e:
                    print(f"FAILED RUN: Method={method}, Alpha={alpha}, Seed={seed}")
                    traceback.print_exc()
                    failed_runs += 1

    export_results(DB_PATH, run_stats, detection_stats)

    total_experiment_time = time.time() - experiment_start_time
    print("==================================================")
    print("EXPERIMENT 2 COMPLETE")
    print("=====================")
    print(f"Completed runs: {total_runs - failed_runs}/{total_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Completed rounds: {completed_rounds}")
    print(f"Total runtime: {total_experiment_time:.2f}s ({total_experiment_time/3600:.2f}h)")
    print("")
    print(f"Database: {DB_PATH}")
    print("Run-level CSV: experiment2_label_flip_results.csv")
    print("Round-level CSV: experiment2_label_flip_round_results.csv")
    print("Detection CSV: experiment2_label_flip_detection.csv")
    print("Summary JSON: experiment2_label_flip_summary.json")
    print("=============")

if __name__ == "__main__":
    main()
