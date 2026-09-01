import os
import json
import sqlite3
import statistics
import csv
from attacks import select_malicious_clients

DB_PATH = "fl_metrics_experiment2_label_flip.db"
NUM_CLIENTS = 5
NUM_MALICIOUS = 1
ALPHAS = [0.3, 1.0]
METHODS = ["FedAvg", "Krum", "Median", "Trimmed Mean", "RuleBased", "Agent"]

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

def export():
    print("Exporting results from database...")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1. Round-level CSV
    rounds = cur.execute("SELECT * FROM round_metrics ORDER BY run_id, round").fetchall()
    if rounds:
        with open("experiment2_label_flip_round_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rounds[0].keys())
            for r in rounds:
                writer.writerow(tuple(r))

    # 2. Reconstruct Run Stats and Detection Stats
    run_stats = []
    detection_stats = []
    
    # Get all distinct run_ids that have 8 rounds
    completed_runs = cur.execute("SELECT run_id FROM round_metrics GROUP BY run_id HAVING COUNT(round) = 8").fetchall()
    completed_run_ids = [r["run_id"] for r in completed_runs]
    
    for run_id in completed_run_ids:
        # Get run info from the first round
        info = cur.execute("SELECT method, alpha, seed FROM round_metrics WHERE run_id=? AND round=1", (run_id,)).fetchone()
        method, alpha, seed = info["method"], info["alpha"], info["seed"]
        
        malicious_ids = select_malicious_clients(NUM_CLIENTS, NUM_MALICIOUS, seed)
        
        # Get all rounds for detection
        run_rounds = cur.execute("SELECT round, clean_ids FROM round_metrics WHERE run_id=? ORDER BY round", (run_id,)).fetchall()
        for r_row in run_rounds:
            rnd = r_row["round"]
            clean_ids = json.loads(r_row["clean_ids"]) if r_row["clean_ids"] else []
            anomalous_ids = list(set(range(NUM_CLIENTS)) - set(clean_ids))
            
            tp, fp, tn, fn, prec, rec, det_f1 = compute_detection_metrics(anomalous_ids, malicious_ids, NUM_CLIENTS)
            detection_stats.append({
                "method": method, "alpha": alpha, "seed": seed, "round": rnd,
                "malicious_ids": malicious_ids, "anomalous_ids": anomalous_ids,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": prec, "recall": rec, "f1": det_f1
            })
            
        # Run level stats
        last_round = cur.execute("SELECT accuracy, f1, auc FROM round_metrics WHERE run_id=? AND round=8", (run_id,)).fetchone()
        
        # Agent logs for this run
        agent_logs = cur.execute("SELECT latency_ms, fallback_reason FROM agent_log WHERE run_id=?", (run_id,)).fetchall()
        latencies = [l["latency_ms"]/1000.0 for l in agent_logs if l["latency_ms"] is not None]
        avg_lat = sum(latencies)/len(latencies) if latencies else 0.0
        fallbacks = sum(1 for l in agent_logs if l["fallback_reason"] is not None)
        total_calls = len(agent_logs)
        
        run_stats.append({
            "method": method, "alpha": alpha, "seed": seed, "run_id": run_id,
            "final_accuracy": last_round["accuracy"],
            "final_f1": last_round["f1"],
            "final_auc": last_round["auc"],
            "avg_llm_latency": avg_lat,
            "total_llm_calls": total_calls,
            "fallback_count": fallbacks
        })

    # Export Run-level CSV
    with open("experiment2_label_flip_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "alpha", "seed", "run_id", "final_accuracy", "final_f1", "final_auc", 
                         "average_llm_latency", "total_llm_calls", "fallback_count"])
        for run in run_stats:
            writer.writerow([
                run["method"], run["alpha"], run["seed"], run["run_id"],
                f"{run['final_accuracy']:.2f}", f"{run['final_f1']:.4f}", f"{run['final_auc']:.4f}",
                f"{run['avg_llm_latency']:.2f}", run["total_llm_calls"], run["fallback_count"]
            ])

    # Export Detection CSV
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

    # Summary JSON
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
    print(f"Exported data for {len(completed_run_ids)} fully completed runs.")

if __name__ == "__main__":
    export()
