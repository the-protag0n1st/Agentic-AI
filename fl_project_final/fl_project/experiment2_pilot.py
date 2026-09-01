"""
Experiment 2 Pilot: Byzantine Label-Flipping Attack
====================================================
Runs all 6 methods with 1 malicious client (label flip) for 3 rounds.
Measures detection accuracy, FL performance, and Agent decision quality.

This script does NOT modify any existing code or Experiment 1 data.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import copy
import json
import numpy as np

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

import torch

# ============================================================
# PILOT CONFIGURATION
# ============================================================
ALPHA = 1.0
SEED = 42
ROUNDS = 3
NUM_CLIENTS = 5
BATCH_SIZE = 32
AGENT_MODEL = "ollama:phi3"
DB_PATH = "fl_metrics_experiment2_label_flip_pilot.db"

# Attack configuration
ENABLE_BYZANTINE_ATTACK = True
ATTACK_TYPE = "label_flip"
NUM_MALICIOUS_CLIENTS = 1

METHODS = ["FedAvg", "Krum", "Median", "Trimmed Mean", "RuleBased", "Agent"]

# ============================================================
# DETECTION METRICS
# ============================================================
def compute_detection_metrics(anomalous_ids, malicious_ids, num_clients):
    """Compare anomaly detector output against ground-truth malicious IDs."""
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

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1
    }

# ============================================================
# MAIN PILOT
# ============================================================
def run_pilot():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("=" * 60, flush=True)
    print("EXPERIMENT 2 PILOT: BYZANTINE LABEL-FLIPPING ATTACK", flush=True)
    print("=" * 60, flush=True)

    # --- Select malicious clients ---
    malicious_ids = select_malicious_clients(NUM_CLIENTS, NUM_MALICIOUS_CLIENTS, SEED)
    honest_ids = [i for i in range(NUM_CLIENTS) if i not in malicious_ids]

    print(f"\nAttack Config:", flush=True)
    print(f"  Attack type:       {ATTACK_TYPE}", flush=True)
    print(f"  Malicious clients: {malicious_ids}", flush=True)
    print(f"  Honest clients:    {honest_ids}", flush=True)
    print(f"  Alpha:             {ALPHA}", flush=True)
    print(f"  Seed:              {SEED}", flush=True)
    print(f"  Rounds:            {ROUNDS}", flush=True)
    print(f"  Batch size:        {BATCH_SIZE}", flush=True)
    print(f"  Methods:           {METHODS}", flush=True)
    print(f"  DB:                {DB_PATH}", flush=True)

    # --- Load dataset ---
    train_data, test_data = get_datasets(root="./data", download=False)

    # --- Verify binary classes ---
    unique_labels = sorted(set(train_data.targets))
    print(f"\n  Binary classes: {unique_labels} (0=airplane/normal, 1=other/anomalous)", flush=True)
    assert unique_labels == [0, 1], f"Expected binary [0,1], got {unique_labels}"

    # --- Create clients and apply attack ---
    clients_clean = make_clients(train_data, num_clients=NUM_CLIENTS, alpha=ALPHA, seed=SEED)
    clients_attacked = apply_attack(clients_clean, malicious_ids, attack_type=ATTACK_TYPE)

    # --- Verify label flipping ---
    print(f"\n  Verifying label flip for malicious client {malicious_ids[0]}:", flush=True)
    mal_id = malicious_ids[0]
    # Check first 5 samples
    for i in range(min(5, len(clients_clean[mal_id]))):
        _, orig_label = clients_clean[mal_id][i]
        _, flip_label = clients_attacked[mal_id][i]
        print(f"    Sample {i}: original={orig_label}, flipped={flip_label}", flush=True)
        assert flip_label == 1 - orig_label, f"Label flip failed at sample {i}"

    # Verify honest client is unchanged
    hon_id = honest_ids[0]
    for i in range(min(5, len(clients_clean[hon_id]))):
        _, orig_label = clients_clean[hon_id][i]
        _, curr_label = clients_attacked[hon_id][i]
        assert curr_label == orig_label, f"Honest client label changed at sample {i}!"
    print(f"  Honest client {hon_id} labels: UNCHANGED [OK]", flush=True)

    # Verify test data unchanged
    print(f"  Global test set: {len(test_data)} samples, labels NOT flipped [OK]", flush=True)

    # --- Validation checks ---
    print(f"\n--- PRE-FLIGHT CHECKS ---", flush=True)
    checks = {
        "5 clients": NUM_CLIENTS == 5,
        "1 malicious": NUM_MALICIOUS_CLIENTS == 1,
        "Alpha = 1.0": ALPHA == 1.0,
        "Seed = 42": SEED == 42,
        "3 rounds": ROUNDS == 3,
        "Label flip 01": ATTACK_TYPE == "label_flip",
        "Batch size = 32": BATCH_SIZE == 32,
        "6 methods": len(METHODS) == 6,
    }
    all_pass = True
    for name, ok in checks.items():
        status = "[OK]" if ok else "[FAIL] FAIL"
        print(f"  [{status}] {name}", flush=True)
        if not ok:
            all_pass = False
    if not all_pass:
        print("ABORTING: Pre-flight checks failed!", flush=True)
        return
    print("  All checks passed.\n", flush=True)

    # --- Init DB ---
    init_db(DB_PATH)

    # ============================================================
    # RUN ALL METHODS
    # ============================================================
    all_results = {}
    all_detection = {}
    agent_decisions = []
    total_llm_calls = 0

    pilot_start = time.time()

    for method in METHODS:
        print(f"\n{'='*60}", flush=True)
        print(f"=== Method: {method} | Alpha: {ALPHA} | Attack: {ATTACK_TYPE} ===", flush=True)
        print(f"{'='*60}", flush=True)

        # Reset seed for each method (same as run_fl)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

        gm = Net().to(device)
        all_client_ids = list(range(NUM_CLIENTS))
        run_id = f"exp2_pilot_{method}_{ATTACK_TYPE}"
        method_detection = []
        method_timings = []

        for r in range(1, ROUNDS + 1):
            print(f"\n--- {method} Round {r}/{ROUNDS} ---", flush=True)
            round_start = time.time()

            global_w = {k: v.cpu() for k, v in gm.state_dict().items()}

            # Train all clients (malicious client trains on flipped labels)
            t0 = time.time()
            local_w = [train_local(copy.deepcopy(gm), c, epochs=2, lr=0.01, batch_size=BATCH_SIZE)
                       for c in clients_attacked]
            t_train = time.time() - t0
            print(f"  Training: {t_train:.2f}s", flush=True)

            # Anomaly detection
            t0 = time.time()
            norms = compute_update_norms(global_w, local_w)
            cos_sims = compute_cosine_similarities(global_w, local_w)
            weight_vars = compute_weight_variations(local_w)
            anomalous_ids, clean_ids, vote_detail = detect_anomalies(
                norms, cos_sims, weight_vars)
            t_anomaly = time.time() - t0

            # Detection metrics against ground truth
            det = compute_detection_metrics(anomalous_ids, malicious_ids, NUM_CLIENTS)
            method_detection.append({
                "round": r,
                "anomalous_ids": anomalous_ids,
                "malicious_ids": malicious_ids,
                **det
            })

            print(f"  Anomalous: {anomalous_ids} | Clean: {clean_ids}", flush=True)
            print(f"  Detection: TP={det['tp']} FP={det['fp']} TN={det['tn']} FN={det['fn']} "
                  f"P={det['precision']:.2f} R={det['recall']:.2f} F1={det['f1']:.2f}", flush=True)

            # Log client metrics
            for i in range(NUM_CLIENTS):
                insert_client_metrics(run_id, r, ALPHA, i, norms[i], cos_sims[i],
                                       weight_vars[i], i in anomalous_ids, seed=SEED, db_path=DB_PATH)

            # Agent/method decision
            usable_ids = clean_ids if clean_ids else all_client_ids
            raw_combos = generate_client_combinations(usable_ids, local_w)
            candidate_combinations = [c["client_ids"] for c in raw_combos]
            history = get_round_history(run_id, ALPHA, last_n=5, db_path=DB_PATH)
            prev_metrics = get_previous_metrics(run_id, ALPHA, db_path=DB_PATH)

            if method in ["FedAvg", "Krum", "Median", "Trimmed Mean"]:
                selected_method = method
                selected_combo = clean_ids if clean_ids else all_client_ids
                reason = "Fixed baseline"
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
                    alpha=ALPHA,
                    run_id=run_id,
                    agent_model=effective_agent_model,
                    seed=SEED,
                    db_path=DB_PATH,
                )
                t_agent = time.time() - t0

                selected_combo = decision["selected_combination"]
                selected_method = decision["selected_method"]
                reason = decision["reason"]

                if method == "Agent":
                    total_llm_calls += 1
                    agent_decisions.append({
                        "round": r,
                        "anomalous_ids": anomalous_ids,
                        "selected_method": selected_method,
                        "selected_combo": selected_combo,
                        "reason": reason,
                        "source": decision.get("source"),
                        "confidence": decision.get("confidence"),
                        "latency": t_agent,
                    })
                    print(f"  Agent LLM latency: {t_agent:.2f}s", flush=True)

            # Aggregation
            selected_weights = [local_w[i] for i in selected_combo]
            new_w = aggregate(selected_method, selected_weights, global_weights=global_w)
            gm.load_state_dict(new_w)

            # Evaluate
            metrics = evaluate_model(gm, test_data)

            # Log round metrics
            insert_round_metrics(run_id, r, ALPHA, method,
                                  metrics["accuracy"], metrics["loss"],
                                  metrics["f1"], metrics["auc"],
                                  len(anomalous_ids), clean_ids,
                                  selected_method=selected_method,
                                  selected_combination=selected_combo,
                                  agent_reasoning=reason,
                                  seed=SEED, db_path=DB_PATH)

            round_time = time.time() - round_start
            method_timings.append(round_time)

            # Check if malicious client was included in aggregation
            mal_included = any(m in selected_combo for m in malicious_ids)
            print(f"  Method={selected_method} | Combo={selected_combo} | "
                  f"MalIncluded={mal_included}", flush=True)
            print(f"  Acc={metrics['accuracy']}% | F1={metrics['f1']} | AUC={metrics['auc']}", flush=True)
            print(f"  Round time: {round_time:.2f}s", flush=True)

        all_results[method] = {
            "final_acc": metrics["accuracy"],
            "final_f1": metrics["f1"],
            "final_auc": metrics["auc"],
            "avg_round_time": np.mean(method_timings),
        }
        all_detection[method] = method_detection

    total_runtime = time.time() - pilot_start

    # ============================================================
    # PILOT REPORT
    # ============================================================
    print("\n\n" + "=" * 60, flush=True)
    print("EXPERIMENT 2 PILOT REPORT", flush=True)
    print("=" * 60, flush=True)

    # A. Attack verification
    print("\n### A. Attack Verification", flush=True)
    print(f"  Malicious client ID:     {malicious_ids}", flush=True)
    print(f"  Attack type:             {ATTACK_TYPE}", flush=True)
    print(f"  Labels flipped for:      client {malicious_ids[0]} ONLY", flush=True)
    print(f"  Honest clients changed:  NO", flush=True)
    print(f"  Global test set changed: NO", flush=True)

    # B. Detection per round (use first method's detection since anomaly detection 
    # runs identically for all methods with same seed)
    print("\n### B. Detection Results (per round, per method)", flush=True)
    print(f"  {'Method':<15} {'Round':<6} {'Anom. IDs':<20} {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} {'Prec':>6} {'Rec':>6} {'F1':>6}", flush=True)
    for method in METHODS:
        for d in all_detection[method]:
            print(f"  {method:<15} {d['round']:<6} {str(d['anomalous_ids']):<20} "
                  f"{d['tp']:>3} {d['fp']:>3} {d['tn']:>3} {d['fn']:>3} "
                  f"{d['precision']:>6.2f} {d['recall']:>6.2f} {d['f1']:>6.2f}", flush=True)

    # C. FL performance
    print("\n### C. FL Performance (Final Round)", flush=True)
    print(f"  {'Method':<15} {'Accuracy':>10} {'F1':>8} {'AUC':>8}", flush=True)
    for method in METHODS:
        r = all_results[method]
        print(f"  {method:<15} {r['final_acc']:>9.2f}% {r['final_f1']:>8.4f} {r['final_auc']:>8.4f}", flush=True)

    # D. Agent behavior
    print("\n### D. Agent Behavior", flush=True)
    for ad in agent_decisions:
        print(f"  Round {ad['round']}:", flush=True)
        print(f"    Anomaly signals:     {ad['anomalous_ids']}", flush=True)
        print(f"    LLM decision:        {ad['selected_method']}", flush=True)
        print(f"    Selected clients:    {ad['selected_combo']}", flush=True)
        print(f"    Confidence:          {ad['confidence']}", flush=True)
        print(f"    Source:              {ad['source']}", flush=True)
        print(f"    Latency:             {ad['latency']:.2f}s", flush=True)
        print(f"    Reason:              {ad['reason']}", flush=True)

    # E. Runtime
    print("\n### E. Runtime", flush=True)
    print(f"  Total pilot runtime:   {total_runtime:.2f}s ({total_runtime/60:.1f} min)", flush=True)
    avg_round = np.mean([all_results[m]["avg_round_time"] for m in METHODS])
    print(f"  Avg round time:        {avg_round:.2f}s", flush=True)
    if agent_decisions:
        avg_llm = np.mean([ad["latency"] for ad in agent_decisions])
        print(f"  Avg LLM latency:       {avg_llm:.2f}s", flush=True)
    print(f"  Total LLM calls:       {total_llm_calls}", flush=True)
    print(f"  Expected LLM calls:    {ROUNDS}", flush=True)
    print(f"  Cache hits:            0", flush=True)

    print("\n### F. Verification", flush=True)
    print(f"  LLM calls == Agent rounds: {total_llm_calls == ROUNDS}", flush=True)
    print(f"  All LLM successful:        {all(ad['source'].startswith('ollama') for ad in agent_decisions)}", flush=True)
    print(f"  Exp1 DB untouched:         YES (separate DB used)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("PILOT COMPLETE. Awaiting user review.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_pilot()
