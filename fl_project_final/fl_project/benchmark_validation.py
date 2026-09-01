import sys
sys.stdout.reconfigure(line_buffering=True)
import time
import copy
import json

from models import Net, train_local, evaluate_model, device
from anomaly import (compute_update_norms, compute_cosine_similarities,
                      compute_weight_variations, detect_anomalies)
from combinations import generate_client_combinations
from aggregation import aggregate
from agent import agent_select, _build_prompt
from database import (init_db, insert_client_metrics, insert_round_metrics,
                       get_round_history, get_previous_metrics)
from dataset import get_datasets, make_clients

import torch

# --- Configuration ---
ALPHA = 1.0
SEED = 42
ROUNDS = 3
NUM_CLIENTS = 5
BATCH_SIZE = 32
AGENT_MODEL = "ollama:phi3"
DB_PATH = "benchmark_validation.db"

def run_benchmark():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("=" * 60, flush=True)
    print("FINAL RUNTIME VALIDATION BENCHMARK", flush=True)
    print("=" * 60, flush=True)
    print(f"Config: alpha={ALPHA}, seed={SEED}, rounds={ROUNDS}, clients={NUM_CLIENTS}", flush=True)
    print(f"Batch size: {BATCH_SIZE}, Agent: {AGENT_MODEL}", flush=True)
    print(f"CUDA: {torch.cuda.is_available()}, Device: {device}", flush=True)
    print("=" * 60, flush=True)

    # --- Dataset loading ---
    t0 = time.time()
    train_data, test_data = get_datasets(root="./data", download=False)
    t_dataset = time.time() - t0
    print(f"\nDataset loading: {t_dataset:.2f}s", flush=True)

    # --- Client making ---
    t0 = time.time()
    clients = make_clients(train_data, num_clients=NUM_CLIENTS, alpha=ALPHA, seed=SEED)
    t_clients = time.time() - t0
    print(f"Client partitioning: {t_clients:.2f}s", flush=True)

    # --- Init DB ---
    init_db(DB_PATH)

    gm = Net().to(device)
    all_client_ids = list(range(NUM_CLIENTS))
    run_id = "benchmark_val"

    # Per-round timing storage
    round_timings = []
    llm_calls = 0
    llm_successes = 0
    llm_fallbacks = 0
    cache_hits = 0  # Must remain 0

    total_start = time.time()

    for r in range(1, ROUNDS + 1):
        print(f"\n{'='*40}", flush=True)
        print(f"--- Round {r}/{ROUNDS} ---", flush=True)
        round_start = time.time()
        timings = {}

        # 1. Local training
        global_w = {k: v.cpu() for k, v in gm.state_dict().items()}
        t0 = time.time()
        local_w = [train_local(copy.deepcopy(gm), c, epochs=2, lr=0.01, batch_size=BATCH_SIZE)
                   for c in clients]
        timings["local_training"] = time.time() - t0
        print(f"  Local training: {timings['local_training']:.2f}s", flush=True)

        # 2. Anomaly detection
        t0 = time.time()
        norms = compute_update_norms(global_w, local_w)
        cos_sims = compute_cosine_similarities(global_w, local_w)
        weight_vars = compute_weight_variations(local_w)
        anomalous_ids, clean_ids, _votes = detect_anomalies(norms, cos_sims, weight_vars)
        timings["anomaly_detection"] = time.time() - t0
        print(f"  Anomaly detection: {timings['anomaly_detection']:.4f}s", flush=True)
        print(f"    Anomalous: {anomalous_ids} | Clean: {clean_ids}", flush=True)

        # 3. Client metrics logging
        t0 = time.time()
        for i in range(NUM_CLIENTS):
            insert_client_metrics(run_id, r, ALPHA, i, norms[i], cos_sims[i],
                                   weight_vars[i], i in anomalous_ids, seed=SEED, db_path=DB_PATH)
        timings["client_logging"] = time.time() - t0

        # 4. Prompt construction (timed separately from LLM call)
        usable_ids = clean_ids if clean_ids else all_client_ids
        raw_combos = generate_client_combinations(usable_ids, local_w)
        candidate_combinations = [c["client_ids"] for c in raw_combos]
        history = get_round_history(run_id, ALPHA, last_n=5, db_path=DB_PATH)
        prev_metrics = get_previous_metrics(run_id, ALPHA, db_path=DB_PATH)

        payload = {
            "round": r,
            "anomalous_clients": list(anomalous_ids),
            "candidate_combinations": [list(c) for c in candidate_combinations],
            "history": history,
            "metrics": prev_metrics,
            "all_client_ids": list(all_client_ids),
            "alpha": ALPHA,
        }
        t0 = time.time()
        prompt = _build_prompt(payload)
        timings["prompt_construction"] = time.time() - t0
        print(f"  Prompt construction: {timings['prompt_construction']:.4f}s", flush=True)

        # 5. Agent decision (full LLM call)
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
            agent_model=AGENT_MODEL,
            seed=SEED,
            db_path=DB_PATH,
        )
        timings["llm_inference"] = time.time() - t0
        llm_calls += 1
        if decision.get("source", "").startswith("ollama"):
            llm_successes += 1
        else:
            llm_fallbacks += 1
        print(f"  LLM inference: {timings['llm_inference']:.2f}s", flush=True)
        print(f"    Source: {decision.get('source')}", flush=True)
        print(f"    Method: {decision['selected_method']}", flush=True)
        print(f"    Combo: {decision['selected_combination']}", flush=True)

        # 6. Aggregation
        t0 = time.time()
        selected_combo = decision["selected_combination"]
        selected_method = decision["selected_method"]
        selected_weights = [local_w[i] for i in selected_combo]
        new_w = aggregate(selected_method, selected_weights, global_weights=global_w)
        gm.load_state_dict(new_w)
        timings["aggregation"] = time.time() - t0
        print(f"  Aggregation: {timings['aggregation']:.4f}s", flush=True)

        # 7. Evaluation
        t0 = time.time()
        metrics = evaluate_model(gm, test_data)
        timings["evaluation"] = time.time() - t0
        print(f"  Evaluation: {timings['evaluation']:.2f}s", flush=True)
        print(f"    Acc={metrics['accuracy']}% | F1={metrics['f1']} | AUC={metrics['auc']}", flush=True)

        # 8. Round metrics logging
        t0 = time.time()
        insert_round_metrics(run_id, r, ALPHA, "Agent",
                              metrics["accuracy"], metrics["loss"],
                              metrics["f1"], metrics["auc"],
                              len(anomalous_ids), clean_ids,
                              selected_method=selected_method,
                              selected_combination=selected_combo,
                              agent_reasoning=decision["reason"],
                              seed=SEED, db_path=DB_PATH)
        timings["db_logging"] = time.time() - t0
        print(f"  DB logging: {timings['db_logging']:.4f}s", flush=True)

        timings["total_round"] = time.time() - round_start
        print(f"  TOTAL ROUND: {timings['total_round']:.2f}s", flush=True)

        round_timings.append(timings)

    total_runtime = time.time() - total_start

    # --- Summary ---
    print("\n" + "=" * 60, flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print("=" * 60, flush=True)

    avg = lambda key: sum(t[key] for t in round_timings) / len(round_timings)

    print(f"\nPer-round averages:", flush=True)
    print(f"  Local training:     {avg('local_training'):.2f}s", flush=True)
    print(f"  Anomaly detection:  {avg('anomaly_detection'):.4f}s", flush=True)
    print(f"  Prompt construction:{avg('prompt_construction'):.4f}s", flush=True)
    print(f"  LLM inference:      {avg('llm_inference'):.2f}s", flush=True)
    print(f"  Aggregation:        {avg('aggregation'):.4f}s", flush=True)
    print(f"  Evaluation:         {avg('evaluation'):.2f}s", flush=True)
    print(f"  DB logging:         {avg('db_logging'):.4f}s", flush=True)
    print(f"  Total round:        {avg('total_round'):.2f}s", flush=True)

    print(f"\nTotals:", flush=True)
    print(f"  Total runtime:      {total_runtime:.2f}s", flush=True)
    print(f"  Total LLM calls:    {llm_calls}", flush=True)
    print(f"  Expected LLM calls: {ROUNDS}", flush=True)
    print(f"  Successful LLM:     {llm_successes}", flush=True)
    print(f"  Fallback decisions: {llm_fallbacks}", flush=True)
    print(f"  Cache hits:         {cache_hits}", flush=True)

    print(f"\nVerification:", flush=True)
    print(f"  LLM calls == rounds: {llm_calls == ROUNDS}", flush=True)
    print(f"  Cache hits == 0:     {cache_hits == 0}", flush=True)
    print(f"  All LLM successful:  {llm_successes == ROUNDS}", flush=True)

    # Cleanup benchmark db
    import os
    os.remove(DB_PATH)
    print(f"\nCleaned up {DB_PATH}", flush=True)

if __name__ == "__main__":
    run_benchmark()
