import sys
sys.stdout.reconfigure(line_buffering=True)
import time
import copy
import uuid

from models import Net, train_local, evaluate_model, device
from anomaly import (compute_update_norms, compute_cosine_similarities,
                      compute_weight_variations, detect_anomalies)
from aggregation import aggregate
from agent import agent_select
from dataset import get_datasets, make_clients

def profile_round():
    print("=== Device Diagnostics ===", flush=True)
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

    print("Loading datasets...", flush=True)
    t0 = time.time()
    train_data, test_data = get_datasets(root="./data", download=True)
    t1 = time.time()
    print(f"Dataset loading: {t1-t0:.2f}s", flush=True)
    
    print("Making clients...", flush=True)
    t0 = time.time()
    clients = make_clients(train_data, num_clients=5, alpha=1.0, seed=42)
    t1 = time.time()
    print(f"Client making: {t1-t0:.2f}s", flush=True)
    
    gm = Net().to(device)
    
    for r in range(1, 4):
        print(f"\n--- Round {r} ---", flush=True)
        round_start = time.time()
        
        global_w = {k: v.cpu() for k, v in gm.state_dict().items()}
        
        t0 = time.time()
        local_w = []
        for i, c in enumerate(clients):
            w = train_local(copy.deepcopy(gm), c, epochs=2, lr=0.01, batch_size=32)
            local_w.append(w)
        t1 = time.time()
        print(f"Local training (5 clients): {t1-t0:.2f}s", flush=True)
        
        t0 = time.time()
        norms = compute_update_norms(global_w, local_w)
        cos_sims = compute_cosine_similarities(global_w, local_w)
        weight_vars = compute_weight_variations(local_w)
        anomalous_ids, clean_ids, _ = detect_anomalies(norms, cos_sims, weight_vars)
        t1 = time.time()
        print(f"Anomaly detection: {t1-t0:.2f}s", flush=True)
        
        t0 = time.time()
        candidate_combinations = [clean_ids] if clean_ids else [list(range(5))]
        
        decision = agent_select(
            round_num=r,
            anomalous_clients=anomalous_ids,
            candidate_combinations=candidate_combinations,
            history=[],
            metrics={},
            all_client_ids=list(range(5)),
            alpha=1.0,
            run_id="test",
            agent_model="ollama:phi3",
            seed=42,
            db_path="fl_metrics.db"
        )
        t1 = time.time()
        print(f"Agent decision (Ollama): {t1-t0:.2f}s", flush=True)
        
        t0 = time.time()
        selected_combo = decision["selected_combination"]
        selected_method = decision["selected_method"]
        selected_weights = [local_w[i] for i in selected_combo]
        new_w = aggregate(selected_method, selected_weights, global_weights=global_w)
        gm.load_state_dict(new_w)
        t1 = time.time()
        print(f"Aggregation: {t1-t0:.2f}s", flush=True)
        
        t0 = time.time()
        metrics = evaluate_model(gm, test_data)
        t1 = time.time()
        print(f"Evaluation: {t1-t0:.2f}s", flush=True)
        
        print(f"Total round time: {time.time()-round_start:.2f}s", flush=True)

if __name__ == '__main__':
    profile_round()
