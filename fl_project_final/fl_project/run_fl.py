import copy
import uuid

from models import Net, train_local, evaluate_model, device
from anomaly import (compute_update_norms, compute_cosine_similarities,
                      compute_weight_variations, detect_anomalies)
from combinations import generate_client_combinations
from aggregation import aggregate
from agent import agent_select
from database import (init_db, insert_client_metrics, insert_round_metrics,
                       get_round_history, get_previous_metrics)


def run_fl(clients, test_data, rounds=10, alpha=1.0, agent_model="ollama:phi3", method=None,
           run_id=None, seed=42, local_epochs=2, local_lr=0.01, local_batch_size=32,
           norm_z_threshold=2.0, cosine_threshold=-0.2, var_z_threshold=2.0,
           min_votes=1, db_path="fl_metrics.db", verbose=True,
           include_history=True, include_anomaly_signals=True,
           allow_client_selection=True, allow_aggregation_selection=True):
    import torch as _torch
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed(seed)

    init_db(db_path)
    if run_id is None:
        run_id = f"a{alpha}_{str(uuid.uuid4())[:6]}"

    gm = Net().to(device)
    logs = {"acc": [], "loss": [], "f1": [], "auc": [], "method": [],
            "n_anomalous": [], "selected_combination": []}

    all_client_ids = list(range(len(clients)))

    for r in range(1, rounds + 1):
        if verbose:
            print(f"\n--- Run {run_id} | alpha={alpha} | Round {r}/{rounds} ---")

        global_w = {k: v.cpu() for k, v in gm.state_dict().items()}

        local_w = [train_local(copy.deepcopy(gm), c, epochs=local_epochs, lr=local_lr, batch_size=local_batch_size)
                   for c in clients]

        norms = compute_update_norms(global_w, local_w)
        cos_sims = compute_cosine_similarities(global_w, local_w)
        weight_vars = compute_weight_variations(local_w)
        anomalous_ids, clean_ids, _votes = detect_anomalies(
            norms, cos_sims, weight_vars,
            norm_z_threshold=norm_z_threshold,
            cosine_threshold=cosine_threshold,
            var_z_threshold=var_z_threshold,
            min_votes=min_votes,
        )

        for i in range(len(clients)):
            insert_client_metrics(run_id, r, alpha, i, norms[i], cos_sims[i],
                                   weight_vars[i], i in anomalous_ids, seed=seed, db_path=db_path)

        if verbose:
            print(f"  Anomalous: {anomalous_ids} | Clean: {clean_ids}")

        usable_ids = clean_ids if clean_ids else all_client_ids
        raw_combos = generate_client_combinations(usable_ids, local_w)
        candidate_combinations = [c["client_ids"] for c in raw_combos]

        history = get_round_history(run_id, alpha, last_n=5, db_path=db_path) if include_history else []
        prev_metrics = get_previous_metrics(run_id, alpha, db_path=db_path) if include_history else {}

        if method in ["FedAvg", "Krum", "Median", "Trimmed Mean", "FedProx"]:
            selected_method = method
            selected_combo = clean_ids if clean_ids else all_client_ids
            reason = "Fixed baseline"
        else:
            effective_agent_model = "RuleBased" if method == "RuleBased" else agent_model
            
            payload_anomalous = anomalous_ids if include_anomaly_signals else []
            payload_candidates = candidate_combinations if allow_client_selection else [all_client_ids]
            
            decision = agent_select(
                round_num=r,
                anomalous_clients=payload_anomalous,
                candidate_combinations=payload_candidates,
                history=history,
                metrics=prev_metrics,
                all_client_ids=all_client_ids,
                alpha=alpha,
                run_id=run_id,
                agent_model=effective_agent_model,
                seed=seed,
                db_path=db_path,
            )
            
            if not allow_client_selection:
                selected_combo = clean_ids if clean_ids else all_client_ids
            else:
                selected_combo = decision["selected_combination"]
                
            if not allow_aggregation_selection:
                selected_method = "FedAvg"
            else:
                selected_method = decision["selected_method"]
                
            reason = decision["reason"]

        selected_weights = [local_w[i] for i in selected_combo]
        new_w = aggregate(selected_method, selected_weights, global_weights=global_w)
        gm.load_state_dict(new_w)

        metrics = evaluate_model(gm, test_data)

        insert_round_metrics(run_id, r, alpha, method,
                              metrics["accuracy"], metrics["loss"],
                              metrics["f1"], metrics["auc"],
                              len(anomalous_ids), clean_ids,
                              selected_method=selected_method,
                              selected_combination=selected_combo,
                              agent_reasoning=reason,
                              seed=seed,
                              db_path=db_path)

        logs["acc"].append(metrics["accuracy"])
        logs["loss"].append(metrics["loss"])
        logs["f1"].append(metrics["f1"])
        logs["auc"].append(metrics["auc"])
        logs["method"].append(selected_method)
        logs["n_anomalous"].append(len(anomalous_ids))
        logs["selected_combination"].append(selected_combo)

        if verbose:
            print(f"  Method={selected_method} | Combo={selected_combo} | "
                  f"Acc={metrics['accuracy']}% | F1={metrics['f1']} | AUC={metrics['auc']}")

    logs["run_id"] = run_id
    logs["global_state"] = gm.state_dict()
    logs["final_metrics"] = metrics
    return logs


def run_experiment(alphas, num_clients=5, rounds=10, methods=["Agent"],
                    agent_model="ollama:phi3", db_path="fl_metrics.db", seed=42, normal_classes=None,
                    download=True, root="./data",
                    include_history=True, include_anomaly_signals=True,
                    allow_client_selection=True, allow_aggregation_selection=True,
                    local_batch_size=32):
    from dataset import get_datasets, make_clients

    train_data, test_data = get_datasets(normal_classes=normal_classes,
                                          root=root, download=download)
    results = {}
    for alpha in alphas:
        results[alpha] = {}
        for method in methods:
            print(f"\n=== Starting Method: {method} | Alpha: {alpha} ===")
            clients = make_clients(train_data, num_clients=num_clients, alpha=alpha, seed=seed)
            try:
                logs = run_fl(clients, test_data, rounds=rounds, alpha=alpha,
                               agent_model=agent_model, method=method, run_id=None, seed=seed, db_path=db_path,
                               include_history=include_history,
                               include_anomaly_signals=include_anomaly_signals,
                               allow_client_selection=allow_client_selection,
                               allow_aggregation_selection=allow_aggregation_selection,
                               local_batch_size=local_batch_size)
                results[alpha][method] = logs
            except Exception as e:
                import traceback
                print(f"FAILED RUN: method={method} alpha={alpha} seed={seed}")
                traceback.print_exc()

    return results, train_data, test_data
