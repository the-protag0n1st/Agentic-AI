"""Experiment 3 Execution Engine: Dynamic Heterogeneity & Byzantine Schedule.

Executes the 7 canonical FL methods across dynamic regimes:
  - Rounds 1-5:   alpha=100.0 (near-IID)
  - Rounds 6-10:  alpha=1.0   (moderate non-IID)
  - Rounds 11-15: alpha=0.2   (severe non-IID)
  - Rounds 16-20: alpha=0.2 + 2 Byzantine attackers (-3.0 * clean update)

Features:
  - Safe, deterministic resume from exact (method, seed) state checkpoints
  - Quota-aware rate-limit header tracking and safe stopping
  - Training-derived validation split for Counterfactual Oracle evaluation
  - Strict separation from frozen Experiment 1 & 2 databases
  - Real dynamic partition switching per regime
  - Exact candidate pool enforcement and deterministic validation/repair
  - Same-model fairness for SingleShotLLM and ReflectiveAgent
"""

import copy
import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Subset

from aggregation import aggregate
from anomaly import compute_cosine_similarities, compute_update_norms, compute_weight_variations, detect_anomalies
from candidates import generate_experiment3_candidates
from dataset import get_datasets, make_clients
from experiment3_config import (
    ATTACK_SCALE,
    BYZANTINE_BUDGET,
    DB_PATH,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    EVAL_BATCH_SIZE,
    GROQ_DAILY_REQUEST_LIMIT,
    GROQ_DAILY_TOKEN_LIMIT,
    GROQ_MIN_REMAINING_REQUESTS,
    GROQ_MIN_REMAINING_TOKENS,
    GROQ_REQUEST_SAFETY_MARGIN,
    GROQ_TOKEN_SAFETY_MARGIN,
    HISTORY_LENGTH,
    LOCAL_BATCH_SIZE,
    LOCAL_EPOCHS,
    LOCAL_LR,
    MAX_CANDIDATES,
    METHODS,
    NUM_CLIENTS,
    NUM_ROUNDS,
    TRIM_RATIO,
    VAL_SPLIT_SEED,
    VAL_SPLIT_SIZE,
    get_attacker_ids,
    get_environment_metadata,
    get_regime_for_round,
)
from experiment3_db import (
    commit_round_transaction,
    get_completed_rounds,
    init_experiment3_db,
    insert_llm_call_log,
    insert_run_record,
    load_latest_checkpoint,
)
from feasibility import repair_decision, validate_decision
from models import Net, device, evaluate_model, train_local
from oracle import CounterfactualOracle
from quota import QuotaPauseException, QuotaTracker
from agent.agentic_ai import ReflectiveAgent
from agent.agentic_controller import AgenticController
from agent.history import AgentHistory
from agent.llm_client import GroqLLM, LLMClient, MockLLM
from agent.schemas import AgentResult, ClientTelemetry, Decision, RoundTelemetry
from agent.single_shot import SingleShotAgent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_training_and_validation_splits(root: str = "./data", download: bool = False):
    """Split CIFAR-10 training set into train_pool and fixed val_subset for the oracle.
    
    The official CIFAR-10 test set is left untouched for final evaluation.
    """
    full_train, official_test = get_datasets(root=root, download=download)
    n_total = len(full_train)

    rng = np.random.default_rng(VAL_SPLIT_SEED)
    indices = list(range(n_total))
    rng.shuffle(indices)

    val_indices = indices[:VAL_SPLIT_SIZE]
    train_indices = indices[VAL_SPLIT_SIZE:]

    val_subset = Subset(full_train, val_indices)
    val_subset.targets = [full_train.targets[i] for i in val_indices]
    train_pool = Subset(full_train, train_indices)
    train_pool.targets = [full_train.targets[i] for i in train_indices]

    return train_pool, val_subset, official_test


def execute_byzantine_attack(
    local_weights: List[Dict[str, torch.Tensor]],
    global_weights: Dict[str, torch.Tensor],
    attacker_ids: List[int],
    attack_scale: float = ATTACK_SCALE,
) -> List[Dict[str, torch.Tensor]]:
    """Inject malicious updates post-training: malicious_update = -3.0 * clean_update."""
    attacked_weights = []
    attacker_set = set(attacker_ids)

    for i, w in enumerate(local_weights):
        if i in attacker_set:
            attacked_w = {}
            for k in w:
                clean_delta = w[k].float() - global_weights[k].float()
                attacked_w[k] = (global_weights[k].float() + attack_scale * clean_delta).to(w[k].dtype)
            attacked_weights.append(attacked_w)
        else:
            attacked_weights.append(w)

    return attacked_weights


def rule_based_decision(
    candidate_decisions: List[Decision],
    anomalous_ids: List[int],
    history: AgentHistory,
    alpha: float,
    all_client_ids: List[int],
    byzantine_budget: int = BYZANTINE_BUDGET,
    trim_ratio: float = TRIM_RATIO,
) -> Decision:
    """Deterministic, rule-based baseline decision using anomaly signals and history."""
    cmap = {c.candidate_id or f"C{i}": c for i, c in enumerate(candidate_decisions)}
    persistent = [cid for cid in anomalous_ids if history.has_persistent_anomaly(cid, min_rounds=2)]
    clean_clients = sorted(list(set(all_client_ids) - set(persistent if persistent else anomalous_ids)))

    if persistent:
        target_method = "Krum" if len(clean_clients) >= (2 * byzantine_budget + 3) else "TrimmedMean"
        target_clients = clean_clients
    elif alpha <= 0.2:
        target_method = "TrimmedMean"
        target_clients = clean_clients if clean_clients else all_client_ids
    elif anomalous_ids:
        target_method = "TrimmedMean"
        target_clients = clean_clients if clean_clients else all_client_ids
    else:
        target_method = "FedAvg"
        target_clients = all_client_ids

    best_dec, _, _ = repair_decision(
        None, cmap, target_clients=target_clients, target_method=target_method
    )
    return best_dec


def restore_history_from_checkpoint(history: AgentHistory, hist_data: List[Dict[str, Any]]):
    """Restore AgentHistory state from serialized checkpoint list."""
    history._rounds.clear()
    history._decisions.clear()
    for item in hist_data:
        clients = [
            ClientTelemetry(**c) if isinstance(c, dict) else c
            for c in item.get("clients", [])
        ]
        scalar_metrics = {
            k: float(v) for k, v in item.get("global_metrics", {}).items()
            if isinstance(v, (int, float))
        }
        rt = RoundTelemetry(
            round_num=item["round_num"],
            alpha=item.get("alpha"),
            clients=clients,
            global_metrics=scalar_metrics,
        )
        history._rounds.append(rt)


def run_experiment3_single_method(
    method: str,
    seed: int,
    train_pool,
    val_subset,
    test_data,
    llm_client: Optional[LLMClient] = None,
    rounds: int = NUM_ROUNDS,
    num_clients: int = NUM_CLIENTS,
    db_path: str = DB_PATH,
    resume: bool = True,
    quota_tracker: Optional[QuotaTracker] = None,
    quota_safe: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute a single method for a given seed in Experiment 3 with resume and quota safety."""
    set_seed(seed)
    init_experiment3_db(db_path)
    all_client_ids = list(range(num_clients))
    run_id = f"exp3_{method}_s{seed}_{int(time.time()*1000)%1000000}"

    # 1. Inspect DB for completed rounds for this exact (method, seed)
    completed_rounds = get_completed_rounds(db_path, method, seed) if resume else []
    if resume and len(completed_rounds) >= rounds:
        print(f"Experiment 3: seed={seed}, method={method} already completed ({len(completed_rounds)}/{rounds} rounds). Skipping.")
        return {
            "run_id": run_id,
            "method": method,
            "seed": seed,
            "status": "already_completed",
            "rounds_completed": len(completed_rounds),
            "quota_paused": False,
        }

    start_round = 1
    gm = Net().to(device)
    oracle = CounterfactualOracle(val_subset=val_subset)
    history = AgentHistory(history_length=HISTORY_LENGTH)

    if resume and completed_rounds:
        last_completed = max(completed_rounds)
        start_round = last_completed + 1
        print(f"Resuming Experiment 3: seed={seed}, method={method}, starting from round={start_round}")

        # Load latest checkpoint state
        chk = load_latest_checkpoint(db_path, method, seed)
        if chk:
            gm.load_state_dict(chk["model_weights"])
            restore_history_from_checkpoint(history, chk.get("history_json", []))
            oracle.cumulative_regret = chk.get("cumulative_regret", 0.0)
            if verbose:
                print(f"  Successfully restored model checkpoint and history from round {chk['round']}.")
    else:
        print(f"Starting Experiment 3: seed={seed}, method={method} from round=1")

    # Initialize Active LLM with quota tracker if method requires LLM
    active_llm = llm_client
    if method in ("ReflectiveAgent", "SingleShotLLM", "AgenticAI") and active_llm is None:
        active_llm = GroqLLM(
            model=DEFAULT_LLM_MODEL,
            quota_tracker=quota_tracker,
            quota_safe=quota_safe,
        )
    elif active_llm is not None and quota_tracker is not None:
        active_llm.quota_tracker = quota_tracker
        if hasattr(active_llm, "quota_safe"):
            active_llm.quota_safe = quota_safe

    # Log run record
    env_meta = get_environment_metadata()
    model_name = getattr(active_llm, "model", "N/A") if active_llm else "N/A"
    provider_name = "mock" if isinstance(active_llm, MockLLM) else ("groq" if isinstance(active_llm, GroqLLM) else "rule_or_fixed")
    insert_run_record(run_id, method, seed, model_name, provider_name, env_meta, db_path=db_path)

    # Pre-generate deterministic client partitions for each alpha regime (Section 11)
    partitions = {
        100.0: make_clients(train_pool, num_clients=num_clients, alpha=100.0, seed=seed),
        1.0: make_clients(train_pool, num_clients=num_clients, alpha=1.0, seed=seed),
        0.2: make_clients(train_pool, num_clients=num_clients, alpha=0.2, seed=seed),
    }
    attacker_ids = get_attacker_ids(seed=seed, num_clients=num_clients, num_attackers=BYZANTINE_BUDGET)

    # Initialize Agent if applicable
    reflective_agent = None
    single_shot_agent = None
    agentic_controller = None
    if method == "AgenticAI":
        agentic_controller = AgenticController(
            llm_client=active_llm,
            byzantine_budget=BYZANTINE_BUDGET,
            trim_ratio=TRIM_RATIO,
            history_length=HISTORY_LENGTH,
        )
    elif method == "ReflectiveAgent":
        reflective_agent = ReflectiveAgent(
            llm_client=active_llm,
            byzantine_budget=BYZANTINE_BUDGET,
            trim_ratio=TRIM_RATIO,
            history_length=HISTORY_LENGTH,
        )
    elif method == "SingleShotLLM":
        single_shot_agent = SingleShotAgent(
            llm_client=active_llm,
            byzantine_budget=BYZANTINE_BUDGET,
            trim_ratio=TRIM_RATIO,
            history_length=HISTORY_LENGTH,
        )

    round_logs = []
    quota_paused = False

    for r in range(start_round, rounds + 1):
        t_round_start = time.time()
        regime = get_regime_for_round(r)
        alpha = regime["alpha"]
        attack_active = regime["attack_active"]
        active_clients = partitions[alpha]

        if verbose:
            print(f"\n--- [{method} | Seed {seed}] Round {r}/{rounds} | alpha={alpha} | attack={attack_active} ---")

        global_w = {k: v.cpu() for k, v in gm.state_dict().items()}

        # 1. Local Training
        t0 = time.time()
        local_w = [
            train_local(copy.deepcopy(gm), client, epochs=LOCAL_EPOCHS, lr=LOCAL_LR, batch_size=LOCAL_BATCH_SIZE)
            for client in active_clients
        ]
        t_train = time.time() - t0

        # 2. Post-Training Attack Injection (Rounds 16-20 only)
        if attack_active:
            local_w = execute_byzantine_attack(local_w, global_w, attacker_ids, attack_scale=ATTACK_SCALE)

        # 3. Anomaly Detection (Attacker IDs are strictly HIDDEN from the agent)
        norms = compute_update_norms(global_w, local_w)
        cos_sims = compute_cosine_similarities(global_w, local_w)
        weight_vars = compute_weight_variations(local_w)
        anomalous_ids, clean_ids, _ = detect_anomalies(norms, cos_sims, weight_vars)

        current_telemetry = [
            ClientTelemetry(
                client_id=i,
                update_norm=norms[i],
                cosine_similarity=cos_sims[i],
                weight_var=weight_vars[i],
                is_flagged_by_detector=(i in anomalous_ids),
            )
            for i in range(num_clients)
        ]

        # 4. Candidate Generation (~10 pre-approved feasible candidates)
        persistent_ids = [cid for cid in anomalous_ids if history.has_persistent_anomaly(cid, min_rounds=2)]
        candidates = generate_experiment3_candidates(
            all_client_ids=all_client_ids,
            flagged_ids=anomalous_ids,
            persistent_ids=persistent_ids,
            byzantine_budget=BYZANTINE_BUDGET,
            trim_ratio=TRIM_RATIO,
            max_candidates=MAX_CANDIDATES,
        )

        agent_result: Optional[AgentResult] = None
        selected_decision: Decision

        # 5. Method Decision with safe QuotaPauseException catching
        try:
            if method == "FixedFedAvg":
                selected_decision = Decision(method="FedAvg", client_ids=all_client_ids, candidate_id="C0")
            elif method == "FixedMedian":
                selected_decision = Decision(method="Median", client_ids=all_client_ids, candidate_id="C1")
            elif method == "FixedTrimmedMean":
                selected_decision = Decision(method="TrimmedMean", client_ids=all_client_ids, candidate_id="C2")
            elif method == "FixedKrum":
                selected_decision = Decision(method="Krum", client_ids=all_client_ids, candidate_id="C3")
            elif method == "RuleBased":
                selected_decision = rule_based_decision(
                    candidates, anomalous_ids, history, alpha, all_client_ids
                )
            elif method == "SingleShotLLM":
                agent_result = single_shot_agent.decide(
                    round_num=r,
                    telemetry=current_telemetry,
                    candidate_decisions=candidates,
                    history=history,
                    alpha=alpha,
                    valid_client_ids=all_client_ids,
                )
                selected_decision = agent_result.final_decision
                # Log single-shot LLM call metadata
                if hasattr(active_llm, "last_call_metadata") and active_llm.last_call_metadata:
                    m = active_llm.last_call_metadata
                    insert_llm_call_log(
                        run_id=run_id, round_num=r, stage="single_shot", model=m.get("model", model_name),
                        prompt_version="v1_single_shot", prompt="[single_shot_prompt]", completion="[completion]",
                        input_tokens=m.get("input_tokens"), output_tokens=m.get("output_tokens"),
                        total_tokens=m.get("total_tokens"), latency_ms=m.get("latency_ms"),
                        retries=m.get("retries", 0), status=m.get("status", "success"),
                        error=m.get("error"), request_id=m.get("request_id"),
                        requests_remaining=m.get("requests_remaining"), requests_limit=m.get("requests_limit"),
                        requests_reset_time=m.get("requests_reset_time"), tokens_remaining=m.get("tokens_remaining"),
                        tokens_limit=m.get("tokens_limit"), tokens_reset_time=m.get("tokens_reset_time"),
                        db_path=db_path,
                    )
            elif method == "ReflectiveAgent":
                agent_result = reflective_agent.decide(
                    round_num=r,
                    telemetry=current_telemetry,
                    candidate_decisions=candidates,
                    history=history,
                    alpha=alpha,
                    valid_client_ids=all_client_ids,
                )
                selected_decision = agent_result.final_decision
                # Log 3-stage LLM call metadata
                for stage_name, lat in [
                    ("analyst", agent_result.timing.analyst_latency_ms),
                    ("proposer", agent_result.timing.proposer_latency_ms),
                    ("critic", agent_result.timing.critic_latency_ms),
                ]:
                    m = getattr(active_llm, "last_call_metadata", {}) or {}
                    insert_llm_call_log(
                        run_id=run_id, round_num=r, stage=stage_name, model=model_name,
                        prompt_version=f"v2_{stage_name}", prompt=f"[{stage_name}_prompt]", completion="[completion]",
                        input_tokens=m.get("input_tokens"), output_tokens=m.get("output_tokens"),
                        total_tokens=m.get("total_tokens"), latency_ms=lat,
                        retries=m.get("retries", 0), status="success" if stage_name not in agent_result.stage_errors else "error",
                        error=agent_result.stage_errors.get(stage_name), request_id=m.get("request_id"),
                        requests_remaining=m.get("requests_remaining"), requests_limit=m.get("requests_limit"),
                        requests_reset_time=m.get("requests_reset_time"), tokens_remaining=m.get("tokens_remaining"),
                        tokens_limit=m.get("tokens_limit"), tokens_reset_time=m.get("tokens_reset_time"),
                        db_path=db_path,
                    )
            elif method == "AgenticAI":
                agent_result = agentic_controller.decide(
                    round_num=r,
                    telemetry=current_telemetry,
                    candidate_decisions=candidates,
                    history=history,
                    alpha=alpha,
                    valid_client_ids=all_client_ids,
                )
                selected_decision = agent_result.final_decision
                # Log Agentic AI stage call metadata
                for stage_name, lat in [
                    ("agentic_analyst", agent_result.timing.analyst_latency_ms),
                    ("agentic_proposer", agent_result.timing.proposer_latency_ms),
                    ("agentic_critic", agent_result.timing.critic_latency_ms),
                ]:
                    m = getattr(active_llm, "last_call_metadata", {}) or {}
                    insert_llm_call_log(
                        run_id=run_id, round_num=r, stage=stage_name, model=model_name,
                        prompt_version=f"v3_{stage_name}", prompt=f"[{stage_name}_prompt]", completion="[completion]",
                        input_tokens=m.get("input_tokens"), output_tokens=m.get("output_tokens"),
                        total_tokens=m.get("total_tokens"), latency_ms=lat,
                        retries=m.get("retries", 0), status="success" if stage_name not in agent_result.stage_errors else "error",
                        error=agent_result.stage_errors.get(stage_name), request_id=m.get("request_id"),
                        requests_remaining=m.get("requests_remaining"), requests_limit=m.get("requests_limit"),
                        requests_reset_time=m.get("requests_reset_time"), tokens_remaining=m.get("tokens_remaining"),
                        tokens_limit=m.get("tokens_limit"), tokens_reset_time=m.get("tokens_reset_time"),
                        db_path=db_path,
                    )
            else:
                raise ValueError(f"Unknown method: {method}")

        except QuotaPauseException as qe:
            print(f"\n[QUOTA PAUSE] {qe}")
            print("Experiment paused safely due to Groq quota. Completed results are preserved and the run can be resumed later.")
            quota_paused = True
            break

        # 6. Counterfactual Oracle Evaluation (on val_subset, reusing already trained updates)
        oracle_eval = oracle.evaluate_candidates(
            base_model=gm,
            candidate_decisions=candidates,
            local_weights=local_w,
            global_weights=global_w,
            selected_decision=selected_decision,
            byzantine_budget=BYZANTINE_BUDGET,
            trim_ratio=TRIM_RATIO,
            metric_key="accuracy",
        )

        # 7. Model Aggregation & Update
        selected_weights = [local_w[i] for i in selected_decision.client_ids]
        new_w = aggregate(
            selected_decision.method,
            selected_weights,
            global_weights=global_w,
            num_byzantine=BYZANTINE_BUDGET,
            trim_ratio=TRIM_RATIO,
        )
        gm.load_state_dict(new_w)

        # 8. Evaluation on untouched official test dataset
        test_metrics = evaluate_model(gm, test_data, batch_size=EVAL_BATCH_SIZE)
        t_round_total = time.time() - t_round_start

        # 9. Update History & Observable Outcome Memory
        scalar_metrics = {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
        history.add_round(
            round_num=r,
            client_telemetry=current_telemetry,
            decision_made=selected_decision,
            alpha=alpha,
            global_metrics=scalar_metrics,
        )

        # Update Observable Outcome Memory for AgenticAI (Strict Oracle Firewall)
        if agentic_controller is not None:
            agentic_controller.record_round_outcome(
                round_num=r,
                selected_decision=selected_decision,
                observed_accuracy=test_metrics.get("accuracy", 0.0),
                observed_loss=test_metrics.get("loss", 0.0),
                observed_f1=test_metrics.get("f1"),
                telemetry=current_telemetry,
                validation_status=agent_result.validation_status if agent_result else "valid",
                tool_calls_used=len(agent_result.tool_calls_made) if agent_result else 0,
                iterations_used=agent_result.iterations_used if agent_result else 1,
            )

        # 10. Atomic Database Commitment (Round record + Agentic log + Checkpoint)
        round_data = {
            "alpha": alpha,
            "attack_active": attack_active,
            "attacker_ids": attacker_ids if attack_active else [],
            "selected_clients": selected_decision.client_ids,
            "aggregation_method": selected_decision.method,
            "train_loss": 0.0,
            "val_metrics": oracle_eval["selected_metrics"],
            "test_metrics": test_metrics,
            "oracle_accuracy": oracle_eval["oracle_score"],
            "oracle_candidate_id": oracle_eval["oracle_candidate_id"],
            "regret": oracle_eval["regret"],
            "cumulative_regret": oracle_eval["cumulative_regret"],
            "round_time_s": t_round_total,
        }

        agentic_data = None
        if agent_result is not None:
            agentic_data = {
                "telemetry": current_telemetry,
                "history": history.get_recent(3),
                "analyst_output": agent_result.analysis,
                "proposals": agent_result.proposals,
                "critic_output": agent_result.critique,
                "final_decision": selected_decision,
                "validation_result": agent_result.validation_status,
                "repair_status": agent_result.repair_status,
                "repair_reason": agent_result.repair_reason,
                "oracle_decision": oracle_eval["oracle_decision"],
                "selected_score": oracle_eval["selected_score"],
                "oracle_score": oracle_eval["oracle_score"],
                "regret": oracle_eval["regret"],
                "cumulative_regret": oracle_eval["cumulative_regret"],
                "iterations_used": getattr(agent_result, "iterations_used", 1),
                "reflection_summary": getattr(agent_result, "reflection_summary", None),
                "assessed_regime": getattr(agent_result, "assessed_regime", None),
                "tool_calls_made": getattr(agent_result, "tool_calls_made", []),
                "llm_calls_used": getattr(agent_result, "llm_calls_used", 0),
            }

        checkpoint_data = {
            "model_weights": copy.deepcopy(gm.state_dict()),
            "history_json": [r.model_dump() for r in history.get_recent(HISTORY_LENGTH)],
            "cumulative_regret": oracle.cumulative_regret,
        }

        commit_round_transaction(
            db_path=db_path,
            run_id=run_id,
            method=method,
            seed=seed,
            round_num=r,
            round_record=round_data,
            agentic_log=agentic_data,
            checkpoint_state=checkpoint_data,
        )

        round_record_summary = {
            "round": r,
            "alpha": alpha,
            "attack_active": attack_active,
            "method": selected_decision.method,
            "clients": selected_decision.client_ids,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "oracle_accuracy": oracle_eval["oracle_score"],
            "regret": oracle_eval["regret"],
            "cumulative_regret": oracle_eval["cumulative_regret"],
            "round_time_s": t_round_total,
        }
        round_logs.append(round_record_summary)

        if verbose:
            print(f"  Decision: [{selected_decision.candidate_id or 'Manual'}] {selected_decision.method} on {selected_decision.client_ids}")
            print(f"  Test Acc: {test_metrics['accuracy']:.2f}% | Test F1: {test_metrics['f1']:.4f}")
            print(f"  Oracle Acc: {oracle_eval['oracle_score']:.2f}% | Regret: {oracle_eval['regret']:.2f}% (Cumul: {oracle_eval['cumulative_regret']:.2f}%)")
            print(f"  Round Time: {t_round_total:.2f}s")

    return {
        "run_id": run_id,
        "method": method,
        "seed": seed,
        "round_logs": round_logs,
        "final_test_accuracy": round_logs[-1]["test_accuracy"] if round_logs else 0.0,
        "cumulative_regret": oracle.cumulative_regret,
        "quota_paused": quota_paused,
        "rounds_completed": len(get_completed_rounds(db_path, method, seed)),
    }


def print_execution_summary(
    seeds: List[int],
    completed_methods: List[str],
    total_rounds_completed: int,
    quota_tracker: Optional[QuotaTracker],
    quota_paused: bool,
):
    """Print standard Experiment 3 execution summary."""
    q_summary = quota_tracker.get_summary() if quota_tracker else {}
    total_calls = q_summary.get("total_calls", 0)
    prompt_tokens = q_summary.get("total_input_tokens", 0)
    completion_tokens = q_summary.get("total_output_tokens", 0)
    total_tokens = q_summary.get("total_tokens", 0)
    avg_tokens = q_summary.get("average_tokens_per_call", 0.0)
    peak_tokens = q_summary.get("peak_tokens_per_call", 0)

    print("\n" + "=" * 70)
    print("## Experiment 3 execution summary\n")
    print(f"Seed: {seeds if len(seeds) > 1 else seeds[0]}")
    print(f"Methods completed: {', '.join(completed_methods) if completed_methods else 'None'}")
    print(f"Rounds completed: {total_rounds_completed}")
    print(f"LLM calls: {total_calls}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens/call: {avg_tokens:.1f}")
    print(f"Peak observed token usage: {peak_tokens}")
    print(f"Quota pause triggered: {'yes' if quota_paused else 'no'}")
    print("Resume supported: yes")
    print("=" * 70)

    if quota_paused:
        print("\nExperiment paused safely due to Groq quota. Completed results are preserved and the run can be resumed later.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 3 benchmark with resume and quota awareness.")
    parser.add_argument("--seed", type=int, default=1, help="Seed to run (e.g. 1)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of seeds to run")
    parser.add_argument("--methods", nargs="+", default=METHODS, help="Methods to run")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS, help="Rounds per run")
    parser.add_argument("--clients", type=int, default=NUM_CLIENTS, help="Number of clients")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Database path")
    parser.add_argument("--data-root", type=str, default="./data", help="CIFAR-10 data root")

    # Resume & Quota flags (default SAFE and RESUMABLE)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from completed rounds (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume; run from round 1")
    parser.add_argument("--quota-safe", dest="quota_safe", action="store_true", default=True, help="Enable proactive quota pause (default: True)")
    parser.add_argument("--no-quota-safe", dest="quota_safe", action="store_false", help="Disable proactive quota pause")
    parser.add_argument("--token-safety-margin", type=float, default=GROQ_TOKEN_SAFETY_MARGIN, help="Token safety margin fraction")
    parser.add_argument("--request-safety-margin", type=float, default=GROQ_REQUEST_SAFETY_MARGIN, help="Request safety margin fraction")

    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [args.seed]
    methods = args.methods

    # Initialize shared QuotaTracker
    quota_tracker = QuotaTracker(
        daily_token_limit=GROQ_DAILY_TOKEN_LIMIT,
        daily_request_limit=GROQ_DAILY_REQUEST_LIMIT,
        token_safety_margin=args.token_safety_margin,
        request_safety_margin=args.request_safety_margin,
        min_remaining_tokens=GROQ_MIN_REMAINING_TOKENS,
        min_remaining_requests=GROQ_MIN_REMAINING_REQUESTS,
    )

    print("=" * 70)
    print("STARTING EXPERIMENT 3 RUNNER (RESUMABLE & QUOTA-AWARE)")
    print(f"Seeds: {seeds}")
    print(f"Methods: {methods}")
    print(f"Rounds: {args.rounds} | Clients: {args.clients}")
    print(f"Resume Enabled: {args.resume} | Quota Safe: {args.quota_safe}")
    print(f"Database: {args.db}")
    print(f"LLM Provider: {DEFAULT_LLM_PROVIDER} | LLM Model: {DEFAULT_LLM_MODEL}")
    print("=" * 70)

    print("\nPreparing training-derived validation and official test splits...")
    train_pool, val_subset, test_data = create_training_and_validation_splits(
        root=args.data_root, download=True
    )
    print(f"  Train pool: {len(train_pool)} samples | Val subset: {len(val_subset)} samples | Test set: {len(test_data)} samples")

    completed_methods = []
    total_rounds_counter = 0
    quota_paused_global = False

    for s in seeds:
        print(f"\n==================== SEED {s} ====================")
        for m in methods:
            print(f"\n>>> Running Method: {m} (Seed {s}) >>>")
            t_start = time.time()

            res = run_experiment3_single_method(
                method=m,
                seed=s,
                train_pool=train_pool,
                val_subset=val_subset,
                test_data=test_data,
                rounds=args.rounds,
                num_clients=args.clients,
                db_path=args.db,
                resume=args.resume,
                quota_tracker=quota_tracker,
                quota_safe=args.quota_safe,
                verbose=True,
            )
            elapsed = time.time() - t_start
            rounds_done = res.get("rounds_completed", 0)
            total_rounds_counter += len(res.get("round_logs", []))

            if rounds_done >= args.rounds and m not in completed_methods:
                completed_methods.append(f"{m} (Seed {s})")

            if res.get("quota_paused"):
                quota_paused_global = True
                print(f"\n[!] Execution paused on {m} (Seed {s}) due to quota threshold.")
                break

            print(f"\n<<< Completed {m} (Seed {s}) in {elapsed:.2f}s | Progress: {rounds_done}/{args.rounds} rounds <<<")

        if quota_paused_global:
            break

    print_execution_summary(
        seeds=seeds,
        completed_methods=completed_methods,
        total_rounds_completed=total_rounds_counter,
        quota_tracker=quota_tracker,
        quota_paused=quota_paused_global,
    )
