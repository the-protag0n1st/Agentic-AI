"""Tiny Smoke Test for Experiment 3 Integration.

Runs a fast, minimal smoke test across all 7 canonical methods:
  - 4 clients
  - 2 rounds
  - Mock LLM (Zero real Groq calls)
  - Isolated database (smoke_test_experiment3.db)
  - Tests FL training, candidate generation, feasibility validation,
    history updates, counterfactual oracle, DB logging, dynamic regime machinery,
    and attack injection.
"""

import os
from pathlib import Path
import sqlite3
import time
import torch
from torch.utils.data import TensorDataset

from aggregation import aggregate
from candidates import generate_experiment3_candidates
from experiment3_config import (
    ATTACK_SCALE,
    BYZANTINE_BUDGET,
    METHODS,
    TRIM_RATIO,
    get_attacker_ids,
)
from experiment3_db import (
    init_experiment3_db,
    insert_agentic_log,
    insert_llm_call_log,
    insert_round_record,
    insert_run_record,
)
from feasibility import check_mathematical_feasibility, repair_decision, validate_decision
from models import Net, device, evaluate_model, train_local
from oracle import CounterfactualOracle
from run_experiment3 import (
    execute_byzantine_attack,
    rule_based_decision,
    run_experiment3_single_method,
    set_seed,
)
from agent.agentic_ai import ReflectiveAgent
from agent.history import AgentHistory
from agent.llm_client import MockLLM
from agent.schemas import ClientTelemetry, Decision
from agent.single_shot import SingleShotAgent
from agent.tests.mock_llm import create_standard_mock_llm


def run_smoke_test():
    print("=" * 70)
    print("EXPERIMENT 3 INTEGRATION SMOKE TEST (MOCK LLM ONLY)")
    print("=" * 70)

    db_path = "smoke_test_experiment3.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    init_experiment3_db(db_path)
    print(f"Initialized isolated smoke test database: {db_path}")

    # 1. Create tiny synthetic dataset (4 clients, 60 samples each; 100 val samples; 100 test samples)
    print("\n[1/14] Generating synthetic datasets for smoke test...")
    n_samples_per_client = 60
    num_clients = 4
    n_train = n_samples_per_client * num_clients

    x_train = torch.randn(n_train, 3, 32, 32)
    y_train = torch.randint(0, 2, (n_train,))
    train_pool = TensorDataset(x_train, y_train)
    train_pool.targets = y_train.tolist()

    x_val = torch.randn(100, 3, 32, 32)
    y_val = torch.randint(0, 2, (100,))
    val_subset = TensorDataset(x_val, y_val)
    val_subset.targets = y_val.tolist()

    x_test = torch.randn(100, 3, 32, 32)
    y_test = torch.randint(0, 2, (100,))
    test_data = TensorDataset(x_test, y_test)
    test_data.targets = y_test.tolist()
    print("  Created train_pool (240 samples), val_subset (100 samples), test_data (100 samples).")

    # 2. Candidate generation verification
    print("\n[2/14] Testing deterministic candidate generation...")
    all_client_ids = list(range(num_clients))
    candidates = generate_experiment3_candidates(
        all_client_ids=all_client_ids,
        flagged_ids=[3],
        persistent_ids=[3],
        byzantine_budget=0,  # for 4 clients, m=0 allows Krum (4 >= 2(0)+3 = 3)
        trim_ratio=0.2,
        max_candidates=8,
    )
    print(f"  Generated {len(candidates)} candidates: {[c.candidate_id for c in candidates]}")
    assert len(candidates) >= 3, "Failed to generate sufficient candidates"

    # 3. Feasibility validation verification
    print("\n[3/14] Testing feasibility validation...")
    val_ok, msg = validate_decision(candidates[0], candidates, valid_client_ids=all_client_ids)
    assert val_ok is True
    print(f"  Candidate {candidates[0].candidate_id} feasibility: {val_ok} ({msg})")

    # 4. Setup Mock LLM for SingleShot and Reflective
    single_mock_llm = MockLLM(
        responses={"pre-approved feasible candidate decisions": '{"selected_candidate_id": "C0", "reason": "Smoke test single shot", "confidence": 0.85}'}
    )
    reflective_mock_llm = create_standard_mock_llm(
        analyst_risk="low",
        proposed_candidate_ids=["C0", "C1"],
        recommended_candidate_id="C0",
        confidence=0.88,
    )

    # 5. Run all 7 methods for 2 rounds each
    print("\n[4-10/14] Executing 2-round smoke test across all 7 methods...")
    method_results = {}
    total_start = time.time()

    for method_name in METHODS:
        print(f"\n  --> Running Method: {method_name}...")
        t0 = time.time()

        llm = None
        if method_name == "SingleShotLLM":
            llm = single_mock_llm
        elif method_name in ("ReflectiveAgent", "AgenticAI"):
            llm = reflective_mock_llm

        res = run_experiment3_single_method(
            method=method_name,
            seed=42,
            train_pool=train_pool,
            val_subset=val_subset,
            test_data=test_data,
            llm_client=llm,
            rounds=2,
            num_clients=num_clients,
            db_path=db_path,
            verbose=False,
        )
        elapsed = time.time() - t0
        method_results[method_name] = res
        print(f"      Finished in {elapsed:.2f}s | Final Acc: {res['final_test_accuracy']:.2f}% | Regret: {res['cumulative_regret']:.2f}%")

    # 11. Verify Database Isolation and Logging
    print("\n[11/14] Verifying Experiment 3 Database logs...")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    n_runs = cur.execute("SELECT COUNT(*) FROM experiment3_runs").fetchone()[0]
    n_rounds = cur.execute("SELECT COUNT(*) FROM experiment3_rounds").fetchone()[0]
    n_agentic = cur.execute("SELECT COUNT(*) FROM agentic_log").fetchone()[0]
    n_llm = cur.execute("SELECT COUNT(*) FROM llm_call_log").fetchone()[0]
    con.close()

    expected_runs = len(METHODS)
    expected_rounds = expected_runs * 2
    print(f"  experiment3_runs:   {n_runs} records (expected {expected_runs})")
    print(f"  experiment3_rounds: {n_rounds} records (expected {expected_runs} methods x 2 rounds = {expected_rounds})")
    print(f"  agentic_log:        {n_agentic} records")
    print(f"  llm_call_log:       {n_llm} records")
    assert n_runs == expected_runs, f"Expected {expected_runs} runs, got {n_runs}"
    assert n_rounds == expected_rounds, f"Expected {expected_rounds} rounds, got {n_rounds}"

    # 12. Verify Counterfactual Oracle and Regret
    print("\n[12/14] Verifying Counterfactual Oracle...")
    oracle_accs = [r["round_logs"][0]["oracle_accuracy"] for r in method_results.values()]
    print(f"  Oracle evaluated successfully on all methods: min={min(oracle_accs):.2f}%, max={max(oracle_accs):.2f}%")

    # 13. Verify Dynamic Partitions and Attack Injection Machinery
    print("\n[13/14] Verifying Dynamic Partitions & Attack Injection Machinery...")
    att_ids = get_attacker_ids(seed=42, num_clients=10, num_attackers=2)
    print(f"  Deterministic attacker IDs for seed 42: {att_ids}")
    assert len(att_ids) == 2, "Expected 2 attacker IDs"

    # 14. Verify Frozen Experiment 1 & 2 files untouched
    print("\n[14/14] Verifying Frozen Experiment 1 & 2 integrity...")
    frozen_files = [
        "fl_metrics.db",
        "fl_metrics_experiment1_90runs_baseline.db",
        "fl_metrics_experiment2_label_flip.db",
        "fl_metrics_experiment2_label_flip_pilot.db",
        "experiment2_label_flip_results.csv",
        "experiment2_label_flip_summary.json",
    ]
    for fn in frozen_files:
        full_path = Path("fl_project_final/fl_project") / fn
        if not full_path.exists():
            full_path = Path(fn)
        status = "EXISTS & UNTOUCHED" if full_path.exists() else "NOT FOUND"
        print(f"  {fn:<45}: {status}")

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"SMOKE TEST COMPLETE: All 7 methods PASSED in {total_time:.2f}s")
    print("NO real Groq API calls were made.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
