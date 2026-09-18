"""Unit and Integration Tests for Experiment 3 Infrastructure."""

import os
import sqlite3
import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset, Subset

from aggregation import aggregate, fedavg, krum, trimmed_mean
from candidates import generate_experiment3_candidates
from experiment3_config import (
    ATTACK_SCALE,
    BYZANTINE_BUDGET,
    METHODS,
    TRIM_RATIO,
    get_attacker_ids,
    get_regime_for_round,
)
from experiment3_db import (
    init_experiment3_db,
    insert_agentic_log,
    insert_llm_call_log,
    insert_round_record,
    insert_run_record,
)
from feasibility import (
    check_mathematical_feasibility,
    repair_decision,
    validate_decision,
)
from oracle import CounterfactualOracle
from agent.agentic_ai import ReflectiveAgent
from agent.history import AgentHistory
from agent.llm_client import MockLLM
from agent.schemas import ClientTelemetry, Decision
from agent.single_shot import SingleShotAgent
from agent.tests.mock_llm import create_standard_mock_llm


# ==============================================================================
# 1. AGGREGATION TESTS
# ==============================================================================
def test_krum_self_distance_exclusion():
    """Verify that Krum excludes the client's self-distance (d=0) from nearest neighbors."""
    w0 = {"w": torch.tensor([0.0, 0.0])}
    w1 = {"w": torch.tensor([1.0, 1.0])}
    w2 = {"w": torch.tensor([1.1, 1.1])}
    w3 = {"w": torch.tensor([10.0, 10.0])}
    w4 = {"w": torch.tensor([10.1, 10.1])}

    # For 5 clients with m=1, nearest neighbors count = n - m - 2 = 5 - 1 - 2 = 2
    # w1 and w2 are close together.
    res = krum([w0, w1, w2, w3, w4], num_byzantine=1)
    # The winner should be w1 or w2 (closest cluster), NOT corrupted by self-distance
    assert torch.allclose(res["w"], w1["w"]) or torch.allclose(res["w"], w2["w"])


def test_krum_parameter_propagation():
    """Verify num_byzantine is properly passed to krum via aggregate()."""
    w_list = [{"w": torch.tensor([float(i)])} for i in range(10)]
    # Should run without error and respect num_byzantine=2
    res = aggregate("Krum", w_list, num_byzantine=2)
    assert "w" in res


def test_weighted_fedavg():
    """Verify FedAvg correctly weights updates by client sample counts."""
    w1 = {"layer": torch.tensor([10.0])}
    w2 = {"layer": torch.tensor([20.0])}

    # Equal counts -> mean is 15.0
    res_eq = fedavg([w1, w2], sample_counts=[100, 100])
    assert torch.isclose(res_eq["layer"], torch.tensor([15.0]))

    # 3:1 weighting -> (10*3 + 20*1)/4 = 50/4 = 12.5
    res_weighted = fedavg([w1, w2], sample_counts=[300, 100])
    assert torch.isclose(res_weighted["layer"], torch.tensor([12.5]))

    # Default unweighted (backward compatible)
    res_def = fedavg([w1, w2])
    assert torch.isclose(res_def["layer"], torch.tensor([15.0]))


def test_trimmed_mean_feasibility():
    """Verify TrimmedMean requires n >= 2k + 1."""
    ok, _ = check_mathematical_feasibility("TrimmedMean", [0, 1, 2, 3, 4], trim_ratio=0.2)
    assert ok is True

    # 2 clients with trim_ratio=0.5 -> k = floor(2*0.5)=1 -> 2k+1 = 3 > 2 -> False
    ok_bad, reason = check_mathematical_feasibility("TrimmedMean", [0, 1], trim_ratio=0.5)
    assert ok_bad is False
    assert "TrimmedMean requires" in reason


# ==============================================================================
# 2. AGENT & FEASIBILITY TESTS
# ==============================================================================
def test_candidate_generation_deterministic():
    """Candidate generation produces deterministic pool of ~10 feasible candidates."""
    all_clients = list(range(10))
    cands1 = generate_experiment3_candidates(all_clients, flagged_ids=[7, 8], persistent_ids=[7])
    cands2 = generate_experiment3_candidates(all_clients, flagged_ids=[7, 8], persistent_ids=[7])

    assert len(cands1) <= 10
    assert len(cands1) >= 4
    assert [c.candidate_id for c in cands1] == [c.candidate_id for c in cands2]
    assert [c.method for c in cands1] == [c.method for c in cands2]


def test_single_shot_llm_execution():
    """Verify SingleShotAgent executes exactly 1 LLM call and returns validated decision."""
    candidates = [
        Decision(method="FedAvg", client_ids=list(range(10)), candidate_id="C0"),
        Decision(method="Median", client_ids=list(range(10)), candidate_id="C1"),
    ]
    mock_llm = MockLLM(responses={"pre-approved feasible candidate decisions": '{"selected_candidate_id": "C1", "reason": "Robust median", "confidence": 0.9}'})
    agent = SingleShotAgent(llm_client=mock_llm)

    res = agent.decide(
        round_num=1,
        telemetry=[ClientTelemetry(client_id=i) for i in range(10)],
        candidate_decisions=candidates,
    )
    assert res.final_decision.candidate_id == "C1"
    assert res.final_decision.method == "Median"
    assert len(mock_llm.call_history) == 1


def test_reflective_agent_three_llm_calls():
    """Verify ReflectiveAgent executes exactly 3 LLM calls: Analyst -> Proposer -> Critic."""
    mock_llm = create_standard_mock_llm(
        analyst_risk="medium",
        suspected_byzantine=[9],
        proposed_candidate_ids=["C1", "C2"],
        recommended_candidate_id="C1",
    )
    candidates = [
        Decision(method="FedAvg", client_ids=list(range(10)), candidate_id="C0"),
        Decision(method="Median", client_ids=list(range(9)), candidate_id="C1"),
        Decision(method="TrimmedMean", client_ids=list(range(9)), candidate_id="C2"),
    ]
    agent = ReflectiveAgent(llm_client=mock_llm)

    res = agent.decide(
        round_num=2,
        telemetry=[ClientTelemetry(client_id=i) for i in range(10)],
        candidate_decisions=candidates,
    )
    assert len(mock_llm.call_history) == 3
    assert res.final_decision.candidate_id == "C1"


# ==============================================================================
# 3. EXPERIMENT 3 REGIMES, ATTACKS & ORACLE
# ==============================================================================
def test_regime_schedule():
    """Verify the 4-regime schedule mapping across rounds 1-20."""
    r1 = get_regime_for_round(1)
    assert r1["alpha"] == 100.0 and r1["attack_active"] is False

    r6 = get_regime_for_round(6)
    assert r6["alpha"] == 1.0 and r6["attack_active"] is False

    r11 = get_regime_for_round(11)
    assert r11["alpha"] == 0.2 and r11["attack_active"] is False

    r16 = get_regime_for_round(16)
    assert r16["alpha"] == 0.2 and r16["attack_active"] is True and r16["num_attackers"] == 2


def test_attacker_ids_deterministic_and_matched():
    """Attacker IDs are deterministic for a given seed and match across calls."""
    att_s1_a = get_attacker_ids(seed=1, num_clients=10, num_attackers=2)
    att_s1_b = get_attacker_ids(seed=1, num_clients=10, num_attackers=2)
    att_s2 = get_attacker_ids(seed=2, num_clients=10, num_attackers=2)

    assert att_s1_a == att_s1_b
    assert len(att_s1_a) == 2
    assert att_s1_a != att_s2  # Different seeds get distinct attacker pairs


def test_post_training_attack_math():
    """Verify attack formula: malicious_update = -3.0 * clean_update."""
    from run_experiment3 import execute_byzantine_attack

    global_w = {"layer": torch.tensor([10.0])}
    clean_local_w = [
        {"layer": torch.tensor([12.0])},  # client 0 (clean delta = +2.0)
        {"layer": torch.tensor([14.0])},  # client 1 (clean delta = +4.0)
    ]
    # Attack client 1 with scale -3.0 -> delta becomes -12.0 -> weight becomes 10.0 - 12.0 = -2.0
    attacked = execute_byzantine_attack(clean_local_w, global_w, attacker_ids=[1], attack_scale=-3.0)

    # Client 0 unchanged
    assert torch.isclose(attacked[0]["layer"], torch.tensor([12.0]))
    # Client 1 scaled negatively
    assert torch.isclose(attacked[1]["layer"], torch.tensor([-2.0]))


def test_counterfactual_oracle_regret_computation():
    """Verify Counterfactual Oracle accurately computes regret and cumulative regret."""
    from models import Net
    # Create synthetic dataset with 20 samples
    x = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 2, (20,))
    val_set = TensorDataset(x, y)

    oracle = CounterfactualOracle(val_subset=val_set)
    gm = Net()
    global_w = {k: v.cpu() for k, v in gm.state_dict().items()}
    local_w = [copy_weights(global_w) for _ in range(5)]

    cands = [
        Decision(method="FedAvg", client_ids=[0, 1, 2, 3, 4], candidate_id="C0"),
        Decision(method="Median", client_ids=[0, 1, 2, 3, 4], candidate_id="C1"),
    ]
    selected = Decision(method="FedAvg", client_ids=[0, 1, 2, 3, 4], candidate_id="C0")

    eval_res = oracle.evaluate_candidates(gm, cands, local_w, global_w, selected)
    assert "regret" in eval_res
    assert eval_res["regret"] >= 0.0
    assert eval_res["cumulative_regret"] >= 0.0


def copy_weights(w):
    return {k: v.clone() for k, v in w.items()}


def test_experiment3_database_isolation(tmp_path):
    """Verify Experiment 3 tables are created and populated cleanly in isolated DB."""
    test_db = str(tmp_path / "test_exp3.db")
    init_experiment3_db(test_db)

    con = sqlite3.connect(test_db)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    con.close()

    assert "experiment3_runs" in tables
    assert "experiment3_rounds" in tables
    assert "agentic_log" in tables
    assert "llm_call_log" in tables

    # Test insertion
    insert_run_record("run_1", "ReflectiveAgent", 42, "llama-3.3-70b-versatile", "groq", {"test": True}, db_path=test_db)
    insert_round_record("run_1", 1, 100.0, False, [], [0, 1, 2], "FedAvg", round_time_s=1.2, db_path=test_db)
    insert_agentic_log("run_1", 1, validation_result="valid", repair_status="none", regret=0.0, db_path=test_db)
    insert_llm_call_log("run_1", 1, "analyst", "llama", "v1", "prompt", "resp", latency_ms=120.0, db_path=test_db)

    con = sqlite3.connect(test_db)
    assert con.execute("SELECT COUNT(*) FROM experiment3_runs").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM experiment3_rounds").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM agentic_log").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM llm_call_log").fetchone()[0] == 1
    con.close()
