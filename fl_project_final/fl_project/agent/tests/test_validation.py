"""Unit Tests for Mathematical Feasibility, Validation, and Deterministic Repair."""

import pytest

from ..schemas import Decision
from ..validation import (
    check_mathematical_feasibility,
    repair_decision,
    validate_decision,
)


def test_mathematical_feasibility_fedavg():
    """FedAvg is feasible for any non-empty client set."""
    ok, _ = check_mathematical_feasibility("FedAvg", [0, 1, 2])
    assert ok is True

    ok_empty, _ = check_mathematical_feasibility("FedAvg", [])
    assert ok_empty is False


def test_mathematical_feasibility_trimmed_mean():
    """TrimmedMean feasibility checks trimming ratio math."""
    # 5 clients with trim_ratio=0.2 -> k = 1 -> requires n >= 2*1 + 1 = 3 -> Feasible
    ok, _ = check_mathematical_feasibility("TrimmedMean", [0, 1, 2, 3, 4], trim_ratio=0.2)
    assert ok is True

    # 2 clients with trim_ratio=0.2 -> k = 0 -> Feasible (no trimming)
    ok_small, _ = check_mathematical_feasibility("TrimmedMean", [0, 1], trim_ratio=0.2)
    assert ok_small is True


def test_mathematical_feasibility_krum():
    """Krum requires n >= 2m + 3. For m=1, n >= 5."""
    ok_5, _ = check_mathematical_feasibility("Krum", [0, 1, 2, 3, 4], byzantine_budget=1)
    assert ok_5 is True

    # 4 clients with m=1 is mathematically invalid!
    ok_4, reason = check_mathematical_feasibility("Krum", [0, 1, 2, 3], byzantine_budget=1)
    assert ok_4 is False
    assert "Krum requires |G| >= 5" in reason


def test_validate_decision_valid():
    """Valid decision matching candidate pool passes validation."""
    cands = [
        Decision(method="FedAvg", client_ids=[0, 1, 2], candidate_id="C0"),
        Decision(method="Median", client_ids=[0, 1], candidate_id="C1"),
    ]
    target = Decision(method="FedAvg", client_ids=[0, 1, 2])
    ok, msg = validate_decision(target, cands, valid_client_ids=[0, 1, 2, 3])
    assert ok is True
    assert msg == "Valid"


def test_validate_decision_invalid_client_ids():
    """Decision with non-existent client ID is rejected."""
    cands = [Decision(method="FedAvg", client_ids=[0, 1, 99])]
    target = Decision(method="FedAvg", client_ids=[0, 1, 99])
    ok, msg = validate_decision(target, cands, valid_client_ids=[0, 1, 2, 3])
    assert ok is False
    assert "invalid client IDs" in msg


def test_validate_decision_unapproved_candidate():
    """Decision not in pre-approved candidate pool is rejected."""
    cands = [Decision(method="FedAvg", client_ids=[0, 1, 2])]
    unapproved = Decision(method="FedAvg", client_ids=[0, 1])  # Not in cands
    ok, msg = validate_decision(unapproved, cands, valid_client_ids=[0, 1, 2, 3])
    assert ok is False
    assert "does not match any pre-approved candidate" in msg


def test_repair_decision_exact_match():
    """Repair with existing candidate ID returns exact candidate."""
    cmap = {
        "C0": Decision(method="FedAvg", client_ids=[0, 1, 2, 3], candidate_id="C0"),
        "C1": Decision(method="Median", client_ids=[0, 1, 2], candidate_id="C1"),
    }
    decision, status, reason = repair_decision("C1", cmap)
    assert decision.candidate_id == "C1"
    assert status == "valid"
    assert reason == "none"


def test_repair_decision_hallucinated_candidate_id():
    """Hallucinated candidate ID (e.g. C999) is deterministically repaired to closest candidate."""
    cmap = {
        "C0": Decision(method="FedAvg", client_ids=[0, 1, 2, 3], candidate_id="C0"),
        "C1": Decision(method="TrimmedMean", client_ids=[0, 1, 2], candidate_id="C1"),
    }
    # LLM hallucinates 'C999' with target clients [0, 1, 2]
    repaired, status, reason = repair_decision("C999", cmap, target_clients=[0, 1, 2])
    assert status == "repaired"
    assert "C999" in reason
    # C1 has symmetric difference 0 with [0, 1, 2]
    assert repaired.candidate_id == "C1"


def test_repair_decision_deterministic_tie_break():
    """When two candidates have identical distance, tie-breaks deterministically by ID."""
    cmap = {
        "C1": Decision(method="Median", client_ids=[0, 1], candidate_id="C1"),
        "C0": Decision(method="Median", client_ids=[0, 2], candidate_id="C0"),
    }
    # Target clients [0] has sym_diff=1 with both C0 and C1
    repaired, status, _ = repair_decision("C_UNKNOWN", cmap, target_clients=[0])
    assert status == "repaired"
    # Ties broken alphabetically -> C0 comes before C1
    assert repaired.candidate_id == "C0"
