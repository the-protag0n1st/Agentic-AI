"""Comprehensive Integration & Behavioral Tests for the Reflective Agentic Layer."""

import pytest

from ..agentic_ai import ReflectiveAgent
from ..history import AgentHistory
from ..llm_client import MockLLM
from ..schemas import ClientTelemetry, Decision
from .mock_llm import create_faulty_mock_llm, create_standard_mock_llm


@pytest.fixture
def standard_candidates():
    """Standard pool of 4 candidate decisions for 5 clients."""
    return [
        Decision(method="FedAvg", client_ids=[0, 1, 2, 3, 4], candidate_id="C0"),
        Decision(method="FedAvg", client_ids=[0, 1, 2, 3], candidate_id="C1"),
        Decision(method="Median", client_ids=[0, 1, 2, 3, 4], candidate_id="C2"),
        Decision(method="Krum", client_ids=[0, 1, 2, 3, 4], candidate_id="C3"),
    ]


@pytest.fixture
def standard_telemetry():
    """Standard telemetry for 5 clients."""
    return [
        ClientTelemetry(client_id=0, update_norm=1.2, cosine_similarity=0.92, is_flagged_by_detector=False),
        ClientTelemetry(client_id=1, update_norm=1.1, cosine_similarity=0.88, is_flagged_by_detector=False),
        ClientTelemetry(client_id=2, update_norm=1.3, cosine_similarity=0.90, is_flagged_by_detector=False),
        ClientTelemetry(client_id=3, update_norm=1.2, cosine_similarity=0.85, is_flagged_by_detector=False),
        ClientTelemetry(client_id=4, update_norm=2.8, cosine_similarity=-0.45, is_flagged_by_detector=True),
    ]


def test_full_pipeline_success(standard_candidates, standard_telemetry):
    """Verify complete end-to-end execution of Analyst -> Proposer -> Critic -> Decision."""
    mock_llm = create_standard_mock_llm(
        analyst_risk="high",
        suspected_byzantine=[4],
        proposed_candidate_ids=["C1", "C2"],
        recommended_candidate_id="C1",
        confidence=0.92,
    )
    agent = ReflectiveAgent(llm_client=mock_llm)

    result = agent.decide(
        round_num=3,
        telemetry=standard_telemetry,
        candidate_decisions=standard_candidates,
        alpha=0.3,
        valid_client_ids=[0, 1, 2, 3, 4],
    )

    # 1. Analyst stage executed
    assert result.analysis is not None
    assert result.analysis.risk_level == "high"
    assert result.analysis.suspected_byzantine == [4]

    # 2. Proposer stage executed
    assert result.proposals is not None
    assert len(result.proposals.proposals) == 2
    assert result.proposals.proposals[0].candidate_id == "C1"

    # 3. Critic stage executed
    assert result.critique is not None
    assert result.critique.recommended_candidate_id == "C1"
    assert result.critique.confidence == 0.92

    # 4. Final deterministic decision
    assert result.final_decision.candidate_id == "C1"
    assert result.final_decision.method == "FedAvg"
    assert result.final_decision.client_ids == [0, 1, 2, 3]
    assert result.validation_status == "valid"
    assert result.repair_status == "none"

    # 5. Timing was tracked
    assert result.timing.total_latency_ms >= 0.0


def test_fault_tolerance_malformed_analyst_json(standard_candidates, standard_telemetry):
    """Pipeline survives malformed Analyst JSON via fallback without crashing."""
    mock_llm = create_faulty_mock_llm("malformed_analyst_json")
    agent = ReflectiveAgent(llm_client=mock_llm)

    result = agent.decide(
        round_num=2,
        telemetry=standard_telemetry,
        candidate_decisions=standard_candidates,
    )

    assert "analyst" in result.stage_errors
    assert result.analysis is not None  # Fallback analysis created
    assert result.final_decision is not None  # Pipeline finished safely


def test_fault_tolerance_malformed_proposer_json(standard_candidates, standard_telemetry):
    """Pipeline survives malformed Proposer JSON via fallback proposals."""
    mock_llm = create_faulty_mock_llm("malformed_proposer_json")
    agent = ReflectiveAgent(llm_client=mock_llm)

    result = agent.decide(
        round_num=2,
        telemetry=standard_telemetry,
        candidate_decisions=standard_candidates,
    )

    assert "proposer" in result.stage_errors
    assert result.proposals is not None
    assert len(result.proposals.proposals) >= 1
    assert result.final_decision is not None


def test_fault_tolerance_malformed_critic_json(standard_candidates, standard_telemetry):
    """Pipeline survives malformed Critic output by falling back to top proposal."""
    mock_llm = create_faulty_mock_llm("malformed_critic_json")
    agent = ReflectiveAgent(llm_client=mock_llm)

    result = agent.decide(
        round_num=2,
        telemetry=standard_telemetry,
        candidate_decisions=standard_candidates,
    )

    assert "critic" in result.stage_errors
    assert result.critique is not None
    assert result.final_decision is not None


def test_deterministic_repair_of_hallucinated_candidate(standard_candidates, standard_telemetry):
    """When LLM hallucinates an invalid candidate C999, Python deterministically repairs it."""
    mock_llm = create_faulty_mock_llm("invalid_candidate_id")
    agent = ReflectiveAgent(llm_client=mock_llm)

    result = agent.decide(
        round_num=1,
        telemetry=standard_telemetry,
        candidate_decisions=standard_candidates,
    )

    # Verification: Hallucinated candidate C999 was REJECTED and REPAIRED
    assert result.recommended_candidate_id == "C999"
    assert result.validation_status == "repaired"
    assert result.repair_status == "repaired"
    assert "C999" in (result.repair_reason or "")
    # Executed decision must be a legitimate candidate from standard_candidates
    assert result.final_decision.candidate_id in ["C0", "C1", "C2", "C3"]


def test_transient_vs_persistent_behavior_influences_context():
    """Verify Section 24 requirement: History allows distinguishing transient from persistent anomalies."""
    history = AgentHistory(history_length=3)

    # Scenario A: Client 7 is transient (unusual in round 3, but normal in rounds 1 and 2)
    history.add_round(
        round_num=1,
        client_telemetry=[ClientTelemetry(client_id=7, cosine_similarity=0.90, is_flagged_by_detector=False)],
    )
    history.add_round(
        round_num=2,
        client_telemetry=[ClientTelemetry(client_id=7, cosine_similarity=0.88, is_flagged_by_detector=False)],
    )

    current_tel = [
        ClientTelemetry(client_id=7, cosine_similarity=-0.25, is_flagged_by_detector=True),
        ClientTelemetry(client_id=0, cosine_similarity=0.91, is_flagged_by_detector=False),
    ]

    # In Round 3, client 7 is transient
    assert history.has_persistent_anomaly(7, min_rounds=2) is False

    # Scenario B: Client 7 is persistent (anomalous in rounds 2 and 3)
    history.clear()
    history.add_round(
        round_num=1,
        client_telemetry=[ClientTelemetry(client_id=7, cosine_similarity=-0.35, is_flagged_by_detector=True)],
    )
    history.add_round(
        round_num=2,
        client_telemetry=[ClientTelemetry(client_id=7, cosine_similarity=-0.40, is_flagged_by_detector=True)],
    )

    # In Round 3, client 7 is persistent
    assert history.has_persistent_anomaly(7, min_rounds=2) is True


def test_critic_challenges_unnecessary_exclusion():
    """Verify Section 25 requirement:

    Candidate pool:
      C0 = FedAvg + all clients [0..4]
      C1 = FedAvg + exclude client 4 [0..3]
    When client 4 has only 1 unusual round, Critic prefers C0 or robust aggregation C2,
    challenging unnecessary exclusion.
    """
    candidates = [
        Decision(method="FedAvg", client_ids=[0, 1, 2, 3, 4], candidate_id="C0"),
        Decision(method="FedAvg", client_ids=[0, 1, 2, 3], candidate_id="C1"),
        Decision(method="Median", client_ids=[0, 1, 2, 3, 4], candidate_id="C2"),
    ]

    # Mock LLM where Proposer offered C1, but Critic challenged and chose C2 (Median, retaining all clients)
    mock_llm = create_standard_mock_llm(
        analyst_risk="low",
        heterogeneity_clients=[4],  # Non-IID skew, not Byzantine
        proposed_candidate_ids=["C1", "C2"],
        recommended_candidate_id="C2",
        confidence=0.89,
    )
    agent = ReflectiveAgent(llm_client=mock_llm)

    tel = [
        ClientTelemetry(client_id=i, update_norm=1.0, cosine_similarity=0.90, is_flagged_by_detector=False)
        for i in range(4)
    ]
    tel.append(ClientTelemetry(client_id=4, update_norm=1.5, cosine_similarity=0.60, is_flagged_by_detector=False))

    result = agent.decide(round_num=2, telemetry=tel, candidate_decisions=candidates)

    assert result.final_decision.candidate_id == "C2"
    assert result.final_decision.method == "Median"
    assert 4 in result.final_decision.client_ids  # Client 4 was NOT excluded unnecessarily


def test_reflective_information_flow_across_stages(standard_candidates):
    """Verify Directive 3: Explicit information flow between stages.

    Telemetry + 3 rounds history -> Analyst prompt
    Analyst output -> Proposer prompt
    Proposer proposals + Analyst + History -> Critic prompt
    Critic recommendation -> Deterministic final decision
    """
    history = AgentHistory(history_length=3)

    # 3 prior rounds
    for r in range(1, 4):
        tel_r = [
            ClientTelemetry(client_id=0, update_norm=1.0, cosine_similarity=0.90, is_flagged_by_detector=False),
            ClientTelemetry(client_id=4, update_norm=1.0 + r, cosine_similarity=0.90 - (r * 0.4), is_flagged_by_detector=(r >= 2)),
        ]
        history.add_round(round_num=r, client_telemetry=tel_r)

    # Current round telemetry (Round 4)
    current_tel = [
        ClientTelemetry(client_id=0, update_norm=1.02, cosine_similarity=0.91, is_flagged_by_detector=False),
        ClientTelemetry(client_id=4, update_norm=3.80, cosine_similarity=-0.65, is_flagged_by_detector=True),
    ]

    custom_analyst_msg = "UNIQUE_ANALYST_SYNTHESIS_CLIENT_4_BYZANTINE"
    custom_proposal_reason = "UNIQUE_PROPOSAL_REASON_EXCLUDE_4"

    mock_llm = create_standard_mock_llm(
        analyst_risk="high",
        suspected_byzantine=[4],
        proposed_candidate_ids=["C1"],
        recommended_candidate_id="C1",
        confidence=0.95,
    )
    # Inject unique text to track data propagation across prompt boundaries
    import json
    mock_llm.responses["expert federated learning anomaly & telemetry analyst"] = json.dumps({
        "persistent_anomalies": [4],
        "heterogeneity_clients": [],
        "suspected_byzantine": [4],
        "risk_level": "high",
        "signal_interpretation": custom_analyst_msg,
        "recommended_constraints": ["Exclude client 4"],
    })
    mock_llm.responses["expert federated learning strategy proposer"] = json.dumps({
        "proposals": [{"candidate_id": "C1", "reason": custom_proposal_reason, "confidence": 0.90}]
    })

    agent = ReflectiveAgent(llm_client=mock_llm)
    result = agent.decide(
        round_num=4,
        telemetry=current_tel,
        candidate_decisions=standard_candidates,
        history=history,
        alpha=0.3,
    )

    calls = mock_llm.call_history
    assert len(calls) == 3, f"Expected exactly 3 LLM calls, got {len(calls)}"

    # 1. Check Analyst prompt received round, alpha, telemetry, and 3-round history
    analyst_prompt = calls[0]["prompt"]
    assert "Round: 4" in analyst_prompt
    assert "alpha=0.3" in analyst_prompt
    assert "Client 4" in analyst_prompt
    assert "-0.65" in analyst_prompt  # Current cosine
    assert "Round 1" in analyst_prompt
    assert "Round 2" in analyst_prompt
    assert "Round 3" in analyst_prompt
    assert "trajectory" in analyst_prompt

    # 2. Check Proposer prompt received Analyst output
    proposer_prompt = calls[1]["prompt"]
    assert custom_analyst_msg in proposer_prompt
    assert "Suspected Byzantine: [4]" in proposer_prompt
    assert "Risk Level: high" in proposer_prompt

    # 3. Check Critic prompt received Analyst synthesis AND Proposer proposals AND history
    critic_prompt = calls[2]["prompt"]
    assert custom_analyst_msg in critic_prompt
    assert custom_proposal_reason in critic_prompt
    assert "Proposal [C1]" in critic_prompt
    assert "Client 4" in critic_prompt

    # 4. Check final deterministic decision matches validated choice
    assert result.final_decision.candidate_id == "C1"
    assert result.validation_status == "valid"


def test_llm_cannot_execute_invalid_aggregation_method():
    """Verify Directive 4: Mathematical filter rejects unfeasible or invalid aggregation methods."""
    from ..validation import check_mathematical_feasibility

    # Method not in supported methods
    ok, reason = check_mathematical_feasibility("BogusMethod", [0, 1, 2])
    assert ok is False
    assert "Unsupported aggregation method" in reason

    # Method with insufficient clients (Krum requires n >= 5 for m=1)
    ok_krum, reason_krum = check_mathematical_feasibility("Krum", [0, 1], byzantine_budget=1)
    assert ok_krum is False
    assert "Krum requires" in reason_krum


def test_llm_cannot_execute_invalid_client_ids(standard_candidates):
    """Verify Directive 4: Decision with client IDs outside valid_client_ids is rejected."""
    from ..validation import validate_decision

    bad_decision = Decision(method="FedAvg", client_ids=[0, 1, 999])
    ok, reason = validate_decision(
        bad_decision,
        standard_candidates,
        valid_client_ids=[0, 1, 2, 3, 4],
    )
    assert ok is False
    assert "invalid client IDs: [999]" in reason


def test_llm_cannot_execute_fabricated_subset(standard_candidates):
    """Verify Directive 4: Decision with fabricated subset not in candidates is rejected."""
    from ..validation import validate_decision

    # Subset [0, 2] is not in standard_candidates (which has [0..4] and [0..3])
    fabricated = Decision(method="FedAvg", client_ids=[0, 2])
    ok, reason = validate_decision(
        fabricated,
        standard_candidates,
        valid_client_ids=[0, 1, 2, 3, 4],
    )
    assert ok is False
    assert "does not match any pre-approved candidate" in reason


def test_llm_exhausted_retries_results_in_llm_error_fallback(standard_candidates, standard_telemetry):
    """Verify Directive 1: When LLM retries are exhausted, explicit llm_error_fallback status is returned."""
    # Mock LLM that raises an error simulating exhausted retries
    def error_side_effect(prompt):
        raise RuntimeError("Groq API rate limit: retries exhausted on model llama-3.3-70b-versatile")

    faulty_llm = MockLLM(side_effect=error_side_effect)
    agent = ReflectiveAgent(llm_client=faulty_llm)

    result = agent.decide(
        round_num=1,
        telemetry=standard_telemetry,
        candidate_decisions=standard_candidates,
    )

    assert result.validation_status == "fallback"
    assert result.repair_status == "llm_error_fallback"
    assert "llm_error" in (result.repair_reason or "").lower()
    assert len(result.stage_errors) > 0
    # A safe deterministic decision from the candidate set was still executed
    assert result.final_decision.candidate_id in ["C0", "C1", "C2", "C3"]

