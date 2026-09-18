"""Comprehensive tests for the AgenticController, OutcomeMemory, and ToolRegistry."""

import json
import pytest
from pydantic import ValidationError

from ..agentic_controller import AgenticController
from ..outcome_memory import OutcomeMemory, ObservableOutcome
from ..tools import ToolRegistry, parse_tool_calls, ToolCall
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

def test_outcome_memory_record_and_reflection():
    memory = OutcomeMemory(max_length=3)
    o1 = ObservableOutcome(round_num=1, selected_method="FedAvg", selected_client_ids=[0, 1, 2], observed_accuracy=0.80, observed_loss=0.5, total_client_count=3, flagged_client_count=0)
    memory.record(o1)
    assert memory.generate_reflection() != ""
    o2 = ObservableOutcome(round_num=2, selected_method="Median", selected_client_ids=[0, 1], observed_accuracy=0.85, observed_loss=0.4, accuracy_delta=0.05, total_client_count=3, flagged_client_count=1)
    memory.record(o2)
    reflection = memory.generate_reflection()
    assert "improving" in reflection
    assert "0.80" in reflection or "80.0%" in reflection
    assert "0.85" in reflection or "85.0%" in reflection

def test_outcome_memory_strict_oracle_isolation():
    o1 = ObservableOutcome(
        round_num=1,
        selected_method="FedAvg",
        selected_client_ids=[0, 1, 2],
        observed_accuracy=0.80,
        observed_loss=0.5
    )
    assert not hasattr(o1, 'regret')
    assert not hasattr(o1, 'oracle_candidate')
    
    with pytest.raises(ValidationError):
        ObservableOutcome(
            round_num=1, selected_method="FedAvg", selected_client_ids=[0, 1, 2],
            observed_accuracy=0.80, observed_loss=0.5,
            regret=0.1
        )

def test_tool_registry_all_tools(standard_candidates, standard_telemetry):
    history = AgentHistory()
    history.add_round(1, standard_telemetry)
    memory = OutcomeMemory()
    candidate_map = {c.candidate_id: c for c in standard_candidates}
    registry = ToolRegistry(
        history=history,
        candidate_map=candidate_map,
        outcome_memory=memory,
        telemetry=standard_telemetry,
        previous_telemetry=standard_telemetry
    )
    
    t1 = ToolCall(name="get_client_trajectory", args={"client_id": 4, "num_rounds": 1})
    res1 = registry.execute(t1)
    assert res1.success
    
    t2 = ToolCall(name="check_candidate_feasibility", args={"candidate_id": "C0"})
    res2 = registry.execute(t2)
    assert res2.success
    assert res2.result["feasible"] is True
    
    t3 = ToolCall(name="compare_candidates", args={"candidate_id_a": "C0", "candidate_id_b": "C1"})
    res3 = registry.execute(t3)
    assert res3.success
    assert res3.result["shared_clients"] == [0, 1, 2, 3]
    
    t4 = ToolCall(name="get_outcome_history", args={"num_rounds": 1})
    res4 = registry.execute(t4)
    assert res4.success
    assert type(res4.result) == list
    
    t5 = ToolCall(name="get_regime_indicators", args={})
    res5 = registry.execute(t5)
    assert res5.success
    assert "assessed_regime" in res5.result

def test_tool_budget_exhaustion(standard_candidates, standard_telemetry):
    registry = ToolRegistry(
        history=AgentHistory(), candidate_map={}, outcome_memory=OutcomeMemory(), telemetry=[], max_tool_calls=5
    )
    
    for _ in range(5):
        t = ToolCall(name="get_outcome_history", args={})
        res = registry.execute(t)
        assert res.success is True
        
    t_extra = ToolCall(name="get_outcome_history", args={})
    res_extra = registry.execute(t_extra)
    assert res_extra.success is False
    assert "budget exhausted" in res_extra.error
    assert registry.budget_remaining == 0

def test_parse_tool_calls_formats():
    valid = '{"tool_calls": [{"name": "compare_candidates", "args": {"candidate_id_a": "C0", "candidate_id_b": "C1"}}]}'
    calls = parse_tool_calls(valid)
    assert len(calls) == 1
    assert calls[0].name == "compare_candidates"
    
    md = '```json\n{"tool_calls": [{"name": "check_candidate_feasibility", "args": {"candidate_id": "C0"}}]}\n```'
    calls = parse_tool_calls(md)
    assert len(calls) == 1
    assert calls[0].name == "check_candidate_feasibility"
    
    inv = '{tool_calls: [malformed]}'
    calls = parse_tool_calls(inv)
    assert len(calls) == 0

def test_agentic_controller_full_loop(standard_candidates, standard_telemetry):
    mock_llm = create_standard_mock_llm()
    controller = AgenticController(llm_client=mock_llm)
    
    result = controller.decide(
        round_num=2,
        telemetry=standard_telemetry,
        candidate_decisions=standard_candidates,
    )
    assert result.iterations_used >= 1
    assert result.assessed_regime is not None
    assert result.llm_calls_used >= 3 
    assert result.validation_status == "valid"
    assert result.repair_status == "none"

def test_critic_checklist_rejection_triggers_revision(standard_candidates, standard_telemetry):
    base_llm = create_standard_mock_llm()
    
    failing_critic = json.dumps({
        "recommended_candidate_id": "C1",
        "feasible": True,
        "uses_valid_clients": True,
        "respects_byzantine_constraints": True,
        "addresses_anomaly_evidence": False,
        "evidence_consistent": True,
        "critique": "Rejecting proposal",
    })
    passing_critic = json.dumps({
        "recommended_candidate_id": "C2",
        "feasible": True,
        "uses_valid_clients": True,
        "respects_byzantine_constraints": True,
        "addresses_anomaly_evidence": True,
        "evidence_consistent": True,
        "critique": "Passing proposal",
    })
    
    class RejectingCriticLLM(MockLLM):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.responses = base_llm.responses.copy()
            self.call_history = []
            
        def generate(self, prompt, **kwargs):
            self.call_history.append({"prompt": prompt, "kwargs": kwargs})
            if "adversarial federated learning critic & validator" in kwargs.get("system_prompt", "").lower():
                self.calls += 1
                if self.calls == 1:
                    self.last_response = failing_critic
                    return failing_critic
                self.last_response = passing_critic
                return passing_critic
            resp = base_llm.generate(prompt, **kwargs)
            self.last_response = resp
            return resp

    reject_llm = RejectingCriticLLM()
    controller = AgenticController(llm_client=reject_llm)
    result = controller.decide(
        round_num=2,
        telemetry=standard_telemetry,
        candidate_decisions=standard_candidates,
    )
    assert result.iterations_used == 2
    assert result.final_decision.candidate_id == "C2"

def test_critic_checklist_acceptance_single_iteration(standard_candidates, standard_telemetry):
    mock_llm = create_standard_mock_llm()
    mock_llm.responses["adversarial federated learning critic & validator"] = json.dumps({
        "recommended_candidate_id": "C0",
        "feasible": True,
        "uses_valid_clients": True,
        "respects_byzantine_constraints": True,
        "addresses_anomaly_evidence": True,
        "evidence_consistent": True,
        "critique": "Perfect",
        "confidence": 0.95,
    })
    
    controller = AgenticController(llm_client=mock_llm)
    result = controller.decide(
        round_num=2,
        telemetry=standard_telemetry,
        candidate_decisions=standard_candidates,
    )
    assert result.iterations_used == 1
    assert result.final_decision.candidate_id == "C0"

def test_multi_round_outcome_adaptation(standard_candidates, standard_telemetry):
    mock_llm = create_standard_mock_llm()
    controller = AgenticController(llm_client=mock_llm)
    
    result_r1 = controller.decide(1, standard_telemetry, standard_candidates)
    assert result_r1.reflection_summary is None
    
    controller.record_round_outcome(
        round_num=1,
        selected_decision=result_r1.final_decision,
        observed_accuracy=0.8,
        observed_loss=0.5,
        telemetry=standard_telemetry
    )
    
    result_r2 = controller.decide(2, standard_telemetry, standard_candidates)
    assert result_r2.reflection_summary is not None
    assert "80" in result_r2.reflection_summary or "0.8" in result_r2.reflection_summary

def test_autonomous_regime_assessment(standard_candidates):
    registry = ToolRegistry(AgentHistory(), {}, OutcomeMemory(), [])
    
    attack_tel = [
        ClientTelemetry(client_id=0, update_norm=1.0, cosine_similarity=0.1, is_flagged_by_detector=True),
        ClientTelemetry(client_id=1, update_norm=1.0, cosine_similarity=0.2, is_flagged_by_detector=True),
    ]
    registry._telemetry = attack_tel
    res_attack = registry._tool_get_regime_indicators()
    assert res_attack["assessed_regime"] == "possible_attack"
    
    iid_tel = [
        ClientTelemetry(client_id=0, update_norm=1.0, cosine_similarity=0.9, is_flagged_by_detector=False),
        ClientTelemetry(client_id=1, update_norm=1.0, cosine_similarity=0.92, is_flagged_by_detector=False),
    ]
    registry._telemetry = iid_tel
    res_iid = registry._tool_get_regime_indicators()
    assert res_iid["assessed_regime"] == "near_iid"

def test_computational_budget_enforcement(standard_candidates, standard_telemetry):
    mock_llm = create_standard_mock_llm()
    mock_llm.responses["adversarial federated learning critic & validator"] = json.dumps({
        "recommended_candidate_id": "C0",
        "feasible": False,
        "uses_valid_clients": True,
        "respects_byzantine_constraints": True,
        "addresses_anomaly_evidence": True,
        "evidence_consistent": True,
        "critique": "Failing",
        "confidence": 0.5,
    })
    controller = AgenticController(llm_client=mock_llm)
    result = controller.decide(1, standard_telemetry, standard_candidates)
    
    assert result.iterations_used <= 2
    assert result.llm_calls_used <= 6

def test_fault_recovery_malformed_llm(standard_candidates, standard_telemetry):
    faulty_llm = create_faulty_mock_llm("malformed_analyst_json")
    faulty_llm.responses["expert federated learning strategy proposer"] = "{malformed"
    faulty_llm.responses["adversarial federated learning critic & validator"] = "NO_JSON_JUST_PROSE"
    
    controller = AgenticController(llm_client=faulty_llm)
    result = controller.decide(1, standard_telemetry, standard_candidates)
    
    assert result.validation_status in ["valid", "fallback", "repaired"]
    assert result.final_decision is not None
    assert result.final_decision.candidate_id in ["C0", "C1", "C2", "C3"]
