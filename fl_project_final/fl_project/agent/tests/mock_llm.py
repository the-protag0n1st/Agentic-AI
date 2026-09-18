"""Reusable Mock LLMs for Unit Testing the Agentic AI Pipeline.

Provides canned responses and scriptable behaviors to test valid execution,
malformed JSON handling, out-of-bounds candidate repair, and failure recovery.
"""

import json
from typing import Any, Dict, List, Optional
from ..llm_client import MockLLM


def create_standard_mock_llm(
    analyst_risk: str = "low",
    suspected_byzantine: Optional[List[int]] = None,
    heterogeneity_clients: Optional[List[int]] = None,
    persistent_anomalies: Optional[List[int]] = None,
    proposed_candidate_ids: Optional[List[str]] = None,
    recommended_candidate_id: str = "C0",
    confidence: float = 0.85,
) -> MockLLM:
    """Create a mock LLM that produces valid, well-formed JSON responses for all 3 stages."""
    suspected = suspected_byzantine or []
    hetero = heterogeneity_clients or []
    persist = persistent_anomalies or []
    proposals = proposed_candidate_ids or ["C0", "C1"]

    analyst_resp = json.dumps({
        "persistent_anomalies": persist,
        "heterogeneity_clients": hetero,
        "suspected_byzantine": suspected,
        "risk_level": analyst_risk,
        "signal_interpretation": f"Observed {len(suspected)} suspected threats and {len(hetero)} non-IID skewed clients.",
        "recommended_constraints": ["Maintain at least 3 clients", "Avoid single-round exclusions"],
    })

    proposer_resp = json.dumps({
        "proposals": [
            {
                "candidate_id": cid,
                "reason": f"Justification for proposal {cid} based on risk assessment.",
                "confidence": 0.80,
            }
            for cid in proposals
        ]
    })

    critic_resp = json.dumps({
        "recommended_candidate_id": recommended_candidate_id,
        "critique": f"Challenged all proposals. {recommended_candidate_id} demonstrates the most defensible trade-off.",
        "concerns": ["Slight non-IID variance risk"],
        "confidence": confidence,
    })

    responses = {
        "expert federated learning anomaly & telemetry analyst": analyst_resp,
        "expert federated learning strategy proposer": proposer_resp,
        "adversarial federated learning critic & validator": critic_resp,
    }

    return MockLLM(responses=responses, default_response="{}")


def create_faulty_mock_llm(fault_type: str) -> MockLLM:
    """Create a mock LLM that injects specific failure modes into one or more stages."""
    base_llm = create_standard_mock_llm()

    if fault_type == "malformed_analyst_json":
        base_llm.responses["expert federated learning anomaly & telemetry analyst"] = "{not valid json"
    elif fault_type == "malformed_proposer_json":
        base_llm.responses["expert federated learning strategy proposer"] = "{proposals: [malformed"
    elif fault_type == "malformed_critic_json":
        base_llm.responses["adversarial federated learning critic & validator"] = "NO_JSON_JUST_PROSE"
    elif fault_type == "invalid_candidate_id":
        # Proposer recommends C999, Critic recommends C999
        base_llm.responses["expert federated learning strategy proposer"] = json.dumps({
            "proposals": [{"candidate_id": "C999", "reason": "Hallucinated ID", "confidence": 0.9}]
        })
        base_llm.responses["adversarial federated learning critic & validator"] = json.dumps({
            "recommended_candidate_id": "C999",
            "critique": "Hallucinated recommendation",
            "concerns": [],
            "confidence": 0.9,
        })
    elif fault_type == "missing_critic_field":
        base_llm.responses["adversarial federated learning critic & validator"] = json.dumps({
            "critique": "Missing recommended candidate field",
            "concerns": ["No ID"],
            "confidence": 0.7,
        })
    elif fault_type == "empty_proposals":
        base_llm.responses["expert federated learning strategy proposer"] = json.dumps({
            "proposals": []
        })

    return base_llm
