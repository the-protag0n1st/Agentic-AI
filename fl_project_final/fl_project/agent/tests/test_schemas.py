"""Unit Tests for Pydantic Schemas in the Reflective Agentic Layer."""

import pytest
from pydantic import ValidationError

from ..schemas import (
    AnalystOutput,
    ClientTelemetry,
    CriticOutput,
    Decision,
    Proposal,
    ProposerOutput,
    RoundTelemetry,
)


def test_valid_decision_normalization():
    """Verify decision methods normalize properly."""
    d1 = Decision(method="FedAvg", client_ids=[0, 1, 2])
    assert d1.normalized_method() == "FedAvg"

    d2 = Decision(method="coordinate_wise_median", client_ids=[0, 1, 2])
    assert d2.normalized_method() == "Median"

    d3 = Decision(method="trimmed_mean", client_ids=[0, 1, 2])
    assert d3.normalized_method() == "TrimmedMean"

    d4 = Decision(method="Krum", client_ids=[0, 1, 2])
    assert d4.normalized_method() == "Krum"


def test_valid_analyst_output():
    """Verify valid analyst JSON schema validation."""
    data = {
        "persistent_anomalies": [2, 4],
        "heterogeneity_clients": [1, 3],
        "suspected_byzantine": [4],
        "risk_level": "high",
        "signal_interpretation": "Client 4 shows sustained negative cosine similarity across rounds.",
        "recommended_constraints": ["Exclude client 4", "Ensure min_clients >= 3"],
    }
    analyst_out = AnalystOutput.model_validate(data)
    assert analyst_out.risk_level == "high"
    assert analyst_out.persistent_anomalies == [2, 4]
    assert analyst_out.suspected_byzantine == [4]


def test_invalid_analyst_risk_level():
    """Verify that invalid risk_level values raise ValidationError."""
    data = {
        "persistent_anomalies": [],
        "heterogeneity_clients": [],
        "suspected_byzantine": [],
        "risk_level": "catastrophic",  # Invalid! Only low, medium, high allowed
        "signal_interpretation": "Test",
        "recommended_constraints": [],
    }
    with pytest.raises(ValidationError):
        AnalystOutput.model_validate(data)


def test_analyst_missing_required_field():
    """Verify missing required field raises ValidationError."""
    data = {
        "persistent_anomalies": [],
        "heterogeneity_clients": [],
        # Missing risk_level and signal_interpretation!
    }
    with pytest.raises(ValidationError):
        AnalystOutput.model_validate(data)


def test_valid_proposal_and_proposer_output():
    """Verify valid proposer output schema."""
    p1 = Proposal(candidate_id="C0", reason="Full FedAvg aggregation", confidence=0.9)
    p2 = Proposal(candidate_id="C1", reason="Median aggregation excluding outlier", confidence=0.8)
    out = ProposerOutput(proposals=[p1, p2])
    assert len(out.proposals) == 2
    assert out.proposals[0].candidate_id == "C0"


def test_invalid_proposal_confidence():
    """Verify proposal confidence must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        Proposal(candidate_id="C0", reason="Test", confidence=1.5)  # Out of range!

    with pytest.raises(ValidationError):
        Proposal(candidate_id="C0", reason="Test", confidence=-0.1)  # Out of range!


def test_empty_proposer_proposals_rejected():
    """Verify ProposerOutput requires at least one proposal."""
    with pytest.raises(ValidationError):
        ProposerOutput(proposals=[])


def test_valid_critic_output():
    """Verify valid critic schema validation."""
    data = {
        "recommended_candidate_id": "C1",
        "critique": "Proposal C0 includes client 4 which exhibits persistent divergence.",
        "concerns": ["Over-reliance on FedAvg under attack"],
        "confidence": 0.88,
    }
    critic_out = CriticOutput.model_validate(data)
    assert critic_out.recommended_candidate_id == "C1"
    assert critic_out.confidence == 0.88


def test_invalid_critic_confidence():
    """Verify critic confidence constraints."""
    data = {
        "recommended_candidate_id": "C1",
        "critique": "Test",
        "confidence": 2.0,  # Invalid!
    }
    with pytest.raises(ValidationError):
        CriticOutput.model_validate(data)


def test_client_telemetry_optional_fields():
    """Verify client telemetry handles missing metrics gracefully."""
    t = ClientTelemetry(client_id=3)
    assert t.client_id == 3
    assert t.update_norm is None
    assert t.cosine_similarity is None
    assert t.is_flagged_by_detector is None
