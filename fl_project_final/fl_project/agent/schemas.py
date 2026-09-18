"""Pydantic Schemas for the Reflective Agentic AI Layer.

Defines all typed models for decisions, telemetry, analyst output,
proposals, critiques, validation results, and overall agent outcomes.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class Decision(BaseModel):
    """Canonical representation of an FL aggregation decision."""
    method: str = Field(..., description="Aggregation method name (e.g., FedAvg, Median, TrimmedMean, Krum)")
    client_ids: List[int] = Field(..., description="Selected subset of client IDs")
    candidate_id: Optional[str] = Field(default=None, description="Assigned candidate label (e.g., C0, C1)")

    model_config = ConfigDict(frozen=True)

    def normalized_method(self) -> str:
        """Normalize method names for consistency."""
        clean = self.method.strip().lower().replace(" ", "").replace("_", "")
        if clean in ("fedavg", "average"):
            return "FedAvg"
        if clean in ("median", "coordinatemedian", "coordinatewisemedian"):
            return "Median"
        if clean in ("trimmedmean", "trimmed_mean", "trimmed"):
            return "TrimmedMean"
        if clean in ("krum", "multikrum"):
            return "Krum"
        return self.method

    def sorted_client_ids(self) -> List[int]:
        return sorted(self.client_ids)


class ClientTelemetry(BaseModel):
    """Per-client telemetry for a given communication round."""
    client_id: int
    update_norm: Optional[float] = None
    cosine_similarity: Optional[float] = None
    weight_var: Optional[float] = None
    anomaly_score: Optional[float] = None
    is_flagged_by_detector: Optional[bool] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    auc: Optional[float] = None
    loss_change: Optional[float] = None
    accuracy_change: Optional[float] = None
    previous_anomaly_status: Optional[bool] = None
    previous_selection_status: Optional[bool] = None
    previous_aggregation_participation: Optional[bool] = None


class RoundTelemetry(BaseModel):
    """Aggregated telemetry for an entire communication round."""
    round_num: int
    alpha: Optional[float] = None
    clients: List[ClientTelemetry]
    global_metrics: Optional[Dict[str, float]] = None

    @property
    def client_telemetry(self) -> List[ClientTelemetry]:
        return self.clients


class AnalystOutput(BaseModel):
    """Structured output produced by the Analyst stage."""
    persistent_anomalies: List[int] = Field(
        default_factory=list,
        description="Client IDs exhibiting persistent abnormal behavior across multiple rounds",
    )
    heterogeneity_clients: List[int] = Field(
        default_factory=list,
        description="Client IDs whose unusual signals are likely explained by statistical non-IID data skew",
    )
    suspected_byzantine: List[int] = Field(
        default_factory=list,
        description="Client IDs exhibiting signals consistent with Byzantine/malicious attack behavior",
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        ..., description="Overall adversarial and divergence risk level for this round"
    )
    signal_interpretation: str = Field(
        ..., description="Coherent synthesis distinguishing non-IID heterogeneity from malicious updates"
    )
    recommended_constraints: List[str] = Field(
        default_factory=list,
        description="Concrete guidance for the proposer (e.g. avoid single-round exclusions, ensure min clients)",
    )


class Proposal(BaseModel):
    """A single decision proposal from the Proposer stage."""
    candidate_id: str = Field(..., description="Identifier of the candidate decision (e.g., C0, C1)")
    reason: str = Field(..., description="Justification for selecting this candidate")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score")


class ProposerOutput(BaseModel):
    """Structured output produced by the Proposer stage."""
    proposals: List[Proposal] = Field(
        ..., min_length=1, max_length=5, description="List of 2-3 candidate proposals"
    )


class ProposalAssessment(BaseModel):
    """Critique assessment for an individual proposal."""
    candidate_id: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0, default=0.5)


class CriticOutput(BaseModel):
    """Structured output produced by the Critic / Self-Verification stage."""
    recommended_candidate_id: str = Field(
        ..., description="The candidate ID that survives adversarial critique best"
    )
    critique: str = Field(
        ..., description="Rigorous challenge of proposed exclusions and aggregation suitability"
    )
    concerns: List[str] = Field(
        default_factory=list,
        description="Identified flaws (e.g. non-IID false positive, excessive client exclusion)",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the recommended candidate after critique"
    )
    assessments: Optional[List[ProposalAssessment]] = None


class AgentTiming(BaseModel):
    """Latency measurements for all stages in milliseconds."""
    analyst_latency_ms: float = 0.0
    proposer_latency_ms: float = 0.0
    critic_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class AgentResult(BaseModel):
    """Complete, auditable result of the reflective agentic decision pipeline."""
    analysis: Optional[AnalystOutput] = None
    proposals: Optional[ProposerOutput] = None
    critique: Optional[CriticOutput] = None
    final_decision: Decision
    recommended_candidate_id: Optional[str] = None
    validation_status: Literal["valid", "repaired", "fallback", "error"] = "valid"
    repair_status: str = "none"
    repair_reason: Optional[str] = None
    timing: AgentTiming = Field(default_factory=AgentTiming)
    stage_errors: Dict[str, str] = Field(default_factory=dict)
    # --- Agentic metadata (all optional with defaults for backward compatibility) ---
    iterations_used: int = Field(
        default=1, description="Number of Propose-Critique cycles executed"
    )
    reflection_summary: Optional[str] = Field(
        default=None, description="Observable outcome reflection from previous rounds"
    )
    assessed_regime: Optional[str] = Field(
        default=None, description="Autonomously assessed operational regime from telemetry"
    )
    tool_calls_made: List[str] = Field(
        default_factory=list, description="Tool invocations made during this decision"
    )
    llm_calls_used: int = Field(
        default=0, description="Total LLM interactions consumed for this decision"
    )
