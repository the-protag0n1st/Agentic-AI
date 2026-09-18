"""Standalone Demonstration of the Reflective Agentic AI Layer.

Demonstrates the 4-stage pipeline on synthetic telemetry for 10 clients:
  Telemetry + History -> Analyst -> Proposer -> Critic -> Deterministic Validator & Repair
Does NOT touch the actual FL experiment, datasets, or databases.
"""

from typing import List

from .agentic_ai import ReflectiveAgent
from .history import AgentHistory
from .llm_client import MockLLM
from .schemas import ClientTelemetry, Decision
from .tests.mock_llm import create_standard_mock_llm


def run_demo():
    print("=" * 70)
    print("DEMO: REFLECTIVE AGENTIC AI LAYER FOR FEDERATED LEARNING")
    print("=" * 70)

    # 1. Create synthetic history for 10 clients across rounds 1 and 2
    history = AgentHistory(history_length=3)
    num_clients = 10

    # Round 1: All clean
    r1_tel = [
        ClientTelemetry(
            client_id=i,
            update_norm=1.0 + (i * 0.05),
            cosine_similarity=0.92,
            is_flagged_by_detector=False,
        )
        for i in range(num_clients)
    ]
    history.add_round(round_num=1, client_telemetry=r1_tel, decision_made=Decision(method="FedAvg", client_ids=list(range(10))))

    # Round 2: Client 7 exhibits first sign of divergence (negative cosine)
    r2_tel = [
        ClientTelemetry(
            client_id=i,
            update_norm=1.05 + (i * 0.05),
            cosine_similarity=0.90 if i != 7 else -0.22,
            is_flagged_by_detector=(i == 7),
        )
        for i in range(num_clients)
    ]
    history.add_round(round_num=2, client_telemetry=r2_tel, decision_made=Decision(method="Median", client_ids=list(range(10))))

    # 2. Current Round (Round 3) Telemetry: Client 7 sustains divergence; Client 3 has non-IID variance
    current_telemetry: List[ClientTelemetry] = []
    for i in range(num_clients):
        if i == 7:
            # Persistent Byzantine behavior
            tel = ClientTelemetry(client_id=7, update_norm=3.20, cosine_similarity=-0.48, is_flagged_by_detector=True)
        elif i == 3:
            # Non-IID label skew (higher norm, but positive cosine)
            tel = ClientTelemetry(client_id=3, update_norm=2.10, cosine_similarity=0.74, is_flagged_by_detector=False)
        else:
            tel = ClientTelemetry(client_id=i, update_norm=1.10, cosine_similarity=0.88, is_flagged_by_detector=False)
        current_telemetry.append(tel)

    # 3. Candidate Decisions (6 feasible candidate combinations)
    all_clients = list(range(num_clients))
    clean_clients = [i for i in all_clients if i != 7]

    candidates = [
        Decision(method="FedAvg", client_ids=all_clients, candidate_id="C0"),
        Decision(method="Median", client_ids=all_clients, candidate_id="C1"),
        Decision(method="TrimmedMean", client_ids=all_clients, candidate_id="C2"),
        Decision(method="FedAvg", client_ids=clean_clients, candidate_id="C3"),
        Decision(method="Median", client_ids=clean_clients, candidate_id="C4"),
        Decision(method="Krum", client_ids=clean_clients, candidate_id="C5"),
    ]

    # 4. Instantiate Mock LLM simulating realistic stage outputs
    mock_llm = create_standard_mock_llm(
        analyst_risk="high",
        suspected_byzantine=[7],
        heterogeneity_clients=[3],
        persistent_anomalies=[7],
        proposed_candidate_ids=["C3", "C4"],
        recommended_candidate_id="C4",
        confidence=0.91,
    )

    # 5. Execute ReflectiveAgent
    agent = ReflectiveAgent(llm_client=mock_llm, byzantine_budget=1, history_length=3)
    result = agent.decide(
        round_num=3,
        telemetry=current_telemetry,
        candidate_decisions=candidates,
        history=history,
        alpha=0.2,
        valid_client_ids=all_clients,
    )

    # 6. Display Auditable Results
    print("\n[STAGE 1: ANALYST]")
    if result.analysis:
        print(f"  Risk Level:              {result.analysis.risk_level.upper()}")
        print(f"  Suspected Byzantine:     {result.analysis.suspected_byzantine}")
        print(f"  Heterogeneity (Honest):  {result.analysis.heterogeneity_clients}")
        print(f"  Persistent Anomalies:    {result.analysis.persistent_anomalies}")
        print(f"  Interpretation:          {result.analysis.signal_interpretation}")
        print(f"  Constraints:             {result.analysis.recommended_constraints}")
    print(f"  Latency:                 {result.timing.analyst_latency_ms:.2f} ms")

    print("\n[STAGE 2: PROPOSER]")
    if result.proposals:
        for p in result.proposals.proposals:
            print(f"  Proposal [{p.candidate_id}]: {p.reason} (conf={p.confidence})")
    print(f"  Latency:                 {result.timing.proposer_latency_ms:.2f} ms")

    print("\n[STAGE 3: CRITIC]")
    if result.critique:
        print(f"  Recommended Candidate:   {result.critique.recommended_candidate_id}")
        print(f"  Critique:                {result.critique.critique}")
        print(f"  Concerns:                {result.critique.concerns}")
        print(f"  Confidence:              {result.critique.confidence}")
    print(f"  Latency:                 {result.timing.critic_latency_ms:.2f} ms")

    print("\n[DETERMINISTIC VALIDATION & FINAL DECISION]")
    print(f"  Executed Method:         {result.final_decision.method}")
    print(f"  Selected Client IDs:     {result.final_decision.client_ids}")
    print(f"  Validation Status:       {result.validation_status}")
    print(f"  Repair Status:           {result.repair_status}")
    print(f"  Total Agent Latency:     {result.timing.total_latency_ms:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
