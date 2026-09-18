"""Reflective Agentic AI Controller for Adaptive Federated Learning.

Orchestrates the 4-stage reflective decision process:
  Telemetry + History -> Analyst -> Proposer -> Critic -> Deterministic Validator & Repair
Ensures the LLM acts purely as an advisory intelligence layer while deterministic
code retains absolute authority over mathematical feasibility and final execution.
"""

import time
from typing import Dict, List, Optional, Tuple

from .analyst import Analyst
from .critic import Critic
from .history import AgentHistory
from .llm_client import LLMClient
from .proposer import Proposer
from .schemas import (
    AgentResult,
    AgentTiming,
    AnalystOutput,
    ClientTelemetry,
    CriticOutput,
    Decision,
    ProposerOutput,
)
from .validation import (
    check_mathematical_feasibility,
    repair_decision,
    validate_decision,
)


class ReflectiveAgent:
    """Multi-stage reflective decision layer with deterministic mathematical authority."""

    def __init__(
        self,
        llm_client: LLMClient,
        byzantine_budget: int = 1,
        trim_ratio: float = 0.2,
        history_length: int = 3,
    ):
        self.llm_client = llm_client
        self.byzantine_budget = byzantine_budget
        self.trim_ratio = trim_ratio
        self.history = AgentHistory(history_length=history_length)

        self.analyst = Analyst(llm_client=self.llm_client)
        self.proposer = Proposer(llm_client=self.llm_client)
        self.critic = Critic(llm_client=self.llm_client)

    def _prepare_candidate_map(
        self, candidate_decisions: List[Decision]
    ) -> Tuple[List[Decision], Dict[str, Decision]]:
        """Filter candidates for mathematical feasibility and assign compact IDs C0, C1, ..."""
        feasible_candidates: List[Decision] = []
        candidate_map: Dict[str, Decision] = {}

        for cand in candidate_decisions:
            is_feasible, _ = check_mathematical_feasibility(
                cand.method,
                cand.client_ids,
                byzantine_budget=self.byzantine_budget,
                trim_ratio=self.trim_ratio,
            )
            if is_feasible:
                cid = cand.candidate_id or f"C{len(feasible_candidates)}"
                bound_cand = Decision(
                    method=cand.method,
                    client_ids=cand.client_ids,
                    candidate_id=cid,
                )
                feasible_candidates.append(bound_cand)
                candidate_map[cid] = bound_cand

        # If zero candidates were feasible (e.g. malformed inputs), construct safe FedAvg fallback
        if not feasible_candidates and candidate_decisions:
            first = candidate_decisions[0]
            fallback = Decision(
                method="FedAvg",
                client_ids=first.client_ids,
                candidate_id="C0",
            )
            feasible_candidates.append(fallback)
            candidate_map["C0"] = fallback

        return feasible_candidates, candidate_map

    def decide(
        self,
        round_num: int,
        telemetry: List[ClientTelemetry],
        candidate_decisions: List[Decision],
        history: Optional[AgentHistory] = None,
        alpha: Optional[float] = None,
        valid_client_ids: Optional[List[int]] = None,
    ) -> AgentResult:
        """Execute the complete reflective agentic decision pipeline.

        Args:
            round_num: Current communication round.
            telemetry: List of ClientTelemetry records for the current round.
            candidate_decisions: Pool of candidate (method, client_ids) decisions.
            history: Optional AgentHistory instance (uses internal history if None).
            alpha: Optional Dirichlet heterogeneity parameter.
            valid_client_ids: Optional list of all legitimate client IDs.

        Returns:
            Complete, auditable AgentResult containing stage outputs, timings, and verified decision.
        """
        active_history = history if history is not None else self.history
        stage_errors: Dict[str, str] = {}
        t_start = time.time()

        # 1. Pre-filter candidate decisions and build compact mapping
        feasible_candidates, candidate_map = self._prepare_candidate_map(candidate_decisions)

        # 2. Stage 1: Analyst
        analysis, t_analyst, err_a = self.analyst.analyze(
            round_num=round_num,
            current_telemetry=telemetry,
            history=active_history,
            candidate_decisions=feasible_candidates,
            alpha=alpha,
        )
        if err_a:
            stage_errors["analyst"] = err_a

        # 3. Stage 2: Proposer
        proposals, t_proposer, err_p = self.proposer.propose(
            round_num=round_num,
            analyst_output=analysis,
            candidate_decisions=feasible_candidates,
            candidate_map=candidate_map,
        )
        if err_p:
            stage_errors["proposer"] = err_p

        # 4. Stage 3: Critic
        critique, t_critic, err_c = self.critic.critique(
            round_num=round_num,
            analyst_output=analysis,
            proposer_output=proposals,
            candidate_map=candidate_map,
            current_telemetry=telemetry,
            history=active_history,
        )
        if err_c:
            stage_errors["critic"] = err_c

        # 5. Deterministic Decision Selection & Validation
        rec_cid = critique.recommended_candidate_id
        target_decision = candidate_map.get(rec_cid)

        if stage_errors:
            validation_status = "fallback"
            repair_status = "llm_error_fallback"
            repair_reason = f"llm_error in stage(s): {list(stage_errors.keys())}. Using deterministic safe fallback."
            final_decision, _, _ = repair_decision(
                rec_cid,
                candidate_map,
                target_clients=[c.client_id for c in telemetry] if telemetry else None,
            )
        elif target_decision is not None:
            is_valid, v_reason = validate_decision(
                target_decision,
                feasible_candidates,
                valid_client_ids=valid_client_ids,
                byzantine_budget=self.byzantine_budget,
                trim_ratio=self.trim_ratio,
            )
            if is_valid:
                final_decision = target_decision
                validation_status = "valid"
                repair_status = "none"
                repair_reason = None
            else:
                validation_status = "repaired"
                repair_status = "repaired"
                repair_reason = f"Candidate '{rec_cid}' failed validation: {v_reason}"
                final_decision, _, _ = repair_decision(
                    rec_cid,
                    candidate_map,
                    target_clients=target_decision.client_ids,
                    target_method=target_decision.method,
                )
        else:
            # LLM hallucinated an unknown candidate ID (e.g. 'C999')
            validation_status = "repaired"
            repair_status = "repaired"
            repair_reason = f"Candidate ID '{rec_cid}' does not exist in feasible candidate set."
            final_decision, _, _ = repair_decision(
                rec_cid,
                candidate_map,
                target_clients=[c.client_id for c in telemetry] if telemetry else None,
            )

        t_total = (time.time() - t_start) * 1000.0

        timing = AgentTiming(
            analyst_latency_ms=t_analyst,
            proposer_latency_ms=t_proposer,
            critic_latency_ms=t_critic,
            total_latency_ms=t_total,
        )

        return AgentResult(
            analysis=analysis,
            proposals=proposals,
            critique=critique,
            final_decision=final_decision,
            recommended_candidate_id=rec_cid,
            validation_status=validation_status,
            repair_status=repair_status,
            repair_reason=repair_reason,
            timing=timing,
            stage_errors=stage_errors,
        )
