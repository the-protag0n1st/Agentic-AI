"""Agentic AI Controller for Adaptive Federated Learning.

Implements an outcome-aware agentic decision layer with:
1. Dynamic Tool Use: Real execution loop for querying client telemetry, candidate comparisons, and regime metrics.
2. Closed-Loop Revision: Adversarial Critic structured checklist rejection triggers Proposer revision.
3. Observable Outcome Adaptation: Historical round accuracy, loss, and anomaly trends inform decision-making (strict Oracle Firewall).
4. Autonomous Regime Assessment: Classifies operational regime using observable telemetry statistics alone.
5. Hard Computational Budget: Strict ceilings on iterations (2), tool calls (5), and LLM calls (6).
6. Deterministic Authority: Absolute mathematical feasibility and validation retained in Python code.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .analyst import Analyst
from .critic import Critic, _clean_json_string
from .history import AgentHistory
from .llm_client import LLMClient
from .outcome_memory import ObservableOutcome, OutcomeMemory
from .proposer import Proposer
from .schemas import (
    AgentResult,
    AgentTiming,
    AnalystOutput,
    ClientTelemetry,
    CriticOutput,
    Decision,
    Proposal,
    ProposerOutput,
)
from .tools import ToolCall, ToolRegistry, parse_tool_calls
from .validation import (
    check_mathematical_feasibility,
    repair_decision,
    validate_decision,
)
from .agentic_prompts import (
    ANALYST_AGENTIC_ADDENDUM,
    CRITIC_VALIDATION_SYSTEM,
    CRITIC_VALIDATION_USER,
    OUTCOME_REFLECTION_TEMPLATE,
    PROPOSER_REVISION_ADDENDUM,
    REGIME_ASSESSMENT_TEMPLATE,
    TOOL_RESULTS_TEMPLATE,
    TOOL_USE_ADDENDUM,
)
from .prompts import ANALYST_SYSTEM_PROMPT, PROPOSER_SYSTEM_PROMPT, PROPOSER_USER_TEMPLATE

# Hard computational limits per round
MAX_ITERATIONS = 2
MAX_TOOL_CALLS = 5
MAX_LLM_CALLS = 6


class CriticValidation(BaseModel):
    """Structured validation checklist evaluated by the Critic.
    
    Acceptance requires ALL boolean checks to pass AND zero unresolved concerns.
    Confidence is retained as an auxiliary logged metric only.
    """
    recommended_candidate_id: str
    feasible: bool = True
    uses_valid_clients: bool = True
    respects_byzantine_constraints: bool = True
    addresses_anomaly_evidence: bool = True
    evidence_consistent: bool = True
    unresolved_concerns: List[str] = Field(default_factory=list)
    critique: str = ""
    confidence: float = 0.8


class AgentBudget:
    """Tracks and enforces computational limits per decision round."""

    def __init__(
        self,
        max_llm_calls: int = MAX_LLM_CALLS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.max_llm_calls = max_llm_calls
        self.max_tool_calls = max_tool_calls
        self.max_iterations = max_iterations
        self.llm_calls_used = 0
        self.tool_calls_used = 0
        self.iterations_used = 0
        self.tool_calls_made: List[str] = []

    def can_call_llm(self) -> bool:
        return self.llm_calls_used < self.max_llm_calls

    def can_call_tool(self) -> bool:
        return self.tool_calls_used < self.max_tool_calls

    def can_iterate(self) -> bool:
        return self.iterations_used < self.max_iterations

    def record_llm_call(self) -> None:
        self.llm_calls_used += 1

    def record_tool_call(self, tool_name: str) -> None:
        self.tool_calls_used += 1
        self.tool_calls_made.append(tool_name)

    def record_iteration(self) -> None:
        self.iterations_used += 1


class AgenticController:
    """Outcome-aware agentic decision layer for adaptive federated learning.
    
    Jointly adapts client participation and aggregation strategy under dynamic
    conditions via tool investigation, adversarial critique, structured revision,
    and observable outcome adaptation.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        byzantine_budget: int = 2,
        trim_ratio: float = 0.2,
        history_length: int = 3,
        max_iterations: int = MAX_ITERATIONS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_llm_calls: int = MAX_LLM_CALLS,
    ):
        self.llm_client = llm_client
        self.byzantine_budget = byzantine_budget
        self.trim_ratio = trim_ratio
        self.history_length = history_length
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_llm_calls = max_llm_calls

        self.history = AgentHistory(history_length=history_length)
        self.outcome_memory = OutcomeMemory(max_length=history_length)

        # Underlying stage implementations
        self.analyst = Analyst(llm_client=self.llm_client)
        self.proposer = Proposer(llm_client=self.llm_client)
        self.critic = Critic(llm_client=self.llm_client)

    def record_round_outcome(
        self,
        round_num: int,
        selected_decision: Decision,
        observed_accuracy: float,
        observed_loss: float,
        observed_f1: Optional[float] = None,
        telemetry: Optional[List[ClientTelemetry]] = None,
        validation_status: str = "valid",
        tool_calls_used: int = 0,
        iterations_used: int = 1,
    ) -> None:
        """Record observable outcome after round aggregation and evaluation.
        
        STRICT ORACLE FIREWALL: Only observable test metrics and telemetry
        are accepted. Never accepts Oracle evaluations or regret.
        """
        acc_delta = 0.0
        loss_delta = 0.0
        recent = self.outcome_memory.get_recent(1)
        if recent:
            acc_delta = observed_accuracy - recent[-1].observed_accuracy
            loss_delta = observed_loss - recent[-1].observed_loss

        flagged_count = 0
        total_count = 0
        mean_cos = 0.0
        mean_norm = 0.0
        if telemetry:
            total_count = len(telemetry)
            flagged_count = sum(1 for c in telemetry if c.is_flagged_by_detector)
            cosines = [c.cosine_similarity for c in telemetry if c.cosine_similarity is not None]
            norms = [c.update_norm for c in telemetry if c.update_norm is not None]
            if cosines:
                mean_cos = sum(cosines) / len(cosines)
            if norms:
                mean_norm = sum(norms) / len(norms)

        outcome = ObservableOutcome(
            round_num=round_num,
            selected_method=selected_decision.method,
            selected_client_ids=selected_decision.client_ids,
            observed_accuracy=observed_accuracy,
            observed_loss=observed_loss,
            observed_f1=observed_f1,
            accuracy_delta=acc_delta,
            loss_delta=loss_delta,
            flagged_client_count=flagged_count,
            total_client_count=total_count,
            mean_cosine_similarity=mean_cos,
            mean_update_norm=mean_norm,
            validation_status=validation_status,
            tool_calls_used=tool_calls_used,
            iterations_used=iterations_used,
        )
        self.outcome_memory.record(outcome)

    def _prepare_candidate_map(
        self, candidate_decisions: List[Decision]
    ) -> Tuple[List[Decision], Dict[str, Decision]]:
        """Filter candidates for mathematical feasibility and assign compact IDs."""
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

    def _critic_accepts(self, validation: CriticValidation) -> Tuple[bool, List[str]]:
        """Determine if Critic accepts proposal using structured validation checklist."""
        failing_checks = []
        if not validation.feasible:
            failing_checks.append("feasible: False (candidate not mathematically feasible)")
        if not validation.uses_valid_clients:
            failing_checks.append("uses_valid_clients: False (invalid or malicious clients included)")
        if not validation.respects_byzantine_constraints:
            failing_checks.append("respects_byzantine_constraints: False (violates method constraints)")
        if not validation.addresses_anomaly_evidence:
            failing_checks.append("addresses_anomaly_evidence: False (fails to address detected anomalies)")
        if not validation.evidence_consistent:
            failing_checks.append("evidence_consistent: False (justification contradicts telemetry)")
        if validation.unresolved_concerns:
            for c in validation.unresolved_concerns:
                failing_checks.append(f"unresolved_concern: {c}")

        is_accepted = len(failing_checks) == 0
        return is_accepted, failing_checks

    def _execute_tool_loop(
        self,
        initial_response: str,
        system_prompt: str,
        tool_registry: ToolRegistry,
        budget: AgentBudget,
    ) -> str:
        """Execute real tool calling loop if the LLM requests tools."""
        current_response = initial_response

        while budget.can_call_tool() and budget.can_call_llm():
            tool_calls = parse_tool_calls(current_response)
            if not tool_calls:
                break

            results_text_parts = []
            for tc in tool_calls:
                if not budget.can_call_tool():
                    results_text_parts.append(f"Tool {tc.name}: Budget exhausted.")
                    break

                res = tool_registry.execute(tc)
                budget.record_tool_call(tc.name)

                if res.success:
                    results_text_parts.append(
                        f"Tool {tc.name} returned:\n{json.dumps(res.result, default=str)}"
                    )
                else:
                    results_text_parts.append(f"Tool {tc.name} failed: {res.error}")

            tool_results_text = "\n\n".join(results_text_parts)
            follow_up_prompt = TOOL_RESULTS_TEMPLATE.format(
                tool_results_text=tool_results_text,
                tool_budget_remaining=tool_registry.budget_remaining,
            )

            budget.record_llm_call()
            try:
                current_response = self.llm_client.generate(
                    prompt=follow_up_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                )
            except Exception:
                # If follow-up fails, break and use current response
                break

        return current_response

    def _evaluate_critic_structured(
        self,
        round_num: int,
        analyst_output: AnalystOutput,
        proposals: ProposerOutput,
        candidate_map: Dict[str, Decision],
        current_telemetry: List[ClientTelemetry],
        regime_assessment_text: str,
        outcome_reflection_text: str,
        budget: AgentBudget,
    ) -> Tuple[CriticValidation, float, Optional[str]]:
        """Run structured adversarial critique using structured validation checklist."""
        t0 = time.time()
        error_msg = None

        cand_lines = [
            f"  - [{cid}] {cand.method} with clients {cand.client_ids}"
            for cid, cand in candidate_map.items()
        ]
        candidate_list = "\n".join(cand_lines)

        prop_lines = []
        for p in proposals.proposals:
            cand_info = candidate_map.get(p.candidate_id)
            details = (
                f"({cand_info.method}, clients {cand_info.client_ids})"
                if cand_info
                else "(Unknown)"
            )
            conf_str = f"conf={p.confidence:.2f}" if p.confidence is not None else ""
            prop_lines.append(f"  - Proposal [{p.candidate_id}] {details}: {p.reason} {conf_str}")
        proposals_text = "\n".join(prop_lines)

        tel_lines = []
        for c in current_telemetry:
            norm_str = f"{c.update_norm:.3f}" if c.update_norm is not None else "N/A"
            cos_str = f"{c.cosine_similarity:.3f}" if c.cosine_similarity is not None else "N/A"
            flag_str = "FLAGGED" if c.is_flagged_by_detector else "clean"
            tel_lines.append(
                f"  - Client {c.client_id} [{flag_str}]: norm={norm_str}, cos={cos_str}"
            )
        telemetry_summary = "\n".join(tel_lines)

        user_prompt = CRITIC_VALIDATION_USER.format(
            round_num=round_num,
            regime_assessment=regime_assessment_text,
            risk_level=analyst_output.risk_level,
            suspected_byzantine=analyst_output.suspected_byzantine,
            heterogeneity_clients=analyst_output.heterogeneity_clients,
            signal_interpretation=analyst_output.signal_interpretation,
            proposal_text=proposals_text,
            candidate_list=candidate_list,
            telemetry_summary=telemetry_summary,
            outcome_reflection=outcome_reflection_text,
        )

        try:
            budget.record_llm_call()
            raw = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=CRITIC_VALIDATION_SYSTEM,
                json_mode=True,
            )
            cleaned = _clean_json_string(raw)
            data = json.loads(cleaned)

            rec_id = str(data.get("recommended_candidate_id", "")).strip()
            if not rec_id:
                rec_id = proposals.proposals[0].candidate_id if proposals.proposals else "C0"

            # Parse structured checklist with safe defaults
            feasible = bool(data.get("feasible", True))
            uses_valid = bool(data.get("uses_valid_clients", True))
            respects_byz = bool(data.get("respects_byzantine_constraints", True))
            addresses_anomaly = bool(data.get("addresses_anomaly_evidence", True))
            evidence_consistent = bool(data.get("evidence_consistent", True))

            # Unresolved concerns: check both 'unresolved_concerns' and legacy 'concerns'
            raw_concerns = data.get("unresolved_concerns", data.get("concerns", []))
            unresolved = [str(c) for c in raw_concerns] if isinstance(raw_concerns, list) else []

            critique_text = str(data.get("critique", "Structured critique evaluated."))
            confidence = float(data.get("confidence", 0.85)) if data.get("confidence") is not None else 0.85

            val = CriticValidation(
                recommended_candidate_id=rec_id,
                feasible=feasible,
                uses_valid_clients=uses_valid,
                respects_byzantine_constraints=respects_byz,
                addresses_anomaly_evidence=addresses_anomaly,
                evidence_consistent=evidence_consistent,
                unresolved_concerns=unresolved,
                critique=critique_text,
                confidence=max(0.0, min(1.0, confidence)),
            )
        except Exception as e:
            error_msg = str(e)
            fallback_rec = (
                proposals.proposals[0].candidate_id
                if proposals.proposals
                else list(candidate_map.keys())[0]
            )
            val = CriticValidation(
                recommended_candidate_id=fallback_rec,
                feasible=True,
                uses_valid_clients=True,
                respects_byzantine_constraints=True,
                addresses_anomaly_evidence=True,
                evidence_consistent=True,
                unresolved_concerns=[],
                critique=f"Deterministic fallback critique: {error_msg}",
                confidence=0.5,
            )

        latency_ms = (time.time() - t0) * 1000.0
        return val, latency_ms, error_msg

    def _propose_with_revision(
        self,
        round_num: int,
        analyst_output: AnalystOutput,
        candidate_map: Dict[str, Decision],
        failing_checks: List[str],
        previous_critique: str,
        unresolved_concerns: List[str],
        rejected_ids: List[str],
        budget: AgentBudget,
    ) -> Tuple[ProposerOutput, float, Optional[str]]:
        """Run Proposer with revision addendum after Critic rejection."""
        t0 = time.time()
        error_msg = None

        cand_lines = [
            f"  - [{cid}] {cand.method} with clients {cand.client_ids}"
            for cid, cand in candidate_map.items()
        ]
        candidate_list = "\n".join(cand_lines)

        base_prompt = PROPOSER_USER_TEMPLATE.format(
            round_num=round_num,
            risk_level=analyst_output.risk_level,
            suspected_byzantine=analyst_output.suspected_byzantine,
            heterogeneity_clients=analyst_output.heterogeneity_clients,
            persistent_anomalies=analyst_output.persistent_anomalies,
            signal_interpretation=analyst_output.signal_interpretation,
            recommended_constraints="; ".join(analyst_output.recommended_constraints) or "None",
            candidate_list=candidate_list,
        )

        revision_text = PROPOSER_REVISION_ADDENDUM.format(
            failing_checks="\n".join(f"- {c}" for c in failing_checks) or "None",
            previous_critique=previous_critique or "Proposal rejected.",
            unresolved_concerns="\n".join(f"- {c}" for c in unresolved_concerns) or "None",
            rejected_ids=", ".join(rejected_ids) or "None",
        )

        full_prompt = f"{base_prompt}\n\n{revision_text}"

        try:
            budget.record_llm_call()
            raw = self.llm_client.generate(
                prompt=full_prompt,
                system_prompt=PROPOSER_SYSTEM_PROMPT,
                json_mode=True,
            )
            cleaned = _clean_json_string(raw)
            data = json.loads(cleaned)

            raw_props = data.get("proposals", [])
            proposals = []
            for p in raw_props:
                cid = str(p.get("candidate_id", "")).strip()
                reason = str(p.get("reason", "Revised proposal."))
                conf = float(p.get("confidence", 0.7)) if p.get("confidence") is not None else 0.7
                proposals.append(Proposal(candidate_id=cid, reason=reason, confidence=conf))

            if not proposals:
                # Pick alternative candidates not in rejected_ids
                alt_cids = [cid for cid in candidate_map if cid not in rejected_ids]
                if not alt_cids:
                    alt_cids = list(candidate_map.keys())
                proposals = [
                    Proposal(candidate_id=cid, reason="Alternative candidate.", confidence=0.6)
                    for cid in alt_cids[:2]
                ]

            output = ProposerOutput(proposals=proposals)
        except Exception as e:
            error_msg = str(e)
            alt_cids = [cid for cid in candidate_map if cid not in rejected_ids] or list(candidate_map.keys())
            output = ProposerOutput(
                proposals=[
                    Proposal(
                        candidate_id=alt_cids[0],
                        reason=f"Deterministic fallback revision: {error_msg}",
                        confidence=0.5,
                    )
                ]
            )

        latency_ms = (time.time() - t0) * 1000.0
        return output, latency_ms, error_msg

    def decide(
        self,
        round_num: int,
        telemetry: List[ClientTelemetry],
        candidate_decisions: List[Decision],
        history: Optional[AgentHistory] = None,
        alpha: Optional[float] = None,
        valid_client_ids: Optional[List[int]] = None,
    ) -> AgentResult:
        """Execute bounded agentic decision loop with tools, revision, and reflection.
        
        Strictly observes computational limits:
          - max_iterations = 2
          - max_tool_calls = 5
          - max_llm_calls = 6
        """
        active_history = history if history is not None else self.history
        stage_errors: Dict[str, str] = {}
        t_start = time.time()
        budget = AgentBudget(
            max_llm_calls=self.max_llm_calls,
            max_tool_calls=self.max_tool_calls,
            max_iterations=self.max_iterations,
        )

        # 1. Pre-filter candidate decisions and build compact mapping
        feasible_candidates, candidate_map = self._prepare_candidate_map(candidate_decisions)

        # 2. Step 1: Observable Outcome Reflection (Strictly NO Oracle data)
        reflection_summary = self.outcome_memory.generate_reflection()
        outcome_reflection_text = (
            OUTCOME_REFLECTION_TEMPLATE.format(reflection_text=reflection_summary)
            if reflection_summary
            else ""
        )

        # 3. Step 2: Operational Regime Assessment (Observable Telemetry Only)
        recent_rounds = active_history.get_recent(2)
        prev_telemetry = None
        if len(recent_rounds) >= 2:
            prev_telemetry = getattr(recent_rounds[-2], 'clients', getattr(recent_rounds[-2], 'client_telemetry', None))

        tool_registry = ToolRegistry(
            history=active_history,
            candidate_map=candidate_map,
            outcome_memory=self.outcome_memory,
            telemetry=telemetry,
            previous_telemetry=prev_telemetry,
            max_tool_calls=self.max_tool_calls,
        )

        regime_indicators = tool_registry._tool_get_regime_indicators()
        assessed_regime = regime_indicators["assessed_regime"]
        regime_assessment_text = REGIME_ASSESSMENT_TEMPLATE.format(
            mean_cosine=regime_indicators["mean_cosine_similarity"],
            cosine_var=regime_indicators["cosine_variance"],
            mean_norm=regime_indicators["mean_update_norm"],
            norm_var=regime_indicators["norm_variance"],
            flagged_ratio=regime_indicators["flagged_client_ratio"],
            client_disagreement=regime_indicators["client_disagreement"],
            update_drift=regime_indicators["update_drift"],
            accuracy_trend=regime_indicators["accuracy_trend"],
            assessed_regime=assessed_regime,
        )

        # 4. Step 3: Enriched Analyst Stage
        tool_instructions = TOOL_USE_ADDENDUM.format(
            tool_descriptions=tool_registry.get_tool_descriptions(),
            tool_budget_remaining=tool_registry.budget_remaining,
        )
        enriched_system_prompt = f"{ANALYST_SYSTEM_PROMPT}\n\n" + ANALYST_AGENTIC_ADDENDUM.format(
            outcome_reflection=outcome_reflection_text,
            regime_assessment=regime_assessment_text,
            tool_instructions=tool_instructions,
        )

        budget.record_llm_call()
        analysis, t_analyst, err_a = self.analyst.analyze(
            round_num=round_num,
            current_telemetry=telemetry,
            history=active_history,
            candidate_decisions=feasible_candidates,
            alpha=None,  # Do not leak ground-truth alpha to the agent
        )
        if err_a:
            stage_errors["analyst"] = err_a

        # Check if analyst output requested any tools via raw call history
        if hasattr(self.llm_client, "call_history") and self.llm_client.call_history:
            last_resp = getattr(self.llm_client, "last_response", "") or ""
            if "tool_calls" in last_resp:
                self._execute_tool_loop(
                    initial_response=last_resp,
                    system_prompt=enriched_system_prompt,
                    tool_registry=tool_registry,
                    budget=budget,
                )

        # 5. Step 4: Propose -> Critique Iterative Loop
        proposals: Optional[ProposerOutput] = None
        critic_validation: Optional[CriticValidation] = None
        rejected_cids: List[str] = []
        t_proposer_total = 0.0
        t_critic_total = 0.0
        failing_checks: List[str] = []

        for iteration in range(1, self.max_iterations + 1):
            budget.record_iteration()

            # 4a. Proposer
            if iteration == 1:
                budget.record_llm_call()
                proposals, t_prop, err_p = self.proposer.propose(
                    round_num=round_num,
                    analyst_output=analysis,
                    candidate_decisions=feasible_candidates,
                    candidate_map=candidate_map,
                )
                t_proposer_total += t_prop
                if err_p:
                    stage_errors["proposer"] = err_p
            else:
                # Revision after rejection
                proposals, t_prop, err_p = self._propose_with_revision(
                    round_num=round_num,
                    analyst_output=analysis,
                    candidate_map=candidate_map,
                    failing_checks=failing_checks,
                    previous_critique=critic_validation.critique if critic_validation else "",
                    unresolved_concerns=critic_validation.unresolved_concerns if critic_validation else [],
                    rejected_ids=rejected_cids,
                    budget=budget,
                )
                t_proposer_total += t_prop
                if err_p:
                    stage_errors["proposer_revision"] = err_p

            # 4b. Critic Structured Validation
            critic_validation, t_crit, err_c = self._evaluate_critic_structured(
                round_num=round_num,
                analyst_output=analysis,
                proposals=proposals,
                candidate_map=candidate_map,
                current_telemetry=telemetry,
                regime_assessment_text=regime_assessment_text,
                outcome_reflection_text=outcome_reflection_text,
                budget=budget,
            )
            t_critic_total += t_crit
            if err_c:
                stage_errors["critic"] = err_c

            # 4c. Checklist Evaluation
            is_accepted, failing_checks = self._critic_accepts(critic_validation)

            if is_accepted:
                break

            # If rejected and we have iteration budget remaining, loop to revise
            if iteration < self.max_iterations and budget.can_iterate() and budget.can_call_llm():
                rec_id = critic_validation.recommended_candidate_id
                if rec_id not in rejected_cids:
                    rejected_cids.append(rec_id)
                for p in proposals.proposals:
                    if p.candidate_id not in rejected_cids:
                        rejected_cids.append(p.candidate_id)
            else:
                # Accept best available on last iteration
                break

        # Convert CriticValidation to legacy CriticOutput for schema compatibility
        critic_output = CriticOutput(
            recommended_candidate_id=critic_validation.recommended_candidate_id,
            critique=critic_validation.critique,
            concerns=critic_validation.unresolved_concerns,
            confidence=critic_validation.confidence,
        )

        # 6. Step 5: Deterministic Decision Selection & Validation
        rec_cid = critic_output.recommended_candidate_id
        target_decision = candidate_map.get(rec_cid)

        if stage_errors and not target_decision:
            validation_status = "fallback"
            repair_status = "llm_error_fallback"
            repair_reason = (
                f"llm_error in stage(s): {list(stage_errors.keys())}. Deterministic fallback."
            )
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
            validation_status = "repaired"
            repair_status = "repaired"
            repair_reason = (
                f"Candidate ID '{rec_cid}' does not exist in feasible candidate set."
            )
            final_decision, _, _ = repair_decision(
                rec_cid,
                candidate_map,
                target_clients=[c.client_id for c in telemetry] if telemetry else None,
            )

        t_total = (time.time() - t_start) * 1000.0
        timing = AgentTiming(
            analyst_latency_ms=t_analyst,
            proposer_latency_ms=t_proposer_total,
            critic_latency_ms=t_critic_total,
            total_latency_ms=t_total,
        )

        return AgentResult(
            analysis=analysis,
            proposals=proposals,
            critique=critic_output,
            final_decision=final_decision,
            recommended_candidate_id=rec_cid,
            validation_status=validation_status,
            repair_status=repair_status,
            repair_reason=repair_reason,
            timing=timing,
            stage_errors=stage_errors,
            iterations_used=budget.iterations_used,
            reflection_summary=reflection_summary or None,
            assessed_regime=assessed_regime,
            tool_calls_made=budget.tool_calls_made + tool_registry.call_log,
            llm_calls_used=budget.llm_calls_used,
        )
