"""Single-Shot LLM Agent for Federated Learning.

Executes exactly one LLM call per communication round using the exact same
LLM provider, model, candidate pool, telemetry, and 3-round history available
to the Reflective Agent, serving as a scientifically rigorous ablation of reflection.
"""

import json
import re
import time
from typing import Dict, List, Optional, Tuple

from feasibility import repair_decision, validate_decision
from .history import AgentHistory
from .llm_client import LLMClient
from .schemas import AgentResult, AgentTiming, ClientTelemetry, Decision

SINGLE_SHOT_SYSTEM_PROMPT = """You are an expert Federated Learning Decision Controller.
Your role is to select BOTH the client subset and the aggregation method from the PRE-APPROVED FEASIBLE CANDIDATE SET.

CRITICAL RULES:
1. You MUST select exactly one candidate ID from the provided candidate list (e.g. C0, C1, ...).
2. You are FORBIDDEN from inventing new candidate IDs, new client subsets, or unsupported methods.
3. Distinguish statistical non-IID heterogeneity from adversarial poisoning. A client with unusual norm or angle may simply have label skew.
4. Respond with ONLY a valid, parseable JSON object matching the requested schema. No markdown fences, no preamble."""

SINGLE_SHOT_USER_TEMPLATE = """Communication Round: {round_num} (Heterogeneity alpha={alpha})

Current Client Telemetry:
{telemetry_summary}

Recent Multi-Round Trajectory (Last {history_length} rounds):
{history_summary}

Pre-Approved Feasible Candidate Decisions (SELECT ONE):
{candidate_list}

Select the single best candidate decision.
Respond with a JSON object strictly conforming to this schema:
{{
  "selected_candidate_id": "<exact ID from list, e.g. C0>",
  "reason": "<one sentence justification>",
  "confidence": <float 0.0 to 1.0>
}}"""


def _clean_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


class SingleShotAgent:
    """Single-stage LLM agent for one-shot federated learning decisions."""

    def __init__(
        self,
        llm_client: LLMClient,
        byzantine_budget: int = 2,
        trim_ratio: float = 0.2,
        history_length: int = 3,
    ):
        self.llm_client = llm_client
        self.byzantine_budget = byzantine_budget
        self.trim_ratio = trim_ratio
        self.history = AgentHistory(history_length=history_length)
        self.prompt_version = "v1_single_shot_direct"

    def decide(
        self,
        round_num: int,
        telemetry: List[ClientTelemetry],
        candidate_decisions: List[Decision],
        history: Optional[AgentHistory] = None,
        alpha: Optional[float] = None,
        valid_client_ids: Optional[List[int]] = None,
    ) -> AgentResult:
        """Execute single-shot decision pipeline with deterministic validation & repair."""
        active_history = history if history is not None else self.history
        candidate_map = {c.candidate_id or f"C{i}": c for i, c in enumerate(candidate_decisions)}
        t_start = time.time()

        # 1. Format telemetry
        tel_lines = []
        for c in telemetry:
            norm_str = f"{c.update_norm:.4f}" if c.update_norm is not None else "N/A"
            cos_str = f"{c.cosine_similarity:.4f}" if c.cosine_similarity is not None else "N/A"
            var_str = f"{c.weight_var:.4f}" if c.weight_var is not None else "N/A"
            flag_str = "FLAGGED" if c.is_flagged_by_detector else "clean"
            tel_lines.append(f"  - Client {c.client_id}: norm={norm_str}, cos={cos_str}, var={var_str}, status={flag_str}")
        tel_summary = "\n".join(tel_lines) if tel_lines else "No telemetry."

        # 2. Format history
        hist_lines = []
        recent_rounds = active_history.get_recent(active_history.history_length)
        if not recent_rounds:
            hist_lines.append("  No previous rounds recorded.")
        else:
            for r in recent_rounds:
                flagged = [c.client_id for c in r.clients if c.is_flagged_by_detector]
                hist_lines.append(f"  - Round {r.round_num}: flagged={flagged}")
        hist_summary = "\n".join(hist_lines)

        # 3. Format candidate pool
        cand_lines = [
            f"  - [{cid}] {cand.method} with clients {cand.client_ids}"
            for cid, cand in candidate_map.items()
        ]
        cand_list = "\n".join(cand_lines)

        user_prompt = SINGLE_SHOT_USER_TEMPLATE.format(
            round_num=round_num,
            alpha=alpha if alpha is not None else "unknown",
            telemetry_summary=tel_summary,
            history_length=active_history.history_length,
            history_summary=hist_summary,
            candidate_list=cand_list,
        )

        stage_error: Optional[str] = None
        selected_cid: Optional[str] = None
        t_llm_start = time.time()

        try:
            raw_response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=SINGLE_SHOT_SYSTEM_PROMPT,
                json_mode=True,
            )
            cleaned = _clean_json(raw_response)
            data = json.loads(cleaned)
            selected_cid = str(data.get("selected_candidate_id", "")).strip()
            reason = str(data.get("reason", "No reason provided."))
            confidence = float(data.get("confidence", 0.7)) if data.get("confidence") is not None else 0.7
        except Exception as e:
            stage_error = str(e)
            selected_cid = None
            reason = f"Fallback due to single-shot error: {e}"
            confidence = 0.5

        t_llm = (time.time() - t_llm_start) * 1000.0

        # Deterministic Validation & Repair
        target_decision = candidate_map.get(selected_cid) if selected_cid else None

        if stage_error:
            validation_status = "fallback"
            repair_status = "llm_error_fallback"
            repair_reason = stage_error
            final_decision, _, _ = repair_decision(
                selected_cid, candidate_map,
                target_clients=[c.client_id for c in telemetry] if telemetry else None,
            )
        elif target_decision is not None:
            is_valid, v_reason = validate_decision(
                target_decision, candidate_decisions,
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
                repair_reason = f"Candidate '{selected_cid}' invalid: {v_reason}"
                final_decision, _, _ = repair_decision(
                    selected_cid, candidate_map,
                    target_clients=target_decision.client_ids,
                    target_method=target_decision.method,
                )
        else:
            validation_status = "repaired"
            repair_status = "repaired"
            repair_reason = f"Candidate ID '{selected_cid}' unknown."
            final_decision, _, _ = repair_decision(
                selected_cid, candidate_map,
                target_clients=[c.client_id for c in telemetry] if telemetry else None,
            )

        t_total = (time.time() - t_start) * 1000.0
        timing = AgentTiming(
            analyst_latency_ms=0.0,
            proposer_latency_ms=0.0,
            critic_latency_ms=t_llm,
            total_latency_ms=t_total,
        )

        return AgentResult(
            final_decision=final_decision,
            recommended_candidate_id=selected_cid,
            validation_status=validation_status,
            repair_status=repair_status,
            repair_reason=repair_reason,
            timing=timing,
            stage_errors={"single_shot": stage_error} if stage_error else {},
        )
