"""Analyst Stage: Multi-Round Telemetry Interpretation.

Synthesizes current-round signals, sliding-window trajectories, and candidate
options to assess risk, distinguish non-IID heterogeneity from Byzantine threats,
and recommend strategic constraints without final decision authority.
"""

import json
import re
import time
from typing import List, Optional, Tuple

from .history import AgentHistory
from .llm_client import LLMClient
from .prompts import ANALYST_SYSTEM_PROMPT, ANALYST_USER_TEMPLATE
from .schemas import AnalystOutput, ClientTelemetry, Decision


def _clean_json_string(raw: str) -> str:
    """Strip markdown code block fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


class Analyst:
    """Stage 1: Interprets telemetry and multi-round trajectories."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze(
        self,
        round_num: int,
        current_telemetry: List[ClientTelemetry],
        history: AgentHistory,
        candidate_decisions: List[Decision],
        alpha: Optional[float] = None,
    ) -> Tuple[AnalystOutput, float, Optional[str]]:
        """Run telemetry analysis.

        Returns:
            (AnalystOutput, latency_ms, error_msg_if_any)
        """
        # Format current telemetry
        tel_lines = []
        for c in current_telemetry:
            norm_str = f"{c.update_norm:.4f}" if c.update_norm is not None else "N/A"
            cos_str = f"{c.cosine_similarity:.4f}" if c.cosine_similarity is not None else "N/A"
            var_str = f"{c.weight_var:.4f}" if c.weight_var is not None else "N/A"
            flag_str = "FLAGGED" if c.is_flagged_by_detector else "clean"
            tel_lines.append(
                f"  - Client {c.client_id}: norm={norm_str}, cosine={cos_str}, var={var_str}, detector={flag_str}"
            )
        telemetry_summary = "\n".join(tel_lines) if tel_lines else "No current client telemetry."

        # Format history summary
        hist_lines = []
        recent_rounds = history.get_recent(history.history_length)
        if not recent_rounds:
            hist_lines.append("  No previous rounds recorded (Round 1).")
        else:
            for r in recent_rounds:
                flagged_ids = [c.client_id for c in r.clients if c.is_flagged_by_detector]
                hist_lines.append(
                    f"  - Round {r.round_num}: flagged={flagged_ids}, total_clients={len(r.clients)}"
                )
                for cid in history.get_all_client_ids():
                    traj = history.get_client_trajectory(cid, n=3)
                    if traj:
                        t_summary = ", ".join(
                            f"R{t['round']}:cos={t['cosine']:.2f}" if t.get('cosine') is not None else f"R{t['round']}:N/A"
                            for t in traj
                        )
                        hist_lines.append(f"    Client {cid} trajectory -> [{t_summary}]")
        history_summary = "\n".join(hist_lines)

        # Format candidates
        cand_lines = [
            f"  [{c.candidate_id or f'C{i}'}] {c.method} on clients {c.client_ids}"
            for i, c in enumerate(candidate_decisions)
        ]
        candidate_summary = "\n".join(cand_lines) if cand_lines else "No candidates available."

        user_prompt = ANALYST_USER_TEMPLATE.format(
            round_num=round_num,
            alpha=alpha if alpha is not None else "unknown",
            telemetry_summary=telemetry_summary,
            history_length=history.history_length,
            history_summary=history_summary,
            candidate_summary=candidate_summary,
        )

        t0 = time.time()
        error_msg = None
        try:
            raw_response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=ANALYST_SYSTEM_PROMPT,
                json_mode=True,
            )
            cleaned = _clean_json_string(raw_response)
            data = json.loads(cleaned)
            analysis = AnalystOutput.model_validate(data)
        except Exception as e:
            error_msg = str(e)
            # Safe deterministic fallback
            flagged = [c.client_id for c in current_telemetry if c.is_flagged_by_detector]
            analysis = AnalystOutput(
                persistent_anomalies=[],
                heterogeneity_clients=[],
                suspected_byzantine=flagged,
                risk_level="medium" if flagged else "low",
                signal_interpretation=f"Fallback analysis due to parsing/LLM error: {error_msg}",
                recommended_constraints=["Ensure aggregation method is mathematically feasible"],
            )

        latency_ms = (time.time() - t0) * 1000.0
        return analysis, latency_ms, error_msg
