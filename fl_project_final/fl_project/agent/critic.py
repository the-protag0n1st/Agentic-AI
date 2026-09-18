"""Critic Stage: Adversarial Self-Verification & Proposal Challenge.

Scrutinizes Proposer recommendations against telemetry evidence, historical trends,
heterogeneity risks, and mathematical feasibility, selecting the most defensible option.
"""

import json
import re
import time
from typing import Dict, List, Optional, Tuple

from .history import AgentHistory
from .llm_client import LLMClient
from .prompts import CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE
from .schemas import AnalystOutput, ClientTelemetry, CriticOutput, Decision, ProposerOutput


def _clean_json_string(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


class Critic:
    """Stage 3: Adversarial validation and critique of proposed aggregation decisions."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def critique(
        self,
        round_num: int,
        analyst_output: AnalystOutput,
        proposer_output: ProposerOutput,
        candidate_map: Dict[str, Decision],
        current_telemetry: List[ClientTelemetry],
        history: AgentHistory,
    ) -> Tuple[CriticOutput, float, Optional[str]]:
        """Run critique on proposals.

        Returns:
            (CriticOutput, latency_ms, error_msg_if_any)
        """
        # Format candidate list
        cand_lines = [
            f"  - [{cid}] {cand.method} with clients {cand.client_ids}"
            for cid, cand in candidate_map.items()
        ]
        candidate_list = "\n".join(cand_lines)

        # Format proposals text
        prop_lines = []
        for p in proposer_output.proposals:
            cand_info = candidate_map.get(p.candidate_id)
            details = f"({cand_info.method}, clients {cand_info.client_ids})" if cand_info else "(Unknown)"
            conf_str = f"conf={p.confidence:.2f}" if p.confidence is not None else ""
            prop_lines.append(f"  - Proposal [{p.candidate_id}] {details}: {p.reason} {conf_str}")
        proposals_text = "\n".join(prop_lines)

        # Format telemetry & history context
        tel_lines = []
        for c in current_telemetry:
            norm_str = f"{c.update_norm:.3f}" if c.update_norm is not None else "N/A"
            cos_str = f"{c.cosine_similarity:.3f}" if c.cosine_similarity is not None else "N/A"
            traj = history.get_client_trajectory(c.client_id, n=3)
            hist_cos = [f"R{t['round']}:{t['cosine']:.2f}" for t in traj if t.get('cosine') is not None]
            tel_lines.append(
                f"  - Client {c.client_id}: norm={norm_str}, cos={cos_str}, hist_traj=[{', '.join(hist_cos)}]"
            )
        telemetry_summary = "\n".join(tel_lines)

        user_prompt = CRITIC_USER_TEMPLATE.format(
            round_num=round_num,
            risk_level=analyst_output.risk_level,
            suspected_byzantine=analyst_output.suspected_byzantine,
            heterogeneity_clients=analyst_output.heterogeneity_clients,
            signal_interpretation=analyst_output.signal_interpretation,
            proposals_text=proposals_text,
            candidate_list=candidate_list,
            telemetry_summary=telemetry_summary,
        )

        t0 = time.time()
        error_msg = None
        try:
            raw_response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=CRITIC_SYSTEM_PROMPT,
                json_mode=True,
            )
            cleaned = _clean_json_string(raw_response)
            data = json.loads(cleaned)

            rec_id = str(data.get("recommended_candidate_id", "")).strip()
            critique_text = str(data.get("critique", "No critique provided."))
            concerns = [str(c) for c in data.get("concerns", [])]
            conf = float(data.get("confidence", 0.7)) if data.get("confidence") is not None else 0.7
            conf = max(0.0, min(1.0, conf))

            if not rec_id:
                raise ValueError("Critic output missing 'recommended_candidate_id'.")

            output = CriticOutput(
                recommended_candidate_id=rec_id,
                critique=critique_text,
                concerns=concerns,
                confidence=conf,
            )

        except Exception as e:
            error_msg = str(e)
            # Safe fallback: pick the first proposal's candidate ID or first available candidate
            fallback_rec = (
                proposer_output.proposals[0].candidate_id
                if proposer_output.proposals
                else list(candidate_map.keys())[0]
            )
            output = CriticOutput(
                recommended_candidate_id=fallback_rec,
                critique=f"Deterministic fallback critique due to LLM/parse error: {error_msg}",
                concerns=["Fallback executed without adversarial critique"],
                confidence=0.5,
            )

        latency_ms = (time.time() - t0) * 1000.0
        return output, latency_ms, error_msg
