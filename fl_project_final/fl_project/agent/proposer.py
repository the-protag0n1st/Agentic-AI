"""Proposer Stage: Candidate Selection from Feasible Space.

Considers the Analyst's risk assessment and recommends 2 to 3 candidate
decisions chosen strictly from the pre-approved feasible candidate set.
"""

import json
import re
import time
from typing import Dict, List, Optional, Tuple

from .llm_client import LLMClient
from .prompts import PROPOSER_SYSTEM_PROMPT, PROPOSER_USER_TEMPLATE
from .schemas import AnalystOutput, Decision, Proposal, ProposerOutput


def _clean_json_string(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


class Proposer:
    """Stage 2: Generates diverse candidate proposals from the feasible decision space."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def propose(
        self,
        round_num: int,
        analyst_output: AnalystOutput,
        candidate_decisions: List[Decision],
        candidate_map: Dict[str, Decision],
    ) -> Tuple[ProposerOutput, float, Optional[str]]:
        """Propose 2-3 candidate decisions from candidate_map.

        Returns:
            (ProposerOutput, latency_ms, error_msg_if_any)
        """
        cand_lines = [
            f"  - [{cid}] {cand.method} with clients {cand.client_ids}"
            for cid, cand in candidate_map.items()
        ]
        candidate_list = "\n".join(cand_lines)

        user_prompt = PROPOSER_USER_TEMPLATE.format(
            round_num=round_num,
            risk_level=analyst_output.risk_level,
            suspected_byzantine=analyst_output.suspected_byzantine,
            heterogeneity_clients=analyst_output.heterogeneity_clients,
            persistent_anomalies=analyst_output.persistent_anomalies,
            signal_interpretation=analyst_output.signal_interpretation,
            recommended_constraints="; ".join(analyst_output.recommended_constraints) or "None",
            candidate_list=candidate_list,
        )

        t0 = time.time()
        error_msg = None
        try:
            raw_response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=PROPOSER_SYSTEM_PROMPT,
                json_mode=True,
            )
            cleaned = _clean_json_string(raw_response)
            data = json.loads(cleaned)

            proposals_raw = data.get("proposals", [])
            if not proposals_raw or not isinstance(proposals_raw, list):
                raise ValueError("Proposer output missing non-empty 'proposals' list.")

            proposals = []
            for p in proposals_raw:
                cid = str(p.get("candidate_id", "")).strip()
                reason = str(p.get("reason", "No reason provided."))
                conf = float(p.get("confidence", 0.7)) if p.get("confidence") is not None else 0.7
                conf = max(0.0, min(1.0, conf))
                proposals.append(Proposal(candidate_id=cid, reason=reason, confidence=conf))

            output = ProposerOutput(proposals=proposals)

        except Exception as e:
            error_msg = str(e)
            # Safe deterministic fallback: propose the first 1 or 2 available candidates
            available_cids = list(candidate_map.keys())
            fallback_proposals = []
            if available_cids:
                fallback_proposals.append(
                    Proposal(
                        candidate_id=available_cids[0],
                        reason=f"Deterministic fallback proposal C0: {error_msg}",
                        confidence=0.5,
                    )
                )
            if len(available_cids) > 1:
                fallback_proposals.append(
                    Proposal(
                        candidate_id=available_cids[1],
                        reason="Deterministic fallback alternative candidate",
                        confidence=0.5,
                    )
                )
            output = ProposerOutput(proposals=fallback_proposals)

        latency_ms = (time.time() - t0) * 1000.0
        return output, latency_ms, error_msg
