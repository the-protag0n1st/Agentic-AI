"""Deterministic Feasibility and Validation Layer for Federated Learning.

Ensures that executable aggregation decisions, represented strictly as (method, subset)
pairs or Decision objects, satisfy mathematical constraints and match pre-approved
candidate sets before server execution.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
from agent.schemas import Decision

SUPPORTED_METHODS: Set[str] = {"FedAvg", "Median", "TrimmedMean", "Krum"}


def normalize_method_name(method: str) -> str:
    clean = method.strip().lower().replace(" ", "").replace("_", "")
    if clean in ("fedavg", "average"):
        return "FedAvg"
    if clean in ("median", "coordinatemedian", "coordinatewisemedian"):
        return "Median"
    if clean in ("trimmedmean", "trimmed_mean", "trimmed"):
        return "TrimmedMean"
    if clean in ("krum", "multikrum"):
        return "Krum"
    return method


def check_mathematical_feasibility(
    method: str,
    client_ids: List[int],
    byzantine_budget: int = 1,
    trim_ratio: float = 0.2,
) -> Tuple[bool, str]:
    n = len(client_ids)
    if n == 0:
        return False, "Selected client subset is empty (|G| = 0)."

    norm_method = normalize_method_name(method)

    if norm_method not in SUPPORTED_METHODS:
        return False, f"Unsupported aggregation method: '{method}'."

    if norm_method in ("FedAvg", "Median"):
        return True, "Feasible"

    if norm_method == "TrimmedMean":
        k = int(n * trim_ratio)
        if k > 0 and n < (2 * k + 1):
            return False, f"TrimmedMean requires |G| >= {2 * k + 1} for trim_ratio={trim_ratio}, but got |G|={n}."
        return True, "Feasible"

    if norm_method == "Krum":
        required_n = 2 * byzantine_budget + 3
        if n < required_n:
            return False, f"Krum requires |G| >= {required_n} for byzantine_budget={byzantine_budget}, but got |G|={n}."
        return True, "Feasible"

    return False, f"Unrecognized aggregation method: {norm_method}"


def validate_decision(
    decision: Union[Decision, Tuple[str, List[int]]],
    candidate_decisions: List[Decision],
    valid_client_ids: Optional[List[int]] = None,
    byzantine_budget: int = 1,
    trim_ratio: float = 0.2,
) -> Tuple[bool, str]:
    if isinstance(decision, tuple):
        method, client_ids = decision[0], list(decision[1])
        dec_obj = Decision(method=method, client_ids=client_ids)
    else:
        dec_obj = decision

    if valid_client_ids is not None:
        valid_set = set(valid_client_ids)
        invalid_ids = [cid for cid in dec_obj.client_ids if cid not in valid_set]
        if invalid_ids:
            return False, f"Decision contains invalid client IDs: {invalid_ids} (allowed: {sorted(valid_set)})"

    feasible, reason = check_mathematical_feasibility(
        dec_obj.method, dec_obj.client_ids, byzantine_budget=byzantine_budget, trim_ratio=trim_ratio
    )
    if not feasible:
        return False, reason

    norm_method = dec_obj.normalized_method()
    decision_clients = set(dec_obj.client_ids)

    matched = False
    for cand in candidate_decisions:
        if cand.normalized_method() == norm_method and set(cand.client_ids) == decision_clients:
            matched = True
            break

    if not matched:
        return False, "Decision does not match any pre-approved candidate in the candidate pool."

    return True, "Valid"


def repair_decision(
    recommended_candidate_id: Optional[str],
    candidate_map: Dict[str, Decision],
    target_clients: Optional[List[int]] = None,
    target_method: Optional[str] = None,
) -> Tuple[Decision, str, str]:
    if not candidate_map:
        raise ValueError("Cannot repair decision: candidate_map is empty.")

    if recommended_candidate_id and recommended_candidate_id in candidate_map:
        return candidate_map[recommended_candidate_id], "valid", "none"

    reason = f"Candidate ID '{recommended_candidate_id}' not found in candidate pool."

    target_set = set(target_clients) if target_clients is not None else set()
    norm_target_method = (
        normalize_method_name(target_method) if target_method else None
    )

    scored_candidates = []
    for cid, cand in candidate_map.items():
        cand_set = set(cand.client_ids)
        sym_diff = len(cand_set ^ target_set) if target_set else 0
        method_mismatch = 0 if (norm_target_method and cand.normalized_method() == norm_target_method) else 1
        scored_candidates.append(((sym_diff, method_mismatch, cid), cid, cand))

    scored_candidates.sort(key=lambda x: x[0])
    best_decision = scored_candidates[0][2]

    return best_decision, "repaired", reason
