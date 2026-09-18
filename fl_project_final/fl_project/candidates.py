"""Deterministic Candidate Generation for Federated Learning.

Generates a compact, diverse pool of ~10 pre-approved candidate decisions
using stable IDs (C0, C1, ...). Ensures candidates are mathematically feasible
and provides strategic variety (full set baselines, clean subsets, leave-one-out).
"""

from typing import Dict, List, Optional, Set
from agent.schemas import Decision
from feasibility import check_mathematical_feasibility


def generate_experiment3_candidates(
    all_client_ids: List[int],
    flagged_ids: Optional[List[int]] = None,
    persistent_ids: Optional[List[int]] = None,
    byzantine_budget: int = 2,
    trim_ratio: float = 0.2,
    max_candidates: int = 10,
) -> List[Decision]:
    """Generate a stable, deterministic set of ~10 candidate aggregation decisions.

    Guarantees:
      - All candidates satisfy check_mathematical_feasibility.
      - Assigned sequential stable IDs: C0, C1, ...
      - Includes full-set baselines (FedAvg, Median, TrimmedMean, Krum if feasible).
      - If anomalies/persistent clients exist, includes clean subsets across methods.
      - Provides targeted exclusions of suspicious clients.
    """
    flagged_set = set(flagged_ids or [])
    persistent_set = set(persistent_ids or [])
    all_set = set(all_client_ids)
    sorted_all = sorted(all_client_ids)

    # 1. Base candidates on all clients
    raw_candidates = [
        ("FedAvg", sorted_all),
        ("Median", sorted_all),
        ("TrimmedMean", sorted_all),
    ]

    # Krum on all clients (feasible if n >= 2m + 3)
    krum_all_feasible, _ = check_mathematical_feasibility(
        "Krum", sorted_all, byzantine_budget=byzantine_budget
    )
    if krum_all_feasible:
        raw_candidates.append(("Krum", sorted_all))

    # 2. Subset excluding persistent anomalies (if any)
    clean_persistent = sorted(list(all_set - persistent_set))
    if persistent_set and len(clean_persistent) >= 1:
        raw_candidates.append(("FedAvg", clean_persistent))
        raw_candidates.append(("TrimmedMean", clean_persistent))
        raw_candidates.append(("Median", clean_persistent))
        krum_clean_p, _ = check_mathematical_feasibility(
            "Krum", clean_persistent, byzantine_budget=byzantine_budget
        )
        if krum_clean_p:
            raw_candidates.append(("Krum", clean_persistent))

    # 3. Subset excluding all flagged anomalies (if different from persistent)
    clean_all = sorted(list(all_set - flagged_set))
    if flagged_set and clean_all != clean_persistent and len(clean_all) >= 1:
        raw_candidates.append(("FedAvg", clean_all))
        raw_candidates.append(("TrimmedMean", clean_all))
        raw_candidates.append(("Median", clean_all))
        krum_clean_a, _ = check_mathematical_feasibility(
            "Krum", clean_all, byzantine_budget=byzantine_budget
        )
        if krum_clean_a:
            raw_candidates.append(("Krum", clean_all))

    # 4. Leave-one-out candidates for individual suspicious clients
    suspicious_ordered = list(persistent_set) + [cid for cid in flagged_set if cid not in persistent_set]
    for cid in suspicious_ordered:
        loo_subset = sorted([c for c in all_client_ids if c != cid])
        if len(loo_subset) >= 1:
            raw_candidates.append(("TrimmedMean", loo_subset))
            raw_candidates.append(("Median", loo_subset))

    # 5. If still under max_candidates (e.g. clean rounds with no anomalies),
    # provide diverse conservative subsets (leave-one-out on edge clients)
    if len(raw_candidates) < max_candidates:
        for cid in sorted_all:
            loo_subset = sorted([c for c in all_client_ids if c != cid])
            if len(loo_subset) >= 1:
                raw_candidates.append(("FedAvg", loo_subset))
                raw_candidates.append(("TrimmedMean", loo_subset))
            if len(raw_candidates) >= max_candidates * 2:
                break

    # 6. Deduplicate while preserving order and verifying mathematical feasibility
    seen = set()
    validated_candidates: List[Decision] = []

    for method, subset in raw_candidates:
        key = (method, tuple(sorted(subset)))
        if key in seen:
            continue
        seen.add(key)

        is_feasible, _ = check_mathematical_feasibility(
            method, subset, byzantine_budget=byzantine_budget, trim_ratio=trim_ratio
        )
        if is_feasible:
            cid_label = f"C{len(validated_candidates)}"
            validated_candidates.append(
                Decision(method=method, client_ids=subset, candidate_id=cid_label)
            )
            if len(validated_candidates) >= max_candidates:
                break

    return validated_candidates
