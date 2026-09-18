"""Counterfactual Oracle for Federated Learning Decision Evaluation.

Evaluates all pre-approved candidate decisions on a fixed validation subset
reusing already-trained client updates (zero retraining overhead).
Computes instantaneous and cumulative regret strictly for post-hoc evaluation.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple
import torch

from aggregation import aggregate
from agent.schemas import Decision
from models import evaluate_model, device


class CounterfactualOracle:
    """Evaluates decision quality against counterfactual alternatives on validation data."""

    def __init__(self, val_subset):
        self.val_subset = val_subset
        self.cumulative_regret = 0.0

    def evaluate_candidates(
        self,
        base_model: torch.nn.Module,
        candidate_decisions: List[Decision],
        local_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        selected_decision: Decision,
        byzantine_budget: int = 2,
        trim_ratio: float = 0.2,
        metric_key: str = "accuracy",
    ) -> Dict[str, Any]:
        """Evaluate each feasible candidate on val_subset and compute regret.

        Args:
            base_model: Current global neural network architecture.
            candidate_decisions: List of Decision objects to evaluate.
            local_weights: List of client state dicts (already trained).
            global_weights: Previous global weights state dict.
            selected_decision: The actual Decision chosen by the agent or baseline.
            byzantine_budget: Configured Byzantine budget for Krum.
            trim_ratio: Configured trim ratio for TrimmedMean.
            metric_key: Metric to maximize ('accuracy' or 'f1').

        Returns:
            Dict containing oracle_decision, oracle_score, selected_score,
            instantaneous regret, cumulative_regret, and candidate scores.
        """
        candidate_evaluations: List[Dict[str, Any]] = []
        best_cand: Optional[Decision] = None
        best_score = -float("inf")
        best_metrics: Dict[str, float] = {}

        selected_norm_method = selected_decision.normalized_method()
        selected_clients_set = set(selected_decision.client_ids)
        selected_score = 0.0
        selected_metrics: Dict[str, float] = {}

        for cand in candidate_decisions:
            weights_subset = [local_weights[i] for i in cand.client_ids]
            method_name = cand.method

            try:
                aggregated_w = aggregate(
                    method_name,
                    weights_subset,
                    global_weights=global_weights,
                    num_byzantine=byzantine_budget,
                    trim_ratio=trim_ratio,
                )

                test_model = copy.deepcopy(base_model)
                test_model.load_state_dict(aggregated_w)
                metrics = evaluate_model(test_model, self.val_subset)
                score = float(metrics.get(metric_key, 0.0))

            except Exception as e:
                score = -1.0
                metrics = {"error": str(e), "accuracy": 0.0, "loss": 99.0}

            cand_record = {
                "candidate_id": cand.candidate_id,
                "method": cand.method,
                "client_ids": cand.client_ids,
                "score": score,
                "metrics": metrics,
            }
            candidate_evaluations.append(cand_record)

            if score > best_score:
                best_score = score
                best_cand = cand
                best_metrics = metrics

            # Check if this candidate matches the chosen decision
            if (cand.normalized_method() == selected_norm_method and
                    set(cand.client_ids) == selected_clients_set):
                selected_score = score
                selected_metrics = metrics

        # Fallback if selected decision was not in candidate list
        if not selected_metrics:
            try:
                weights_subset = [local_weights[i] for i in selected_decision.client_ids]
                agg_w = aggregate(
                    selected_decision.method,
                    weights_subset,
                    global_weights=global_weights,
                    num_byzantine=byzantine_budget,
                    trim_ratio=trim_ratio,
                )
                test_model = copy.deepcopy(base_model)
                test_model.load_state_dict(agg_w)
                selected_metrics = evaluate_model(test_model, self.val_subset)
                selected_score = float(selected_metrics.get(metric_key, 0.0))
            except Exception:
                selected_score = 0.0

        instantaneous_regret = max(0.0, best_score - selected_score)
        self.cumulative_regret += instantaneous_regret

        return {
            "oracle_decision": best_cand,
            "oracle_candidate_id": best_cand.candidate_id if best_cand else None,
            "oracle_score": best_score,
            "oracle_metrics": best_metrics,
            "selected_score": selected_score,
            "selected_metrics": selected_metrics,
            "regret": instantaneous_regret,
            "cumulative_regret": self.cumulative_regret,
            "candidate_evaluations": candidate_evaluations,
        }

    def reset(self):
        """Reset cumulative regret tracking."""
        self.cumulative_regret = 0.0
