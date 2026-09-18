"""Observable Outcome Memory & Reflection for Outcome-Aware Adaptation.

Stores ONLY post-decision observable information available to a real FL server.
NEVER stores Oracle decisions, Oracle accuracy, or regret values.
Oracle evaluation is strictly an offline experimental metric.
"""

from collections import deque
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ObservableOutcome(BaseModel):
    """Observable outcome from a completed FL round.
    Contains ONLY information available to a real FL server."""
    model_config = {"extra": "forbid"}

    round_num: int
    selected_method: str
    selected_client_ids: List[int]
    observed_accuracy: float
    observed_loss: float
    observed_f1: Optional[float] = None
    accuracy_delta: float = 0.0          # vs. previous round
    loss_delta: float = 0.0
    flagged_client_count: int = 0
    total_client_count: int = 0
    mean_cosine_similarity: float = 0.0
    mean_update_norm: float = 0.0
    validation_status: str = "valid"     # valid / repaired / fallback
    tool_calls_used: int = 0
    iterations_used: int = 1


class OutcomeMemory:
    """Bounded FIFO memory of observable round outcomes.
    
    Stores ONLY post-decision observable information.
    NEVER stores Oracle decisions, Oracle accuracy, or regret.
    """
    
    def __init__(self, max_length: int = 3):
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        self.max_length = max_length
        self._outcomes: deque[ObservableOutcome] = deque(maxlen=max_length)
    
    def record(self, outcome: ObservableOutcome) -> None:
        """Record an observable outcome from a completed round."""
        self._outcomes.append(outcome)
    
    def get_recent(self, n: Optional[int] = None) -> List[ObservableOutcome]:
        """Return the most recent N outcomes in chronological order."""
        limit = n if n is not None else self.max_length
        outcomes_list = list(self._outcomes)
        return outcomes_list[-limit:]
    
    def generate_reflection(self) -> str:
        """Generate natural-language reflection from observable outcomes.
        
        Analyzes:
        - Accuracy trend (improving / stable / degrading)
        - Whether recent decisions led to improvement or degradation
        - Method selection patterns
        - Anomaly trend (flagged ratio changing)
        - Client participation patterns
        
        Returns a concise paragraph for prompt injection.
        Returns empty string if no outcomes recorded yet.
        """
        outcomes = list(self._outcomes)
        if not outcomes:
            return ""
        
        parts = []
        
        # Accuracy trend
        if len(outcomes) >= 2:
            first_acc = outcomes[0].observed_accuracy
            last_acc = outcomes[-1].observed_accuracy
            delta = last_acc - first_acc
            n_rounds = len(outcomes)
            
            if delta > 0.005:
                trend = "improving"
            elif delta < -0.005:
                trend = "degrading"
            else:
                trend = "stable"
            
            parts.append(
                f"Over the last {n_rounds} rounds, accuracy has been {trend} "
                f"(from {first_acc:.1%} to {last_acc:.1%})."
            )
        
        # Most recent outcome
        latest = outcomes[-1]
        parts.append(
            f"Previous decision: {latest.selected_method} on clients "
            f"{sorted(latest.selected_client_ids)} produced accuracy {latest.observed_accuracy:.1%}"
            f" (change: {latest.accuracy_delta:+.1%})."
        )
        
        # Anomaly trend
        if len(outcomes) >= 2:
            prev_flagged_ratio = (
                outcomes[-2].flagged_client_count / max(outcomes[-2].total_client_count, 1)
            )
            curr_flagged_ratio = (
                latest.flagged_client_count / max(latest.total_client_count, 1)
            )
            if curr_flagged_ratio > prev_flagged_ratio + 0.05:
                parts.append(
                    f"Flagged client ratio increased from {prev_flagged_ratio:.0%} "
                    f"to {curr_flagged_ratio:.0%}, suggesting growing instability."
                )
            elif curr_flagged_ratio < prev_flagged_ratio - 0.05:
                parts.append(
                    f"Flagged client ratio decreased from {prev_flagged_ratio:.0%} "
                    f"to {curr_flagged_ratio:.0%}, suggesting improving stability."
                )
        
        # Method selection pattern
        if len(outcomes) >= 2:
            methods = [o.selected_method for o in outcomes]
            if len(set(methods)) == 1:
                parts.append(
                    f"The same method ({methods[0]}) has been used for "
                    f"{len(methods)} consecutive rounds."
                )
        
        # Validation status warnings
        recent_fallbacks = sum(1 for o in outcomes if o.validation_status == "fallback")
        if recent_fallbacks > 0:
            parts.append(
                f"Warning: {recent_fallbacks} of the last {len(outcomes)} rounds "
                f"required fallback to deterministic selection."
            )
        
        return " ".join(parts)
    
    def get_history_dicts(self, n: int = 3) -> List[Dict]:
        """Return outcome history as plain dicts for tool access."""
        recent = self.get_recent(n)
        return [
            {
                "round": o.round_num,
                "selected_method": o.selected_method,
                "selected_clients": sorted(o.selected_client_ids),
                "observed_accuracy": round(o.observed_accuracy, 4),
                "observed_loss": round(o.observed_loss, 4),
                "accuracy_delta": round(o.accuracy_delta, 4),
                "flagged_client_count": o.flagged_client_count,
                "mean_cosine": round(o.mean_cosine_similarity, 4),
            }
            for o in recent
        ]
    
    def clear(self) -> None:
        """Clear all stored outcomes."""
        self._outcomes.clear()
    
    def __len__(self) -> int:
        return len(self._outcomes)
