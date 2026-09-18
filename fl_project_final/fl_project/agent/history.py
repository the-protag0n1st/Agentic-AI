"""Multi-Round Telemetry Memory & Trajectory Tracking.

Maintains a sliding window of historical telemetry across communication rounds,
allowing the agent to distinguish single-round transient outliers from persistent
adversarial behavior without hardcoding client count assumptions.
"""

from collections import deque
from typing import Dict, List, Optional
from .schemas import ClientTelemetry, Decision, RoundTelemetry


class AgentHistory:
    """Sliding-window buffer storing round-by-round client telemetry and decisions."""

    def __init__(self, history_length: int = 3):
        if history_length < 1:
            raise ValueError("history_length must be at least 1")
        self.history_length = history_length
        self._rounds: deque[RoundTelemetry] = deque(maxlen=history_length)
        self._decisions: deque[Optional[Decision]] = deque(maxlen=history_length)

    def add_round(
        self,
        round_num: int,
        client_telemetry: List[ClientTelemetry],
        decision_made: Optional[Decision] = None,
        alpha: Optional[float] = None,
        global_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        scalar_metrics = {
            k: float(v) for k, v in (global_metrics or {}).items()
            if isinstance(v, (int, float))
        }
        round_data = RoundTelemetry(
            round_num=round_num,
            alpha=alpha,
            clients=client_telemetry,
            global_metrics=scalar_metrics,
        )
        self._rounds.append(round_data)
        self._decisions.append(decision_made)

    def get_recent(self, n: Optional[int] = None) -> List[RoundTelemetry]:
        """Return the most recent N rounds of telemetry (chronological order)."""
        limit = n if n is not None else self.history_length
        rounds_list = list(self._rounds)
        return rounds_list[-limit:]

    def get_client_trajectory(self, client_id: int, n: Optional[int] = None) -> List[Dict]:
        """Extract historical metrics trajectory for a specific client across recent rounds."""
        limit = n if n is not None else self.history_length
        recent_rounds = self.get_recent(limit)
        trajectory = []

        for r in recent_rounds:
            client_record = next((c for c in r.clients if c.client_id == client_id), None)
            if client_record:
                trajectory.append({
                    "round": r.round_num,
                    "norm": client_record.update_norm,
                    "cosine": client_record.cosine_similarity,
                    "var": client_record.weight_var,
                    "flagged": client_record.is_flagged_by_detector,
                    "anomaly_score": client_record.anomaly_score,
                })
        return trajectory

    def has_persistent_anomaly(self, client_id: int, min_rounds: int = 2) -> bool:
        """Return True if the client was flagged anomalous in >= min_rounds of recent history."""
        trajectory = self.get_client_trajectory(client_id)
        if not trajectory:
            return False
        flagged_count = sum(1 for step in trajectory if step.get("flagged") is True)
        return flagged_count >= min_rounds

    def get_all_client_ids(self) -> List[int]:
        """Extract unique client IDs observed across all stored rounds."""
        client_ids = set()
        for r in self._rounds:
            for c in r.clients:
                client_ids.add(c.client_id)
        return sorted(client_ids)

    def clear(self) -> None:
        """Clear all stored historical records."""
        self._rounds.clear()
        self._decisions.clear()

    def __len__(self) -> int:
        return len(self._rounds)
