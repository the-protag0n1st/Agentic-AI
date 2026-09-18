"""Unit Tests for Multi-Round Telemetry History & Trajectory Tracking."""

import pytest

from ..history import AgentHistory
from ..schemas import ClientTelemetry, Decision


def test_history_window_length_maintained():
    """History buffer maintains max length and drops oldest rounds FIFO."""
    hist = AgentHistory(history_length=3)
    assert len(hist) == 0

    for r in range(1, 6):
        tel = [
            ClientTelemetry(client_id=0, update_norm=1.0 + r, cosine_similarity=0.9),
            ClientTelemetry(client_id=1, update_norm=2.0 + r, cosine_similarity=0.8),
        ]
        dec = Decision(method="FedAvg", client_ids=[0, 1])
        hist.add_round(round_num=r, client_telemetry=tel, decision_made=dec)

    # Max length is 3, so only rounds 3, 4, 5 are kept
    assert len(hist) == 3
    recent = hist.get_recent()
    assert [r.round_num for r in recent] == [3, 4, 5]


def test_dynamic_client_support():
    """History supports varying client counts without hardcoded 5-client assumptions."""
    hist = AgentHistory(history_length=3)

    # Round 1 has 10 clients
    tel_10 = [ClientTelemetry(client_id=i, update_norm=float(i)) for i in range(10)]
    hist.add_round(round_num=1, client_telemetry=tel_10)

    # Round 2 has 12 clients
    tel_12 = [ClientTelemetry(client_id=i, update_norm=float(i * 2)) for i in range(12)]
    hist.add_round(round_num=2, client_telemetry=tel_12)

    all_clients = hist.get_all_client_ids()
    assert len(all_clients) == 12
    assert all_clients == list(range(12))


def test_transient_vs_persistent_anomaly_detection():
    """History correctly differentiates transient (1-round) vs persistent (2+ rounds) anomalies."""
    hist = AgentHistory(history_length=3)

    # Round 1: Client 7 is clean
    hist.add_round(
        round_num=1,
        client_telemetry=[
            ClientTelemetry(client_id=0, is_flagged_by_detector=False),
            ClientTelemetry(client_id=7, is_flagged_by_detector=False, cosine_similarity=0.85),
        ],
    )

    # Round 2: Client 7 is clean
    hist.add_round(
        round_num=2,
        client_telemetry=[
            ClientTelemetry(client_id=0, is_flagged_by_detector=False),
            ClientTelemetry(client_id=7, is_flagged_by_detector=False, cosine_similarity=0.82),
        ],
    )

    # Round 3: Client 7 is flagged for the first time (TRANSIENT)
    hist.add_round(
        round_num=3,
        client_telemetry=[
            ClientTelemetry(client_id=0, is_flagged_by_detector=False),
            ClientTelemetry(client_id=7, is_flagged_by_detector=True, cosine_similarity=-0.3),
        ],
    )

    # Transient: Flagged in only 1 of 3 rounds
    assert hist.has_persistent_anomaly(client_id=7, min_rounds=2) is False

    # Round 4: Client 7 is flagged again (PERSISTENT across rounds 3 and 4)
    hist.add_round(
        round_num=4,
        client_telemetry=[
            ClientTelemetry(client_id=0, is_flagged_by_detector=False),
            ClientTelemetry(client_id=7, is_flagged_by_detector=True, cosine_similarity=-0.4),
        ],
    )

    # Persistent: Flagged in 2 of last 3 rounds (rounds 3 and 4)
    assert hist.has_persistent_anomaly(client_id=7, min_rounds=2) is True


def test_client_trajectory_extraction():
    """Extracts chronological metric sequence for a single client."""
    hist = AgentHistory(history_length=3)
    for r in range(1, 4):
        hist.add_round(
            round_num=r,
            client_telemetry=[
                ClientTelemetry(
                    client_id=2,
                    update_norm=r * 0.5,
                    cosine_similarity=1.0 - (r * 0.1),
                    is_flagged_by_detector=(r == 3),
                )
            ],
        )

    traj = hist.get_client_trajectory(client_id=2)
    assert len(traj) == 3
    assert [t["round"] for t in traj] == [1, 2, 3]
    assert [t["norm"] for t in traj] == [0.5, 1.0, 1.5]
    assert traj[2]["flagged"] is True
