"""
Byzantine attack implementations for Federated Learning experiments.

Attacks modify only local client training data.
Global test data is NEVER modified.
"""

from torch.utils.data import Dataset


class LabelFlipDataset(Dataset):
    """Wraps a dataset (typically a Subset) and flips binary labels 0↔1.
    
    Only affects __getitem__ — the underlying dataset is NOT mutated.
    This ensures honest clients sharing the same base dataset are unaffected.
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # Binary flip: 0 → 1, 1 → 0
        return img, 1 - label


def apply_attack(clients, malicious_ids, attack_type="label_flip"):
    """Return a new client list with the attack applied to malicious clients.
    
    The original client Subset objects are NOT mutated.
    Honest clients are returned unchanged.
    
    Args:
        clients: list of Subset objects from make_clients()
        malicious_ids: list of client indices to attack
        attack_type: "label_flip" or None
    
    Returns:
        new_clients: list with attacked clients wrapped
    """
    if attack_type is None or not malicious_ids:
        return clients

    malicious_set = set(malicious_ids)
    new_clients = []

    for i, client in enumerate(clients):
        if i in malicious_set:
            if attack_type == "label_flip":
                new_clients.append(LabelFlipDataset(client))
            else:
                raise ValueError(f"Unknown attack_type: {attack_type}")
        else:
            new_clients.append(client)

    return new_clients


def select_malicious_clients(num_clients, num_malicious, seed):
    """Deterministically select malicious client IDs based on seed.
    
    Uses a simple deterministic formula so the same seed always
    produces the same malicious client set.
    """
    import numpy as np
    rng = np.random.default_rng(seed + 9999)  # Offset to avoid collision with data partitioning RNG
    all_ids = list(range(num_clients))
    rng.shuffle(all_ids)
    return sorted(all_ids[:num_malicious])
