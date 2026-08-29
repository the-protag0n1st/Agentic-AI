from itertools import combinations


def generate_client_combinations(clean_ids, local_weights, min_clients=2, max_combos=10):
    if len(clean_ids) < min_clients:
        return [{"client_ids": list(clean_ids),
                 "weights": [local_weights[i] for i in clean_ids]}]

    combos = []
    for r in range(len(clean_ids), min_clients - 1, -1):
        for subset in combinations(clean_ids, r):
            combos.append({
                "client_ids": list(subset),
                "weights": [local_weights[i] for i in subset],
            })
            if len(combos) >= max_combos:
                return combos
    return combos