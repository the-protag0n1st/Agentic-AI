import copy
import torch


def fedavg(weights_list, sample_counts=None, **kwargs):
    """Computes coordinate-wise average of client updates.
    
    If sample_counts is provided, computes weighted average proportional to client samples.
    Otherwise computes unweighted mean (backward compatible with Experiments 1 & 2).
    """
    avg = copy.deepcopy(weights_list[0])
    if sample_counts is not None and len(sample_counts) == len(weights_list) and sum(sample_counts) > 0:
        total_samples = float(sum(sample_counts))
        weights = [c / total_samples for c in sample_counts]
        for k in avg:
            stacked = torch.stack([w[k].float() * weights[i] for i, w in enumerate(weights_list)])
            avg[k] = stacked.sum(dim=0).to(weights_list[0][k].dtype)
    else:
        for k in avg:
            stacked = torch.stack([w[k].float() for w in weights_list])
            avg[k] = stacked.mean(dim=0).to(weights_list[0][k].dtype)
    return avg


def fedsgd(weights_list, global_weights, server_lr=1.0, **kwargs):
    new_w = copy.deepcopy(global_weights)
    for k in global_weights:
        deltas = torch.stack([(w[k].float() - global_weights[k].float()) for w in weights_list])
        mean_delta = deltas.mean(dim=0)
        new_w[k] = (global_weights[k].float() + server_lr * mean_delta).to(global_weights[k].dtype)
    return new_w


def median_aggr(weights_list, **kwargs):
    agg = copy.deepcopy(weights_list[0])
    for k in agg:
        stacked = torch.stack([w[k].float() for w in weights_list])
        agg[k] = stacked.median(dim=0).values.to(weights_list[0][k].dtype)
    return agg


def trimmed_mean(weights_list, trim_ratio=0.2, **kwargs):
    n = len(weights_list)
    k = max(int(n * trim_ratio), 0)
    agg = copy.deepcopy(weights_list[0])
    for key in agg:
        stacked = torch.stack([w[key].float() for w in weights_list], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if 2 * k < n and k > 0:
            trimmed = sorted_vals[k: n - k]
        else:
            trimmed = sorted_vals
        agg[key] = trimmed.mean(dim=0).to(weights_list[0][key].dtype)
    return agg


def _flatten(w):
    return torch.cat([w[k].float().reshape(-1) for k in w])


def krum(weights_list, num_byzantine=1, **kwargs):
    """Krum aggregation excluding self-distance from nearest neighbor calculation.
    
    For n clients and byzantine budget m, calculates distance to nearest n - m - 2
    OTHER clients (excluding client i's distance of 0 to itself).
    """
    n = len(weights_list)
    flat = [_flatten(w) for w in weights_list]
    dists = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = torch.norm(flat[i] - flat[j]) ** 2
    num_neighbors = max(n - num_byzantine - 2, 1)
    scores = []
    for i in range(n):
        # Exclude self-distance dists[i, i] == 0.0
        other_dists = [dists[i, j] for j in range(n) if i != j]
        sorted_d, _ = torch.sort(torch.tensor(other_dists))
        scores.append(sorted_d[:num_neighbors].sum().item())
    best_idx = int(torch.tensor(scores).argmin())
    return copy.deepcopy(weights_list[best_idx])


# NOTE ON FEDPROX:
# FedProx is a client-side optimization algorithm that introduces a proximal
# term mu/2 * ||w - w_t||^2 during local SGD. At the server level, FedProx uses
# standard FedAvg aggregation. FedProx is NOT an aggregation rule and is not
# part of Experiment 3's 7 canonical methods (FixedFedAvg, FixedMedian,
# FixedTrimmedMean, FixedKrum, RuleBased, SingleShotLLM, ReflectiveAgent).
# Therefore, FedProx is deliberately not included as an aggregation method.

ALL_METHODS_NO_GLOBAL = {
    "FedAvg": lambda weights_list, **kw: fedavg(weights_list, **kw),
    "Median": lambda weights_list, **kw: median_aggr(weights_list, **kw),
    "Trimmed Mean": lambda weights_list, **kw: trimmed_mean(weights_list, **kw),
    "TrimmedMean": lambda weights_list, **kw: trimmed_mean(weights_list, **kw),
    "Krum": lambda weights_list, **kw: krum(weights_list, **kw),
}

AVAILABLE_METHODS = ["FedAvg", "FedSGD", "Median", "Trimmed Mean", "TrimmedMean", "Krum"]


def aggregate(method, weights_list, global_weights=None, **kwargs):
    if method == "FedSGD":
        if global_weights is None:
            raise ValueError("FedSGD requires global_weights")
        return fedsgd(weights_list, global_weights, **kwargs)
    if method not in ALL_METHODS_NO_GLOBAL:
        raise ValueError(f"Unknown aggregation method: {method}")
    return ALL_METHODS_NO_GLOBAL[method](weights_list, **kwargs)
