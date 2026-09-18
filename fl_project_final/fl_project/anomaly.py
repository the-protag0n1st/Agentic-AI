import numpy as np
import torch


def compute_update_norms(global_weights, local_weights_list):
    norms = []
    for w in local_weights_list:
        sq_sum = 0.0
        for k in global_weights:
            diff = w[k].float() - global_weights[k].float()
            sq_sum += torch.sum(diff ** 2).item()
        norms.append(float(np.sqrt(sq_sum)))
    return norms


def zscore(values):
    arr = np.array(values, dtype=float)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = arr.std() if arr.std() > 1e-8 else 1.0
    return (arr - med) / scale


def compute_cosine_similarities(global_weights, local_weights_list):
    def flatten_update(w):
        parts = [(w[k].float() - global_weights[k].float()).reshape(-1) for k in global_weights]
        return torch.cat(parts)

    updates = [flatten_update(w) for w in local_weights_list]
    # Use coordinate-wise median reference to ensure robust cosine similarity calculation
    # even when attacker count (f >= 3) would otherwise invert the unweighted mean update
    ref_u = torch.stack(updates).median(dim=0).values
    sims = []
    for u in updates:
        num = torch.dot(u, ref_u)
        den = (torch.norm(u) * torch.norm(ref_u)) + 1e-8
        sims.append(float((num / den).item()))
    return sims


def compute_weight_variations(local_weights_list, layer_key=None):
    variations = []
    for w in local_weights_list:
        if layer_key is None:
            weight_keys = [k for k in w if k.endswith("weight")]
            key = weight_keys[-2] if len(weight_keys) >= 2 else weight_keys[-1]
        else:
            key = layer_key
        vals = w[key].float().cpu().numpy().flatten()
        variations.append(float(np.std(vals)))
    return variations


def detect_anomalies(norms, cosine_sims, weight_vars,
                      norm_z_threshold=2.0,
                      cosine_threshold=-0.2,
                      var_z_threshold=2.0,
                      min_votes=1):
    """Fuse the three per-client signals (norm z-score, cosine similarity,
    weight-variation z-score) into a flagged/clean decision.

    This is configurable multi-signal voting, not majority voting: a client
    is flagged once at least `min_votes` of the 3 signals fire. With the
    default min_votes=1, any single signal is enough to flag a client;
    raising min_votes (e.g. to 2) requires agreement across signals instead.
    """
    z_norms = zscore(norms)
    z_vars = zscore(weight_vars)
    anomalous_ids, clean_ids, vote_detail = [], [], []

    for i in range(len(norms)):
        votes = {
            "norm_flag": bool(z_norms[i] > norm_z_threshold),
            "cosine_flag": bool(cosine_sims[i] < cosine_threshold),
            "var_flag": bool(z_vars[i] > var_z_threshold),
        }
        n_votes = sum(votes.values())
        flagged = n_votes >= min_votes
        votes["n_votes"] = n_votes
        votes["flagged"] = flagged
        vote_detail.append(votes)
        (anomalous_ids if flagged else clean_ids).append(i)

    return anomalous_ids, clean_ids, vote_detail