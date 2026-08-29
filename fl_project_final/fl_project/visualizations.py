import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as auc_fn

from database import fetch_dataframe

METHOD_COLORS = {
    "FedAvg": "#4C72B0", "FedSGD": "#937860", "Median": "#DD8452",
    "Trimmed Mean": "#55A868", "Krum": "#C44E52",
}


def plot_data_distribution(counts, title="Client class distribution"):
    fig, ax = plt.subplots(figsize=(8, 4))
    num_clients, num_classes = counts.shape
    bottom = np.zeros(num_clients)
    labels = ["Normal", "Anomaly"] if num_classes == 2 else [str(c) for c in range(num_classes)]
    colors = ["#4C72B0", "#C44E52"]
    for c in range(num_classes):
        ax.bar(range(num_clients), counts[:, c], bottom=bottom,
               label=labels[c], color=colors[c % len(colors)])
        bottom += counts[:, c]
    ax.set_xlabel("Client"); ax.set_ylabel("# samples")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    return fig


def plot_accuracy_loss(logs, alpha):
    rounds = range(1, len(logs["acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = [METHOD_COLORS.get(m, "gray") for m in logs["method"]]
    axes[0].bar(rounds, logs["acc"], color=colors)
    axes[0].set_title(f"Accuracy per round (alpha={alpha})")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Accuracy (%)")
    axes[1].plot(rounds, logs["loss"], marker="o", color="#C44E52")
    axes[1].set_title(f"Loss per round (alpha={alpha})")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Loss")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in METHOD_COLORS.values()]
    fig.legend(handles, METHOD_COLORS.keys(), title="Method", loc="upper center",
               ncol=len(METHOD_COLORS), bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout()
    return fig


def plot_anomaly_heatmap(run_id, alpha, db_path="fl_metrics.db"):
    df = fetch_dataframe(
        "SELECT round, client_id, norm, flagged FROM client_metrics "
        "WHERE run_id=? AND alpha=? ORDER BY round, client_id",
        params=(run_id, alpha), db_path=db_path)
    if df.empty:
        return None

    pivot_norm = df.pivot(index="client_id", columns="round", values="norm")
    pivot_flag = df.pivot(index="client_id", columns="round", values="flagged")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axes[0].imshow(pivot_norm.values, aspect="auto", cmap="YlOrRd")
    axes[0].set_title(f"L2 norm heatmap (alpha={alpha})")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Client")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(pivot_flag.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_title("Flagged clients (red = anomalous)")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Client")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    return fig


def plot_method_selection_history(run_id, alpha, db_path="fl_metrics.db"):
    df = fetch_dataframe(
        "SELECT round, method, accuracy FROM round_metrics "
        "WHERE run_id=? AND alpha=? ORDER BY round",
        params=(run_id, alpha), db_path=db_path)
    if df.empty:
        return None

    colors = [METHOD_COLORS.get(m, "gray") for m in df["method"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["round"], df["accuracy"], color=colors, alpha=0.85)
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Agent-selected method per round (alpha={alpha})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in METHOD_COLORS.values()]
    ax.legend(handles, METHOD_COLORS.keys(), title="Method")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_roc(labels, probs, title="ROC curve"):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc_fn(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#4C72B0")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    return fig


def plot_alpha_comparison(results):
    fig, ax = plt.subplots(figsize=(7, 4))
    alphas = list(results.keys())
    final_accs = [results[a]["acc"][-1] for a in alphas]
    ax.plot([str(a) for a in alphas], final_accs, marker="o", color="#55A868")
    ax.set_xlabel("Alpha (lower = more non-IID)")
    ax.set_ylabel("Final round accuracy (%)")
    ax.set_title("Final accuracy vs. data heterogeneity")
    plt.tight_layout()
    return fig


def plot_method_usage(run_id=None, alpha=None, db_path="fl_metrics.db"):
    query = "SELECT selected_method FROM round_metrics"
    params = ()
    if run_id is not None and alpha is not None:
        query += " WHERE run_id=? AND alpha=?"
        params = (run_id, alpha)

    df = fetch_dataframe(query, params=params, db_path=db_path)
    if df.empty:
        return None

    counts = df["selected_method"].fillna("Unknown").value_counts()
    colors = [METHOD_COLORS.get(m, "gray") for m in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_xlabel("Aggregation method"); ax.set_ylabel("Rounds selected")
    title = "Aggregation method usage"
    if run_id is not None:
        title += f" (run={run_id}, alpha={alpha})"
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_combination_usage(run_id=None, alpha=None, top_n=15, db_path="fl_metrics.db"):
    query = "SELECT selected_combination FROM round_metrics"
    params = ()
    if run_id is not None and alpha is not None:
        query += " WHERE run_id=? AND alpha=?"
        params = (run_id, alpha)

    df = fetch_dataframe(query, params=params, db_path=db_path)
    if df.empty:
        return None

    combo_labels = []
    for raw in df["selected_combination"].dropna():
        try:
            ids = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        combo_labels.append("-".join(f"C{i}" for i in ids))

    if not combo_labels:
        return None

    counts = Counter(combo_labels).most_common(top_n)
    labels, values = zip(*counts)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_xlabel("Selected client combination"); ax.set_ylabel("Rounds selected")
    title = "Client combination usage"
    if run_id is not None:
        title += f" (run={run_id}, alpha={alpha})"
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_client_anomaly_frequency(run_id=None, alpha=None, db_path="fl_metrics.db"):
    query = "SELECT client_id, flagged FROM client_metrics"
    params = ()
    if run_id is not None and alpha is not None:
        query += " WHERE run_id=? AND alpha=?"
        params = (run_id, alpha)

    df = fetch_dataframe(query, params=params, db_path=db_path)
    if df.empty:
        return None

    freq = df[df["flagged"] == 1].groupby("client_id").size()
    all_clients = sorted(df["client_id"].unique())
    counts = [int(freq.get(c, 0)) for c in all_clients]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([f"C{c}" for c in all_clients], counts, color="#C44E52")
    ax.set_xlabel("Client"); ax.set_ylabel("Times flagged anomalous")
    title = "Client anomaly frequency"
    if run_id is not None:
        title += f" (run={run_id}, alpha={alpha})"
    ax.set_title(title)
    plt.tight_layout()
    return fig