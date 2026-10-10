"""
Comprehensive Experiment 3 Analysis & Research Paper Figures Generator.

Generates all findings, tables, statistical tests, and publication-quality plots
for the Federated Learning Agentic AI benchmark paper.

Completed data: 8 methods × 3 seeds × 20 rounds = 480 round records
Methods: FixedFedAvg, FixedMedian, FixedTrimmedMean, FixedKrum, RuleBased, SingleShotLLM, ReflectiveAgent, AgenticAI
"""

import csv
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

warnings.filterwarnings("ignore")

# Try importing plotting/stats libraries
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available; skipping plots.")

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not available; skipping statistical tests.")

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path(r"c:\Users\akhil\Downloads\fl_project_final (2)\fl_project_final\fl_project")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "paper_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ROUNDS_CSV = RESULTS_DIR / "experiment3_rounds.csv"
RUNS_CSV = RESULTS_DIR / "experiment3_runs.csv"
AGENTIC_CSV = RESULTS_DIR / "agentic_log.csv"
LLM_LOG_CSV = RESULTS_DIR / "llm_call_log.csv"

METHOD_DISPLAY = {
    "FixedFedAvg": "FedAvg",
    "FixedMedian": "Median",
    "FixedTrimmedMean": "TrimmedMean",
    "FixedKrum": "Krum",
    "RuleBased": "Rule-Based",
    "SingleShotLLM": "SingleShot-LLM",
    "ReflectiveAgent": "Reflective-Agent",
    "AgenticAI": "Agentic-AI",
}

METHOD_ORDER = list(METHOD_DISPLAY.keys())

METHOD_COLORS = {
    "FixedFedAvg": "#1f77b4",
    "FixedMedian": "#ff7f0e",
    "FixedTrimmedMean": "#2ca02c",
    "FixedKrum": "#d62728",
    "RuleBased": "#9467bd",
    "SingleShotLLM": "#8c564b",
    "ReflectiveAgent": "#e377c2",
    "AgenticAI": "#17becf",
}

REGIME_LABELS = {
    (1, 5): "Near-IID\n(a=100)",
    (6, 10): "Moderate\n(a=1.0)",
    (11, 15): "Severe\n(a=0.2)",
    (16, 20): "Byzantine\n(a=0.2+Attack)",
}

REGIME_RANGES = [(1, 5), (6, 10), (11, 15), (16, 20)]

# =============================================================================
# DATA LOADING
# =============================================================================
def load_rounds_data():
    """Load and organize all round-level data."""
    data = []
    with open(ROUNDS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Parse method and seed from run_id
    # Format: exp3_{method}_s{seed} or exp3_{method}_s{seed}_{timestamp}
    for row in data:
        rid = row["run_id"]
        parts = rid.split("_s")
        row["method"] = parts[0].replace("exp3_", "")
        seed_part = parts[1].split("_")[0] if len(parts) > 1 else "?"
        row["seed"] = int(seed_part)
        row["round_num"] = int(row["round"])
        row["test_acc"] = float(row["test_accuracy"]) if row["test_accuracy"] else 0.0
        row["test_f1_val"] = float(row["test_f1"]) if row["test_f1"] else 0.0
        row["oracle_acc"] = float(row["oracle_accuracy"]) if row["oracle_accuracy"] else 0.0
        row["regret_val"] = float(row["regret"]) if row["regret"] else 0.0
        row["cumul_regret"] = float(row["cumulative_regret"]) if row["cumulative_regret"] else 0.0
        row["alpha_val"] = float(row["alpha"]) if row["alpha"] else 0.0
        row["attack"] = row["attack_active"] == "1"
        row["round_time"] = float(row["round_time_s"]) if row["round_time_s"] else 0.0

    return data


def load_agentic_log():
    """Load agentic decision log."""
    data = []
    with open(AGENTIC_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def load_llm_log():
    """Load LLM call log."""
    data = []
    with open(LLM_LOG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def group_by_method_seed(data):
    """Group rounds by (method, seed)."""
    groups = defaultdict(list)
    for row in data:
        groups[(row["method"], row["seed"])].append(row)
    # Sort each group by round
    for key in groups:
        groups[key].sort(key=lambda x: x["round_num"])
    return groups


def get_latest_run_per_method_seed(data):
    """For method/seed combos with multiple run_ids, keep only the latest complete run."""
    # Group by run_id
    by_run = defaultdict(list)
    for row in data:
        by_run[row["run_id"]].append(row)

    # For each (method, seed), find the run_id that covers all 20 rounds
    method_seed_runs = defaultdict(list)
    for rid, rows in by_run.items():
        m = rows[0]["method"]
        s = rows[0]["seed"]
        method_seed_runs[(m, s)].append((rid, rows))

    filtered = []
    for (m, s), runs in method_seed_runs.items():
        # Find the run that has the most rounds (up to 20)
        best_run = None
        best_count = 0
        for rid, rows in runs:
            if len(rows) >= best_count:
                best_count = len(rows)
                best_run = rows

        # Some methods have split runs (e.g., SingleShotLLM_s2 with rounds 1-10 in one run and 11-20 in another)
        if best_count < 20:
            # Try merging all runs
            all_rounds = {}
            for rid, rows in runs:
                for row in rows:
                    rn = row["round_num"]
                    all_rounds[rn] = row
            if len(all_rounds) >= 20:
                best_run = [all_rounds[r] for r in sorted(all_rounds.keys())]
                best_count = len(best_run)

        if best_run:
            filtered.extend(best_run)

    return filtered


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================
def compute_method_stats(groups):
    """Compute per-method statistics across seeds."""
    results = {}
    for method in METHOD_ORDER:
        accs = []
        f1s = []
        regrets = []
        final_accs = []
        final_f1s = []
        final_cumul_regrets = []
        round_times = []

        # Per-regime accuracy
        regime_accs = {r: [] for r in REGIME_RANGES}

        for seed in [1, 2, 3]:
            if (method, seed) not in groups:
                continue
            rounds = groups[(method, seed)]
            for row in rounds:
                accs.append(row["test_acc"])
                f1s.append(row["test_f1_val"])
                regrets.append(row["regret_val"])
                round_times.append(row["round_time"])

                rn = row["round_num"]
                for (rs, re) in REGIME_RANGES:
                    if rs <= rn <= re:
                        regime_accs[(rs, re)].append(row["test_acc"])
                        break

            if rounds:
                last = rounds[-1]
                final_accs.append(last["test_acc"])
                final_f1s.append(last["test_f1_val"])
                final_cumul_regrets.append(last["cumul_regret"])

        results[method] = {
            "all_accs": np.array(accs),
            "all_f1s": np.array(f1s),
            "all_regrets": np.array(regrets),
            "final_accs": np.array(final_accs),
            "final_f1s": np.array(final_f1s),
            "final_cumul_regrets": np.array(final_cumul_regrets),
            "round_times": np.array(round_times),
            "regime_accs": {k: np.array(v) for k, v in regime_accs.items()},
            "mean_acc": np.mean(accs) if accs else 0,
            "std_acc": np.std(accs) if accs else 0,
            "mean_f1": np.mean(f1s) if f1s else 0,
            "std_f1": np.std(f1s) if f1s else 0,
            "mean_regret": np.mean(regrets) if regrets else 0,
            "std_regret": np.std(regrets) if regrets else 0,
            "mean_final_acc": np.mean(final_accs) if final_accs else 0,
            "std_final_acc": np.std(final_accs) if final_accs else 0,
            "mean_final_cumul_regret": np.mean(final_cumul_regrets) if final_cumul_regrets else 0,
            "std_final_cumul_regret": np.std(final_cumul_regrets) if final_cumul_regrets else 0,
            "mean_round_time": np.mean(round_times) if round_times else 0,
        }
    return results


def cohens_d(x, y):
    diff = x - y
    if len(diff) == 0 or np.std(diff, ddof=1) == 0:
        return 0.0
    return np.mean(diff) / np.std(diff, ddof=1)


def run_statistical_tests(groups):
    """Wilcoxon signed-rank tests + Cohen's d: Agentic methods vs baselines."""
    if not HAS_SCIPY:
        return {}

    agentic_methods = ["AgenticAI", "ReflectiveAgent", "SingleShotLLM"]
    baseline_methods = ["FixedFedAvg", "FixedMedian", "FixedTrimmedMean", "FixedKrum", "RuleBased"]

    results = {}
    for am in agentic_methods:
        for bm in baseline_methods:
            # Get paired final accuracies
            am_accs = []
            bm_accs = []
            for seed in [1, 2, 3]:
                if (am, seed) in groups and (bm, seed) in groups:
                    am_rounds = groups[(am, seed)]
                    bm_rounds = groups[(bm, seed)]
                    if am_rounds and bm_rounds:
                        am_accs.append(am_rounds[-1]["test_acc"])
                        bm_accs.append(bm_rounds[-1]["test_acc"])

            am_accs = np.array(am_accs)
            bm_accs = np.array(bm_accs)
            key = f"{am}_vs_{bm}"

            if len(am_accs) >= 3:
                try:
                    stat, p = scipy_stats.wilcoxon(am_accs, bm_accs)
                    d = cohens_d(am_accs, bm_accs)
                    results[key] = {
                        "n": len(am_accs),
                        "am_mean": np.mean(am_accs),
                        "bm_mean": np.mean(bm_accs),
                        "diff": np.mean(am_accs) - np.mean(bm_accs),
                        "wilcoxon_stat": stat,
                        "p_value": p,
                        "cohens_d": d,
                        "significant": p < 0.05,
                    }
                except Exception as e:
                    results[key] = {"error": str(e), "n": len(am_accs)}
            else:
                results[key] = {"error": "Insufficient paired seeds", "n": len(am_accs)}

    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def set_paper_style():
    """Set matplotlib defaults for IEEE/ACM publication quality."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_1_convergence_curves(groups):
    """Fig 1: Test accuracy convergence across all 20 rounds, per method (mean ± std across seeds)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for method in METHOD_ORDER:
        round_accs = defaultdict(list)
        for seed in [1, 2, 3]:
            if (method, seed) not in groups:
                continue
            for row in groups[(method, seed)]:
                round_accs[row["round_num"]].append(row["test_acc"])

        rounds_sorted = sorted(round_accs.keys())
        means = [np.mean(round_accs[r]) for r in rounds_sorted]
        stds = [np.std(round_accs[r]) for r in rounds_sorted]

        color = METHOD_COLORS[method]
        label = METHOD_DISPLAY[method]
        ax.plot(rounds_sorted, means, marker="o", markersize=4, color=color, label=label, linewidth=1.8)
        ax.fill_between(rounds_sorted,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)

    # Regime shading
    regime_colors = ["#e6f3e6", "#fff3cd", "#fde8e8", "#f8d7da"]
    for i, ((rs, re), label) in enumerate(REGIME_LABELS.items()):
        ax.axvspan(rs - 0.5, re + 0.5, alpha=0.2, color=regime_colors[i], zorder=0)
        ax.text((rs + re) / 2, ax.get_ylim()[0] + 1 if ax.get_ylim()[0] > 0 else 60,
                label, ha="center", fontsize=8, style="italic")

    ax.axvline(x=15.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Byzantine attack onset")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Figure 1: Convergence Under Dynamic Heterogeneity & Byzantine Attacks")
    ax.legend(loc="lower right", ncol=2, framealpha=0.9)
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks(range(1, 21))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_convergence_curves.png")
    plt.close(fig)
    print("  Saved fig1_convergence_curves.png")


def plot_2_regime_accuracy_bars(method_stats):
    """Fig 2: Per-regime accuracy comparison (grouped bar chart)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

    for idx, (regime_range, regime_label) in enumerate(REGIME_LABELS.items()):
        ax = axes[idx]
        methods = []
        means = []
        stds = []
        colors = []

        for method in METHOD_ORDER:
            st = method_stats[method]
            accs = st["regime_accs"].get(regime_range, np.array([]))
            if len(accs) > 0:
                methods.append(METHOD_DISPLAY[method])
                means.append(np.mean(accs))
                stds.append(np.std(accs))
                colors.append(METHOD_COLORS[method])

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_title(regime_label.replace("\n", " "), fontsize=11)
        if idx == 0:
            ax.set_ylabel("Test Accuracy (%)")

    fig.suptitle("Figure 2: Per-Regime Test Accuracy Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_regime_accuracy_bars.png")
    plt.close(fig)
    print("  Saved fig2_regime_accuracy_bars.png")


def plot_3_cumulative_regret(groups):
    """Fig 3: Cumulative regret trajectories."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for method in METHOD_ORDER:
        round_regrets = defaultdict(list)
        for seed in [1, 2, 3]:
            if (method, seed) not in groups:
                continue
            for row in groups[(method, seed)]:
                round_regrets[row["round_num"]].append(row["cumul_regret"])

        rounds_sorted = sorted(round_regrets.keys())
        means = [np.mean(round_regrets[r]) for r in rounds_sorted]
        stds = [np.std(round_regrets[r]) for r in rounds_sorted]

        color = METHOD_COLORS[method]
        label = METHOD_DISPLAY[method]
        ax.plot(rounds_sorted, means, marker="s", markersize=3, color=color, label=label, linewidth=1.8)
        ax.fill_between(rounds_sorted,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color=color)

    ax.axvline(x=15.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Cumulative Regret (%)")
    ax.set_title("Figure 3: Cumulative Oracle Regret Over 20 Rounds")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks(range(1, 21))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_cumulative_regret.png")
    plt.close(fig)
    print("  Saved fig3_cumulative_regret.png")


def plot_4_final_accuracy_boxplot(groups):
    """Fig 4: Final round accuracy boxplot across seeds."""
    fig, ax = plt.subplots(figsize=(10, 5))

    data_list = []
    labels = []
    colors = []
    for method in METHOD_ORDER:
        final_accs = []
        for seed in [1, 2, 3]:
            if (method, seed) in groups:
                rounds = groups[(method, seed)]
                if rounds:
                    final_accs.append(rounds[-1]["test_acc"])
        data_list.append(final_accs)
        labels.append(METHOD_DISPLAY[method])
        colors.append(METHOD_COLORS[method])

    bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, d in enumerate(data_list):
        x = np.random.normal(i + 1, 0.04, len(d))
        ax.scatter(x, d, color="black", s=30, zorder=5, alpha=0.7)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Figure 4: Final Round (R=20) Test Accuracy Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_final_accuracy_boxplot.png")
    plt.close(fig)
    print("  Saved fig4_final_accuracy_boxplot.png")


def plot_5_byzantine_resilience(groups):
    """Fig 5: Pre-attack vs post-attack accuracy drop analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    pre_attack_means = []
    post_attack_means = []
    drops = []

    for method in METHOD_ORDER:
        pre_accs = []
        post_accs = []
        for seed in [1, 2, 3]:
            if (method, seed) not in groups:
                continue
            for row in groups[(method, seed)]:
                rn = row["round_num"]
                if 11 <= rn <= 15:
                    pre_accs.append(row["test_acc"])
                elif 16 <= rn <= 20:
                    post_accs.append(row["test_acc"])

        if pre_accs and post_accs:
            pre_mean = np.mean(pre_accs)
            post_mean = np.mean(post_accs)
            methods.append(METHOD_DISPLAY[method])
            pre_attack_means.append(pre_mean)
            post_attack_means.append(post_mean)
            drops.append(pre_mean - post_mean)

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width / 2, pre_attack_means, width, label="Pre-Attack (R11-15)", color="#2ca02c", alpha=0.8)
    bars2 = ax.bar(x + width / 2, post_attack_means, width, label="Post-Attack (R16-20)", color="#d62728", alpha=0.8)

    # Add drop annotations
    for i, drop in enumerate(drops):
        y_pos = max(pre_attack_means[i], post_attack_means[i]) + 0.5
        color = "red" if drop > 0 else "green"
        ax.annotate(f"{'−' if drop > 0 else '+'}{abs(drop):.1f}%",
                    xy=(x[i], y_pos), ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Figure 5: Byzantine Resilience — Pre vs Post-Attack Accuracy")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_byzantine_resilience.png")
    plt.close(fig)
    print("  Saved fig5_byzantine_resilience.png")


def plot_6_regret_per_round_heatmap(groups):
    """Fig 6: Instantaneous regret heatmap (method × round)."""
    fig, ax = plt.subplots(figsize=(14, 5))

    matrix = []
    y_labels = []
    for method in METHOD_ORDER:
        row_vals = []
        for rn in range(1, 21):
            vals = []
            for seed in [1, 2, 3]:
                if (method, seed) in groups:
                    for row in groups[(method, seed)]:
                        if row["round_num"] == rn:
                            vals.append(row["regret_val"])
            row_vals.append(np.mean(vals) if vals else 0)
        matrix.append(row_vals)
        y_labels.append(METHOD_DISPLAY[method])

    matrix = np.array(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(20))
    ax.set_xticklabels(range(1, 21))
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Method")
    ax.set_title("Figure 6: Per-Round Oracle Regret Heatmap (Mean Across Seeds)")
    plt.colorbar(im, ax=ax, label="Regret (%)")

    # Add regime boundary lines
    for boundary in [4.5, 9.5, 14.5]:
        ax.axvline(x=boundary, color="white", linewidth=2)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_regret_heatmap.png")
    plt.close(fig)
    print("  Saved fig6_regret_heatmap.png")


def plot_7_f1_comparison(groups):
    """Fig 7: F1-score comparison across methods and regimes."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for method in METHOD_ORDER:
        round_f1s = defaultdict(list)
        for seed in [1, 2, 3]:
            if (method, seed) not in groups:
                continue
            for row in groups[(method, seed)]:
                round_f1s[row["round_num"]].append(row["test_f1_val"])

        rounds_sorted = sorted(round_f1s.keys())
        means = [np.mean(round_f1s[r]) for r in rounds_sorted]

        color = METHOD_COLORS[method]
        label = METHOD_DISPLAY[method]
        ax.plot(rounds_sorted, means, marker="^", markersize=3, color=color, label=label, linewidth=1.5)

    ax.axvline(x=15.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("Figure 7: Macro F1-Score Across Rounds")
    ax.legend(loc="lower right", ncol=2, framealpha=0.9)
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks(range(1, 21))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_f1_comparison.png")
    plt.close(fig)
    print("  Saved fig7_f1_comparison.png")


def plot_8_round_time_comparison(method_stats):
    """Fig 8: Computational cost (mean round time) per method."""
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = []
    times = []
    colors = []

    for method in METHOD_ORDER:
        st = method_stats[method]
        methods.append(METHOD_DISPLAY[method])
        times.append(st["mean_round_time"])
        colors.append(METHOD_COLORS[method])

    bars = ax.bar(methods, times, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.1f}s", ha="center", fontsize=9)

    ax.set_ylabel("Mean Round Time (seconds)")
    ax.set_title("Figure 8: Computational Cost Per Method")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_round_time.png")
    plt.close(fig)
    print("  Saved fig8_round_time.png")


def plot_9_accuracy_vs_regret_scatter(method_stats):
    """Fig 9: Mean accuracy vs cumulative regret scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for method in METHOD_ORDER:
        st = method_stats[method]
        ax.scatter(st["mean_final_cumul_regret"], st["mean_final_acc"],
                   s=150, color=METHOD_COLORS[method], edgecolor="black",
                   linewidth=1, zorder=5, label=METHOD_DISPLAY[method])
        ax.annotate(METHOD_DISPLAY[method],
                    (st["mean_final_cumul_regret"], st["mean_final_acc"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.set_xlabel("Mean Cumulative Regret (%) at Round 20")
    ax.set_ylabel("Mean Final Test Accuracy (%)")
    ax.set_title("Figure 9: Accuracy–Regret Trade-off")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig9_accuracy_vs_regret.png")
    plt.close(fig)
    print("  Saved fig9_accuracy_vs_regret.png")


def plot_10_spider_chart(method_stats, groups):
    """Fig 10: Radar/spider chart comparing methods on 5 normalized dimensions."""
    categories = ["Final Acc", "F1-Score", "Byz Resilience", "Low Regret", "Speed"]
    N = len(categories)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Compute normalized scores for each method
    for method in METHOD_ORDER:
        st = method_stats[method]

        # Accuracy (normalize 70-100 range)
        acc_score = (st["mean_final_acc"] - 70) / 30 * 100 if st["mean_final_acc"] > 0 else 0
        acc_score = min(max(acc_score, 0), 100)

        # F1 (normalize 0.8-1.0 range)
        f1_score = (st["mean_f1"] - 0.8) / 0.2 * 100 if st["mean_f1"] > 0 else 0
        f1_score = min(max(f1_score, 0), 100)

        # Byzantine resilience: compute accuracy drop (R11-15 vs R16-20)
        pre = st["regime_accs"].get((11, 15), np.array([]))
        post = st["regime_accs"].get((16, 20), np.array([]))
        if len(pre) > 0 and len(post) > 0:
            drop = np.mean(pre) - np.mean(post)
            byz_score = max(0, 100 - drop * 10)  # Lower drop = higher score
        else:
            byz_score = 50

        # Low regret (invert: lower regret = higher score)
        regret_score = max(0, 100 - st["mean_final_cumul_regret"] * 2)

        # Speed (invert: faster = higher)
        speed_score = max(0, 100 - st["mean_round_time"] * 0.5)

        values = [acc_score, f1_score, byz_score, regret_score, speed_score]
        values += values[:1]  # Close the polygon

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        ax.plot(angles, values, "o-", linewidth=1.5, color=METHOD_COLORS[method],
                label=METHOD_DISPLAY[method], markersize=4)
        ax.fill(angles, values, alpha=0.08, color=METHOD_COLORS[method])

    ax.set_xticks([n / float(N) * 2 * np.pi for n in range(N)])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title("Figure 10: Multi-Dimensional Method Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig10_spider_chart.png")
    plt.close(fig)
    print("  Saved fig10_spider_chart.png")


# =============================================================================
# TABLE GENERATION
# =============================================================================
def generate_table1_overall(method_stats):
    """Table 1: Overall performance summary."""
    lines = []
    lines.append("=" * 120)
    lines.append("TABLE 1: Overall Performance Summary (Mean ± Std Across 3 Seeds, 20 Rounds)")
    lines.append("=" * 120)
    lines.append(f"{'Method':<20s} {'Acc (All)':<18s} {'F1 (All)':<18s} {'Final Acc':<18s} {'Cumul Regret':<18s} {'Time (s/rnd)':<12s}")
    lines.append("-" * 120)

    for method in METHOD_ORDER:
        st = method_stats[method]
        lines.append(
            f"{METHOD_DISPLAY[method]:<20s} "
            f"{st['mean_acc']:.2f} ± {st['std_acc']:.2f}   "
            f"{st['mean_f1']:.4f} ± {st['std_f1']:.4f}   "
            f"{st['mean_final_acc']:.2f} ± {st['std_final_acc']:.2f}   "
            f"{st['mean_final_cumul_regret']:.2f} ± {st['std_final_cumul_regret']:.2f}   "
            f"{st['mean_round_time']:.1f}"
        )
    lines.append("=" * 120)
    return "\n".join(lines)


def generate_table2_regime(method_stats):
    """Table 2: Per-regime accuracy."""
    lines = []
    lines.append("=" * 120)
    lines.append("TABLE 2: Per-Regime Mean Test Accuracy (%) ± Std")
    lines.append("=" * 120)
    header = f"{'Method':<20s}"
    for regime in REGIME_RANGES:
        label = REGIME_LABELS[regime].replace("\n", " ")
        header += f" {label:<22s}"
    lines.append(header)
    lines.append("-" * 120)

    for method in METHOD_ORDER:
        st = method_stats[method]
        row = f"{METHOD_DISPLAY[method]:<20s}"
        for regime in REGIME_RANGES:
            accs = st["regime_accs"].get(regime, np.array([]))
            if len(accs) > 0:
                row += f" {np.mean(accs):.2f} ± {np.std(accs):.2f}       "
            else:
                row += f" {'N/A':<22s}"
        lines.append(row)
    lines.append("=" * 120)
    return "\n".join(lines)


def generate_table3_statistical(test_results):
    """Table 3: Statistical significance tests."""
    lines = []
    lines.append("=" * 120)
    lines.append("TABLE 3: Statistical Significance (Wilcoxon Signed-Rank Test + Cohen's d)")
    lines.append("=" * 120)
    lines.append(f"{'Comparison':<40s} {'N':<5s} {'Mean Diff':<12s} {'p-value':<12s} {'Cohen d':<12s} {'Significant?':<12s}")
    lines.append("-" * 120)

    for key, val in sorted(test_results.items()):
        if "error" in val:
            lines.append(f"{key:<40s} {val.get('n', 0):<5d} {val['error']}")
        else:
            sig = "Yes *" if val["significant"] else "No"
            lines.append(
                f"{key:<40s} {val['n']:<5d} {val['diff']:+.2f}%      "
                f"{val['p_value']:.4f}      {val['cohens_d']:+.3f}       {sig}"
            )
    lines.append("=" * 120)
    return "\n".join(lines)


def generate_table4_byzantine_drop(groups):
    """Table 4: Byzantine attack impact — accuracy drop per method."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 4: Byzantine Attack Impact (R11-15 Pre-Attack vs R16-20 Post-Attack)")
    lines.append("=" * 100)
    lines.append(f"{'Method':<20s} {'Pre-Attack Acc':<18s} {'Post-Attack Acc':<18s} {'Drop (%)':<12s} {'Resilience':<12s}")
    lines.append("-" * 100)

    for method in METHOD_ORDER:
        pre_accs = []
        post_accs = []
        for seed in [1, 2, 3]:
            if (method, seed) not in groups:
                continue
            for row in groups[(method, seed)]:
                rn = row["round_num"]
                if 11 <= rn <= 15:
                    pre_accs.append(row["test_acc"])
                elif 16 <= rn <= 20:
                    post_accs.append(row["test_acc"])

        if pre_accs and post_accs:
            pre = np.mean(pre_accs)
            post = np.mean(post_accs)
            drop = pre - post
            resilience = "High" if abs(drop) < 2 else ("Medium" if abs(drop) < 5 else "Low")
            lines.append(
                f"{METHOD_DISPLAY[method]:<20s} {pre:.2f}%           {post:.2f}%           "
                f"{drop:+.2f}%       {resilience}"
            )
    lines.append("=" * 100)
    return "\n".join(lines)


def generate_table5_agentic_quality(agentic_data):
    """Table 5: Agentic method quality — decision patterns, repair rates."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 5: Agentic Decision Quality Summary")
    lines.append("=" * 100)

    # Count decisions per method
    methods_counts = defaultdict(int)
    repair_counts = defaultdict(int)
    iteration_sums = defaultdict(float)
    llm_call_sums = defaultdict(int)

    for row in agentic_data:
        rid = row["run_id"]
        # Parse method
        parts = rid.split("_s")
        method = parts[0].replace("exp3_", "")
        methods_counts[method] += 1

        if row.get("repair_status") and row["repair_status"] != "none" and row["repair_status"] != "":
            repair_counts[method] += 1

        try:
            iteration_sums[method] += float(row.get("iterations_used", 1) or 1)
        except (ValueError, TypeError):
            iteration_sums[method] += 1

        try:
            llm_call_sums[method] += int(row.get("llm_calls_used", 0) or 0)
        except (ValueError, TypeError):
            pass

    lines.append(f"{'Method':<20s} {'Total Decisions':<18s} {'Repairs':<12s} {'Repair Rate':<14s} {'Avg Iterations':<16s} {'LLM Calls':<12s}")
    lines.append("-" * 100)

    for method in ["SingleShotLLM", "ReflectiveAgent", "AgenticAI"]:
        total = methods_counts.get(method, 0)
        repairs = repair_counts.get(method, 0)
        repair_rate = repairs / total if total > 0 else 0
        avg_iter = iteration_sums.get(method, 0) / total if total > 0 else 0
        llm_calls = llm_call_sums.get(method, 0)
        lines.append(
            f"{METHOD_DISPLAY.get(method, method):<20s} {total:<18d} {repairs:<12d} "
            f"{repair_rate:.1%}          {avg_iter:.1f}              {llm_calls}"
        )
    lines.append("=" * 100)
    return "\n".join(lines)


def generate_table6_llm_cost(llm_data):
    """Table 6: LLM API cost breakdown."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 6: LLM API Cost Breakdown")
    lines.append("=" * 100)

    stage_stats = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "latency": []})

    for row in llm_data:
        stage = row.get("stage", "unknown")
        stage_stats[stage]["calls"] += 1
        try:
            stage_stats[stage]["input_tokens"] += int(row.get("input_tokens", 0) or 0)
            stage_stats[stage]["output_tokens"] += int(row.get("output_tokens", 0) or 0)
            stage_stats[stage]["total_tokens"] += int(row.get("total_tokens", 0) or 0)
        except (ValueError, TypeError):
            pass
        try:
            lat = float(row.get("latency_ms", 0) or 0)
            if lat > 0:
                stage_stats[stage]["latency"].append(lat)
        except (ValueError, TypeError):
            pass

    lines.append(f"{'Stage':<22s} {'Calls':<8s} {'Input Tokens':<14s} {'Output Tokens':<15s} {'Total Tokens':<14s} {'Avg Latency(ms)':<16s}")
    lines.append("-" * 100)

    total_calls = 0
    total_in = 0
    total_out = 0
    total_tok = 0
    for stage, st in sorted(stage_stats.items()):
        avg_lat = np.mean(st["latency"]) if st["latency"] else 0
        lines.append(
            f"{stage:<22s} {st['calls']:<8d} {st['input_tokens']:<14d} {st['output_tokens']:<15d} "
            f"{st['total_tokens']:<14d} {avg_lat:<16.0f}"
        )
        total_calls += st["calls"]
        total_in += st["input_tokens"]
        total_out += st["output_tokens"]
        total_tok += st["total_tokens"]

    lines.append("-" * 100)
    lines.append(f"{'TOTAL':<22s} {total_calls:<8d} {total_in:<14d} {total_out:<15d} {total_tok:<14d}")
    lines.append("=" * 100)
    return "\n".join(lines)


# =============================================================================
# KEY FINDINGS
# =============================================================================
def generate_key_findings(method_stats, groups, test_results):
    """Generate textual summary of key findings."""
    lines = []
    lines.append("=" * 80)
    lines.append("KEY FINDINGS")
    lines.append("=" * 80)

    # Finding 1: Best overall method
    best_method = max(METHOD_ORDER, key=lambda m: method_stats[m]["mean_final_acc"])
    best_acc = method_stats[best_method]["mean_final_acc"]
    lines.append(f"\n1. BEST OVERALL METHOD: {METHOD_DISPLAY[best_method]}")
    lines.append(f"   Final test accuracy: {best_acc:.2f}% (mean across 3 seeds)")
    lines.append(f"   Cumulative regret: {method_stats[best_method]['mean_final_cumul_regret']:.2f}%")

    # Finding 2: Byzantine resilience ranking
    lines.append(f"\n2. BYZANTINE RESILIENCE RANKING:")
    drops = []
    for method in METHOD_ORDER:
        pre = method_stats[method]["regime_accs"].get((11, 15), np.array([]))
        post = method_stats[method]["regime_accs"].get((16, 20), np.array([]))
        if len(pre) > 0 and len(post) > 0:
            drop = np.mean(pre) - np.mean(post)
            drops.append((method, drop))
    drops.sort(key=lambda x: x[1])
    for rank, (m, d) in enumerate(drops, 1):
        lines.append(f"   {rank}. {METHOD_DISPLAY[m]:<20s} drop = {d:+.2f}%")

    # Finding 3: Regret analysis
    lines.append(f"\n3. ORACLE REGRET ANALYSIS:")
    regret_ranking = sorted(METHOD_ORDER, key=lambda m: method_stats[m]["mean_final_cumul_regret"])
    for rank, m in enumerate(regret_ranking, 1):
        lines.append(f"   {rank}. {METHOD_DISPLAY[m]:<20s} cumulative regret = {method_stats[m]['mean_final_cumul_regret']:.2f}%")

    # Finding 4: Agentic vs baselines
    lines.append(f"\n4. AGENTIC AI vs BASELINES:")
    agentic_methods = ["AgenticAI", "ReflectiveAgent", "SingleShotLLM"]
    baseline_methods = ["FixedFedAvg", "FixedMedian", "FixedTrimmedMean", "FixedKrum"]
    for am in agentic_methods:
        am_acc = method_stats[am]["mean_final_acc"]
        improvements = []
        for bm in baseline_methods:
            bm_acc = method_stats[bm]["mean_final_acc"]
            improvements.append(am_acc - bm_acc)
        lines.append(f"   {METHOD_DISPLAY[am]:<20s} vs baselines: avg improvement = {np.mean(improvements):+.2f}%")

    # Finding 5: Computational cost
    lines.append(f"\n5. COMPUTATIONAL COST:")
    for method in METHOD_ORDER:
        lines.append(f"   {METHOD_DISPLAY[method]:<20s} avg round time = {method_stats[method]['mean_round_time']:.1f}s")

    # Finding 6: Heterogeneity impact
    lines.append(f"\n6. DATA HETEROGENEITY IMPACT:")
    for method in METHOD_ORDER:
        iid_acc = method_stats[method]["regime_accs"].get((1, 5), np.array([]))
        severe_acc = method_stats[method]["regime_accs"].get((11, 15), np.array([]))
        if len(iid_acc) > 0 and len(severe_acc) > 0:
            degradation = np.mean(iid_acc) - np.mean(severe_acc)
            lines.append(f"   {METHOD_DISPLAY[method]:<20s} IID → Severe non-IID degradation = {degradation:+.2f}%")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("EXPERIMENT 3: COMPREHENSIVE ANALYSIS & PAPER OUTPUT GENERATOR")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    raw_data = load_rounds_data()
    filtered_data = get_latest_run_per_method_seed(raw_data)
    groups = group_by_method_seed(filtered_data)

    print(f"  Loaded {len(filtered_data)} round records from {len(groups)} (method, seed) combos")
    for (m, s), rounds in sorted(groups.items()):
        print(f"    {m:<20s} seed={s} rounds={len(rounds)}")

    agentic_data = load_agentic_log()
    llm_data = load_llm_log()
    print(f"  Agentic log: {len(agentic_data)} entries")
    print(f"  LLM call log: {len(llm_data)} entries")

    # Compute statistics
    print("\n[2/6] Computing statistics...")
    method_stats = compute_method_stats(groups)

    # Statistical tests
    print("\n[3/6] Running statistical tests...")
    test_results = run_statistical_tests(groups)

    # Generate tables
    print("\n[4/6] Generating tables...")
    tables = []
    t1 = generate_table1_overall(method_stats)
    tables.append(t1)
    print(t1)

    t2 = generate_table2_regime(method_stats)
    tables.append(t2)
    print(t2)

    t3 = generate_table3_statistical(test_results)
    tables.append(t3)
    print(t3)

    t4 = generate_table4_byzantine_drop(groups)
    tables.append(t4)
    print(t4)

    t5 = generate_table5_agentic_quality(agentic_data)
    tables.append(t5)
    print(t5)

    t6 = generate_table6_llm_cost(llm_data)
    tables.append(t6)
    print(t6)

    # Key findings
    findings = generate_key_findings(method_stats, groups, test_results)
    tables.append(findings)
    print(findings)

    # Save all text outputs
    with open(OUTPUT_DIR / "all_tables_and_findings.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(tables))
    print(f"\n  Saved all tables to {OUTPUT_DIR / 'all_tables_and_findings.txt'}")

    # Generate plots
    if HAS_MPL:
        print("\n[5/6] Generating publication-quality plots...")
        set_paper_style()
        plot_1_convergence_curves(groups)
        plot_2_regime_accuracy_bars(method_stats)
        plot_3_cumulative_regret(groups)
        plot_4_final_accuracy_boxplot(groups)
        plot_5_byzantine_resilience(groups)
        plot_6_regret_per_round_heatmap(groups)
        plot_7_f1_comparison(groups)
        plot_8_round_time_comparison(method_stats)
        plot_9_accuracy_vs_regret_scatter(method_stats)
        plot_10_spider_chart(method_stats, groups)
    else:
        print("\n[5/6] Skipping plots (matplotlib not available)")

    # Save raw data as JSON for portability
    print("\n[6/6] Exporting JSON summary...")
    summary_json = {}
    for method in METHOD_ORDER:
        st = method_stats[method]
        summary_json[method] = {
            "display_name": METHOD_DISPLAY[method],
            "mean_overall_accuracy": round(st["mean_acc"], 2),
            "std_overall_accuracy": round(st["std_acc"], 2),
            "mean_overall_f1": round(st["mean_f1"], 4),
            "mean_final_accuracy": round(st["mean_final_acc"], 2),
            "std_final_accuracy": round(st["std_final_acc"], 2),
            "mean_cumulative_regret": round(st["mean_final_cumul_regret"], 2),
            "std_cumulative_regret": round(st["std_final_cumul_regret"], 2),
            "mean_round_time_s": round(st["mean_round_time"], 1),
            "regime_accuracy": {
                f"R{rs}-{re}": round(float(np.mean(st["regime_accs"].get((rs, re), [0]))), 2)
                for (rs, re) in REGIME_RANGES
            },
        }

    with open(OUTPUT_DIR / "experiment3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"  Saved experiment3_summary.json")

    print("\n" + "=" * 70)
    print(f"ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print(f"  - 10 publication-quality figures (PNG, 300 DPI)")
    print(f"  - 6 tables + key findings (TXT)")
    print(f"  - JSON summary for further processing")
    print("=" * 70)


if __name__ == "__main__":
    main()
