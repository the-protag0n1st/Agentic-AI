import os
import numpy as np
import torch

from run_fl import run_experiment
from dataset import client_class_distribution, make_clients
import visualizations as viz

SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(alphas=(0.3, 1.0, 100.0), num_clients=5, rounds=8, use_agent=True,
        db_path="fl_metrics.db", show_plots=True, normal_classes=None):
    set_seed()

    if os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY found -> LLM agent active (llama-3.3-70b-versatile).")
    else:
        print("No GROQ_API_KEY -> using rule-based fallback.")

    results, train_data, test_data = run_experiment(
        alphas=list(alphas), num_clients=num_clients, rounds=rounds,
        use_agent=use_agent, db_path=db_path, seed=SEED,
        normal_classes=normal_classes,
    )

    print("\n=== Summary ===")
    summary = {}
    for alpha, logs in results.items():
        summary[alpha] = {
            "final_acc": logs["acc"][-1],
            "final_f1": logs["f1"][-1],
            "final_auc": logs["auc"][-1],
            "methods_used": logs["method"],
        }
        print(f"alpha={alpha}: acc={logs['acc'][-1]}%  "
              f"f1={logs['f1'][-1]}  auc={logs['auc'][-1]}  "
              f"methods={logs['method']}")

    if show_plots:
        for alpha, logs in results.items():
            clients = make_clients(train_data, num_clients=num_clients, alpha=alpha, seed=SEED)
            counts = client_class_distribution(clients, train_data)
            viz.plot_data_distribution(counts, title=f"Client distribution (alpha={alpha})")
            viz.plot_accuracy_loss(logs, alpha)
            viz.plot_anomaly_heatmap(logs["run_id"], alpha, db_path=db_path)
            viz.plot_method_selection_history(logs["run_id"], alpha, db_path=db_path)
            viz.plot_method_usage(logs["run_id"], alpha, db_path=db_path)
            viz.plot_combination_usage(logs["run_id"], alpha, db_path=db_path)
            viz.plot_client_anomaly_frequency(logs["run_id"], alpha, db_path=db_path)
        viz.plot_alpha_comparison(results)

        import matplotlib.pyplot as plt
        plt.show()

    return results, summary, train_data, test_data


if __name__ == "__main__":
    run()
