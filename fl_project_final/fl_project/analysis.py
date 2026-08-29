import os
import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

DB_PATH = "fl_metrics.db"

def cohens_d(x, y):
    # Cohen's d for paired samples
    diff = x - y
    if len(diff) == 0 or np.std(diff, ddof=1) == 0:
        return 0.0
    return np.mean(diff) / np.std(diff, ddof=1)

def analyze_performance(db_path=DB_PATH):
    print("\n--- Performance Analysis ---")
    if not os.path.exists(db_path):
        print("Database not found.")
        return

    con = sqlite3.connect(db_path)
    # Get final round (max round per run_id)
    query = """
    SELECT r.run_id, r.alpha, r.method, r.seed, r.accuracy, r.f1, r.auc
    FROM round_metrics r
    INNER JOIN (
        SELECT run_id, MAX(round) as max_round
        FROM round_metrics
        GROUP BY run_id
    ) max_r ON r.run_id = max_r.run_id AND r.round = max_r.max_round
    """
    df = pd.read_sql(query, con)
    con.close()

    if df.empty:
        print("No metrics found in database.")
        return

    # Group by method and alpha
    grouped = df.groupby(['method', 'alpha'])
    stats = grouped.agg(
        n_seeds=('run_id', 'count'),
        acc_mean=('accuracy', 'mean'), acc_std=('accuracy', 'std'),
        f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
        auc_mean=('auc', 'mean'), auc_std=('auc', 'std')
    ).reset_index()

    for alpha in stats['alpha'].unique():
        print(f"\nAlpha: {alpha}")
        alpha_stats = stats[stats['alpha'] == alpha]
        for _, row in alpha_stats.iterrows():
            print(f"  Method: {row['method']} (n={row['n_seeds']})")
            print(f"    Acc: {row['acc_mean']:.4f} ± {row['acc_std']:.4f}")
            print(f"    F1:  {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
            print(f"    AUC: {row['auc_mean']:.4f} ± {row['auc_std']:.4f}")

    return df

def analyze_statistical_tests(df):
    print("\n--- Statistical Testing (Agent vs Baselines) ---")
    if df is None or df.empty or 'seed' not in df.columns:
        print("Data or seed tracking is missing.")
        return
    
    alphas = df['alpha'].unique()
    baselines = ["FedAvg", "Krum", "Median", "Trimmed Mean", "RuleBased"]
    
    for alpha in alphas:
        print(f"\nAlpha: {alpha}")
        df_a = df[df['alpha'] == alpha]
        agent_data = df_a[df_a['method'] == 'Agent']
        
        if len(agent_data) == 0:
            print("  No Agent data for this alpha.")
            continue
            
        for base in baselines:
            base_data = df_a[df_a['method'] == base]
            if len(base_data) == 0:
                continue
                
            merged = pd.merge(agent_data, base_data, on='seed', suffixes=('_agent', '_base'))
            n = len(merged)
            
            if n < 3:
                print(f"  Agent vs {base}: Insufficient matched seeds for Wilcoxon (valid pairs={n})")
                continue
                
            x = merged['accuracy_agent'].values
            y = merged['accuracy_base'].values
            
            try:
                stat, p = wilcoxon(x, y)
                d = cohens_d(x, y)
                print(f"  Agent vs {base} (valid pairs={n}):")
                print(f"    Wilcoxon p-value: {p:.4f}")
                print(f"    Cohen's d:        {d:.4f}")
                if p < 0.05:
                    print("    Result: Significant difference")
                else:
                    print("    Result: Not significant")
            except Exception as e:
                print(f"  Agent vs {base} (valid pairs={n}): Wilcoxon failed ({e})")

def plot_convergence(db_path=DB_PATH):
    print("\n--- Generating Convergence Plots ---")
    if not os.path.exists(db_path):
        return
        
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT run_id, round, alpha, method, accuracy FROM round_metrics", con)
    con.close()
    
    if df.empty:
        return
        
    alphas = df['alpha'].unique()
    for alpha in alphas:
        df_a = df[df['alpha'] == alpha]
        
        plt.figure(figsize=(10, 6))
        methods = df_a['method'].unique()
        
        for method in methods:
            df_m = df_a[df_a['method'] == method]
            # Group by round
            round_stats = df_m.groupby('round')['accuracy'].agg(['mean', 'std']).reset_index()
            
            plt.plot(round_stats['round'], round_stats['mean'], label=method)
            plt.fill_between(round_stats['round'], 
                             round_stats['mean'] - round_stats['std'].fillna(0),
                             round_stats['mean'] + round_stats['std'].fillna(0),
                             alpha=0.2)
                             
        plt.title(f'Convergence (Alpha={alpha})')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = f"convergence_alpha_{alpha}.png"
        plt.savefig(plot_path)
        print(f"  Saved plot to {plot_path}")

def analyze_agent_decisions(db_path=DB_PATH):
    print("\n--- Agent Decision Quality ---")
    if not os.path.exists(db_path):
        return
        
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM agent_log", con)
    con.close()
    
    if df.empty:
        print("No agent logs found.")
        return
        
    llm_df = df[df['source'] != 'rule_based'].copy()
    
    total_llm = len(llm_df)
    if total_llm == 0:
        print("No actual LLM decisions found.")
        return

    valid_decisions = llm_df['valid'].sum()
    invalid_decisions = total_llm - valid_decisions
    
    fallback_count = len(llm_df[llm_df['fallback_reason'].notnull()])
    fallback_rate = fallback_count / total_llm
    schema_validity_rate = valid_decisions / total_llm
    
    avg_latency = llm_df['latency_ms'].mean()
    
    llm_df['confidence'] = pd.to_numeric(llm_df['confidence'], errors='coerce')
    avg_confidence = llm_df['confidence'].mean()
    
    print(f"Total LLM decisions:     {total_llm}")
    print(f"Valid decisions:         {valid_decisions}")
    print(f"Invalid decisions:       {invalid_decisions}")
    print(f"Fallback count:          {fallback_count}")
    print(f"Fallback rate:           {fallback_rate:.2%}")
    print(f"Schema validity rate:    {schema_validity_rate:.2%}")
    print(f"Average latency:         {avg_latency:.2f} ms")
    print(f"Average confidence:      {avg_confidence:.4f}")
    
    print("\nMethod-selection frequency:")
    method_counts = llm_df['method_chosen'].value_counts()
    for m, c in method_counts.items():
        print(f"  {m}: {c} ({(c/total_llm):.2%})")
        
    print("\nClient-selection frequency (unique combinations):")
    combo_counts = llm_df['selected_combination'].value_counts().head(5)
    for combo, c in combo_counts.items():
        print(f"  {combo}: {c} ({(c/total_llm):.2%})")

if __name__ == "__main__":
    df = analyze_performance()
    analyze_statistical_tests(df)
    analyze_agent_decisions()
    plot_convergence()
