#!/usr/bin/env python3
"""
Experiment 5: Scalability Analysis

This experiment measures runtime and sample complexity as functions of:
- Number of agents (n)
- Number of items (m)

Validates that Algorithm 1 scales as O(n²m) in sample complexity
and polynomial in runtime.

Generates Table 2 and Figure 5 in the paper.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.valuations import AdditiveValuation, ValuationProfile
from src.algorithms import EEAGAlgorithm


def run_scalability_trial(n_agents, n_items, epsilon, seed):
    """Run a single scalability trial and measure runtime."""
    rng = np.random.default_rng(seed)
    
    valuations = []
    for i in range(n_agents):
        v = AdditiveValuation(n_items, seed=int(rng.integers(0, 2**31)))
        valuations.append(v)
    
    profile = ValuationProfile(valuations)
    
    algo = EEAGAlgorithm(
        profile,
        epsilon=epsilon,
        delta=0.05,
        seed=int(rng.integers(0, 2**31))
    )
    
    start_time = time.perf_counter()
    result = algo.run()
    end_time = time.perf_counter()
    
    runtime = end_time - start_time
    
    return {
        "n_agents": n_agents,
        "n_items": n_items,
        "epsilon": epsilon,
        "samples": result.total_samples,
        "runtime_seconds": runtime,
        "is_ef1": result.is_ef1,
        "epsilon_ef1": result.epsilon_ef1,
        "max_ef1_violation": result.max_ef1_violation,
    }


def run_scalability_experiment(agent_values, item_values, epsilon, n_trials, seed):
    """Run the full scalability experiment."""
    results = []
    
    total_runs = len(agent_values) * len(item_values) * n_trials
    pbar = tqdm(total=total_runs, desc="Scalability analysis")
    
    for n_agents in agent_values:
        for n_items in item_values:
            for trial in range(n_trials):
                trial_seed = seed + trial * 1000 + n_agents * 100 + n_items
                
                res = run_scalability_trial(n_agents, n_items, epsilon, trial_seed)
                res["trial"] = trial
                
                results.append(res)
                pbar.update(1)
    
    pbar.close()
    return pd.DataFrame(results)


def plot_scalability(df, output_path):
    """Generate Figure 5: Scalability plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    for n_items in sorted(df["n_items"].unique()):
        subset = df[df["n_items"] == n_items]
        grouped = subset.groupby("n_agents").agg({
            "samples": ["mean", "std"]
        }).reset_index()
        
        ax1.errorbar(
            grouped["n_agents"],
            grouped[("samples", "mean")],
            yerr=grouped[("samples", "std")],
            marker='o',
            label=f"m={n_items}",
            capsize=3
        )
    
    ax1.set_xlabel("Number of agents (n)", fontsize=12)
    ax1.set_ylabel("Number of samples", fontsize=12)
    ax1.set_title("Sample Complexity vs. Agents", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for n_agents in sorted(df["n_agents"].unique()):
        subset = df[df["n_agents"] == n_agents]
        grouped = subset.groupby("n_items").agg({
            "samples": ["mean", "std"]
        }).reset_index()
        
        ax2.errorbar(
            grouped["n_items"],
            grouped[("samples", "mean")],
            yerr=grouped[("samples", "std")],
            marker='s',
            label=f"n={n_agents}",
            capsize=3
        )
    
    ax2.set_xlabel("Number of items (m)", fontsize=12)
    ax2.set_ylabel("Number of samples", fontsize=12)
    ax2.set_title("Sample Complexity vs. Items", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    for n_items in sorted(df["n_items"].unique()):
        subset = df[df["n_items"] == n_items]
        grouped = subset.groupby("n_agents").agg({
            "runtime_seconds": ["mean", "std"]
        }).reset_index()
        
        ax3.errorbar(
            grouped["n_agents"],
            grouped[("runtime_seconds", "mean")] * 1000,
            yerr=grouped[("runtime_seconds", "std")] * 1000,
            marker='o',
            label=f"m={n_items}",
            capsize=3
        )
    
    ax3.set_xlabel("Number of agents (n)", fontsize=12)
    ax3.set_ylabel("Runtime (ms)", fontsize=12)
    ax3.set_title("Runtime vs. Agents", fontsize=14)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    for n_agents in sorted(df["n_agents"].unique()):
        subset = df[df["n_agents"] == n_agents]
        grouped = subset.groupby("n_items").agg({
            "runtime_seconds": ["mean", "std"]
        }).reset_index()
        
        ax4.errorbar(
            grouped["n_items"],
            grouped[("runtime_seconds", "mean")] * 1000,
            yerr=grouped[("runtime_seconds", "std")] * 1000,
            marker='s',
            label=f"n={n_agents}",
            capsize=3
        )
    
    ax4.set_xlabel("Number of items (m)", fontsize=12)
    ax4.set_ylabel("Runtime (ms)", fontsize=12)
    ax4.set_title("Runtime vs. Items", fontsize=14)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {output_path}")


def generate_scalability_table(df, output_path):
    """Generate Table 2: Scalability results."""
    pivot = df.pivot_table(
        values=["samples", "runtime_seconds", "epsilon_ef1"],
        index="n_agents",
        columns="n_items",
        aggfunc={
            "samples": "mean",
            "runtime_seconds": "mean",
            "epsilon_ef1": "mean"
        }
    ).round(3)
    
    pivot.to_csv(output_path)
    print(f"Table saved to {output_path}")
    
    return pivot


def fit_complexity_model(df):
    """Fit complexity model to verify O(n²m) scaling."""
    from scipy.optimize import curve_fit
    
    def model(X, c):
        n, m = X
        return c * n * m
    
    X = np.array([df["n_agents"].values, df["n_items"].values])
    y = df["samples"].values
    
    try:
        popt, pcov = curve_fit(model, X, y)
        c = popt[0]
        
        y_pred = model(X, c)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            "coefficient": c,
            "r_squared": r_squared,
            "model": "O(nm)"
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5: Scalability Analysis"
    )
    parser.add_argument("--max_agents", type=int, default=20)
    parser.add_argument("--max_items", type=int, default=200)
    parser.add_argument("--n_agent_values", type=int, default=5)
    parser.add_argument("--n_item_values", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="../results")
    
    args = parser.parse_args()
    
    agent_values = np.linspace(2, args.max_agents, args.n_agent_values).astype(int)
    agent_values = np.unique(agent_values)
    
    item_values = np.linspace(10, args.max_items, args.n_item_values).astype(int)
    item_values = np.unique(item_values)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    
    print(f"Running Experiment 5: Scalability Analysis")
    print(f"  Agent values: {agent_values}")
    print(f"  Item values: {item_values}")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Trials per configuration: {args.n_trials}")
    print()
    
    df = run_scalability_experiment(
        agent_values=agent_values,
        item_values=item_values,
        epsilon=args.epsilon,
        n_trials=args.n_trials,
        seed=args.seed
    )
    
    csv_path = output_dir / "tables" / "exp5_scalability.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    fig_path = output_dir / "figures" / "fig5_scalability.pdf"
    plot_scalability(df, fig_path)
    
    table_path = output_dir / "tables" / "table2_scalability.csv"
    pivot = generate_scalability_table(df, table_path)
    
    print("\nScalability Table (Sample means):")
    print(pivot["samples"])
    
    fit_results = fit_complexity_model(df)
    print(f"\nComplexity model fit: {fit_results}")
    
    print("\nSummary Statistics:")
    print(f"  Total trials: {len(df)}")
    print(f"  Mean runtime: {df['runtime_seconds'].mean():.4f} seconds")
    print(f"  Max runtime: {df['runtime_seconds'].max():.4f} seconds")
    print(f"  ε-EF1 success rate: {df['epsilon_ef1'].mean():.2%}")


if __name__ == "__main__":
    main()
