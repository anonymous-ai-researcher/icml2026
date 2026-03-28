#!/usr/bin/env python3
"""
Experiment 3: Robustness of EF1 to Estimation Noise

This experiment demonstrates the key insight from Section 4:
EF1's existential quantifier structure (∃g) provides inherent robustness
to estimation errors, unlike exact EF which fails immediately under noise.

Generates Figure 3 in the paper.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.valuations import AdditiveValuation, ValuationProfile
from src.algorithms import Allocation
from src.algorithms.greedy_ef1 import round_robin_ef1
from src.fairness import (
    compute_envy_matrix,
    compute_ef1_violations,
    compute_efx_violations,
    is_envy_free,
    FairnessMetrics
)


def add_noise_to_values(true_values, noise_level, rng):
    """Add multiplicative noise to value estimates."""
    noise = rng.uniform(1 - noise_level, 1 + noise_level, true_values.shape)
    noisy = true_values * noise
    return np.clip(noisy, 0, 1)


def run_robustness_trial(n_agents, n_items, noise_level, seed):
    """Run a single robustness trial."""
    rng = np.random.default_rng(seed)
    
    valuations = []
    for i in range(n_agents):
        v = AdditiveValuation(n_items, seed=int(rng.integers(0, 2**31)))
        valuations.append(v)
    profile = ValuationProfile(valuations)
    
    true_values = np.zeros((n_agents, n_items))
    for a in range(n_agents):
        for i in range(n_items):
            true_values[a, i] = profile[a].value({i})
    
    noisy_values = add_noise_to_values(true_values, noise_level, rng)
    
    allocation = round_robin_ef1(set(range(n_items)), n_agents, noisy_values)
    
    metrics_true = FairnessMetrics(allocation, true_values)
    metrics_noisy = FairnessMetrics(allocation, noisy_values)
    
    return {
        "noise_level": noise_level,
        "is_ef_true": metrics_true.is_ef(),
        "is_ef1_true": metrics_true.is_ef1(),
        "is_efx_true": metrics_true.is_efx(),
        "max_envy_true": metrics_true.max_envy(),
        "max_ef1_violation_true": metrics_true.max_ef1_violation(),
        "max_efx_violation_true": metrics_true.max_efx_violation(),
        "is_ef_noisy": metrics_noisy.is_ef(),
        "is_ef1_noisy": metrics_noisy.is_ef1(),
        "nash_welfare": metrics_true.nash_welfare(),
        "estimation_error": np.mean(np.abs(true_values - noisy_values)),
    }


def run_robustness_experiment(n_agents, n_items, noise_levels, n_trials, seed):
    """Run the full robustness experiment."""
    results = []
    
    total_runs = len(noise_levels) * n_trials
    pbar = tqdm(total=total_runs, desc="Robustness analysis")
    
    for noise in noise_levels:
        for trial in range(n_trials):
            trial_seed = seed + trial * 1000 + int(noise * 10000)
            
            res = run_robustness_trial(n_agents, n_items, noise, trial_seed)
            res["trial"] = trial
            res["n_agents"] = n_agents
            res["n_items"] = n_items
            
            results.append(res)
            pbar.update(1)
    
    pbar.close()
    return pd.DataFrame(results)


def plot_robustness(df, output_path):
    """Generate Figure 3: Robustness analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    grouped = df.groupby("noise_level").agg({
        "is_ef_true": "mean",
        "is_ef1_true": "mean",
        "is_efx_true": "mean",
        "max_envy_true": ["mean", "std"],
        "max_ef1_violation_true": ["mean", "std"],
        "max_efx_violation_true": ["mean", "std"],
        "nash_welfare": ["mean", "std"],
    }).reset_index()
    
    ax1 = axes[0]
    noise = grouped["noise_level"]
    ax1.plot(noise * 100, grouped[("is_ef_true", "mean")] * 100, 
             marker='X', label="Exact EF", color='#E63946', linewidth=2)
    ax1.plot(noise * 100, grouped[("is_ef1_true", "mean")] * 100,
             marker='o', label="EF1", color='#2A9D8F', linewidth=2)
    ax1.plot(noise * 100, grouped[("is_efx_true", "mean")] * 100,
             marker='s', label="EFX", color='#E9C46A', linewidth=2)
    
    ax1.set_xlabel("Noise level (%)", fontsize=12)
    ax1.set_ylabel("Success rate (%)", fontsize=12)
    ax1.set_title("Fairness Achievement under Noise", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.errorbar(noise * 100, grouped[("max_envy_true", "mean")],
                 yerr=grouped[("max_envy_true", "std")],
                 marker='X', label="Max Envy", color='#E63946', 
                 capsize=3, linewidth=2)
    ax2.errorbar(noise * 100, grouped[("max_ef1_violation_true", "mean")],
                 yerr=grouped[("max_ef1_violation_true", "std")],
                 marker='o', label="Max EF1 Violation", color='#2A9D8F',
                 capsize=3, linewidth=2)
    
    ax2.set_xlabel("Noise level (%)", fontsize=12)
    ax2.set_ylabel("Violation magnitude", fontsize=12)
    ax2.set_title("Fairness Violations under Noise", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.errorbar(noise * 100, grouped[("nash_welfare", "mean")],
                 yerr=grouped[("nash_welfare", "std")],
                 marker='D', color='#457B9D', capsize=3, linewidth=2)
    
    ax3.set_xlabel("Noise level (%)", fontsize=12)
    ax3.set_ylabel("Nash welfare", fontsize=12)
    ax3.set_title("Welfare Degradation under Noise", fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {output_path}")


def plot_robustness_heatmap(df, output_path):
    """Generate supplementary heatmap of EF1 violations."""
    pivot = df.pivot_table(
        values="max_ef1_violation_true",
        index="noise_level",
        aggfunc=["mean", "std"]
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    noise_levels = df["noise_level"].unique()
    means = df.groupby("noise_level")["max_ef1_violation_true"].mean()
    stds = df.groupby("noise_level")["max_ef1_violation_true"].std()
    
    bars = ax.bar(range(len(noise_levels)), means, yerr=stds, 
                  capsize=5, color='#2A9D8F', alpha=0.8)
    
    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f"{n*100:.0f}%" for n in noise_levels])
    ax.set_xlabel("Noise level", fontsize=12)
    ax.set_ylabel("Mean EF1 violation", fontsize=12)
    ax.set_title("EF1 Violations by Noise Level", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Robustness Analysis"
    )
    parser.add_argument("--n_agents", type=int, default=6)
    parser.add_argument("--n_items", type=int, default=30)
    parser.add_argument("--noise_levels", type=str, default="0.01,0.05,0.1,0.15,0.2,0.25,0.3",
                        help="Comma-separated noise levels")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="../results")
    
    args = parser.parse_args()
    
    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    
    print(f"Running Experiment 3: Robustness Analysis")
    print(f"  Agents: {args.n_agents}, Items: {args.n_items}")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Trials per noise level: {args.n_trials}")
    print()
    
    df = run_robustness_experiment(
        n_agents=args.n_agents,
        n_items=args.n_items,
        noise_levels=noise_levels,
        n_trials=args.n_trials,
        seed=args.seed
    )
    
    csv_path = output_dir / "tables" / "exp3_robustness.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    fig_path = output_dir / "figures" / "fig3_robustness.pdf"
    plot_robustness(df, fig_path)
    
    fig_path2 = output_dir / "figures" / "fig3_robustness_bars.pdf"
    plot_robustness_heatmap(df, fig_path2)
    
    print("\nSummary Statistics:")
    summary = df.groupby("noise_level").agg({
        "is_ef1_true": "mean",
        "max_ef1_violation_true": ["mean", "std"],
        "nash_welfare": "mean",
    }).round(4)
    print(summary)


if __name__ == "__main__":
    main()
