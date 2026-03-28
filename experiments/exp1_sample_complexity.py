#!/usr/bin/env python3
"""
Experiment 1: Sample Complexity vs. Target Accuracy

This experiment validates the theoretical sample complexity bounds from Theorem 4.3:
- Upper bound: O(n²m/ε²) for general submodular valuations
- Tight bounds for specific classes (Table 1)

Generates Figure 1 in the paper.
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

from src.valuations import (
    AdditiveValuation, 
    UnitDemandValuation, 
    CoverageValuation,
    SubmodularValuation,
    ValuationProfile
)
from src.algorithms import EEAGAlgorithm, Allocation
from src.fairness import FairnessMetrics


def run_single_trial(n_agents, n_items, epsilon, valuation_class, seed):
    """Run a single trial and return results."""
    rng = np.random.default_rng(seed)
    
    val_classes = {
        "additive": AdditiveValuation,
        "unit_demand": UnitDemandValuation,
        "coverage": CoverageValuation,
        "submodular": SubmodularValuation,
    }
    
    ValClass = val_classes.get(valuation_class, AdditiveValuation)
    
    valuations = []
    for i in range(n_agents):
        v = ValClass(n_items, seed=int(rng.integers(0, 2**31)))
        valuations.append(v)
    
    profile = ValuationProfile(valuations)
    
    algo = EEAGAlgorithm(
        profile,
        epsilon=epsilon,
        delta=0.05,
        seed=int(rng.integers(0, 2**31))
    )
    
    result = algo.run()
    
    return {
        "epsilon": epsilon,
        "samples": result.total_samples,
        "is_ef1": result.is_ef1,
        "epsilon_ef1": result.epsilon_ef1,
        "max_ef1_violation": result.max_ef1_violation,
        "estimation_error_mean": result.estimation_errors["mean"],
        "estimation_error_max": result.estimation_errors["max"],
    }


def theoretical_bound(n_agents, n_items, epsilon, valuation_class):
    """Compute theoretical sample complexity bound."""
    delta = 0.05
    base = (1 / (2 * (epsilon/4)**2)) * np.log(2 * n_agents * n_items / delta)
    
    if valuation_class == "unit_demand":
        return n_agents * n_items * base
    elif valuation_class == "coverage":
        return n_agents * n_items * np.log(n_items) * base
    else:
        return n_agents * n_items * base


def run_experiment(n_agents, n_items, epsilon_values, n_trials, 
                   valuation_class, base_seed=42):
    """Run the full experiment."""
    results = []
    
    total_runs = len(epsilon_values) * n_trials
    pbar = tqdm(total=total_runs, desc="Running trials")
    
    for eps in epsilon_values:
        for trial in range(n_trials):
            seed = base_seed + trial * 1000 + int(eps * 10000)
            
            res = run_single_trial(n_agents, n_items, eps, valuation_class, seed)
            res["trial"] = trial
            res["n_agents"] = n_agents
            res["n_items"] = n_items
            res["valuation_class"] = valuation_class
            res["theoretical"] = theoretical_bound(n_agents, n_items, eps, valuation_class)
            
            results.append(res)
            pbar.update(1)
    
    pbar.close()
    return pd.DataFrame(results)


def plot_results(df, output_path):
    """Generate Figure 1: Sample complexity vs epsilon."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    grouped = df.groupby("epsilon").agg({
        "samples": ["mean", "std"],
        "theoretical": "mean",
        "epsilon_ef1": "mean",
    }).reset_index()
    grouped.columns = ["epsilon", "samples_mean", "samples_std", 
                       "theoretical", "success_rate"]
    
    ax1 = axes[0]
    ax1.errorbar(grouped["epsilon"], grouped["samples_mean"], 
                 yerr=grouped["samples_std"], 
                 marker='o', capsize=3, label="Empirical", color='#2E86AB')
    ax1.plot(grouped["epsilon"], grouped["theoretical"], 
             '--', marker='s', label="Theoretical", color='#A23B72')
    
    ax1.set_xlabel(r"Target accuracy $\varepsilon$", fontsize=12)
    ax1.set_ylabel("Number of samples", fontsize=12)
    ax1.set_title("Sample Complexity vs. Accuracy", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(grouped["epsilon"], grouped["success_rate"] * 100, 
             marker='o', color='#28A745', linewidth=2)
    ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, 
                label=r"$1-\delta = 95\%$")
    
    ax2.set_xlabel(r"Target accuracy $\varepsilon$", fontsize=12)
    ax2.set_ylabel(r"$\varepsilon$-EF1 success rate (%)", fontsize=12)
    ax2.set_title("Success Rate vs. Accuracy", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_xscale('log')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Sample Complexity vs. Accuracy"
    )
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--n_items", type=int, default=20)
    parser.add_argument("--eps_range", type=str, default="0.01,0.5",
                        help="Epsilon range as 'min,max'")
    parser.add_argument("--n_eps", type=int, default=8,
                        help="Number of epsilon values")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--valuation", type=str, default="additive",
                        choices=["additive", "unit_demand", "coverage", "submodular"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="../results")
    
    args = parser.parse_args()
    
    eps_min, eps_max = map(float, args.eps_range.split(","))
    epsilon_values = np.logspace(np.log10(eps_min), np.log10(eps_max), args.n_eps)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    
    print(f"Running Experiment 1: Sample Complexity")
    print(f"  Agents: {args.n_agents}, Items: {args.n_items}")
    print(f"  Epsilon values: {epsilon_values}")
    print(f"  Trials per epsilon: {args.n_trials}")
    print(f"  Valuation class: {args.valuation}")
    print()
    
    df = run_experiment(
        n_agents=args.n_agents,
        n_items=args.n_items,
        epsilon_values=epsilon_values,
        n_trials=args.n_trials,
        valuation_class=args.valuation,
        base_seed=args.seed
    )
    
    csv_path = output_dir / "tables" / "exp1_sample_complexity.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    fig_path = output_dir / "figures" / "fig1_sample_complexity.pdf"
    plot_results(df, fig_path)
    
    print("\nSummary Statistics:")
    summary = df.groupby("epsilon").agg({
        "samples": ["mean", "std"],
        "epsilon_ef1": "mean",
        "estimation_error_mean": "mean",
    })
    print(summary)


if __name__ == "__main__":
    main()
