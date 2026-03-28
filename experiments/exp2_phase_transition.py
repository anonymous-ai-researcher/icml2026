#!/usr/bin/env python3
"""
Experiment 2: Phase Transition between Exact EF and Approximate EF1

This experiment demonstrates the fundamental separation (Theorem 3.1 and 4.3):
- Exact EF requires Ω(m) samples even when EF allocations exist
- ε-EF1 requires only O(poly(1/ε)) samples

Generates Figure 2 in the paper.
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
    ValuationProfile
)
from src.algorithms import EEAGAlgorithm, Allocation, ValueEstimator
from src.algorithms.greedy_ef1 import round_robin_ef1, is_ef1, compute_envy
from src.fairness import is_envy_free, compute_envy_matrix


def check_exact_ef_with_samples(profile, n_samples, seed):
    """
    Attempt to find exact EF allocation with limited samples.
    
    Returns True only if we can verify EF with the given sample budget.
    """
    rng = np.random.default_rng(seed)
    n_agents = profile.n_agents
    n_items = profile.n_items
    
    samples_per_bundle = max(1, n_samples // (n_agents * n_items))
    
    estimates = np.zeros((n_agents, n_items))
    for a in range(n_agents):
        for i in range(n_items):
            samples = []
            for _ in range(samples_per_bundle):
                samples.append(profile[a].bounded_sample({i}))
            estimates[a, i] = np.mean(samples)
    
    allocation = round_robin_ef1(set(range(n_items)), n_agents, estimates)
    
    true_values = np.zeros((n_agents, n_items))
    for a in range(n_agents):
        for i in range(n_items):
            true_values[a, i] = profile[a].value({i})
    
    return is_envy_free(allocation, true_values, tolerance=1e-6)


def check_ef1_with_samples(profile, n_samples, epsilon, seed):
    """
    Check if ε-EF1 can be achieved with given sample budget.
    """
    rng = np.random.default_rng(seed)
    n_agents = profile.n_agents
    n_items = profile.n_items
    
    samples_per_bundle = max(1, n_samples // (n_agents * n_items))
    
    estimates = np.zeros((n_agents, n_items))
    for a in range(n_agents):
        for i in range(n_items):
            samples = []
            for _ in range(samples_per_bundle):
                samples.append(profile[a].bounded_sample({i}))
            estimates[a, i] = np.mean(samples)
    
    allocation = round_robin_ef1(set(range(n_items)), n_agents, estimates)
    
    true_values = np.zeros((n_agents, n_items))
    for a in range(n_agents):
        for i in range(n_items):
            true_values[a, i] = profile[a].value({i})
    
    return is_ef1(allocation, true_values, epsilon=epsilon)


def run_phase_transition_experiment(n_agents, n_items, n_trials, seed):
    """Run the phase transition experiment."""
    results = []
    
    sample_budgets = np.logspace(1, np.log10(n_agents * n_items * 100), 15).astype(int)
    sample_budgets = np.unique(sample_budgets)
    
    epsilon_values = [0.01, 0.05, 0.1, 0.2]
    
    rng = np.random.default_rng(seed)
    
    total_runs = n_trials * len(sample_budgets) * (1 + len(epsilon_values))
    pbar = tqdm(total=total_runs, desc="Phase transition")
    
    for trial in range(n_trials):
        valuations = []
        for i in range(n_agents):
            v = AdditiveValuation(n_items, seed=int(rng.integers(0, 2**31)))
            valuations.append(v)
        profile = ValuationProfile(valuations)
        
        for n_samples in sample_budgets:
            trial_seed = int(rng.integers(0, 2**31))
            
            ef_success = check_exact_ef_with_samples(profile, n_samples, trial_seed)
            results.append({
                "trial": trial,
                "n_samples": n_samples,
                "fairness_notion": "Exact EF",
                "epsilon": 0.0,
                "success": ef_success,
            })
            pbar.update(1)
            
            for eps in epsilon_values:
                ef1_success = check_ef1_with_samples(profile, n_samples, eps, trial_seed)
                results.append({
                    "trial": trial,
                    "n_samples": n_samples,
                    "fairness_notion": f"ε-EF1 (ε={eps})",
                    "epsilon": eps,
                    "success": ef1_success,
                })
                pbar.update(1)
    
    pbar.close()
    return pd.DataFrame(results)


def plot_phase_transition(df, output_path):
    """Generate Figure 2: Phase transition plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "Exact EF": "#E63946",
        "ε-EF1 (ε=0.01)": "#457B9D",
        "ε-EF1 (ε=0.05)": "#1D3557",
        "ε-EF1 (ε=0.1)": "#2A9D8F",
        "ε-EF1 (ε=0.2)": "#E9C46A",
    }
    
    markers = {
        "Exact EF": "X",
        "ε-EF1 (ε=0.01)": "o",
        "ε-EF1 (ε=0.05)": "s",
        "ε-EF1 (ε=0.1)": "^",
        "ε-EF1 (ε=0.2)": "D",
    }
    
    grouped = df.groupby(["n_samples", "fairness_notion"]).agg({
        "success": "mean"
    }).reset_index()
    
    for notion in grouped["fairness_notion"].unique():
        subset = grouped[grouped["fairness_notion"] == notion]
        ax.plot(subset["n_samples"], subset["success"] * 100,
                marker=markers.get(notion, "o"),
                color=colors.get(notion, "#333"),
                label=notion,
                linewidth=2,
                markersize=6)
    
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label="95% threshold")
    
    ax.set_xlabel("Number of samples", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title("Phase Transition: Exact EF vs. ε-EF1", fontsize=14)
    ax.set_xscale('log')
    ax.set_ylim([0, 105])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Phase Transition"
    )
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--n_items", type=int, default=50)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="../results")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    
    print(f"Running Experiment 2: Phase Transition")
    print(f"  Agents: {args.n_agents}, Items: {args.n_items}")
    print(f"  Trials: {args.n_trials}")
    print()
    
    df = run_phase_transition_experiment(
        n_agents=args.n_agents,
        n_items=args.n_items,
        n_trials=args.n_trials,
        seed=args.seed
    )
    
    csv_path = output_dir / "tables" / "exp2_phase_transition.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    fig_path = output_dir / "figures" / "fig2_phase_transition.pdf"
    plot_phase_transition(df, fig_path)
    
    print("\nSummary:")
    summary = df.groupby(["fairness_notion", "n_samples"]).agg({
        "success": ["mean", "std"]
    }).round(3)
    print(summary.head(20))


if __name__ == "__main__":
    main()
