#!/usr/bin/env python3
"""
Experiment 4: Sample Complexity Across Valuation Classes

This experiment validates the class-specific bounds from Table 1:
- Unit-demand: Θ(nm/ε²)
- Coverage: Θ(nm log m/ε²)  
- Additive: Θ(nm/ε²)
- General submodular: O(n²m/ε²)

Generates Figure 4 in the paper.
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
    BudgetAdditiveValuation,
    ValuationProfile
)
from src.algorithms import EEAGAlgorithm


def get_valuation_class(class_name):
    """Get valuation class by name."""
    classes = {
        "additive": AdditiveValuation,
        "unit_demand": UnitDemandValuation,
        "coverage": CoverageValuation,
        "submodular": SubmodularValuation,
        "budget_additive": BudgetAdditiveValuation,
    }
    return classes.get(class_name, AdditiveValuation)


def theoretical_complexity(n_agents, n_items, epsilon, class_name):
    """Compute theoretical sample complexity for each class."""
    delta = 0.05
    log_term = np.log(2 * n_agents * n_items / delta)
    base = 1 / (2 * (epsilon/4)**2)
    
    if class_name == "unit_demand":
        return n_agents * n_items * base * log_term
    elif class_name == "coverage":
        return n_agents * n_items * np.log(n_items + 1) * base * log_term
    elif class_name == "additive":
        return n_agents * n_items * base * log_term
    elif class_name == "submodular" or class_name == "budget_additive":
        return n_agents * n_agents * n_items * base * log_term
    else:
        return n_agents * n_items * base * log_term


def run_single_trial(n_agents, n_items, epsilon, val_class_name, seed):
    """Run a single trial for a specific valuation class."""
    rng = np.random.default_rng(seed)
    ValClass = get_valuation_class(val_class_name)
    
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
        "valuation_class": val_class_name,
        "n_agents": n_agents,
        "n_items": n_items,
        "epsilon": epsilon,
        "samples": result.total_samples,
        "is_ef1": result.is_ef1,
        "epsilon_ef1": result.epsilon_ef1,
        "max_ef1_violation": result.max_ef1_violation,
        "theoretical": theoretical_complexity(n_agents, n_items, epsilon, val_class_name),
    }


def run_valuation_comparison(n_agents, m_values, epsilon, n_trials, 
                             valuation_classes, seed):
    """Run comparison across valuation classes."""
    results = []
    
    total_runs = len(valuation_classes) * len(m_values) * n_trials
    pbar = tqdm(total=total_runs, desc="Valuation comparison")
    
    for val_class in valuation_classes:
        for n_items in m_values:
            for trial in range(n_trials):
                trial_seed = seed + trial * 1000 + n_items * 100
                
                res = run_single_trial(n_agents, n_items, epsilon, val_class, trial_seed)
                res["trial"] = trial
                
                results.append(res)
                pbar.update(1)
    
    pbar.close()
    return pd.DataFrame(results)


def plot_valuation_comparison(df, output_path):
    """Generate Figure 4: Sample complexity by valuation class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        "additive": "#2E86AB",
        "unit_demand": "#A23B72",
        "coverage": "#28A745",
        "submodular": "#E9C46A",
        "budget_additive": "#F18F01",
    }
    
    markers = {
        "additive": "o",
        "unit_demand": "s",
        "coverage": "^",
        "submodular": "D",
        "budget_additive": "v",
    }
    
    labels = {
        "additive": "Additive",
        "unit_demand": "Unit-demand",
        "coverage": "Coverage",
        "submodular": "Submodular",
        "budget_additive": "Budget-additive",
    }
    
    ax1 = axes[0]
    
    for val_class in df["valuation_class"].unique():
        subset = df[df["valuation_class"] == val_class]
        grouped = subset.groupby("n_items").agg({
            "samples": ["mean", "std"]
        }).reset_index()
        
        ax1.errorbar(
            grouped["n_items"],
            grouped[("samples", "mean")],
            yerr=grouped[("samples", "std")],
            marker=markers.get(val_class, "o"),
            color=colors.get(val_class, "#333"),
            label=labels.get(val_class, val_class),
            linewidth=2,
            capsize=3
        )
    
    ax1.set_xlabel("Number of items (m)", fontsize=12)
    ax1.set_ylabel("Number of samples", fontsize=12)
    ax1.set_title("Empirical Sample Complexity by Valuation Class", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    
    for val_class in df["valuation_class"].unique():
        subset = df[df["valuation_class"] == val_class]
        grouped = subset.groupby("n_items").agg({
            "samples": "mean",
            "theoretical": "mean"
        }).reset_index()
        
        ratio = grouped["samples"] / grouped["theoretical"]
        
        ax2.plot(
            grouped["n_items"],
            ratio,
            marker=markers.get(val_class, "o"),
            color=colors.get(val_class, "#333"),
            label=labels.get(val_class, val_class),
            linewidth=2
        )
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label="Ratio = 1")
    
    ax2.set_xlabel("Number of items (m)", fontsize=12)
    ax2.set_ylabel("Empirical / Theoretical ratio", fontsize=12)
    ax2.set_title("Tightness of Bounds", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {output_path}")


def generate_table1(df, output_path):
    """Generate Table 1: Sample complexity bounds."""
    summary = df.groupby("valuation_class").agg({
        "samples": ["mean", "std"],
        "theoretical": "mean",
        "epsilon_ef1": "mean",
    }).round(2)
    
    summary.columns = ["Empirical Mean", "Empirical Std", "Theoretical", "Success Rate"]
    summary["Ratio"] = (summary["Empirical Mean"] / summary["Theoretical"]).round(3)
    
    summary.to_csv(output_path)
    print(f"Table saved to {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Valuation Class Comparison"
    )
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--m_range", type=str, default="10,100",
                        help="Range of m values as 'min,max'")
    parser.add_argument("--n_m", type=int, default=6,
                        help="Number of m values")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="../results")
    
    args = parser.parse_args()
    
    m_min, m_max = map(int, args.m_range.split(","))
    m_values = np.logspace(np.log10(m_min), np.log10(m_max), args.n_m).astype(int)
    m_values = np.unique(m_values)
    
    valuation_classes = ["additive", "unit_demand", "coverage", "submodular"]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    
    print(f"Running Experiment 4: Valuation Class Comparison")
    print(f"  Agents: {args.n_agents}")
    print(f"  Items (m): {m_values}")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Valuation classes: {valuation_classes}")
    print()
    
    df = run_valuation_comparison(
        n_agents=args.n_agents,
        m_values=m_values,
        epsilon=args.epsilon,
        n_trials=args.n_trials,
        valuation_classes=valuation_classes,
        seed=args.seed
    )
    
    csv_path = output_dir / "tables" / "exp4_valuation_classes.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    fig_path = output_dir / "figures" / "fig4_valuation_comparison.pdf"
    plot_valuation_comparison(df, fig_path)
    
    table_path = output_dir / "tables" / "table1_complexity_bounds.csv"
    summary = generate_table1(df, table_path)
    
    print("\nTable 1: Sample Complexity Bounds")
    print(summary)


if __name__ == "__main__":
    main()
