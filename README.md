# Learning to Compensate: Sample Complexity of EEAG under Unknown Submodular Valuations

This repository contains the implementation and experimental evaluation for the paper "Learning to Compensate: Sample Complexity of Envy Elimination under Unknown Submodular Valuations" (ICML 2026 Submission).

## Overview

We study the sample complexity of achieving envy-free allocations when agent valuations must be learned from noisy observations. Our main contributions:

1. **Lower bound for exact EF**: We prove that exact envy-freeness requires Ω(m) queries even when EF allocations exist (Theorem 3.1).

2. **Upper bound for ε-EF1**: We show that ε-EF1 can be achieved with Õ(n²m/ε²) samples for general submodular valuations (Theorem 4.3).

3. **Class-specific bounds**: We establish tight bounds for specific valuation classes:
   - Unit-demand: Θ(nm/ε²)
   - Coverage: Θ(nm log m/ε²)

4. **Robustness of EF1**: We demonstrate that EF1's existential quantifier structure provides inherent robustness to estimation errors.

## Repository Structure

```
eeag-learning/
├── src/
│   ├── valuations/          # Valuation function implementations
│   │   ├── base.py          # Abstract base class
│   │   ├── additive.py      # Additive valuations
│   │   ├── unit_demand.py   # Unit-demand valuations
│   │   ├── coverage.py      # Coverage valuations
│   │   └── submodular.py    # General submodular valuations
│   ├── algorithms/          # Core algorithms
│   │   ├── eeag.py          # EEAG algorithm (Algorithm 1)
│   │   ├── greedy_ef1.py    # Greedy EF1 allocation
│   │   └── estimation.py    # Value estimation with samples
│   ├── fairness/            # Fairness metrics
│   │   ├── envy.py          # Envy computation
│   │   └── ef1.py           # EF1 verification
│   └── utils/               # Utilities
│       └── sampling.py      # Noisy sampling oracle
├── experiments/             # Experiment scripts
│   ├── configs/             # Experiment configurations
│   ├── run_all.sh           # Master script to run all experiments
│   ├── exp1_sample_complexity.py
│   ├── exp2_phase_transition.py
│   ├── exp3_robustness.py
│   ├── exp4_valuation_classes.py
│   └── exp5_scalability.py
├── results/                 # Experimental results
│   ├── figures/             # Generated figures
│   └── tables/              # Result tables (CSV)
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/eeag-learning.git
cd eeag-learning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Requirements

- Python >= 3.8
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- Pandas >= 1.3.0
- tqdm >= 4.62.0
- seaborn >= 0.11.0

## Running Experiments

### Quick Start

To reproduce all experiments from the paper:

```bash
cd experiments
bash run_all.sh
```

This will run all five experiments and generate figures in `results/figures/`.

### Individual Experiments

**Experiment 1: Sample Complexity vs. Accuracy (Figure 1)**
```bash
python experiments/exp1_sample_complexity.py --n_agents 4 --n_items 20 --eps_range 0.01,0.5 --n_trials 100
```

**Experiment 2: Phase Transition (Figure 2)**
```bash
python experiments/exp2_phase_transition.py --n_agents 4 --n_items 50 --n_trials 50
```

**Experiment 3: EF1 Robustness (Figure 3)**
```bash
python experiments/exp3_robustness.py --n_agents 6 --n_items 30 --noise_levels 0.01,0.05,0.1,0.2
```

**Experiment 4: Valuation Class Comparison (Figure 4)**
```bash
python experiments/exp4_valuation_classes.py --n_agents 4 --m_range 10,100 --n_trials 50
```

**Experiment 5: Scalability (Table 1)**
```bash
python experiments/exp5_scalability.py --max_agents 20 --max_items 200
```

### Configuration Files

Experiment parameters can also be specified via YAML config files:

```bash
python experiments/exp1_sample_complexity.py --config experiments/configs/exp1_default.yaml
```

## Main Results

### Sample Complexity Bounds (Table 1)

| Valuation Class | Upper Bound | Lower Bound | Gap |
|-----------------|-------------|-------------|-----|
| Unit-demand     | O(nm/ε²)    | Ω(nm/ε²)    | Tight |
| Coverage        | O(nm log m/ε²) | Ω(nm log m/ε²) | Tight |
| Additive        | O(nm/ε²)    | Ω(nm/ε²)    | Tight |
| Submodular      | O(n²m/ε²)   | Ω(nm/ε²)    | O(n) |

### Key Experimental Findings

1. **Phase Transition**: Sharp transition from Ω(m) for exact EF to poly(1/ε) for ε-EF1.

2. **Robustness**: EF1 violations remain bounded even under 20% estimation noise, while exact EF fails immediately.

3. **Scalability**: Algorithm 1 scales linearly in m and quadratically in n, matching theoretical predictions.

## Algorithm Overview

### Algorithm 1: Explore-then-Exploit for ε-EF1

```
Input: n agents, m items, pool P, accuracy ε, confidence δ
Output: ε-EF1 allocation

Phase 1 (Exploration):
  For each agent a and relevant bundle S:
    Draw T = O(ε⁻² log(nm/δ)) samples
    Compute empirical mean v̂_a(S)

Phase 2 (Exploitation):
  Run greedy EF1 using estimated values v̂
  Return allocation
```

## License

This code is released under the MIT License. See `LICENSE` for details.

## Reproducibility

All experiments were run on a machine with:
- CPU: Intel Xeon E5-2680 v4 @ 2.40GHz (28 cores)
- RAM: 128GB
- OS: Ubuntu 20.04 LTS
- Python 3.9.7

Random seeds are fixed for reproducibility. Expected runtime for all experiments: ~4 hours.

## Contact

For questions about the code, please open an issue on GitHub.
