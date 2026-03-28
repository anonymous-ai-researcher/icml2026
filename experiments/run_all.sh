#!/bin/bash
#
# Run all experiments for the paper
# "Learning to Compensate: Sample Complexity of EEAG under Unknown Submodular Valuations"
#
# Usage: bash run_all.sh [--quick]
#   --quick: Run with reduced trials for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="../results"
mkdir -p "$RESULTS_DIR/figures"
mkdir -p "$RESULTS_DIR/tables"

QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo "Running in quick mode (reduced trials)"
fi

if $QUICK_MODE; then
    N_TRIALS=10
    N_TRIALS_SCALE=5
else
    N_TRIALS=100
    N_TRIALS_SCALE=20
fi

echo "========================================"
echo "EEAG Learning: Experiment Suite"
echo "========================================"
echo "Output directory: $RESULTS_DIR"
echo "Trials per experiment: $N_TRIALS"
echo ""

START_TIME=$(date +%s)

echo "[1/5] Running Experiment 1: Sample Complexity vs. Accuracy"
echo "----------------------------------------"
python exp1_sample_complexity.py \
    --n_agents 4 \
    --n_items 20 \
    --eps_range 0.02,0.3 \
    --n_eps 6 \
    --n_trials $N_TRIALS \
    --valuation additive \
    --output_dir "$RESULTS_DIR"
echo ""

echo "[2/5] Running Experiment 2: Phase Transition"
echo "----------------------------------------"
python exp2_phase_transition.py \
    --n_agents 4 \
    --n_items 30 \
    --n_trials $(($N_TRIALS / 2)) \
    --output_dir "$RESULTS_DIR"
echo ""

echo "[3/5] Running Experiment 3: Robustness Analysis"
echo "----------------------------------------"
python exp3_robustness.py \
    --n_agents 6 \
    --n_items 24 \
    --noise_levels 0.02,0.05,0.1,0.15,0.2,0.25 \
    --n_trials $N_TRIALS \
    --output_dir "$RESULTS_DIR"
echo ""

echo "[4/5] Running Experiment 4: Valuation Class Comparison"
echo "----------------------------------------"
python exp4_valuation_classes.py \
    --n_agents 4 \
    --m_range 10,80 \
    --n_m 5 \
    --epsilon 0.1 \
    --n_trials $(($N_TRIALS / 2)) \
    --output_dir "$RESULTS_DIR"
echo ""

echo "[5/5] Running Experiment 5: Scalability Analysis"
echo "----------------------------------------"
python exp5_scalability.py \
    --max_agents 16 \
    --max_items 100 \
    --n_agent_values 4 \
    --n_item_values 4 \
    --epsilon 0.1 \
    --n_trials $N_TRIALS_SCALE \
    --output_dir "$RESULTS_DIR"
echo ""

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo "Total runtime: ${DURATION} seconds"
echo ""
echo "Generated files:"
echo "  Figures: $RESULTS_DIR/figures/"
ls -la "$RESULTS_DIR/figures/" 2>/dev/null || echo "    (none)"
echo ""
echo "  Tables: $RESULTS_DIR/tables/"
ls -la "$RESULTS_DIR/tables/" 2>/dev/null || echo "    (none)"
echo ""
echo "To view figures, open the PDF files in $RESULTS_DIR/figures/"
