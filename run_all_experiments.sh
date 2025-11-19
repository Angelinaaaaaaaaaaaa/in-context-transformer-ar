#!/bin/bash
# Master script to run all experiments

echo "=========================================="
echo "RUNNING ALL EXPERIMENTS"
echo "=========================================="

# Check if device argument is provided
DEVICE=${1:-cpu}
echo "Using device: $DEVICE"

# H1: Scaling experiments
echo ""
echo "=========================================="
echo "H1: SCALING EXPERIMENTS"
echo "=========================================="
python experiments/h1_scaling.py \
    --p_values 1 2 5 10 \
    --seeds 3 \
    --n_train 50000 \
    --n_val 5000 \
    --n_test 10000 \
    --device $DEVICE \
    --save_dir results/h1_scaling

# H2: Noise robustness experiments
echo ""
echo "=========================================="
echo "H2: NOISE ROBUSTNESS EXPERIMENTS"
echo "=========================================="
python experiments/h2_noise.py \
    --p_values 1 2 5 10 \
    --noise_levels 0.0 0.1 0.3 0.5 \
    --seeds 3 \
    --n_train 50000 \
    --n_val 5000 \
    --n_test 10000 \
    --device $DEVICE \
    --save_dir results/h2_noise

# H3: Mechanism analysis (run for p=5 as example)
echo ""
echo "=========================================="
echo "H3: MECHANISM ANALYSIS EXPERIMENTS"
echo "=========================================="
python experiments/h3_mechanism.py \
    --p 5 \
    --seeds 3 \
    --n_train 50000 \
    --n_val 5000 \
    --n_test 1000 \
    --device $DEVICE \
    --save_dir results/h3_mechanism

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Results saved in:"
echo "  - results/h1_scaling/"
echo "  - results/h2_noise/"
echo "  - results/h3_mechanism/"
