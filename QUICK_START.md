# Quick Start Guide

Get up and running in 5 minutes!

## 1. Install (1 minute)

```bash
cd /Users/runjiezhang/Desktop/182
pip install -r requirements.txt
python -c "import nltk; nltk.download('gutenberg')"
```

## 2. Test (2 minutes)

```bash
python test_implementation.py
```

This verifies all components work correctly.

## 3. Run Quick Experiment (2 minutes)

Test H1 with minimal settings:

```bash
python experiments/h1_scaling.py \
    --p_values 1 2 \
    --seeds 1 \
    --n_train 1000 \
    --n_val 200 \
    --n_test 500 \
    --device cpu
```

Results will be in `results/h1_scaling/`.

## What Just Happened?

1. Generated 1000 training sequences for AR(1) and AR(2)
2. Trained a small Transformer on each
3. Evaluated on 500 test sequences
4. Compared against Oracle, OLS, and Last-value baselines
5. Saved results and plots

## Next Steps

### View Results

```bash
# Check aggregated results
cat results/h1_scaling/aggregated_results.json

# View plot
open results/h1_scaling/scaling_results.png  # macOS
# or
xdg-open results/h1_scaling/scaling_results.png  # Linux
```

### Run Full Experiments

See [USAGE.md](USAGE.md) for complete instructions.

### Explore Code

- **Data generation:** [data/ar_process.py](data/ar_process.py)
- **Model:** [models/transformer.py](models/transformer.py)
- **Training:** [training/trainer.py](training/trainer.py)
- **Analysis:** [analysis/attention.py](analysis/attention.py)

## Common Commands

```bash
# H1: Scaling (quick)
python experiments/h1_scaling.py --p_values 1 2 --seeds 1 --n_train 1000 --n_val 200 --n_test 500

# H2: Noise (quick)
python experiments/h2_noise.py --p_values 1 2 --noise_levels 0.0 0.3 --seeds 1 --n_train 1000 --n_val 200 --n_test 500

# H3: Mechanism (quick)
python experiments/h3_mechanism.py --p 3 --seeds 1 --n_train 1000 --n_val 200 --n_test 100

# Demo: AR(1) linguistic
python demo_ar1_linguistic.py
```

## Help

```bash
# Get help for any experiment
python experiments/h1_scaling.py --help
python experiments/h2_noise.py --help
python experiments/h3_mechanism.py --help
```

## Troubleshooting

**Out of memory?**
```bash
# Reduce batch size or dataset size
python experiments/h1_scaling.py --n_train 500 --n_val 100 --n_test 200
```

**Too slow?**
```bash
# Use GPU if available
python experiments/h1_scaling.py --device cuda
```

**Import errors?**
```bash
# Ensure you're in the right directory
cd /Users/runjiezhang/Desktop/182
python test_implementation.py
```

## Expected Output (Quick Test)

```
==========================================
Running experiment for p=1, seed=0
==========================================
Generating training data...
Training model...
Model parameters: 1,234,567
Training: 100%|██████████| 50/50 [01:23<00:00,  2.50s/it]
Evaluating model...

==========================================
Results for p=1, seed=0
==========================================
1-step MSE: 0.123456
1-step Oracle MSE: 0.067890
1-step Relative Error: 1.8192
10-step Rollout MSE: 0.234567
10-step Relative Error: 2.1234
```

**Good performance indicators:**
- Relative error < 3.0 for small p
- Training converges (val loss decreases)
- Results are reproducible across seeds

## Full-Scale Run

Once you've validated with quick tests:

```bash
# Full H1 experiment (~50 GPU-hours per p value)
python experiments/h1_scaling.py \
    --p_values 1 2 5 10 \
    --seeds 3 \
    --n_train 50000 \
    --n_val 5000 \
    --n_test 10000 \
    --device cuda

# Or run everything at once
./run_all_experiments.sh cuda
```

## Documentation

- [README.md](README.md) - Project overview
- [USAGE.md](USAGE.md) - Detailed usage guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details

## Questions?

Check the test script for examples:
```bash
cat test_implementation.py
```

Happy experimenting! 🚀
