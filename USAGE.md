# Usage Guide

This guide provides step-by-step instructions for running the experiments.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('gutenberg')"
```

### 2. Test Installation

Run the test script to verify everything works:

```bash
python test_implementation.py
```

This will test all components: data generation, model, training, evaluation, and analysis.

### 3. Run Demo (Optional)

Try the AR(1) linguistic baseline replication:

```bash
python demo_ar1_linguistic.py
```

This demonstrates that linguistic text is better modeled by AR(1) than random sequences.

## Running Experiments

### Option 1: Run All Experiments

Use the master script to run all experiments sequentially:

```bash
# On CPU
./run_all_experiments.sh cpu

# On GPU
./run_all_experiments.sh cuda
```

**Note:** Running all experiments with default settings will take approximately 600 GPU-hours (or much longer on CPU). Consider reducing `n_train`, `n_val`, `n_test`, or number of seeds for faster testing.

### Option 2: Run Individual Experiments

#### H1: Scaling Experiments

Test how prediction accuracy changes as AR order p increases.

```bash
python experiments/h1_scaling.py \
    --p_values 1 2 5 10 \
    --seeds 3 \
    --n_train 50000 \
    --n_val 5000 \
    --n_test 10000 \
    --device cpu \
    --save_dir results/h1_scaling
```

**Quick test** (much faster, for development):
```bash
python experiments/h1_scaling.py \
    --p_values 1 2 \
    --seeds 1 \
    --n_train 1000 \
    --n_val 200 \
    --n_test 500 \
    --device cpu
```

#### H2: Noise Robustness Experiments

Test sensitivity to observation noise for different AR orders.

```bash
python experiments/h2_noise.py \
    --p_values 1 2 5 10 \
    --noise_levels 0.0 0.1 0.3 0.5 \
    --seeds 3 \
    --n_train 50000 \
    --n_val 5000 \
    --n_test 10000 \
    --device cpu \
    --save_dir results/h2_noise
```

**Quick test:**
```bash
python experiments/h2_noise.py \
    --p_values 1 2 \
    --noise_levels 0.0 0.3 \
    --seeds 1 \
    --n_train 1000 \
    --n_val 200 \
    --n_test 500 \
    --device cpu
```

#### H3: Mechanism Analysis Experiments

Investigate attention head specialization by lag.

```bash
python experiments/h3_mechanism.py \
    --p 5 \
    --seeds 3 \
    --n_train 50000 \
    --n_val 5000 \
    --n_test 1000 \
    --device cpu \
    --save_dir results/h3_mechanism
```

**Quick test:**
```bash
python experiments/h3_mechanism.py \
    --p 3 \
    --seeds 1 \
    --n_train 1000 \
    --n_val 200 \
    --n_test 100 \
    --device cpu
```

## Understanding the Results

### H1 Results

After running H1 experiments, you'll find:

- `results/h1_scaling/aggregated_results.json` - Summary statistics
- `results/h1_scaling/scaling_results.png` - Visualization
- `results/h1_scaling/p{p}_seed{seed}/` - Individual run results

**Key metrics:**
- `rel_error_1step`: Relative error vs. oracle for 1-step prediction
- `rel_error_rollout`: Relative error vs. oracle for 10-step rollout

**Expected results (H1):**
- p ≤ 5: rel_error ≈ 2× (within 2× of oracle)
- p > 5: rel_error increases approximately linearly

### H2 Results

After running H2 experiments, you'll find:

- `results/h2_noise/aggregated_results.json` - Summary statistics
- `results/h2_noise/noise_robustness.png` - Visualization

**Expected results (H2):**
- Larger p → steeper error vs. noise curves
- Noise sensitivity increases with AR order

### H3 Results

After running H3 experiments, you'll find:

- `results/h3_mechanism/aggregated_h3_p{p}.json` - Summary statistics
- `results/h3_mechanism/avg_attention_heatmap_p{p}.png` - Attention patterns
- `results/h3_mechanism/avg_ablation_results_p{p}.png` - Ablation results

**Expected results (H3):**
- Attention heads specialize to different lags
- Ablating lag-specific heads → selective performance drops

## Customization

### Modifying Model Architecture

Edit model hyperparameters in the experiment scripts or pass them as arguments:

```python
model = GPTModel(
    d_input=d,
    d_model=256,      # Model dimension
    n_layers=6,       # Number of layers
    n_heads=8,        # Number of attention heads
    d_ff=1024,        # Feed-forward dimension
    max_seq_len=T,
    dropout=0.1
)
```

### Changing Data Parameters

Modify in experiment scripts:

```python
# Data generation
p = 5              # AR order
d = 5              # State dimension
T = 100            # Sequence length
context_len = 70   # Context window
noise_std = 0.1    # Noise level
```

### Training Parameters

Adjust in the Trainer:

```python
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    context_len=context_len,
    lr=3e-4,              # Learning rate
    batch_size=64,        # Batch size
    max_epochs=100,       # Maximum epochs
    patience=10,          # Early stopping patience
    device=device
)
```

## GPU Usage

To use GPU, ensure PyTorch is installed with CUDA support:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Run experiments on GPU
python experiments/h1_scaling.py --device cuda
```

For multi-GPU setups, you can modify the code to use `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel`.

## Troubleshooting

### Memory Issues

If you run out of memory:

1. Reduce batch size: `--batch_size 32` (or modify in code)
2. Reduce model size: decrease `d_model`, `n_layers`, or `d_ff`
3. Reduce dataset size: `--n_train 10000`

### Slow Training

If training is too slow:

1. Use GPU: `--device cuda`
2. Reduce dataset size
3. Reduce `max_epochs` or increase `patience` for earlier stopping
4. Use smaller model

### NLTK/Gutenberg Errors

If `demo_ar1_linguistic.py` fails:

```bash
python -c "import nltk; nltk.download('gutenberg'); nltk.download('punkt')"
```

## Tips for Research

1. **Start small:** Test with 1-2 seeds and small datasets first
2. **Monitor training:** Check `training_history` in saved metrics
3. **Compare baselines:** Always compare against Oracle, OLS, and Last-value
4. **Visualize results:** Use the plotting functions extensively
5. **Statistical significance:** Run multiple seeds (≥3) for confidence intervals

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@article{sander2024transformers,
  title={How do Transformers Perform In-Context Autoregressive Learning?},
  author={Sander, M. et al.},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

For issues or questions:
1. Check the README.md
2. Review test_implementation.py for examples
3. Examine the code documentation
4. Run experiments with `--help` flag for options
