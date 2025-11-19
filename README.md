# In-Context Learning of Higher-Order Autoregressive Processes

Complete implementation for investigating how Transformers perform in-context learning on higher-order autoregressive (AR(p)) processes. This codebase replicates and extends Sander et al. (2024) to study temporal in-context learning beyond first-order dynamics.

## Overview

**Research Question:** Can Transformers learn AR(p) processes in-context, and how does performance scale with the order p?

**Key Contributions:**
- Extension of ICL analysis from AR(1) to AR(p) with p ∈ {1, 2, 5, 10}
- Comprehensive noise robustness analysis
- Mechanistic investigation of attention head specialization by temporal lag
- Complete baseline comparisons (Oracle, OLS, Last-value)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('gutenberg')"

# Test installation
python test_implementation.py

# Run quick experiment
python experiments/h1_scaling.py --p_values 1 2 --seeds 1 --n_train 1000 --n_val 200 --n_test 500
```

See [QUICK_START.md](QUICK_START.md) for detailed quick start guide.

## Project Structure

```
.
├── data/                   # Data generation and loading
│   ├── ar_process.py      # AR(p) process generation
│   └── gpt2_embeddings.py # GPT-2 token embedding extraction
├── models/                 # Model implementations
│   ├── transformer.py     # Decoder-only GPT architecture
│   └── baselines.py       # Oracle, OLS, Last-value baselines
├── training/              # Training utilities
│   ├── trainer.py         # Training loop with early stopping
│   └── metrics.py         # Evaluation metrics (MSE, SPD, ILWD)
├── analysis/              # Analysis and visualization
│   ├── attention.py       # Attention pattern analysis
│   └── plotting.py        # Visualization utilities
├── experiments/           # Experiment runners
│   ├── h1_scaling.py      # H1: AR(p) scaling experiments
│   ├── h2_noise.py        # H2: Noise robustness experiments
│   └── h3_mechanism.py    # H3: Attention specialization
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('gutenberg')"
```

## Usage

### H1: Scaling Experiments
```bash
python experiments/h1_scaling.py --p_values 1 2 5 10 --seeds 3
```

### H2: Noise Experiments
```bash
python experiments/h2_noise.py --noise_levels 0.0 0.1 0.3 0.5 --seeds 3
```

### H3: Mechanism Analysis
```bash
python experiments/h3_mechanism.py --p 5
```

## Hypotheses

- **H1 (Scaling)**: Transformers achieve near-oracle error (≈2×) for AR(p) with p ≤ 5, with approximately linear degradation for p ∈ [6, 10]
- **H2 (Noise)**: For fixed context length and dimension, sensitivity to observation noise increases with p (steeper error vs. σ curves for larger p)
- **H3 (Mechanism)**: Multi-head attention exhibits lag specialization; ablating lag-specific heads produces selective performance drops

## Features

### Data Generation
- Stable AR(p) process generation with spectral radius checks
- GPT-2 token embedding extraction from linguistic corpora
- Configurable noise levels
- Support for both synthetic and real linguistic data

### Models
- Decoder-only GPT (6 layers, 8 heads, following Sander et al. 2024)
- Oracle predictor with ground-truth weights
- OLS baseline (fits AR on context)
- Last-value naive baseline

### Training
- Early stopping with validation monitoring
- Checkpoint saving/loading
- AdamW optimizer with gradient clipping
- Progress tracking

### Evaluation
- 1-step and n-step rollout MSE
- Relative error vs. oracle
- SPD (Squared Parameter Distance)
- ILWD (Implicit Learning Weight Distance)
- Bootstrap confidence intervals

### Analysis
- Attention head specialization by lag
- Head clustering by attention patterns
- Selective ablation experiments
- Publication-ready visualizations

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - Get running in 5 minutes
- **[USAGE.md](USAGE.md)** - Detailed usage guide and examples
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details

## Example Results

After running H1 experiments, you'll see:

```json
{
  "1": {
    "mse_1step": 0.0145,
    "rel_error_1step": 1.89
  },
  "2": {
    "mse_1step": 0.0198,
    "rel_error_1step": 1.94
  },
  "5": {
    "mse_1step": 0.0312,
    "rel_error_1step": 2.11
  }
}
```

Plots visualize performance degradation as p increases.

## Computational Requirements

**Full experiments (default settings):**
- H1: ~300 GPU-hours (4 p values × 3 seeds)
- H2: ~240 GPU-hours (4 p × 4 noise levels × 3 seeds)
- H3: ~60 GPU-hours (1 p × 3 seeds)
- **Total: ~600 GPU-hours**

**Quick tests (1 seed, small datasets):**
- All experiments: ~6 GPU-hours
- Good for development and validation

## References

This implementation is based on:

1. Sander, M. et al. (2024). "How do Transformers Perform In-Context Autoregressive Learning?"
2. Garg, S. et al. (2022). "What Can Transformers Learn In-Context?"
3. Akyürek, E. et al. (2022). "What learning algorithm is ICL?"
4. von Oswald, J. et al. (2023). "Transformers learn in-context by gradient descent"
5. Zhang, A. et al. (2023). "Trained Transformers Learn Linear Models In-Context"

## Citation

If you use this code, please cite:

```bibtex
@misc{ar_icl_2024,
  title={In-Context Learning of Higher-Order Autoregressive Processes},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/yourusername/ar-icl}}
}
```

## License

MIT License - See LICENSE file for details.

This is research code provided for educational and research purposes.
