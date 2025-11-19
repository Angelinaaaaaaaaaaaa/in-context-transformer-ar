"""
H3: Mechanism Analysis Experiments
Investigate whether attention heads specialize by lag.

Hypothesis: Multi-head attention exhibits lag specialization; ablating lag-specific
heads produces selective performance drops.
"""
import torch
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
from typing import Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import generate_ar_dataset
from models import GPTModel, OraclePredictor, OLSPredictor
from training import Trainer, ARDataset
from analysis import (
    analyze_head_specialization,
    cluster_attention_heads,
    analyze_lag_specific_ablation,
    plot_attention_heatmap,
    plot_head_clustering,
    plot_ablation_results
)


def train_model(p: int,
               d: int,
               T: int,
               context_len: int,
               n_train: int,
               n_val: int,
               seed: int,
               device: str = 'cpu',
               save_dir: Optional[str] = None):
    """
    Train a model for mechanism analysis.

    Args:
        p: AR order
        d: State dimension
        T: Sequence length
        context_len: Context window length
        n_train: Number of training sequences
        n_val: Number of validation sequences
        seed: Random seed
        device: Device to use
        save_dir: Directory to save model

    Returns:
        Trained model
    """
    print(f"\n{'='*60}")
    print(f"Training model for p={p}, seed={seed}")
    print(f"{'='*60}")

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate data
    print(f"Generating training data...")
    train_sequences, _ = generate_ar_dataset(
        n_sequences=n_train, p=p, d=d, T=T, noise_std=0.0,
        same_dynamics=False, seed=seed
    )

    print(f"Generating validation data...")
    val_sequences, _ = generate_ar_dataset(
        n_sequences=n_val, p=p, d=d, T=T, noise_std=0.0,
        same_dynamics=False, seed=seed + 1
    )

    # Create datasets
    train_dataset = ARDataset(train_sequences)
    val_dataset = ARDataset(val_sequences)

    # Create model
    model = GPTModel(
        d_input=d,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=T,
        dropout=0.1
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    model_save_dir = None
    if save_dir is not None:
        model_save_dir = os.path.join(save_dir, f"p{p}_seed{seed}")
        os.makedirs(model_save_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        context_len=context_len,
        lr=3e-4,
        batch_size=64,
        max_epochs=100,
        patience=10,
        device=device,
        save_dir=model_save_dir
    )

    # Train
    print(f"Training model...")
    history = trainer.train(verbose=True)

    return model, history


def analyze_model(model,
                 p: int,
                 d: int,
                 T: int,
                 context_len: int,
                 n_test: int,
                 seed: int,
                 device: str = 'cpu',
                 save_dir: Optional[str] = None) -> dict:
    """
    Analyze attention patterns and run ablation experiments.

    Args:
        model: Trained model
        p: AR order
        d: State dimension
        T: Sequence length
        context_len: Context window length
        n_test: Number of test sequences
        seed: Random seed
        device: Device to use
        save_dir: Directory to save results

    Returns:
        Dictionary of analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing model for p={p}")
    print(f"{'='*60}")

    # Set random seeds
    np.random.seed(seed + 100)
    torch.manual_seed(seed + 100)

    # Generate test data
    print(f"Generating test data for analysis...")
    test_sequences, test_weights = generate_ar_dataset(
        n_sequences=n_test, p=p, d=d, T=T, noise_std=0.0,
        same_dynamics=False, seed=seed + 100
    )
    test_sequences = torch.tensor(test_sequences, dtype=torch.float32)

    # Move model to device
    model = model.to(device)

    # 1. Analyze head specialization
    print(f"\nAnalyzing attention head specialization...")
    max_lag = min(p + 5, 10)  # Analyze a few lags beyond p
    lag_attention, dominant_lags = analyze_head_specialization(
        model=model,
        sequences=test_sequences[:100],  # Use subset for efficiency
        layer_idx=-1,  # Analyze last layer
        max_lag=max_lag
    )

    print(f"Lag attention shape: {lag_attention.shape}")
    print(f"Dominant lags per head: {dominant_lags}")

    # 2. Cluster attention heads
    print(f"\nClustering attention heads...")
    n_clusters = min(3, model.n_heads)  # At most 3 clusters
    cluster_labels, cluster_centers = cluster_attention_heads(
        lag_attention, n_clusters=n_clusters
    )

    print(f"Cluster assignments: {cluster_labels}")

    # 3. Lag-specific ablation
    print(f"\nRunning lag-specific ablation experiments...")
    context = test_sequences[:100, :context_len, :].to(device)
    target = test_sequences[:100, context_len, :].to(device)

    lag_ablation_results = analyze_lag_specific_ablation(
        model=model,
        sequences=context,
        target=target,
        lag_attention=lag_attention,
        layer_idx=-1
    )

    print(f"Ablation results:")
    for lag, mse_increase in sorted(lag_ablation_results.items()):
        print(f"  Lag {lag}: MSE increase = {mse_increase:.6f}")

    # 4. Save results
    results = {
        'lag_attention': lag_attention.tolist(),
        'dominant_lags': dominant_lags.tolist(),
        'cluster_labels': cluster_labels.tolist(),
        'cluster_centers': cluster_centers.tolist(),
        'lag_ablation_results': lag_ablation_results
    }

    if save_dir is not None:
        results_path = os.path.join(save_dir, f'h3_analysis_p{p}_seed{seed}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved analysis results to {results_path}")

        # Generate plots
        print(f"\nGenerating plots...")

        # Attention heatmap
        heatmap_path = os.path.join(save_dir, f'attention_heatmap_p{p}_seed{seed}.png')
        plot_attention_heatmap(
            lag_attention,
            dominant_lags,
            save_path=heatmap_path
        )

        # Head clustering
        clustering_path = os.path.join(save_dir, f'head_clustering_p{p}_seed{seed}.png')
        plot_head_clustering(
            lag_attention,
            cluster_labels,
            cluster_centers,
            save_path=clustering_path
        )

        # Ablation results
        ablation_path = os.path.join(save_dir, f'ablation_results_p{p}_seed{seed}.png')
        plot_ablation_results(
            lag_ablation_results,
            save_path=ablation_path
        )

    return results


def aggregate_results(all_results: list) -> dict:
    """
    Aggregate analysis results across seeds.

    Args:
        all_results: List of result dictionaries from different seeds

    Returns:
        Aggregated statistics
    """
    n_seeds = len(all_results)

    # Aggregate lag attention patterns (average across seeds)
    lag_attention_list = [np.array(r['lag_attention']) for r in all_results]
    avg_lag_attention = np.mean(lag_attention_list, axis=0)
    std_lag_attention = np.std(lag_attention_list, axis=0)

    # Aggregate ablation results
    all_lag_ablations = {}
    for r in all_results:
        for lag, mse_increase in r['lag_ablation_results'].items():
            lag = int(lag)
            if lag not in all_lag_ablations:
                all_lag_ablations[lag] = []
            all_lag_ablations[lag].append(mse_increase)

    avg_ablation = {lag: np.mean(values) for lag, values in all_lag_ablations.items()}
    std_ablation = {lag: np.std(values) for lag, values in all_lag_ablations.items()}

    aggregated = {
        'avg_lag_attention': avg_lag_attention.tolist(),
        'std_lag_attention': std_lag_attention.tolist(),
        'avg_ablation': avg_ablation,
        'std_ablation': std_ablation,
        'n_seeds': n_seeds
    }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='H3: Mechanism Analysis Experiments')
    parser.add_argument('--p', type=int, default=5,
                       help='AR order to analyze')
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of random seeds')
    parser.add_argument('--d', type=int, default=5,
                       help='State dimension')
    parser.add_argument('--T', type=int, default=100,
                       help='Sequence length')
    parser.add_argument('--context_len', type=int, default=70,
                       help='Context window length')
    parser.add_argument('--n_train', type=int, default=50000,
                       help='Number of training sequences')
    parser.add_argument('--n_val', type=int, default=5000,
                       help='Number of validation sequences')
    parser.add_argument('--n_test', type=int, default=1000,
                       help='Number of test sequences for analysis')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--save_dir', type=str, default='results/h3_mechanism',
                       help='Directory to save results')

    args = parser.parse_args()

    print("="*60)
    print("H3: MECHANISM ANALYSIS EXPERIMENTS")
    print("="*60)
    print(f"p = {args.p}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Run experiments for each seed
    all_results = []

    for seed in range(args.seeds):
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")

        # Train model
        model, history = train_model(
            p=args.p,
            d=args.d,
            T=args.T,
            context_len=args.context_len,
            n_train=args.n_train,
            n_val=args.n_val,
            seed=seed,
            device=args.device,
            save_dir=args.save_dir
        )

        # Analyze model
        results = analyze_model(
            model=model,
            p=args.p,
            d=args.d,
            T=args.T,
            context_len=args.context_len,
            n_test=args.n_test,
            seed=seed,
            device=args.device,
            save_dir=args.save_dir
        )

        all_results.append(results)

    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)

    aggregated = aggregate_results(all_results)

    # Save aggregated results
    aggregated_path = os.path.join(args.save_dir, f'aggregated_h3_p{args.p}.json')
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated results to {aggregated_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nAverage ablation MSE increases:")
    for lag in sorted(aggregated['avg_ablation'].keys()):
        avg = aggregated['avg_ablation'][lag]
        std = aggregated['std_ablation'][lag]
        print(f"  Lag {lag}: {avg:.6f} ± {std:.6f}")

    # Plot aggregated results
    print("\nGenerating aggregated plots...")

    # Average attention heatmap
    avg_lag_attention = np.array(aggregated['avg_lag_attention'])
    dominant_lags = np.argmax(avg_lag_attention, axis=1) + 1

    heatmap_path = os.path.join(args.save_dir, f'avg_attention_heatmap_p{args.p}.png')
    plot_attention_heatmap(
        avg_lag_attention,
        dominant_lags,
        save_path=heatmap_path
    )

    # Average ablation results
    ablation_path = os.path.join(args.save_dir, f'avg_ablation_results_p{args.p}.png')
    plot_ablation_results(
        aggregated['avg_ablation'],
        save_path=ablation_path
    )

    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()
