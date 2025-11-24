"""
H1: Scaling Experiments
Test how prediction accuracy changes as AR order p increases.

Hypothesis: Transformers achieve near-oracle error (within ≈ 2×) for AR(p) with p ≤ 5,
with approximately linear degradation for p ∈ [6, 10].
"""
import torch
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
from typing import Optional, Tuple, List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import generate_ar_dataset
from models import GPTModel, OraclePredictor, OLSPredictor, LastValuePredictor
from training import Trainer, ARDataset, evaluate_model
from analysis import plot_scaling_results

def load_dataset(data_dir: str, p: int, seed: int, split: str) -> Tuple[torch.Tensor, List[List[np.ndarray]]]:
    """
    Helper to load pre-generated data.
    Filename format: ar_p{p}_seed{seed}_{split}.pt
    """
    filename = f"ar_p{p}_seed{seed}_{split}.pt"
    path = os.path.join(data_dir, filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Please make sure you ran 'generate_data.py' and the data is in '{data_dir}'."
        )
        
    # [FIX]: Add weights_only=False to allow loading numpy arrays
    data = torch.load(path, weights_only=False) 
    return data["sequences"], data["weights"]

def run_experiment(p: int,
                  d: int,
                  T: int,
                  context_len: int,
                  n_train: int,
                  n_val: int,
                  n_test: int,
                  seed: int,
                  device: str = 'cpu',
                  save_dir: Optional[str] = None) -> dict:
    """
    Run experiment for a single AR order p.

    Args:
        p: AR order
        d: State dimension
        T: Sequence length
        context_len: Context window length
        n_train: Number of training sequences
        n_val: Number of validation sequences
        n_test: Number of test sequences
        seed: Random seed
        device: Device to use
        save_dir: Directory to save results

    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Running experiment for p={p}, seed={seed}")
    print(f"{'='*60}")

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir = 'data/data_cach_h1'
    # --- LOAD DATA ---
    # Reads from: data/data_cache/ar_p{p}_seed{seed}_{split}.pt
    train_sequences, train_weights = load_dataset(data_dir, p, seed, "train")
    val_sequences, val_weights     = load_dataset(data_dir, p, seed, "val")
    test_sequences, test_weights   = load_dataset(data_dir, p, seed, "test")

    # Check dimensions
    assert train_sequences.shape[-1] == d, f"Dimension mismatch: Data has {train_sequences.shape[-1]}, args have {d}"

    # Create datasets
    train_dataset = ARDataset(train_sequences)
    val_dataset = ARDataset(val_sequences)

    # Create model
    model = GPTModel(
        d_input=d,
        d_model=256,
        n_layers=1,
        n_heads=2,
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
        batch_size=1280,
        max_epochs=300,
        patience=30,
        device=device,
        save_dir=model_save_dir
    )

    # Train
    print(f"Training model...")
    history = trainer.train(verbose=True)

    # Create baselines
    # Use first test sequence weights for oracle (representative)
    oracle = OraclePredictor(test_weights[0])
    ols = OLSPredictor(p)
    last_val = LastValuePredictor()

    # Evaluate
    print(f"Evaluating model...")
    test_sequences_tensor = torch.tensor(test_sequences, dtype=torch.float32)

    metrics = evaluate_model(
        model=model,
        test_sequences=test_sequences_tensor,
        test_weights=test_weights,
        context_len=context_len,
        p=p,
        oracle_predictor=oracle,
        ols_predictor=ols,
        device=device
    )

    # Add training history
    metrics['training_history'] = history

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for p={p}, seed={seed}")
    print(f"{'='*60}")
    print(f"1-step MSE: {metrics['mse_1step']:.6f}")
    print(f"1-step Oracle MSE: {metrics['mse_1step_oracle']:.6f}")
    print(f"1-step OLS MSE: {metrics['mse_1step_ols']:.6f}")
    print(f"1-step Relative Error: {metrics['rel_error_1step']:.4f}")
    print(f"10-step Rollout MSE: {metrics['mse_rollout']:.6f}")
    print(f"10-step Oracle MSE: {metrics['mse_rollout_oracle']:.6f}")
    print(f"10-step Relative Error: {metrics['rel_error_rollout']:.4f}")

    # Save results
    if save_dir is not None:
        results_path = os.path.join(model_save_dir, 'metrics.json')
        with open(results_path, 'w') as f:
            # Convert history to serializable format
            serializable_metrics = {k: v for k, v in metrics.items() if k != 'training_history'}
            serializable_metrics['training_history'] = {
                'train_losses': [float(x) for x in history['train_losses']],
                'val_losses': [float(x) for x in history['val_losses']],
                'best_val_loss': float(history['best_val_loss']),
                'n_epochs': int(history['n_epochs'])
            }
            json.dump(serializable_metrics, f, indent=2)
        print(f"Saved results to {results_path}")

    return metrics


def aggregate_results(all_results: dict, p_values: list) -> dict:
    """
    Aggregate results across seeds.

    Args:
        all_results: Dictionary mapping (p, seed) -> metrics
        p_values: List of p values

    Returns:
        Aggregated metrics
    """
    aggregated = {}

    for p in p_values:
        # Gather metrics for this p across seeds
        metrics_list = [all_results[seed] for seed in all_results.keys() if seed[0] == p]

        if len(metrics_list) == 0:
            continue

        # Compute mean and std
        aggregated[p] = {}
        for key in ['mse_1step', 'mse_1step_oracle', 'mse_1step_ols',
                   'mse_rollout', 'mse_rollout_oracle', 'mse_rollout_ols',
                   'rel_error_1step', 'rel_error_rollout']:
            values = [m[key] for m in metrics_list]
            aggregated[p][key] = np.mean(values)
            aggregated[p][f'{key}_std'] = np.std(values)

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='H1: AR(p) Scaling Experiments')
    parser.add_argument('--p_values', type=int, nargs='+', default=[1, 2, 5, 10],
                       help='AR orders to test')
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
    parser.add_argument('--n_test', type=int, default=10000,
                       help='Number of test sequences')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--save_dir', type=str, default='results/h1_scaling',
                       help='Directory to save results')

    args = parser.parse_args()

    print("="*60)
    print("H1: AR(p) SCALING EXPERIMENTS")
    print("="*60)
    print(f"p values: {args.p_values}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Run experiments
    all_results = {}

    for p in args.p_values:
        for seed in range(args.seeds):
            metrics = run_experiment(
                p=p,
                d=args.d,
                T=args.T,
                context_len=args.context_len,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
                seed=seed,
                device=args.device,
                save_dir=args.save_dir
            )
            all_results[(p, seed)] = metrics

    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)

    aggregated = aggregate_results(all_results, args.p_values)

    # Save aggregated results
    aggregated_path = os.path.join(args.save_dir, 'aggregated_results.json')
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated results to {aggregated_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for p in args.p_values:
        if p not in aggregated:
            continue
        print(f"\np = {p}:")
        print(f"  1-step relative error: {aggregated[p]['rel_error_1step']:.4f} ± {aggregated[p]['rel_error_1step_std']:.4f}")
        print(f"  10-step relative error: {aggregated[p]['rel_error_rollout']:.4f} ± {aggregated[p]['rel_error_rollout_std']:.4f}")

    # Plot results
    print("\nGenerating plots...")
    plot_path = os.path.join(args.save_dir, 'scaling_results.png')
    plot_scaling_results(aggregated, save_path=plot_path)

    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()