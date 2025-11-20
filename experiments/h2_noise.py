"""
H2: Noise Robustness Experiments
Test sensitivity to observation noise for different AR orders.

Hypothesis: For fixed context length and dimension, sensitivity to observation noise
increases with p (steeper error vs. σ curves for larger p).
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
from models import GPTModel, OraclePredictor, OLSPredictor
from training import Trainer, ARDataset, evaluate_model
from analysis import plot_noise_robustness

def load_dataset(data_dir: str, p: int, seed: int, split: str) -> Tuple[torch.Tensor, List[List[np.ndarray]]]:
    """Helper to load pre-generated data."""
    filename = f"ar_p{p}_seed{seed}_{split}.pt"
    path = os.path.join(data_dir, filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Please make sure you ran 'generate_data.py' and the data is in '{data_dir}'."
        )
    
    # [FIX]: Add weights_only=False here as well
    data = torch.load(path, weights_only=False)
    return data["sequences"], data["weights"]

def run_experiment(p: int,
                  noise_std: float,
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
    Run experiment for a single AR order p and noise level.

    Following the paper: train on clean data, test on noisy data.

    Args:
        p: AR order
        noise_std: Noise standard deviation for test data
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
    print(f"Running experiment for p={p}, noise={noise_std}, seed={seed}")
    print(f"{'='*60}")

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir = 'data/data_cache'
    
    train_sequences, train_weights = load_dataset(data_dir, p, seed, "train")
    val_sequences, val_weights     = load_dataset(data_dir, p, seed, "val")
    test_sequences_clean, test_weights = load_dataset(data_dir, p, seed, "test")

    # --- ADD NOISE TO TEST DATA ---
    if not torch.is_tensor(test_sequences_clean):
        test_sequences_clean = torch.tensor(test_sequences_clean, dtype=torch.float32)
    else:
        test_sequences_clean = test_sequences_clean.float()

    if noise_std > 0.0:
        # print(f"  Adding observation noise (std={noise_std}) to test set...")
        noise = torch.randn_like(test_sequences_clean) * noise_std
        test_sequences = test_sequences_clean + noise
    else:
        test_sequences = test_sequences_clean

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
        model_save_dir = os.path.join(save_dir, f"p{p}_noise{noise_std}_seed{seed}")
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
    print(f"Training model on clean data...")
    history = trainer.train(verbose=True)

    # Create baselines
    oracle = OraclePredictor(test_weights[0])
    ols = OLSPredictor(p)

    # Evaluate on noisy test data
    print(f"Evaluating on noisy test data (σ={noise_std})...")
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

    # Add training history and noise level
    metrics['training_history'] = history
    metrics['noise_std'] = noise_std

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for p={p}, noise={noise_std}, seed={seed}")
    print(f"{'='*60}")
    print(f"1-step MSE: {metrics['mse_1step']:.6f}")
    print(f"1-step Oracle MSE: {metrics['mse_1step_oracle']:.6f}")
    print(f"1-step Relative Error: {metrics['rel_error_1step']:.4f}")
    print(f"10-step Rollout MSE: {metrics['mse_rollout']:.6f}")
    print(f"10-step Relative Error: {metrics['rel_error_rollout']:.4f}")

    # Save results
    if save_dir is not None:
        results_path = os.path.join(model_save_dir, 'metrics.json')
        with open(results_path, 'w') as f:
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


def aggregate_results(all_results: dict, p_values: list, noise_levels: list) -> dict:
    """
    Aggregate results across seeds.

    Args:
        all_results: Dictionary mapping (p, noise, seed) -> metrics
        p_values: List of p values
        noise_levels: List of noise levels

    Returns:
        Aggregated metrics nested as {noise: {p: metrics}}
    """
    aggregated = {}

    for noise in noise_levels:
        aggregated[noise] = {}
        for p in p_values:
            # Gather metrics for this (noise, p) across seeds
            metrics_list = [all_results[key] for key in all_results.keys()
                          if key[0] == p and key[1] == noise]

            if len(metrics_list) == 0:
                continue

            # Compute mean and std
            aggregated[noise][p] = {}
            for key in ['mse_1step', 'mse_1step_oracle', 'mse_1step_ols',
                       'mse_rollout', 'mse_rollout_oracle', 'mse_rollout_ols',
                       'rel_error_1step', 'rel_error_rollout']:
                values = [m[key] for m in metrics_list]
                aggregated[noise][p][key] = np.mean(values)
                aggregated[noise][p][f'{key}_std'] = np.std(values)

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='H2: Noise Robustness Experiments')
    parser.add_argument('--p_values', type=int, nargs='+', default=[1, 2, 5, 10],
                       help='AR orders to test')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.1, 0.3, 0.5],
                       help='Noise standard deviations to test')
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
    parser.add_argument('--save_dir', type=str, default='results/h2_noise',
                       help='Directory to save results')

    args = parser.parse_args()

    print("="*60)
    print("H2: NOISE ROBUSTNESS EXPERIMENTS")
    print("="*60)
    print(f"p values: {args.p_values}")
    print(f"Noise levels: {args.noise_levels}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Run experiments
    all_results = {}

    for p in args.p_values:
        for noise in args.noise_levels:
            for seed in range(args.seeds):
                metrics = run_experiment(
                    p=p,
                    noise_std=noise,
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
                all_results[(p, noise, seed)] = metrics

    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)

    aggregated = aggregate_results(all_results, args.p_values, args.noise_levels)

    # Save aggregated results
    aggregated_path = os.path.join(args.save_dir, 'aggregated_results.json')
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated results to {aggregated_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for noise in args.noise_levels:
        print(f"\nNoise σ = {noise}:")
        for p in args.p_values:
            if noise not in aggregated or p not in aggregated[noise]:
                continue
            print(f"  p={p}: rel_error = {aggregated[noise][p]['rel_error_1step']:.4f} ± {aggregated[noise][p]['rel_error_1step_std']:.4f}")

    # Plot results
    print("\nGenerating plots...")
    plot_path = os.path.join(args.save_dir, 'noise_robustness.png')
    plot_noise_robustness(aggregated, save_path=plot_path)

    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()