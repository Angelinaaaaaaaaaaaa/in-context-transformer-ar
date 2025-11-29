"""
H2: Noise Robustness Experiments
Test sensitivity to observation noise for different AR orders.

Hypothesis: For fixed context length and dimension, sensitivity to observation noise
increases with p (steeper error vs. σ curves for larger p).
"""
import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm  # noqa: F401

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import generate_ar_dataset  # noqa: F401
from models import GPTModel, OraclePredictor, OLSPredictor
from training import Trainer, ARDataset
from training.metrics import evaluate_model
from analysis import plot_noise_robustness


def load_dataset(data_dir: str, p: int, seed: int, split: str) -> Tuple[torch.Tensor, List[List[np.ndarray]]]:
    """
    Load pre-generated AR(p) data.

    Expects files: ar_p{p}_seed{seed}_{split}.pt with keys:
      - "sequences": (N, T, d)
      - "weights":   length-N list of AR weight lists
    """
    filename = f"ar_p{p}_seed{seed}_{split}.pt"
    path = os.path.join(data_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Please run the data generation script so that cached data "
            f"is placed in '{data_dir}'."
        )

    data = torch.load(path, weights_only=False)
    return data["sequences"], data["weights"]


def run_experiment(
    p: int,
    noise_std: float,
    d: int,
    T: int,
    context_len: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    device: str = "cpu",
    save_dir: Optional[str] = None,
) -> dict:
    """
    Run a single H2 experiment for AR order p and noise level σ.

    Clean-train / noisy-test:
    - train on clean data (σ_train = 0)
    - test on noisy data (σ_test = noise_std)
    """
    print(f"\n{'=' * 60}")
    print(f"Running H2 experiment for p={p}, noise={noise_std}, seed={seed}")
    print(f"{'=' * 60}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir = "data/data_cache"

    train_sequences, train_weights = load_dataset(data_dir, p, seed, "train")
    val_sequences, val_weights = load_dataset(data_dir, p, seed, "val")
    test_sequences_clean, test_weights = load_dataset(data_dir, p, seed, "test")

    if not torch.is_tensor(test_sequences_clean):
        test_sequences_clean = torch.as_tensor(test_sequences_clean, dtype=torch.float32)
    else:
        test_sequences_clean = test_sequences_clean.float()

    if noise_std > 0.0:
        noise = torch.randn_like(test_sequences_clean) * float(noise_std)
        test_sequences = test_sequences_clean + noise
    else:
        test_sequences = test_sequences_clean

    train_dataset = ARDataset(train_sequences)
    val_dataset = ARDataset(val_sequences)

    model = GPTModel(
        d_input=d,
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=1024,
        max_seq_len=T,
        dropout=0.1,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        save_dir=model_save_dir,
    )

    print("Training model on clean data...")
    history = trainer.train(verbose=True)

    oracle = OraclePredictor(test_weights[0])
    ols = OLSPredictor(p)

    print(f"Evaluating on noisy test data (σ={noise_std})...")
    test_sequences_tensor = torch.as_tensor(test_sequences, dtype=torch.float32)

    metrics = evaluate_model(
        model=model,
        test_sequences=test_sequences_tensor,
        test_weights=test_weights,
        context_len=context_len,
        p=p,
        oracle_predictor=oracle,
        ols_predictor=ols,
        device=device,
    )

    metrics["training_history"] = history
    metrics["noise_std"] = float(noise_std)

    print(f"\n{'=' * 60}")
    print(f"Results for p={p}, noise={noise_std}, seed={seed}")
    print(f"{'=' * 60}")
    print(f"1-step MSE (model):    {metrics['mse_1step']:.6f}")
    print(f"1-step MSE (oracle):   {metrics['mse_1step_oracle']:.6f}")
    print(f"1-step MSE (OLS):      {metrics['mse_1step_ols']:.6f}")
    print(f"1-step MSE (last):     {metrics['mse_1step_last']:.6f}")
    print(f"1-step rel error:      {metrics['rel_error_1step']:.4f}")
    print(f"10-step rollout MSE:   {metrics['mse_rollout']:.6f}")
    print(f"10-step rollout rel e: {metrics['rel_error_rollout']:.4f}")
    if metrics.get("spd") is not None:
        print(f"SPD (implicit vs true): {metrics['spd']:.6f}")
    if metrics.get("ilwd") is not None:
        print(f"ILWD (implicit vs OLS): {metrics['ilwd']:.6f}")

    if save_dir is not None:
        results_path = os.path.join(model_save_dir, "metrics.json")
        serializable_metrics = {
            k: v for k, v in metrics.items() if k != "training_history"
        }
        serializable_metrics["training_history"] = {
            "train_losses": [float(x) for x in history["train_losses"]],
            "val_losses": [float(x) for x in history["val_losses"]],
            "best_val_loss": float(history["best_val_loss"]),
            "n_epochs": int(history["n_epochs"]),
        }
        with open(results_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        print(f"Saved results to {results_path}")

    return metrics


def aggregate_results(
    all_results: dict,
    p_values: list,
    noise_levels: list,
) -> dict:
    """
    Aggregate results across seeds for each (p, σ) pair.

    Returns nested dict: {noise: {p: metrics}}.
    """
    aggregated: dict = {}
    metric_keys = [
        "mse_1step",
        "mse_1step_oracle",
        "mse_1step_ols",
        "mse_1step_last",
        "mse_rollout",
        "mse_rollout_oracle",
        "mse_rollout_ols",
        "mse_rollout_last",
        "rel_error_1step",
        "rel_error_rollout",
        "rel_error_1step_last",
        "rel_error_rollout_last",
        "spd",
        "ilwd",
    ]

    for noise in noise_levels:
        aggregated[noise] = {}
        for p in p_values:
            metrics_list = [
                all_results[key]
                for key in all_results.keys()
                if key[0] == p and key[1] == noise
            ]
            if not metrics_list:
                continue

            aggregated[noise][p] = {}
            for key in metric_keys:
                values = [m[key] for m in metrics_list if key in m and m[key] is not None]
                if not values:
                    continue
                aggregated[noise][p][key] = float(np.mean(values))
                aggregated[noise][p][f"{key}_std"] = float(np.std(values))

    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="H2: Noise Robustness Experiments")
    parser.add_argument(
        "--p_values",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10],
        help="AR orders to test",
    )
    parser.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.3, 0.5],
        help="Noise standard deviations to test",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=5,
        help="State dimension",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=100,
        help="Sequence length",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=70,
        help="Context window length",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=50000,
        help="Number of training sequences (for reference with cached data)",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=5000,
        help="Number of validation sequences (for reference with cached data)",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=10000,
        help="Number of test sequences (for reference with cached data)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/h2_noise",
        help="Directory to save results",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("H2: NOISE ROBUSTNESS EXPERIMENTS")
    print("=" * 60)
    print(f"p values: {args.p_values}")
    print(f"Noise levels: {args.noise_levels}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    all_results: dict = {}
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
                    save_dir=args.save_dir,
                )
                all_results[(p, noise, seed)] = metrics

    print("\n" + "=" * 60)
    print("AGGREGATING RESULTS")
    print("=" * 60)
    aggregated = aggregate_results(all_results, args.p_values, args.noise_levels)

    aggregated_path = os.path.join(args.save_dir, "aggregated_results.json")
    with open(aggregated_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated results to {aggregated_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for noise in args.noise_levels:
        print(f"\nNoise σ = {noise}:")
        if noise not in aggregated:
            continue
        for p in args.p_values:
            if p not in aggregated[noise]:
                continue
            row = aggregated[noise][p]
            if "rel_error_1step" in row:
                mean = row["rel_error_1step"]
                std = row["rel_error_1step_std"]
                print(f"  p={p}: 1-step rel error = {mean:.4f} ± {std:.4f}")

    print("\nGenerating plots...")
    plot_path = os.path.join(args.save_dir, "noise_robustness.png")
    plot_noise_robustness(aggregated, save_path=plot_path)

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
