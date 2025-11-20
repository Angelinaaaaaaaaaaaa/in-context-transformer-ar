import torch
import numpy as np
import argparse
import os
import json
from typing import Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import generate_ar_dataset
from models import GPTModel, OraclePredictor, OLSPredictor
from training import Trainer, ARDataset, evaluate_model
from analysis import plot_noise_robustness


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
    print(f"\n{'=' * 60}")
    print(f"Running experiment for p={p}, noise={noise_std}, seed={seed}")
    print(f"{'=' * 60}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Generating clean training data...")
    train_sequences, train_weights = generate_ar_dataset(
        n_sequences=n_train,
        p=p,
        d=d,
        T=T,
        noise_std=0.0,
        same_dynamics=True,
        seed=seed,
    )
    print("Generating clean validation data...")
    val_sequences, val_weights = generate_ar_dataset(
        n_sequences=n_val,
        p=p,
        d=d,
        T=T,
        noise_std=0.0,
        same_dynamics=True,
        seed=seed + 1,
    )
    print(f"Generating noisy test data (σ={noise_std})...")
    test_sequences, test_weights = generate_ar_dataset(
        n_sequences=n_test,
        p=p,
        d=d,
        T=T,
        noise_std=noise_std,
        same_dynamics=True,
        seed=seed + 2,
    )
    train_dataset = ARDataset(train_sequences)
    val_dataset = ARDataset(val_sequences)
    model = GPTModel(
        d_input=d,
        max_seq_len=T,
        dropout=0.1,
    )
    print(f"Model parameters: {sum(p_.numel() for p_ in model.parameters()):,}")
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
        batch_size=64,
        max_epochs=100,
        patience=10,
        device=device,
        save_dir=model_save_dir,
    )
    print("Training model on clean data...")
    history = trainer.train(verbose=True)
    oracle = OraclePredictor(test_weights[0])
    ols = OLSPredictor(p)
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
        device=device,
    )
    metrics["training_history"] = history
    metrics["noise_std"] = noise_std
    print(f"\n{'=' * 60}")
    print(f"Results for p={p}, noise={noise_std}, seed={seed}")
    print(f"{'=' * 60}")
    print(f"1-step MSE: {metrics['mse_1step']:.6f}")
    print(f"1-step Oracle MSE: {metrics['mse_1step_oracle']:.6f}")
    print(f"1-step Relative Error: {metrics['rel_error_1step']:.4f}")
    print(f"10-step Rollout MSE: {metrics['mse_rollout']:.6f}")
    print(f"10-step Relative Error: {metrics['rel_error_rollout']:.4f}")
    if save_dir is not None:
        results_path = os.path.join(model_save_dir, "metrics.json")
        with open(results_path, "w") as f:
            serializable_metrics = {k: v for k, v in metrics.items() if k != "training_history"}
            serializable_metrics["training_history"] = {
                "train_losses": [float(x) for x in history["train_losses"]],
                "val_losses": [float(x) for x in history["val_losses"]],
                "best_val_loss": float(history["best_val_loss"]),
                "n_epochs": int(history["n_epochs"]),
            }
            json.dump(serializable_metrics, f, indent=2)
        print(f"Saved results to {results_path}")
    return metrics


def aggregate_results(all_results: dict, p_values: list, noise_levels: list) -> dict:
    aggregated = {}
    for noise in noise_levels:
        aggregated[noise] = {}
        for p in p_values:
            metrics_list = [
                all_results[key]
                for key in all_results.keys()
                if key[0] == p and key[1] == noise
            ]
            if len(metrics_list) == 0:
                continue
            aggregated[noise][p] = {}
            for key in [
                "mse_1step",
                "mse_1step_oracle",
                "mse_1step_ols",
                "mse_rollout",
                "mse_rollout_oracle",
                "mse_rollout_ols",
                "rel_error_1step",
                "rel_error_rollout",
            ]:
                values = [m[key] for m in metrics_list]
                aggregated[noise][p][key] = np.mean(values)
                aggregated[noise][p][f"{key}_std"] = np.std(values)
    return aggregated


def main():
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
        help="Number of training sequences",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=5000,
        help="Number of validation sequences",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=10000,
        help="Number of test sequences",
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
        for p in args.p_values:
            if noise not in aggregated or p not in aggregated[noise]:
                continue
            print(
                f"  p={p}: rel_error = {aggregated[noise][p]['rel_error_1step']:.4f} ± {aggregated[noise][p]['rel_error_1step_std']:.4f}"
            )
    print("\nGenerating plots...")
    plot_path = os.path.join(args.save_dir, "noise_robustness.png")
    plot_noise_robustness(aggregated, save_path=plot_path)
    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
