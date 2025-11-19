"""
Visualization utilities for results and attention patterns.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


def set_style():
    """Set consistent plot style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16


def plot_scaling_results(results: Dict[int, Dict],
                         save_path: Optional[str] = None):
    """
    Plot H1 scaling results: performance vs. AR order p.

    Args:
        results: Dictionary mapping p -> metrics dict
        save_path: Path to save figure
    """
    set_style()

    p_values = sorted(results.keys())

    # Extract metrics
    mse_1step = [results[p]['mse_1step'] for p in p_values]
    mse_1step_oracle = [results[p]['mse_1step_oracle'] for p in p_values]
    mse_1step_ols = [results[p]['mse_1step_ols'] for p in p_values]

    mse_rollout = [results[p]['mse_rollout'] for p in p_values]
    mse_rollout_oracle = [results[p]['mse_rollout_oracle'] for p in p_values]

    rel_error_1step = [results[p]['rel_error_1step'] for p in p_values]
    rel_error_rollout = [results[p]['rel_error_rollout'] for p in p_values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Absolute MSE
    axes[0].plot(p_values, mse_1step, 'o-', label='Transformer (1-step)', linewidth=2)
    axes[0].plot(p_values, mse_rollout, 's-', label='Transformer (10-step)', linewidth=2)
    axes[0].plot(p_values, mse_1step_oracle, 'x--', label='Oracle (1-step)', linewidth=2, alpha=0.7)
    axes[0].plot(p_values, mse_rollout_oracle, '+--', label='Oracle (10-step)', linewidth=2, alpha=0.7)
    axes[0].plot(p_values, mse_1step_ols, 'd--', label='OLS (1-step)', linewidth=2, alpha=0.7)

    axes[0].set_xlabel('AR Order (p)')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('A) Prediction Error vs. AR Order')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Panel B: Relative error
    axes[1].plot(p_values, rel_error_1step, 'o-', label='1-step', linewidth=2)
    axes[1].plot(p_values, rel_error_rollout, 's-', label='10-step', linewidth=2)
    axes[1].axhline(y=2.0, color='r', linestyle='--', label='2× Oracle', alpha=0.5)

    axes[1].set_xlabel('AR Order (p)')
    axes[1].set_ylabel('Relative Error (vs. Oracle)')
    axes[1].set_title('B) Relative Performance vs. AR Order')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_noise_robustness(results: Dict[float, Dict[int, Dict]],
                          save_path: Optional[str] = None):
    """
    Plot H2 noise robustness results: performance vs. noise level for different p.

    Args:
        results: Dictionary mapping noise_std -> (p -> metrics dict)
        save_path: Path to save figure
    """
    set_style()

    noise_levels = sorted(results.keys())

    # Get p values from first noise level
    p_values = sorted(results[noise_levels[0]].keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: MSE vs. noise for different p
    for p in p_values:
        mse_values = [results[noise][p]['mse_1step'] for noise in noise_levels]
        axes[0].plot(noise_levels, mse_values, 'o-', label=f'p={p}', linewidth=2)

    axes[0].set_xlabel('Noise Standard Deviation (σ)')
    axes[0].set_ylabel('1-Step MSE')
    axes[0].set_title('A) Noise Sensitivity by AR Order')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel B: Relative error vs. noise
    for p in p_values:
        rel_errors = [results[noise][p]['rel_error_1step'] for noise in noise_levels]
        axes[1].plot(noise_levels, rel_errors, 'o-', label=f'p={p}', linewidth=2)

    axes[1].axhline(y=2.0, color='r', linestyle='--', label='2× Oracle', alpha=0.5)
    axes[1].set_xlabel('Noise Standard Deviation (σ)')
    axes[1].set_ylabel('Relative Error (vs. Oracle)')
    axes[1].set_title('B) Relative Performance vs. Noise')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_attention_heatmap(lag_attention: np.ndarray,
                          dominant_lags: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None):
    """
    Plot attention heatmap showing head specialization to different lags.

    Args:
        lag_attention: Lag attention matrix of shape (n_heads, max_lag)
        dominant_lags: Dominant lag for each head
        save_path: Path to save figure
    """
    set_style()

    n_heads, max_lag = lag_attention.shape

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create heatmap
    im = ax.imshow(lag_attention, aspect='auto', cmap='viridis', interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(max_lag))
    ax.set_yticks(np.arange(n_heads))
    ax.set_xticklabels(np.arange(1, max_lag + 1))
    ax.set_yticklabels(np.arange(n_heads))

    ax.set_xlabel('Lag')
    ax.set_ylabel('Attention Head')
    ax.set_title('Attention Head Specialization by Lag')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Attention Weight')

    # Mark dominant lags
    if dominant_lags is not None:
        for head, lag in enumerate(dominant_lags):
            ax.plot(lag - 1, head, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_head_clustering(lag_attention: np.ndarray,
                        cluster_labels: np.ndarray,
                        cluster_centers: np.ndarray,
                        save_path: Optional[str] = None):
    """
    Plot clustering of attention heads based on lag patterns.

    Args:
        lag_attention: Lag attention matrix of shape (n_heads, max_lag)
        cluster_labels: Cluster assignment for each head
        cluster_centers: Cluster centers
        save_path: Path to save figure
    """
    set_style()

    n_clusters = len(cluster_centers)
    max_lag = lag_attention.shape[1]
    lags = np.arange(1, max_lag + 1)

    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 4))

    if n_clusters == 1:
        axes = [axes]

    for cluster_idx in range(n_clusters):
        ax = axes[cluster_idx]

        # Plot heads in this cluster
        head_indices = np.where(cluster_labels == cluster_idx)[0]
        for head_idx in head_indices:
            ax.plot(lags, lag_attention[head_idx], alpha=0.3, color='gray')

        # Plot cluster center
        ax.plot(lags, cluster_centers[cluster_idx], 'r-', linewidth=3, label='Cluster center')

        ax.set_xlabel('Lag')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Cluster {cluster_idx + 1} ({len(head_indices)} heads)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_ablation_results(lag_ablation_results: Dict[int, float],
                         save_path: Optional[str] = None):
    """
    Plot lag-specific ablation results for H3.

    Args:
        lag_ablation_results: Dictionary mapping lag -> MSE increase
        save_path: Path to save figure
    """
    set_style()

    lags = sorted(lag_ablation_results.keys())
    mse_increases = [lag_ablation_results[lag] for lag in lags]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(lags, mse_increases, alpha=0.7, color='steelblue', edgecolor='black')

    ax.set_xlabel('Lag')
    ax.set_ylabel('MSE Increase After Ablating Lag-Specific Heads')
    ax.set_title('Impact of Ablating Lag-Specialized Heads')
    ax.grid(True, alpha=0.3, axis='y')

    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_training_curves(history: Dict,
                        save_path: Optional[str] = None):
    """
    Plot training and validation loss curves.

    Args:
        history: Training history with 'train_losses' and 'val_losses'
        save_path: Path to save figure
    """
    set_style()

    epochs = np.arange(1, len(history['train_losses']) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, history['train_losses'], 'o-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_losses'], 's-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_ar1_linguistic_comparison(losses_ordered: np.ndarray,
                                  losses_shuffled: np.ndarray,
                                  save_path: Optional[str] = None):
    """
    Plot AR(1) fitting loss comparison for linguistic data (replication of Sander et al.).

    Args:
        losses_ordered: AR(1) losses for ordered sequences
        losses_shuffled: AR(1) losses for shuffled sequences
        save_path: Path to save figure
    """
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(losses_ordered, bins=30, alpha=0.6, label='Ordered (linguistic)', color='blue', density=True)
    ax.hist(losses_shuffled, bins=30, alpha=0.6, label='Shuffled (control)', color='red', density=True)

    ax.axvline(losses_ordered.mean(), color='blue', linestyle='--', linewidth=2, label=f'Ordered mean: {losses_ordered.mean():.4f}')
    ax.axvline(losses_shuffled.mean(), color='red', linestyle='--', linewidth=2, label=f'Shuffled mean: {losses_shuffled.mean():.4f}')

    ax.set_xlabel('AR(1) Fitting Loss')
    ax.set_ylabel('Density')
    ax.set_title('AR(1) Model Fits Linguistic Structure Better Than Random')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test plotting functions with dummy data
    print("Testing plotting functions...")

    set_style()

    # Test scaling plot
    print("\nTesting scaling plot...")
    results_scaling = {
        1: {'mse_1step': 0.1, 'mse_1step_oracle': 0.05, 'mse_1step_ols': 0.12,
            'mse_rollout': 0.2, 'mse_rollout_oracle': 0.1,
            'rel_error_1step': 2.0, 'rel_error_rollout': 2.0},
        2: {'mse_1step': 0.15, 'mse_1step_oracle': 0.07, 'mse_1step_ols': 0.18,
            'mse_rollout': 0.3, 'mse_rollout_oracle': 0.14,
            'rel_error_1step': 2.14, 'rel_error_rollout': 2.14},
        5: {'mse_1step': 0.25, 'mse_1step_oracle': 0.10, 'mse_1step_ols': 0.30,
            'mse_rollout': 0.5, 'mse_rollout_oracle': 0.2,
            'rel_error_1step': 2.5, 'rel_error_rollout': 2.5},
    }
    plot_scaling_results(results_scaling)

    # Test attention heatmap
    print("\nTesting attention heatmap...")
    lag_attention = np.random.rand(8, 10)
    dominant_lags = np.argmax(lag_attention, axis=1) + 1
    plot_attention_heatmap(lag_attention, dominant_lags)

    print("\nPlotting tests completed!")
