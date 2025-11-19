"""
Attention pattern analysis for investigating lag specialization (H3).
"""
import torch
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
import warnings


def aggregate_attention_by_lag(attention_weights: torch.Tensor,
                               max_lag: int = 10) -> np.ndarray:
    """
    Aggregate attention weights by lag distance.

    Args:
        attention_weights: Attention weights of shape (batch, n_heads, seq_len, seq_len)
        max_lag: Maximum lag to consider

    Returns:
        Array of shape (n_heads, max_lag) with average attention to each lag
    """
    batch_size, n_heads, seq_len, _ = attention_weights.shape

    # Average over batch
    attn = attention_weights.mean(dim=0).cpu().numpy()  # (n_heads, seq_len, seq_len)

    # For each head, compute average attention to each lag
    lag_attention = np.zeros((n_heads, max_lag))

    for head in range(n_heads):
        for lag in range(1, max_lag + 1):
            # For each position, look at attention to position at 'lag' distance back
            lag_weights = []
            for pos in range(lag, seq_len):
                lag_weights.append(attn[head, pos, pos - lag])

            if len(lag_weights) > 0:
                lag_attention[head, lag - 1] = np.mean(lag_weights)

    return lag_attention


def analyze_head_specialization(model,
                               sequences: torch.Tensor,
                               layer_idx: int = -1,
                               max_lag: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze whether attention heads specialize to specific lags.

    Args:
        model: Trained model
        sequences: Sequences to analyze of shape (n_sequences, seq_len, d)
        layer_idx: Layer index to analyze (-1 for last layer)
        max_lag: Maximum lag to analyze

    Returns:
        - lag_attention: Average attention by lag for each head, shape (n_heads, max_lag)
        - head_dominant_lag: Dominant lag for each head, shape (n_heads,)
    """
    model.eval()
    device = next(model.parameters()).device
    sequences = sequences.to(device)

    with torch.no_grad():
        # Forward pass with attention
        _, all_attention = model(sequences, return_attention=True)

        # Get attention from specified layer
        if layer_idx == -1:
            layer_idx = len(all_attention) - 1
        attention_weights = all_attention[layer_idx]  # (batch, n_heads, seq_len, seq_len)

    # Aggregate by lag
    lag_attention = aggregate_attention_by_lag(attention_weights, max_lag)

    # Find dominant lag for each head
    head_dominant_lag = np.argmax(lag_attention, axis=1) + 1  # +1 because lag is 1-indexed

    return lag_attention, head_dominant_lag


def cluster_attention_heads(lag_attention: np.ndarray,
                           n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster attention heads based on their lag attention patterns.

    Args:
        lag_attention: Lag attention matrix of shape (n_heads, max_lag)
        n_clusters: Number of clusters

    Returns:
        - cluster_labels: Cluster assignment for each head
        - cluster_centers: Cluster centers of shape (n_clusters, max_lag)
    """
    n_heads, max_lag = lag_attention.shape

    if n_heads < n_clusters:
        warnings.warn(f"Number of heads ({n_heads}) < n_clusters ({n_clusters})")
        n_clusters = n_heads

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(lag_attention)
    cluster_centers = kmeans.cluster_centers_

    return cluster_labels, cluster_centers


def ablate_attention_heads(model,
                          sequences: torch.Tensor,
                          heads_to_ablate: List[Tuple[int, int]],
                          target: torch.Tensor) -> float:
    """
    Ablate specific attention heads and measure performance drop.

    Args:
        model: Model to ablate
        sequences: Input sequences of shape (batch, seq_len, d)
        heads_to_ablate: List of (layer_idx, head_idx) tuples to ablate
        target: Target values of shape (batch, d) for next-step prediction

    Returns:
        MSE after ablation
    """
    model.eval()
    device = next(model.parameters()).device
    sequences = sequences.to(device)
    target = target.to(device)

    # Store original forward method
    original_forwards = {}

    # Modify attention blocks to zero out specific heads
    for layer_idx, head_idx in heads_to_ablate:
        block = model.blocks[layer_idx]
        attention_module = block.attention

        # Store original forward
        if layer_idx not in original_forwards:
            original_forwards[layer_idx] = attention_module.forward

        # Create ablation wrapper
        def create_ablated_forward(original_forward, head_to_ablate):
            def ablated_forward(x, mask=None, return_attention=False):
                output, attn_weights = original_forward(x, mask, return_attention=True)

                # Zero out specific head in output
                batch_size, seq_len, d_model = output.shape
                n_heads = attention_module.n_heads
                d_head = d_model // n_heads

                # Reshape to access individual heads
                output_heads = output.view(batch_size, seq_len, n_heads, d_head)
                output_heads[:, :, head_to_ablate, :] = 0
                output = output_heads.view(batch_size, seq_len, d_model)

                if return_attention:
                    return output, attn_weights
                return output, None

            return ablated_forward

        # Replace forward method
        attention_module.forward = create_ablated_forward(original_forwards[layer_idx], head_idx)

    # Forward pass with ablation
    with torch.no_grad():
        output, _ = model(sequences)
        predictions = output[:, -1, :]  # Last position
        mse = torch.mean((predictions - target) ** 2).item()

    # Restore original forward methods
    for layer_idx in original_forwards:
        model.blocks[layer_idx].attention.forward = original_forwards[layer_idx]

    return mse


def compute_head_importance(model,
                           sequences: torch.Tensor,
                           target: torch.Tensor,
                           baseline_mse: float) -> np.ndarray:
    """
    Compute importance of each attention head by ablation.

    Args:
        model: Model to analyze
        sequences: Input sequences
        target: Target values
        baseline_mse: MSE without ablation

    Returns:
        Importance matrix of shape (n_layers, n_heads) measuring MSE increase
    """
    n_layers = model.n_layers
    n_heads = model.n_heads

    importance = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Ablate this head
            ablated_mse = ablate_attention_heads(
                model, sequences, [(layer_idx, head_idx)], target
            )

            # Importance = increase in MSE
            importance[layer_idx, head_idx] = ablated_mse - baseline_mse

    return importance


def analyze_lag_specific_ablation(model,
                                  sequences: torch.Tensor,
                                  target: torch.Tensor,
                                  lag_attention: np.ndarray,
                                  layer_idx: int = -1) -> Dict[int, float]:
    """
    Ablate heads specialized to each lag and measure performance drop.

    This tests H3: whether ablating lag-specific heads produces selective drops.

    Args:
        model: Model to analyze
        sequences: Input sequences
        target: Target values
        lag_attention: Lag attention matrix from analyze_head_specialization
        layer_idx: Layer to ablate (-1 for last layer)

    Returns:
        Dictionary mapping lag -> MSE increase when ablating heads for that lag
    """
    if layer_idx == -1:
        layer_idx = model.n_layers - 1

    n_heads = lag_attention.shape[0]

    # Baseline MSE (no ablation)
    model.eval()
    device = next(model.parameters()).device
    sequences = sequences.to(device)
    target = target.to(device)

    with torch.no_grad():
        output, _ = model(sequences)
        predictions = output[:, -1, :]
        baseline_mse = torch.mean((predictions - target) ** 2).item()

    # For each lag, find specialized heads and ablate them
    lag_results = {}
    max_lag = lag_attention.shape[1]

    for lag in range(1, max_lag + 1):
        # Find heads most specialized to this lag
        lag_idx = lag - 1
        lag_specialization = lag_attention[:, lag_idx]

        # Get top 2 heads for this lag (or all if < 2)
        n_top = min(2, n_heads)
        top_heads = np.argsort(lag_specialization)[-n_top:]

        # Ablate these heads
        heads_to_ablate = [(layer_idx, int(h)) for h in top_heads]
        ablated_mse = ablate_attention_heads(model, sequences, heads_to_ablate, target)

        lag_results[lag] = ablated_mse - baseline_mse

    return lag_results


if __name__ == "__main__":
    # Test attention analysis
    print("Testing attention analysis...")

    from models.transformer import GPTModel
    from data.ar_process import generate_ar_dataset

    # Generate test data
    p, d, T = 3, 5, 100
    sequences, weights = generate_ar_dataset(
        n_sequences=32, p=p, d=d, T=T, noise_std=0.1, same_dynamics=False, seed=42
    )
    sequences = torch.tensor(sequences, dtype=torch.float32)

    # Create model
    model = GPTModel(
        d_input=d,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=T,
        dropout=0.0  # No dropout for analysis
    )

    print(f"Model created with {model.n_layers} layers and {model.n_heads} heads")

    # Analyze head specialization
    print("\nAnalyzing head specialization...")
    lag_attention, dominant_lags = analyze_head_specialization(
        model, sequences, layer_idx=-1, max_lag=10
    )

    print(f"Lag attention shape: {lag_attention.shape}")
    print(f"Dominant lags: {dominant_lags}")

    # Cluster heads
    print("\nClustering attention heads...")
    cluster_labels, cluster_centers = cluster_attention_heads(lag_attention, n_clusters=3)
    print(f"Cluster labels: {cluster_labels}")
    print(f"Cluster centers shape: {cluster_centers.shape}")

    # Test ablation
    print("\nTesting head ablation...")
    context_len = 70
    context = sequences[:, :context_len, :]
    target = sequences[:, context_len, :]

    # Baseline
    model.eval()
    with torch.no_grad():
        output, _ = model(context)
        baseline_pred = output[:, -1, :]
        baseline_mse = torch.mean((baseline_pred - target) ** 2).item()

    print(f"Baseline MSE: {baseline_mse:.6f}")

    # Ablate head 0 in layer 0
    ablated_mse = ablate_attention_heads(model, context, [(0, 0)], target)
    print(f"MSE after ablating layer 0, head 0: {ablated_mse:.6f}")
    print(f"MSE increase: {ablated_mse - baseline_mse:.6f}")
