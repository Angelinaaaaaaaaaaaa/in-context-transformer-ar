"""
Zero-mean AR(p) process generation with stability checks.

Key design:
- Stable AR(p) dynamics via companion-matrix spectral radius control
- Lower lags are stronger, higher lags are weaker (more realistic AR structure)
- Trajectories are non-degenerate even when noise_std = 0
"""

import numpy as np
from typing import List, Tuple, Optional


# =========================
# 1. Linear-algebra helpers
# =========================

def companion_matrix(weights: List[np.ndarray]) -> np.ndarray:
    """Construct the companion matrix for an AR(p) process."""
    p = len(weights)
    if p == 0:
        raise ValueError("At least one weight matrix is required (p >= 1).")

    d = weights[0].shape[0]
    C = np.zeros((p * d, p * d), dtype=np.float64)

    # first block row: [W1 W2 ... Wp]
    for i, W in enumerate(weights):
        if W.shape != (d, d):
            raise ValueError(f"All weight matrices must be ({d}, {d}), got {W.shape}.")
        C[:d, i * d:(i + 1) * d] = W

    # sub-diagonal identity blocks
    if p > 1:
        C[d:, :d * (p - 1)] = np.eye(d * (p - 1), dtype=np.float64)

    return C


def check_stability(weights: List[np.ndarray],
                    max_spectral_radius: float = 0.95) -> bool:
    """Check stability of an AR(p) process via spectral radius of the companion matrix."""
    C = companion_matrix(weights)
    eigenvalues = np.linalg.eigvals(C)
    spectral_radius = np.max(np.abs(eigenvalues))
    return spectral_radius < max_spectral_radius


# =========================
# 2. Weight generation
# =========================

def generate_stable_ar_weights(
    p: int,
    d: int,
    max_attempts: int = 100,
    base_scale: float = 0.5,
    lag_decay: float = 0.7,
    max_spectral_radius: float = 0.95,
) -> List[np.ndarray]:
    """
    Generate stable AR(p) weights where lower-order lags are stronger.

    - W_1 has largest variance
    - W_k variance decays roughly like lag_decay^(k-1)
    - If the resulting companion matrix is unstable, we iteratively shrink all weights
    """
    if p <= 0:
        raise ValueError("AR order p must be >= 1.")
    if d <= 0:
        raise ValueError("Dimension d must be >= 1.")

    # Base std for lag 1
    base_std0 = base_scale / np.sqrt(d)

    # Initial random weights with decaying lag strength
    weights: List[np.ndarray] = []
    for k in range(p):
        std_k = base_std0 * (lag_decay ** k)  # k=0 -> strongest, k increases -> weaker
        W = np.random.randn(d, d) * std_k
        weights.append(W.astype(np.float64))

    # Iteratively shrink if unstable
    for _ in range(max_attempts):
        C = companion_matrix(weights)
        eigenvalues = np.linalg.eigvals(C)
        rho = np.max(np.abs(eigenvalues))

        if rho < max_spectral_radius:
            # Cast to float32 for consistency with sequences
            return [W.astype(np.float32) for W in weights]

        # Shrink all weights uniformly with a small safety margin
        shrink = max_spectral_radius / (rho + 1e-8)
        shrink *= 0.9
        for k in range(p):
            weights[k] *= shrink

    raise RuntimeError("Could not generate stable AR weights after max_attempts.")


# =========================
# 3. Sequence generation
# =========================

def generate_ar_sequence(
    weights: List[np.ndarray],
    T: int,
    noise_std: float = 1.0,
    initial_states: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a sequence from a zero-mean AR(p) process.

    Model:
        s_t = sum_{i=1}^p W_i s_{t-i} + eps_t
    where eps_t ~ N(0, noise_std^2 I) if noise_std > 0.
    """
    if seed is not None:
        np.random.seed(seed)

    p = len(weights)
    d = weights[0].shape[0]
    if T <= p:
        raise ValueError(f"Sequence length T={T} must be greater than p={p}.")

    # Burn-in only when there is process noise
    burn_in = 500 if noise_std > 0.0 else 0
    total_T = T + burn_in
    sequence = np.zeros((total_T, d), dtype=np.float32)

    if initial_states is None:
        # Non-degenerate initialization even if noise_std = 0
        sequence[:p] = np.random.randn(p, d).astype(np.float32)
    else:
        if initial_states.shape != (p, d):
            raise ValueError(f"initial_states must have shape ({p}, {d}), got {initial_states.shape}.")
        sequence[:p] = initial_states.astype(np.float32)

    for t in range(p, total_T):
        s_t = np.zeros(d, dtype=np.float32)
        for i, W in enumerate(weights):
            s_t += (W @ sequence[t - i - 1]).astype(np.float32)
        if noise_std > 0.0:
            s_t += np.random.randn(d).astype(np.float32) * noise_std
        sequence[t] = s_t

    return sequence[burn_in:]


# =========================
# 4. Dataset generation
# =========================

def generate_ar_dataset(
    n_sequences: int,
    p: int,
    d: int,
    T: int,
    noise_std: float = 1.0,
    same_dynamics: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """
    Generate a dataset of AR(p) sequences.

    If same_dynamics=True:
        - All sequences share the same weights (single AR system)
    Else:
        - Each sequence has its own weights (meta-learning style)
    """
    if seed is not None:
        np.random.seed(seed)

    sequences = np.zeros((n_sequences, T, d), dtype=np.float32)
    weights_list: List[List[np.ndarray]] = []

    if same_dynamics:
        base_weights = generate_stable_ar_weights(p, d, max_spectral_radius=0.95)
        for i in range(n_sequences):
            seq_seed = None if seed is None else seed + i
            seq = generate_ar_sequence(
                base_weights,
                T=T,
                noise_std=noise_std,
                initial_states=None,
                seed=seq_seed,
            )
            sequences[i] = seq
            weights_list.append([W.copy() for W in base_weights])
    else:
        for i in range(n_sequences):
            weights = generate_stable_ar_weights(p, d, max_spectral_radius=0.95)
            seq_seed = None if seed is None else seed + i
            seq = generate_ar_sequence(
                weights,
                T=T,
                noise_std=noise_std,
                initial_states=None,
                seed=seq_seed,
            )
            sequences[i] = seq
            weights_list.append([W.copy() for W in weights])

    return sequences, weights_list


# =========================
# 5. AR(p) least-squares fit
# =========================

def compute_ar_fit_loss(sequence: np.ndarray, p: int) -> Tuple[float, List[np.ndarray]]:
    """
    Fit a zero-intercept AR(p) model to a single sequence and compute MSE.
    """
    T, d = sequence.shape
    if T <= p:
        raise ValueError(f"Sequence length {T} must be greater than AR order {p}.")

    X_list = []
    Y_list = []

    for t in range(p, T):
        x_t = np.concatenate([sequence[t - i - 1] for i in range(p)])
        X_list.append(x_t)
        Y_list.append(sequence[t])

    X = np.array(X_list)
    Y = np.array(Y_list)

    W_flat, *_ = np.linalg.lstsq(X, Y, rcond=None)
    Y_pred = X @ W_flat
    loss = np.mean((Y - Y_pred) ** 2)

    fitted_weights: List[np.ndarray] = []
    for i in range(p):
        W_i_T = W_flat[i * d:(i + 1) * d, :]
        W_i = W_i_T.T
        fitted_weights.append(W_i)

    return loss, fitted_weights


if __name__ == "__main__":
    # Simple sanity check
    p_test, d_test, T_test = 5, 5, 200
    print(f"Testing AR({p_test}) process with dim={d_test}...")

    weights = generate_stable_ar_weights(p_test, d_test, max_spectral_radius=0.98)

    seq_clean = generate_ar_sequence(weights, T_test, noise_std=0.0)
    seq_noisy = generate_ar_sequence(weights, T_test, noise_std=1.0)

    print("\n[Zero Noise]")
    print(f"Mean: {np.mean(seq_clean):.4f}")
    print(f"Std:  {np.std(seq_clean):.4f}")
    print(f"Max Abs: {np.max(np.abs(seq_clean)):.4f}")

    print("\n[Noise Std=1.0]")
    print(f"Mean: {np.mean(seq_noisy):.4f}")
    print(f"Std:  {np.std(seq_noisy):.4f}")

    loss_clean, _ = compute_ar_fit_loss(seq_clean, p_test)
    loss_noisy, _ = compute_ar_fit_loss(seq_noisy, p_test)

    print(f"\nFit MSE (clean): {loss_clean:.6f}")
    print(f"Fit MSE (noisy): {loss_noisy:.6f}")
