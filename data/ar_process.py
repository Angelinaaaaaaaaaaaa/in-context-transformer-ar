"""
Standard Zero-Mean AR(p) process generation with stability checks.

Key features:
- Stable AR(p) dynamics via companion-matrix spectral radius control
- All lags have comparable strength (no artificial decay)
- Zero-mean trajectories that are non-degenerate even when noise_std = 0
"""

import numpy as np
from typing import List, Tuple, Optional


# =========================
# 1. Linear-algebra helpers
# =========================

def companion_matrix(weights: List[np.ndarray]) -> np.ndarray:
    """
    Construct the companion matrix for an AR(p) process.
    """
    p = len(weights)
    if p == 0:
        raise ValueError("At least one weight matrix is required (p >= 1).")

    d = weights[0].shape[0]
    C = np.zeros((p * d, p * d), dtype=np.float64)

    for i, W in enumerate(weights):
        if W.shape != (d, d):
            raise ValueError(f"All weight matrices must be ({d}, {d}), got {W.shape}.")
        C[:d, i * d:(i + 1) * d] = W

    if p > 1:
        C[d:, :d * (p - 1)] = np.eye(d * (p - 1), dtype=np.float64)

    return C


def check_stability(weights: List[np.ndarray],
                    max_spectral_radius: float = 0.95) -> bool:
    """
    Check stability of an AR(p) process via the spectral radius.
    """
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
    max_attempts: int = 1000,
    scale: float = 0.5,
    max_spectral_radius: float = 0.95,
) -> List[np.ndarray]:
    """
    Generate stable AR(p) weights with comparable lag strengths.
    """
    if p <= 0:
        raise ValueError("AR order p must be >= 1.")
    if d <= 0:
        raise ValueError("Dimension d must be >= 1.")

    base_std = 1.0 / np.sqrt(d * p)
    weights: List[np.ndarray] = []

    for _ in range(p):
        W = np.random.randn(d, d) * base_std
        weights.append(W)

    C = companion_matrix(weights)
    eigenvalues = np.linalg.eigvals(C)
    current_radius = np.max(np.abs(eigenvalues))

    if current_radius <= 0:
        current_radius = 1e-6

    target_radius = max_spectral_radius
    lambda_factor = target_radius / current_radius

    final_weights: List[np.ndarray] = []
    for k in range(p):
        scale_k = lambda_factor ** (k + 1)
        final_weights.append(weights[k] * scale_k)

    return final_weights


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
        s_t = Σ_{i=1}^p W_i s_{t-i} + ε_t
    where ε_t ~ N(0, noise_std^2 I) if noise_std > 0.
    """
    if seed is not None:
        np.random.seed(seed)

    p = len(weights)
    d = weights[0].shape[0]
    if T <= p:
        raise ValueError(f"Sequence length T={T} must be greater than p={p}.")

    if noise_std > 0.0:
        burn_in = 500
    else:
        burn_in = 0

    total_T = T + burn_in
    sequence = np.zeros((total_T, d), dtype=np.float32)

    if initial_states is None:
        init_std = 1.0
        sequence[:p] = np.random.randn(p, d).astype(np.float32) * init_std
    else:
        if initial_states.shape != (p, d):
            raise ValueError(
                f"initial_states must have shape ({p}, {d}), "
                f"got {initial_states.shape}."
            )
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


# =========================
# 6. Verification script
# =========================

if __name__ == "__main__":
    p_test, d_test, T_test = 5, 5, 200
    print(f"Testing Zero-Mean AR({p_test}) process with dim={d_test}...")

    weights = generate_stable_ar_weights(p_test, d_test, max_spectral_radius=0.98)

    seq_clean = generate_ar_sequence(weights, T_test, noise_std=0.0)
    seq_noisy = generate_ar_sequence(weights, T_test, noise_std=1.0)

    print(f"Clean mean: {np.mean(seq_clean):.4f}, std: {np.std(seq_clean):.4f}")
    print(f"Noisy mean: {np.mean(seq_noisy):.4f}, std: {np.std(seq_noisy):.4f}")

    loss_clean, _ = compute_ar_fit_loss(seq_clean, p_test)
    loss_noisy, _ = compute_ar_fit_loss(seq_noisy, p_test)
    print(f"Fit loss (clean): {loss_clean:.4f}")
    print(f"Fit loss (noisy): {loss_noisy:.4f}")
