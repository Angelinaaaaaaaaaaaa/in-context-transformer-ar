"""
AR(p) process generation with stability checks and non-zero mean.

Key design goals:
- Stable AR(p) dynamics via companion-matrix spectral radius control
- All lags have comparable strength (no artificial decay)
- Non-zero mean level so sequences are not clustered near 0
- Dataset can be made "challenging" for simple zero-mean AR fits
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings  # kept in case you want to warn on edge cases

# Global configuration: target mean level for the stationary distribution.
# Signatures must stay fixed, so we expose this via a module-level constant.
TARGET_MEAN_LEVEL: float = 10.0


# =========================
# 1. Linear-algebra helpers
# =========================

def companion_matrix(weights: List[np.ndarray]) -> np.ndarray:
    """
    Construct the companion matrix for an AR(p) process.

    Args:
        weights: List of weight matrices [W1, W2, ..., Wp], each (d, d).

    Returns:
        Companion matrix C of shape (p*d, p*d) such that the stacked state
        [s_t, s_{t-1}, ..., s_{t-p+1}] evolves linearly via C.
    """
    p = len(weights)
    if p == 0:
        raise ValueError("At least one weight matrix is required (p >= 1).")

    d = weights[0].shape[0]
    C = np.zeros((p * d, p * d))

    # First block row: [W1, W2, ..., Wp]
    for i, W in enumerate(weights):
        if W.shape != (d, d):
            raise ValueError(f"All weight matrices must be ({d}, {d}), got {W.shape}.")
        C[:d, i * d:(i + 1) * d] = W

    # Lower block rows: shift identity (stacked state shift)
    if p > 1:
        # This creates:
        # [ I_d  0   ...  0 ]
        # [ 0    I_d ...  0 ]
        # ...
        C[d:, :d * (p - 1)] = np.eye(d * (p - 1))

    return C


def check_stability(weights: List[np.ndarray],
                    max_spectral_radius: float = 0.95) -> bool:
    """
    Check stability of an AR(p) process via the spectral radius of its companion matrix.

    Args:
        weights: List of weight matrices [W1, W2, ..., Wp].
        max_spectral_radius: Maximum allowed spectral radius for stability.

    Returns:
        True if spectral radius < max_spectral_radius, else False.
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
    max_attempts: int = 1000,     # kept for API compatibility (unused)
    scale: float = 0.5,           # kept for API compatibility (we use internal base_std)
    max_spectral_radius: float = 0.95
) -> List[np.ndarray]:
    """
    Generate stable AR(p) weights with COMPARABLE lag strengths.

    Design:
    - All lags are sampled from the same distribution (no artificial decay).
    - We then compute the companion matrix spectral radius r_old.
    - To adjust to a target radius r_new, we scale A_k by (r_new / r_old)^(k+1),
      which is a practical scaling scheme to match the desired radius.

    Args:
        p: AR order.
        d: State dimension.
        max_attempts: Unused, kept only for interface compatibility.
        scale: Unused in the core logic, base_std is auto-scaled by (d, p).
        max_spectral_radius: Target spectral radius (< 1 for stability).

    Returns:
        List [W1, ..., Wp] of shape (d, d) matrices.
    """
    if p <= 0:
        raise ValueError("AR order p must be >= 1.")
    if d <= 0:
        raise ValueError("Dimension d must be >= 1.")
    if max_spectral_radius <= 0 or max_spectral_radius >= 1:
        warnings.warn(
            f"max_spectral_radius={max_spectral_radius} is unusual. "
            f"Typical stable values are in (0, 1).",
            RuntimeWarning,
        )

    # 1. Random initial weights (same distribution for all lags)
    weights: List[np.ndarray] = []
    base_std = 1.0 / np.sqrt(d * p)  # keeps eigenvalues in a reasonable range

    for _ in range(p):
        W = np.random.randn(d, d) * base_std
        weights.append(W)

    # 2. Compute current spectral radius
    C = companion_matrix(weights)
    eigenvalues = np.linalg.eigvals(C)
    current_radius = np.max(np.abs(eigenvalues))

    if current_radius <= 0:
        # Extremely degenerate case; avoid division by zero
        current_radius = 1e-6

    # 3. Adjust spectral radius using λ-scaling across lags
    target_radius = max_spectral_radius
    lambda_factor = target_radius / current_radius

    final_weights: List[np.ndarray] = []
    for k in range(p):
        # Scale A_{k+1} by λ^(k+1)
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
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a sequence from an AR(p) process with a non-zero mean.

    Model:
        s_t = c + Σ_{i=1}^p W_i s_{t-i} + ε_t
    where:
        - c is chosen so that the process has stationary mean ≈ TARGET_MEAN_LEVEL.
        - ε_t ~ N(0, noise_std^2 I_d)

    We also use a burn-in period to reach stationarity.

    Args:
        weights: List of weight matrices [W1, ..., Wp], each (d, d).
        T: Desired sequence length (returned after burn-in).
        noise_std: Standard deviation of Gaussian noise.
        initial_states: Optional array (p, d) for the first p states.
        seed: Optional random seed.

    Returns:
        Sequence array of shape (T, d).
    """
    if seed is not None:
        np.random.seed(seed)

    p = len(weights)
    if p == 0:
        raise ValueError("At least one weight matrix is required (p >= 1).")

    d = weights[0].shape[0]
    if T <= p:
        raise ValueError(f"Sequence length T={T} must be greater than p={p}.")

    # Burn-in for stabilizing around the stationary mean
    burn_in = 500
    total_T = T + burn_in

    # ---- 1. Compute bias c to enforce non-zero stationary mean μ ----
    # Stationary mean μ satisfies:
    #   μ = c + (Σ W_i) μ  =>  c = (I - Σ W_i) μ
    sum_weights = np.sum(weights, axis=0)          # (d, d)
    target_mu = np.ones(d) * TARGET_MEAN_LEVEL     # target mean per coordinate
    bias_vector = (np.eye(d) - sum_weights) @ target_mu  # (d,)

    # ---- 2. Initialize sequence ----
    sequence = np.zeros((total_T, d))

    if initial_states is None:
        # Initialize around target mean, not near 0
        sequence[:p] = target_mu + np.random.randn(p, d) * noise_std
    else:
        if initial_states.shape != (p, d):
            raise ValueError(
                f"initial_states must have shape ({p}, {d}), got {initial_states.shape}"
            )
        sequence[:p] = initial_states

    # ---- 3. Simulate AR(p) dynamics ----
    for t in range(p, total_T):
        # Start with bias
        s_t = bias_vector.copy()

        # Add AR contributions
        for i, W in enumerate(weights):
            # weights[0] corresponds to lag 1 => s_{t-1}
            s_t += W @ sequence[t - i - 1]

        # Add noise
        if noise_std > 0:
            s_t += np.random.randn(d) * noise_std

        sequence[t] = s_t

    # ---- 4. Drop burn-in and return ----
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
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """
    Generate a dataset of AR(p) sequences.

    Args:
        n_sequences: Number of sequences.
        p: AR order.
        d: State dimension.
        T: Sequence length (per sequence).
        noise_std: Noise std for each process.
        same_dynamics:
            - True: All sequences share the same [W1, ..., Wp].
            - False: Each sequence has its own independently sampled weights.
        seed: Optional global seed.

    Returns:
        sequences: Array of shape (n_sequences, T, d).
        weights_list: List of length n_sequences; each entry is [W1, ..., Wp].
    """
    if seed is not None:
        np.random.seed(seed)

    sequences = np.zeros((n_sequences, T, d))
    weights_list: List[List[np.ndarray]] = []

    if same_dynamics:
        base_weights = generate_stable_ar_weights(p, d, max_spectral_radius=0.95)
        # Use copies so each sequence has its own weight matrices (no shared references)
        for _ in range(n_sequences):
            weights_list.append([W.copy() for W in base_weights])
    else:
        for _ in range(n_sequences):
            weights = generate_stable_ar_weights(p, d, max_spectral_radius=0.95)
            weights_list.append(weights)

    # Generate each sequence
    for i in range(n_sequences):
        seq_seed = None if seed is None else seed + i
        sequences[i] = generate_ar_sequence(
            weights_list[i],
            T=T,
            noise_std=noise_std,
            seed=seq_seed
        )

    return sequences, weights_list


# =========================
# 5. AR(p) least-squares fit (oracle)
# =========================

def compute_ar_fit_loss(sequence: np.ndarray, p: int) -> Tuple[float, List[np.ndarray]]:
    """
    Fit a zero-intercept AR(p) model to a single sequence and compute MSE.

    IMPORTANT:
        - The generator now has a non-zero mean (bias).
        - Here we fit a model WITHOUT an intercept (Y = XW),
          so the fit will naturally have higher loss.
        - This is intentional: it makes the dataset "challenging" for naive AR fits.

    Args:
        sequence: Array of shape (T, d).
        p: AR order.

    Returns:
        loss: Mean squared error of LS fit on one-step predictions.
        fitted_weights: List [W1, ..., Wp] each (d, d).
    """
    T, d = sequence.shape
    if T <= p:
        raise ValueError(f"Sequence length {T} must be greater than AR order {p}.")

    X_list = []
    Y_list = []

    # Build design matrix X and target Y
    for t in range(p, T):
        # Concatenate states [s_{t-1}, s_{t-2}, ..., s_{t-p}]
        x_t = np.concatenate([sequence[t - i - 1] for i in range(p)])  # (p*d,)
        X_list.append(x_t)
        Y_list.append(sequence[t])

    X = np.array(X_list)  # (T-p, p*d)
    Y = np.array(Y_list)  # (T-p, d)

    # Least squares: Y ≈ X @ W_flat
    # W_flat: (p*d, d), block rows correspond to lags
    W_flat, *_ = np.linalg.lstsq(X, Y, rcond=None)

    # Predict and compute loss
    Y_pred = X @ W_flat  # (T-p, d)
    loss = np.mean((Y - Y_pred) ** 2)

    # Recover [W1, ..., Wp] from block rows of W_flat
    fitted_weights: List[np.ndarray] = []
    for i in range(p):
        # Block of rows corresponding to lag i+1
        W_i_T = W_flat[i * d:(i + 1) * d, :]  # (d, d), this is W_i^T
        W_i = W_i_T.T                          # (d, d)
        fitted_weights.append(W_i)

    return loss, fitted_weights


# =========================
# 6. Verification script
# =========================

if __name__ == "__main__":
    p_test, d_test, T_test = 10, 10, 200
    print(f"Testing challenging AR({p_test}) process with dim={d_test}...")

    weights = generate_stable_ar_weights(p_test, d_test, max_spectral_radius=0.98)

    C = companion_matrix(weights)
    rad = np.max(np.abs(np.linalg.eigvals(C)))
    print(f"Spectral radius: {rad:.4f} (target ≈ 0.98)")

    norms = [np.linalg.norm(w) for w in weights]
    print("Weight norms (should be comparable across lags):")
    print(f"  Lag 1 norm:  {norms[0]:.4f}")
    print(f"  Lag {p_test} norm: {norms[-1]:.4f}")

    seq = generate_ar_sequence(weights, T_test, noise_std=1.0)

    mean_val = np.mean(seq)
    print(f"Data mean level: {mean_val:.2f} (target ≈ {TARGET_MEAN_LEVEL})")
    print(f"Data min value:  {np.min(seq):.2f}")

    loss, fitted = compute_ar_fit_loss(seq, p_test)
    print(f"Fit loss (zero-mean assumption): {loss:.4f}")
    print(f"Recovered W1 norm:  {np.linalg.norm(fitted[0]):.4f}")
    print(f"Recovered Wp norm: {np.linalg.norm(fitted[-1]):.4f}")
    print("Note: Higher loss is expected because the true process has a bias term.")
