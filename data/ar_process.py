"""
AR(p) process generation with stability checks.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings

import numpy as np
from typing import List, Tuple, Optional
import warnings

# Global Configuration: Set target mean level to prevent data from being close to 0.
# Since function signatures cannot be changed, we control the offset via this module-level constant.
TARGET_MEAN_LEVEL = 10.0 

def companion_matrix(weights: List[np.ndarray]) -> np.ndarray:
    """
    Construct companion matrix for AR(p) process.

    Args:
        weights: List of weight matrices [W1, W2, ..., Wp], each of shape (d, d)

    Returns:
        Companion matrix of shape (p*d, p*d)
    """
    p = len(weights)
    d = weights[0].shape[0]

    # Create companion matrix
    C = np.zeros((p * d, p * d))

    # First row: [W1, W2, ..., Wp]
    for i, W in enumerate(weights):
        C[:d, i*d:(i+1)*d] = W

    # Identity blocks for lower rows
    if p > 1:
        C[d:, :d*(p-1)] = np.eye(d * (p-1))

    return C


def check_stability(weights: List[np.ndarray], max_spectral_radius: float = 0.95) -> bool:
    """
    Check if AR(p) process is stable by computing spectral radius of companion matrix.

    Args:
        weights: List of weight matrices [W1, W2, ..., Wp]
        max_spectral_radius: Maximum allowed spectral radius for stability

    Returns:
        True if stable (spectral radius < max_spectral_radius)
    """
    C = companion_matrix(weights)
    eigenvalues = np.linalg.eigvals(C)
    spectral_radius = np.max(np.abs(eigenvalues))
    return spectral_radius < max_spectral_radius


def generate_stable_ar_weights(p: int, d: int,
                               max_attempts: int = 1000,
                               scale: float = 0.5,
                               max_spectral_radius: float = 0.95) -> List[np.ndarray]:
    """
    Generate stable AR(p) weight matrices with COMPARABLE LAGS.
    
    Changes from original:
    1. Removed 'decay': Lag p is just as strong as Lag 1 (Challenging).
    2. Uses lambda^k scaling to enforce exact spectral radius without destroying structure.

    Args:
        p: Order of AR process
        d: Dimension of state vector
        max_attempts: (Unused, kept for API compatibility)
        scale: Initial variance scale
        max_spectral_radius: Target spectral radius

    Returns:
        List of weight matrices [W1, W2, ..., Wp]
    """
    # 1. Generate random weights from SAME distribution for all lags
    # We normalize by sqrt(d*p) to keep initial eigenvalues reasonable
    weights = []
    base_std = 1.0 / np.sqrt(d * p)
    
    for i in range(p):
        # CRITICAL: No decay factor here. W_p is comparable to W_1.
        W = np.random.randn(d, d) * base_std
        weights.append(W)

    # 2. Compute current spectral radius
    C = companion_matrix(weights)
    eigenvalues = np.linalg.eigvals(C)
    current_radius = np.max(np.abs(eigenvalues))

    if current_radius == 0:
        current_radius = 1e-6

    # 3. Strict Spectral Radius Adjustment
    # To change radius from r_old to r_new, we must scale A_k by (r_new/r_old)^k
    # This is the mathematically correct way to adjust radius for AR(p)
    target_radius = max_spectral_radius
    lambda_factor = target_radius / current_radius
    
    final_weights = []
    for k in range(p):
        # Scale A_{k+1} by lambda^(k+1)
        scale_k = lambda_factor ** (k + 1)
        final_weights.append(weights[k] * scale_k)

    return final_weights


def generate_ar_sequence(weights: List[np.ndarray],
                        T: int,
                        noise_std: float = 1.0,
                        initial_states: Optional[np.ndarray] = None,
                        seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a sequence from AR(p) process.
    
    Changes from original:
    1. Injects a Bias term to ensure data is NOT CLOSE TO 0.
    2. Calculates bias based on TARGET_MEAN_LEVEL.

    Args:
        weights: List of weight matrices [W1, W2, ..., Wp]
        T: Length of sequence
        noise_std: Noise standard deviation
        initial_states: Initial states
        seed: Random seed

    Returns:
        Sequence of shape (T, d)
    """
    if seed is not None:
        np.random.seed(seed)

    p = len(weights)
    d = weights[0].shape[0]
    
    # Burn-in to stabilize the process around the mean
    burn_in = 500
    total_T = T + burn_in

    # 1. Calculate Bias Vector to force Non-Zero Mean
    # Formula: c = (I - sum(A_i)) * mu
    sum_weights = np.sum(weights, axis=0)
    target_mu = np.ones(d) * TARGET_MEAN_LEVEL
    # Bias vector c
    bias_vector = (np.eye(d) - sum_weights) @ target_mu

    # Initialize sequence
    sequence = np.zeros((total_T, d))

    # Set initial states
    # If none provided, initialize around the target mean, not 0
    if initial_states is None:
        sequence[:p] = target_mu + np.random.randn(p, d) * noise_std
    else:
        sequence[:p] = initial_states

    # Generate sequence
    for t in range(p, total_T):
        # AR dynamics: s_t = c + sum(W_i * s_{t-i}) + noise
        s_t = bias_vector.copy() # Start with bias
        
        for i, W in enumerate(weights):
            # weights[0] is lag 1, so index is t-1
            s_t += W @ sequence[t - i - 1]

        # Add noise
        if noise_std > 0:
            s_t += np.random.randn(d) * noise_std

        sequence[t] = s_t

    # Return only the valid sequence after burn-in
    return sequence[burn_in:]


def generate_ar_dataset(n_sequences: int,
                       p: int,
                       d: int,
                       T: int,
                       noise_std: float = 1.0,
                       same_dynamics: bool = True,
                       seed: Optional[int] = None) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """
    Generate dataset of AR(p) sequences.
    (Wrapper function, logic remains largely same but calls updated generators)
    """
    if seed is not None:
        np.random.seed(seed)

    sequences = np.zeros((n_sequences, T, d))
    weights_list = []

    # Generate weights
    if same_dynamics:
        weights = generate_stable_ar_weights(p, d)
        weights_list = [weights] * n_sequences
    else:
        for i in range(n_sequences):
            weights = generate_stable_ar_weights(p, d)
            weights_list.append(weights)

    # Generate sequences
    for i in range(n_sequences):
        sequences[i] = generate_ar_sequence(
            weights_list[i],
            T=T,
            noise_std=noise_std,
            seed=None if seed is None else seed + i
        )

    return sequences, weights_list


def compute_ar_fit_loss(sequence: np.ndarray, p: int) -> Tuple[float, List[np.ndarray]]:
    """
    Fit AR(p) model to sequence.
    
    Note: Because the generated data now has a Non-Zero Mean (Bias), 
    a standard Zero-Mean AR fit (Y = XW) will naturally have a higher loss.
    This is expected and desired for 'Challenging' datasets.
    """
    T, d = sequence.shape

    if T <= p:
        raise ValueError(f"Sequence length {T} must be greater than AR order {p}")

    X = []
    Y = []

    for t in range(p, T):
        x_t = np.concatenate([sequence[t - i - 1] for i in range(p)])
        X.append(x_t)
        Y.append(sequence[t])

    X = np.array(X)  # (T-p, p*d)
    Y = np.array(Y)  # (T-p, d)

    # Solve least squares: Y = X @ W_flat^T
    # Note: We are purposely NOT fitting an intercept here to maintain API consistency
    # and to show the "challenge" of the biased data.
    W_flat = np.linalg.lstsq(X, Y, rcond=None)[0].T

    Y_pred = X @ W_flat.T
    loss = np.mean((Y - Y_pred) ** 2)

    fitted_weights = []
    for i in range(p):
        W_i = W_flat[:, i*d:(i+1)*d]
        fitted_weights.append(W_i)

    return loss, fitted_weights


if __name__ == "__main__":
    # ==========================================
    # Verification Script
    # ==========================================
    
    # 1. High dimension, Long lag, Challenging
    p_test, d_test, T_test = 10, 10, 200
    print(f"Testing Challenging AR({p_test}) process with Dim={d_test}...")
    
    # Generate weights
    weights = generate_stable_ar_weights(p_test, d_test, max_spectral_radius=0.98)
    
    # Verify spectral radius
    C = companion_matrix(weights)
    rad = np.max(np.abs(np.linalg.eigvals(C)))
    print(f"Spectral Radius: {rad:.4f} (Target ~0.98)")
    
    # Verify comparability between lags
    print("Weight Norms (Should be comparable, not decaying to zero):")
    norms = [np.linalg.norm(w) for w in weights]
    print(f"  Lag 1: {norms[0]:.4f}")
    print(f"  Lag {p_test}: {norms[-1]:.4f}")
    
    # Generate data
    seq = generate_ar_sequence(weights, T_test, noise_std=1.0)
    
    # Verify data is not close to 0
    mean_val = np.mean(seq)
    print(f"Data Mean Level: {mean_val:.2f} (Target ~{TARGET_MEAN_LEVEL})")
    print(f"Data Min Value: {np.min(seq):.2f}")
    
    # Fit loss
    loss, _ = compute_ar_fit_loss(seq, p_test)
    print(f"Fit Loss (Zero-Mean Assumption): {loss:.4f}")
    print("Note: High loss is expected because data has a bias/intercept term.")