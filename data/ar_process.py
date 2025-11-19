"""
AR(p) process generation with stability checks.
"""
import numpy as np
from typing import List, Tuple, Optional
import warnings


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


def check_stability(weights: List[np.ndarray], max_spectral_radius=0.95, min_spectral_radius=0.5) -> bool:
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
    return (min_spectral_radius < spectral_radius < max_spectral_radius)

# def generate_stable_ar_weights(p: int, d: int,
#                                max_attempts: int = 1000,
#                                scale: float = 0.3,
#                                max_spectral_radius: float = 0.95) -> List[np.ndarray]:
#     """
#     Generate stable AR(p) weight matrices.

#     Args:
#         p: Order of AR process
#         d: Dimension of state vector
#         max_attempts: Maximum number of attempts to generate stable weights
#         scale: Scale parameter for random weight initialization
#         max_spectral_radius: Maximum allowed spectral radius

#     Returns:
#         List of weight matrices [W1, W2, ..., Wp]

#     Raises:
#         ValueError: If stable weights cannot be generated within max_attempts
#     """
#     for attempt in range(max_attempts):
#         # Generate random weights with decreasing magnitude for higher lags
#         weights = []
#         for i in range(p):
#             # Decay factor: higher lags get smaller weights
#             decay = 1.0 / (i + 1)**0.5
#             W = np.random.randn(d, d) * scale * decay
#             weights.append(W)

#         if check_stability(weights, max_spectral_radius):
#             return weights

#     raise ValueError(f"Could not generate stable AR({p}) weights after {max_attempts} attempts")

def generate_ar_weights_with_target_radius(p, d,
                                           base_scale=0.3,
                                           rho_target=0.8,
                                           max_attempts=1000):
    for attempt in range(max_attempts):
        weights = []
        scale = base_scale / np.sqrt(max(p, 1))
        for i in range(p):
            decay = 1.0 / (i + 1)**1.0
            W = np.random.randn(d, d) * scale * decay
            weights.append(W)

        C = companion_matrix(weights)
        eigs = np.linalg.eigvals(C)
        rho = np.max(np.abs(eigs))

        if rho == 0:
            continue 
        
        alpha = rho_target / rho
        scaled_weights = [alpha * W for W in weights]

        C_scaled = companion_matrix(scaled_weights)
        rho_scaled = np.max(np.abs(np.linalg.eigvals(C_scaled)))

        if rho_scaled < 0.99:
            return scaled_weights

    raise ValueError("Could not generate weights with target radius")



def generate_ar_sequence(weights: List[np.ndarray],
                        T: int,
                        noise_std: float = 0.0,
                        initial_states: Optional[np.ndarray] = None,
                        seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a sequence from AR(p) process.

    Args:
        weights: List of weight matrices [W1, W2, ..., Wp], each of shape (d, d)
        T: Length of sequence to generate
        noise_std: Standard deviation of Gaussian noise
        initial_states: Initial p states of shape (p, d). If None, randomly initialized.
        seed: Random seed for reproducibility

    Returns:
        Sequence of shape (T, d)
    """
    if seed is not None:
        np.random.seed(seed)

    p = len(weights)
    d = weights[0].shape[0]

    # Initialize sequence
    sequence = np.zeros((T, d))

    # Set initial states
    if initial_states is None:
        # Random initialization
        sequence[:p] = np.random.randn(p, d) * 0.1
    else:
        assert initial_states.shape == (p, d), f"Initial states must have shape ({p}, {d})"
        sequence[:p] = initial_states

    # Generate sequence
    for t in range(p, T):
        # AR dynamics: s_t = W1*s_{t-1} + W2*s_{t-2} + ... + Wp*s_{t-p} + noise
        s_t = np.zeros(d)
        for i, W in enumerate(weights):
            s_t += W @ sequence[t - i - 1]

        # Add noise
        if noise_std > 0:
            s_t += np.random.randn(d) * noise_std

        sequence[t] = s_t

    return sequence


def generate_ar_dataset(n_sequences: int,
                       p: int,
                       d: int,
                       T: int,
                       noise_std: float = 0.0,
                       same_dynamics: bool = True,
                       seed: Optional[int] = None) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """
    Generate dataset of AR(p) sequences.

    Args:
        n_sequences: Number of sequences to generate
        p: Order of AR process
        d: Dimension of state vector
        T: Length of each sequence
        noise_std: Standard deviation of Gaussian noise
        same_dynamics: If True, all sequences share the same AR weights
        seed: Random seed for reproducibility

    Returns:
        - sequences: Array of shape (n_sequences, T, d)
        - weights_list: List of weight matrices for each sequence
    """
    if seed is not None:
        np.random.seed(seed)

    sequences = np.zeros((n_sequences, T, d))
    weights_list = []

    # Generate weights
    if same_dynamics:
        # All sequences use the same AR weights
        weights = generate_stable_ar_weights(p, d)
        weights_list = [weights] * n_sequences
    else:
        # Each sequence has different AR weights
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
    Fit AR(p) model to sequence using least squares and return loss.

    Args:
        sequence: Sequence of shape (T, d)
        p: Order of AR process

    Returns:
        - loss: Mean squared error
        - fitted_weights: List of fitted weight matrices [W1, ..., Wp]
    """
    T, d = sequence.shape

    if T <= p:
        raise ValueError(f"Sequence length {T} must be greater than AR order {p}")

    # Construct design matrix X and target Y
    # X[t] = [s_{t-1}, s_{t-2}, ..., s_{t-p}] flattened
    # Y[t] = s_t
    X = []
    Y = []

    for t in range(p, T):
        # Stack past p states
        x_t = np.concatenate([sequence[t - i - 1] for i in range(p)])
        X.append(x_t)
        Y.append(sequence[t])

    X = np.array(X)  # Shape: (T-p, p*d)
    Y = np.array(Y)  # Shape: (T-p, d)

    # Solve least squares: Y = X @ W_flat^T
    # W_flat is (d, p*d) matrix
    W_flat = np.linalg.lstsq(X, Y, rcond=None)[0].T  # Shape: (d, p*d)

    # Compute loss
    Y_pred = X @ W_flat.T
    loss = np.mean((Y - Y_pred) ** 2)

    # Reshape to list of weight matrices
    fitted_weights = []
    for i in range(p):
        W_i = W_flat[:, i*d:(i+1)*d]
        fitted_weights.append(W_i)

    return loss, fitted_weights


if __name__ == "__main__":
    # Test AR(1) process
    print("Testing AR(1) process...")
    p, d, T = 1, 5, 100
    weights = generate_stable_ar_weights(p, d)
    print(f"Generated stable AR({p}) weights")
    print(f"Spectral radius: {np.max(np.abs(np.linalg.eigvals(weights[0]))):.4f}")

    sequence = generate_ar_sequence(weights, T, noise_std=0.1)
    print(f"Generated sequence of shape {sequence.shape}")

    loss, fitted_weights = compute_ar_fit_loss(sequence, p)
    print(f"Fitting loss: {loss:.6f}")
    print(f"Weight error: {np.linalg.norm(fitted_weights[0] - weights[0]):.6f}")

    # Test AR(5) process
    print("\nTesting AR(5) process...")
    p, d, T = 5, 5, 100
    weights = generate_stable_ar_weights(p, d)
    C = companion_matrix(weights)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(C)))
    print(f"Generated stable AR({p}) weights")
    print(f"Spectral radius: {spectral_radius:.4f}")

    sequence = generate_ar_sequence(weights, T, noise_std=0.1)
    print(f"Generated sequence of shape {sequence.shape}")

    # Test dataset generation
    print("\nTesting dataset generation...")
    n_sequences = 100
    sequences, weights_list = generate_ar_dataset(
        n_sequences=n_sequences,
        p=2,
        d=5,
        T=100,
        noise_std=0.1,
        same_dynamics=False,
        seed=42
    )
    print(f"Generated dataset of shape {sequences.shape}")
    print(f"Number of weight sets: {len(weights_list)}")
