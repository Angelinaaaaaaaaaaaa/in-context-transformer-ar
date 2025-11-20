import numpy as np
from typing import List, Tuple, Optional


def companion_matrix(weights: List[np.ndarray]) -> np.ndarray:
    p = len(weights)
    if p == 0:
        raise ValueError("At least one weight matrix is required (p >= 1).")
    d = weights[0].shape[0]
    C = np.zeros((p * d, p * d))
    for i, W in enumerate(weights):
        if W.shape != (d, d):
            raise ValueError(f"All weight matrices must be ({d}, {d}), got {W.shape}.")
        C[:d, i * d:(i + 1) * d] = W
    if p > 1:
        C[d:, :d * (p - 1)] = np.eye(d * (p - 1))
    return C


def check_stability(weights: List[np.ndarray], max_spectral_radius: float = 0.95) -> bool:
    C = companion_matrix(weights)
    eigenvalues = np.linalg.eigvals(C)
    spectral_radius = np.max(np.abs(eigenvalues))
    return spectral_radius < max_spectral_radius


def generate_stable_ar_weights(
    p: int,
    d: int,
    max_attempts: int = 1000,
    scale: float = 0.5,
    max_spectral_radius: float = 0.95,
) -> List[np.ndarray]:
    if p <= 0:
        raise ValueError("AR order p must be >= 1.")
    if d <= 0:
        raise ValueError("Dimension d must be >= 1.")
    weights: List[np.ndarray] = []
    base_std = 1.0 / np.sqrt(d * p)
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


def generate_ar_sequence(
    weights: List[np.ndarray],
    T: int,
    noise_std: float = 1.0,
    initial_states: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    p = len(weights)
    d = weights[0].shape[0]
    if T <= p:
        raise ValueError(f"Sequence length T={T} must be greater than p={p}.")
    burn_in = 500
    total_T = T + burn_in
    sequence = np.zeros((total_T, d))
    if initial_states is None:
        sequence[:p] = np.random.randn(p, d) * noise_std
    else:
        if initial_states.shape != (p, d):
            raise ValueError("initial_states shape mismatch.")
        sequence[:p] = initial_states
    for t in range(p, total_T):
        s_t = np.zeros(d)
        for i, W in enumerate(weights):
            s_t += W @ sequence[t - i - 1]
        if noise_std > 0:
            s_t += np.random.randn(d) * noise_std
        sequence[t] = s_t
    return sequence[burn_in:]


def generate_ar_dataset(
    n_sequences: int,
    p: int,
    d: int,
    T: int,
    noise_std: float = 1.0,
    same_dynamics: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    if seed is not None:
        np.random.seed(seed)
    sequences = np.zeros((n_sequences, T, d))
    weights_list: List[List[np.ndarray]] = []
    if same_dynamics:
        base_weights = generate_stable_ar_weights(p, d, max_spectral_radius=0.95)
        for _ in range(n_sequences):
            weights_list.append([W.copy() for W in base_weights])
    else:
        for _ in range(n_sequences):
            weights = generate_stable_ar_weights(p, d, max_spectral_radius=0.95)
            weights_list.append(weights)
    for i in range(n_sequences):
        seq_seed = None if seed is None else seed + i
        sequences[i] = generate_ar_sequence(
            weights_list[i],
            T=T,
            noise_std=noise_std,
            seed=seq_seed,
        )
    return sequences, weights_list


def compute_ar_fit_loss(sequence: np.ndarray, p: int) -> Tuple[float, List[np.ndarray]]:
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
    p_test, d_test, T_test = 10, 10, 200
    print(f"Testing Zero-Mean AR({p_test}) process with dim={d_test}...")
    weights = generate_stable_ar_weights(p_test, d_test, max_spectral_radius=0.98)
    seq = generate_ar_sequence(weights, T_test, noise_std=1.0)
    mean_val = np.mean(seq)
    print(f"Data mean level: {mean_val:.2f}")
    print(f"Data range:      {np.min(seq):.2f} to {np.max(seq):.2f}")
    loss, fitted = compute_ar_fit_loss(seq, p_test)
    print(f"Fit loss: {loss:.4f}")
    print(f"Recovered W1 norm:  {np.linalg.norm(fitted[0]):.4f}")
