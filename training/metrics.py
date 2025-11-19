"""
Evaluation metrics for in-context learning:
- MSE (1-step and n-step rollout)
- SPD (Squared Parameter Distance)
- ILWD (Implicit Learning Weight Distance)
"""
import torch
import numpy as np
from typing import List, Tuple, Optional


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute mean squared error.

    Args:
        predictions: Predictions of shape (batch, ..., d) or (batch, d)
        targets: Targets of same shape as predictions

    Returns:
        MSE value
    """
    return torch.mean((predictions - targets) ** 2).item()


def compute_relative_error(model_mse: float, oracle_mse: float) -> float:
    """
    Compute relative error compared to oracle.

    Args:
        model_mse: MSE of the model
        oracle_mse: MSE of oracle baseline

    Returns:
        Relative error (model_mse / oracle_mse)
    """
    if oracle_mse == 0:
        return float('inf') if model_mse > 0 else 1.0
    return model_mse / oracle_mse


def extract_implicit_weights(model,
                            context: torch.Tensor,
                            p: int) -> List[np.ndarray]:
    """
    Extract implicit AR(p) weights learned by the model using gradient analysis.

    This follows the approach from Akyürek et al. (2022) and von Oswald et al. (2023),
    which analyzes how the model's prediction changes with respect to past states.

    Args:
        model: Trained model
        context: Context sequence of shape (1, context_len, d)
        p: Order of AR process

    Returns:
        List of implicit weight matrices [W1, ..., Wp]
    """
    model.eval()
    device = context.device
    d = context.shape[-1]

    # Ensure context requires grad
    context = context.clone().detach().requires_grad_(True)

    # Forward pass to get prediction
    with torch.enable_grad():
        output, _ = model(context)
        next_pred = output[:, -1, :]  # (1, d)

    # Compute gradients with respect to past p states
    implicit_weights = []

    for i in range(p):
        if context.shape[1] <= i:
            # Not enough context
            implicit_weights.append(np.zeros((d, d)))
            continue

        # Get state at lag i+1
        state_idx = context.shape[1] - i - 1

        # Compute Jacobian: d(next_pred) / d(context[:, state_idx, :])
        jacobian = torch.zeros(d, d, device=device)

        for dim in range(d):
            # Compute gradient of next_pred[dim] w.r.t. context[:, state_idx, :]
            if context.grad is not None:
                context.grad.zero_()

            next_pred[0, dim].backward(retain_graph=True)

            if context.grad is not None:
                jacobian[dim, :] = context.grad[0, state_idx, :].clone()

        implicit_weights.append(jacobian.detach().cpu().numpy())

    return implicit_weights


def compute_spd(true_weights: List[np.ndarray],
               learned_weights: List[np.ndarray]) -> float:
    """
    Compute Squared Parameter Distance (SPD).

    SPD = sum_i ||W_i - W_i^*||_F^2

    Args:
        true_weights: List of ground-truth weight matrices
        learned_weights: List of learned weight matrices

    Returns:
        SPD value
    """
    assert len(true_weights) == len(learned_weights), "Weight lists must have same length"

    spd = 0.0
    for W_true, W_learned in zip(true_weights, learned_weights):
        diff = W_true - W_learned
        spd += np.sum(diff ** 2)  # Frobenius norm squared

    return spd


def compute_ilwd(implicit_weights: List[np.ndarray],
                ols_weights: List[np.ndarray]) -> float:
    """
    Compute Implicit Learning Weight Distance (ILWD).

    This measures whether the model's implicit weights are closer to OLS
    or some other algorithm (e.g., gradient descent).

    ILWD = sum_i ||W_i^implicit - W_i^OLS||_F^2

    Args:
        implicit_weights: List of implicit weight matrices from model
        ols_weights: List of OLS-fitted weight matrices

    Returns:
        ILWD value
    """
    assert len(implicit_weights) == len(ols_weights), "Weight lists must have same length"

    ilwd = 0.0
    for W_implicit, W_ols in zip(implicit_weights, ols_weights):
        diff = W_implicit - W_ols
        ilwd += np.sum(diff ** 2)  # Frobenius norm squared

    return ilwd


def evaluate_model(model,
                  test_sequences: torch.Tensor,
                  test_weights: List[List[np.ndarray]],
                  context_len: int,
                  p: int,
                  oracle_predictor,
                  ols_predictor,
                  device: str = 'cpu') -> dict:
    """
    Comprehensive evaluation of model performance.

    Args:
        model: Model to evaluate
        test_sequences: Test sequences of shape (n_test, T, d)
        test_weights: Ground-truth AR weights for each sequence
        context_len: Length of context window
        p: Order of AR process
        oracle_predictor: Oracle baseline
        ols_predictor: OLS baseline
        device: Device to run on

    Returns:
        Dictionary of metrics
    """
    model.eval()
    model = model.to(device)

    n_test, T, d = test_sequences.shape
    test_sequences = test_sequences.to(device)

    # Split into context and target
    context = test_sequences[:, :context_len, :]
    target = test_sequences[:, context_len:, :]
    n_pred = target.shape[1]

    # 1-step prediction
    with torch.no_grad():
        model_output, _ = model(context)
        model_next = model_output[:, -1, :]  # (n_test, d)

    oracle_next = oracle_predictor.predict_next(context)
    ols_next = ols_predictor.predict_next(context)

    mse_1step = compute_mse(model_next, target[:, 0, :])
    mse_1step_oracle = compute_mse(oracle_next, target[:, 0, :])
    mse_1step_ols = compute_mse(ols_next, target[:, 0, :])

    # n-step rollout (up to 10 steps or available target length)
    n_rollout = min(10, n_pred)

    with torch.no_grad():
        model_rollout = model.autoregressive_predict(context, n_steps=n_rollout)

    oracle_rollout = oracle_predictor.autoregressive_predict(context, n_steps=n_rollout)
    ols_rollout = ols_predictor.autoregressive_predict(context, n_steps=n_rollout)

    mse_rollout = compute_mse(model_rollout, target[:, :n_rollout, :])
    mse_rollout_oracle = compute_mse(oracle_rollout, target[:, :n_rollout, :])
    mse_rollout_ols = compute_mse(ols_rollout, target[:, :n_rollout, :])

    # Relative errors
    rel_error_1step = compute_relative_error(mse_1step, mse_1step_oracle)
    rel_error_rollout = compute_relative_error(mse_rollout, mse_rollout_oracle)

    metrics = {
        'mse_1step': mse_1step,
        'mse_1step_oracle': mse_1step_oracle,
        'mse_1step_ols': mse_1step_ols,
        'rel_error_1step': rel_error_1step,
        'mse_rollout': mse_rollout,
        'mse_rollout_oracle': mse_rollout_oracle,
        'mse_rollout_ols': mse_rollout_ols,
        'rel_error_rollout': rel_error_rollout,
    }

    # Optionally compute SPD and ILWD on a subset (expensive)
    # Skipping for now to save computation

    return metrics


def bootstrap_confidence_interval(values: np.ndarray,
                                 confidence: float = 0.95,
                                 n_bootstrap: int = 1000,
                                 seed: Optional[int] = None) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: Array of values
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        (lower_bound, upper_bound)
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(values)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return lower, upper


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")

    # Create dummy data
    batch_size = 10
    d = 5
    p = 3

    predictions = torch.randn(batch_size, d)
    targets = torch.randn(batch_size, d)

    # Test MSE
    mse = compute_mse(predictions, targets)
    print(f"MSE: {mse:.6f}")

    # Test relative error
    oracle_mse = 0.5
    rel_error = compute_relative_error(mse, oracle_mse)
    print(f"Relative error: {rel_error:.6f}")

    # Test SPD
    true_weights = [np.random.randn(d, d) for _ in range(p)]
    learned_weights = [np.random.randn(d, d) for _ in range(p)]
    spd = compute_spd(true_weights, learned_weights)
    print(f"SPD: {spd:.6f}")

    # Test ILWD
    implicit_weights = [np.random.randn(d, d) for _ in range(p)]
    ols_weights = [np.random.randn(d, d) for _ in range(p)]
    ilwd = compute_ilwd(implicit_weights, ols_weights)
    print(f"ILWD: {ilwd:.6f}")

    # Test bootstrap CI
    values = np.random.randn(100)
    lower, upper = bootstrap_confidence_interval(values)
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
