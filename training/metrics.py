"""
Evaluation metrics for in-context learning:
- MSE (1-step and n-step rollout)
- SPD (Squared Parameter Distance)
- ILWD (Implicit Learning Weight Distance)
"""
from typing import List, Tuple, Optional

import numpy as np
import torch


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute mean squared error.
    """
    return torch.mean((predictions - targets) ** 2).item()


def compute_relative_error(model_mse: float, oracle_mse: float) -> float:
    """
    Compute relative error compared to oracle.
    """
    if oracle_mse == 0:
        return float("inf") if model_mse > 0 else 1.0
    return model_mse / oracle_mse


def extract_implicit_weights(
    model,
    context: torch.Tensor,
    p: int,
) -> List[np.ndarray]:
    """
    Extract implicit AR(p) weights learned by the model using gradient analysis.

    We approximate W_i via the Jacobian of the next prediction w.r.t. the
    state at lag i.
    """
    model.eval()
    device = context.device
    d = context.shape[-1]

    context = context.clone().detach().to(device).requires_grad_(True)

    # Forward pass to get next-step prediction (1, d)
    with torch.enable_grad():
        out = model(context)
        output = out[0] if isinstance(out, tuple) else out
        next_pred = output[:, -1, :]  # (1, d)

    implicit_weights: List[np.ndarray] = []

    for i in range(p):
        if context.shape[1] <= i:
            implicit_weights.append(np.zeros((d, d)))
            continue

        state_idx = context.shape[1] - i - 1
        jacobian = torch.zeros(d, d, device=device)

        for dim in range(d):
            if context.grad is not None:
                context.grad.zero_()
            if hasattr(model, "zero_grad"):
                model.zero_grad(set_to_none=True)

            next_pred[0, dim].backward(retain_graph=True)

            if context.grad is not None:
                jacobian[dim, :] = context.grad[0, state_idx, :].clone()

        implicit_weights.append(jacobian.detach().cpu().numpy())

    return implicit_weights


def compute_spd(true_weights: List[np.ndarray],
                learned_weights: List[np.ndarray]) -> float:
    """
    SPD = sum_i ||W_i - W_i^*||_F^2
    """
    if len(true_weights) != len(learned_weights):
        raise ValueError("Weight lists must have same length for SPD.")
    spd = 0.0
    for w_true, w_learned in zip(true_weights, learned_weights):
        diff = np.asarray(w_true) - np.asarray(w_learned)
        spd += float(np.sum(diff ** 2))
    return spd


def compute_ilwd(implicit_weights: List[np.ndarray],
                 ols_weights: List[np.ndarray]) -> float:
    """
    ILWD = sum_i ||W_i^implicit - W_i^OLS||_F^2
    """
    if len(implicit_weights) != len(ols_weights):
        raise ValueError("Weight lists must have same length for ILWD.")
    ilwd = 0.0
    for w_imp, w_ols in zip(implicit_weights, ols_weights):
        diff = np.asarray(w_imp) - np.asarray(w_ols)
        ilwd += float(np.sum(diff ** 2))
    return ilwd


def fit_ols_weights_from_context(
    context: np.ndarray,
    p: int,
) -> List[np.ndarray]:
    """
    Fit AR(p) weights W_i via least squares using one context window.

    We solve s_t = sum_i W_i s_{t-i} + eps for t >= p.
    """
    T_ctx, d = context.shape
    if T_ctx <= p:
        raise ValueError(f"Context length {T_ctx} must be larger than p={p}.")

    X_rows = []
    Y_rows = []

    for t in range(p, T_ctx):
        y_t = context[t]  # (d,)
        lags = [context[t - i] for i in range(1, p + 1)]
        x_t = np.concatenate(lags, axis=-1)  # (d * p,)
        X_rows.append(x_t)
        Y_rows.append(y_t)

    X = np.stack(X_rows, axis=0)  # (N, d*p)
    Y = np.stack(Y_rows, axis=0)  # (N, d)

    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (d*p, d)
    W_concat = coef.T  # (d, d*p)

    ols_weights: List[np.ndarray] = []
    for i in range(p):
        start = i * d
        end = (i + 1) * d
        W_i = W_concat[:, start:end]  # (d, d)
        ols_weights.append(W_i)

    return ols_weights


def evaluate_model(
    model,
    test_sequences: torch.Tensor,
    test_weights: List[List[np.ndarray]],
    context_len: int,
    p: int,
    oracle_predictor,
    ols_predictor,
    device: str = "cpu",
) -> dict:
    """
    Comprehensive evaluation of model performance.

    Assumes:
    - oracle_predictor / ols_predictor can handle the provided batch.
    """
    model.eval()
    model = model.to(device)

    if not torch.is_tensor(test_sequences):
        test_sequences = torch.as_tensor(test_sequences, dtype=torch.float32)
    else:
        test_sequences = test_sequences.float()

    n_test, T, d = test_sequences.shape
    test_sequences = test_sequences.to(device)

    context = test_sequences[:, :context_len, :]          # (n_test, context_len, d)
    target = test_sequences[:, context_len:, :]           # (n_test, T-context_len, d)
    n_pred = target.shape[1]

    # ---------- 1-step ----------
    with torch.no_grad():
        out = model(context)
        model_output = out[0] if isinstance(out, tuple) else out
        model_next = model_output[:, -1, :]               # (n_test, d)

    oracle_next = oracle_predictor.predict_next(context)  # (n_test, d)
    ols_next = ols_predictor.predict_next(context)        # (n_test, d)
    last_next = context[:, -1, :]                         # naive baseline

    mse_1step = compute_mse(model_next, target[:, 0, :])
    mse_1step_oracle = compute_mse(oracle_next, target[:, 0, :])
    mse_1step_ols = compute_mse(ols_next, target[:, 0, :])
    mse_1step_last = compute_mse(last_next, target[:, 0, :])

    # ---------- n-step rollout (up to 10) ----------
    n_rollout = min(10, n_pred)

    with torch.no_grad():
        model_rollout = model.autoregressive_predict(context, n_steps=n_rollout)

    oracle_rollout = oracle_predictor.autoregressive_predict(context, n_steps=n_rollout)
    ols_rollout = ols_predictor.autoregressive_predict(context, n_steps=n_rollout)
    last_rollout = context[:, -1:, :].repeat(1, n_rollout, 1)

    mse_rollout = compute_mse(model_rollout, target[:, :n_rollout, :])
    mse_rollout_oracle = compute_mse(oracle_rollout, target[:, :n_rollout, :])
    mse_rollout_ols = compute_mse(ols_rollout, target[:, :n_rollout, :])
    mse_rollout_last = compute_mse(last_rollout, target[:, :n_rollout, :])

    rel_error_1step = compute_relative_error(mse_1step, mse_1step_oracle)
    rel_error_rollout = compute_relative_error(mse_rollout, mse_rollout_oracle)
    rel_error_1step_last = compute_relative_error(mse_1step_last, mse_1step_oracle)
    rel_error_rollout_last = compute_relative_error(mse_rollout_last, mse_rollout_oracle)

    metrics = {
        "mse_1step": mse_1step,
        "mse_1step_oracle": mse_1step_oracle,
        "mse_1step_ols": mse_1step_ols,
        "mse_1step_last": mse_1step_last,
        "rel_error_1step": rel_error_1step,
        "rel_error_1step_last": rel_error_1step_last,
        "mse_rollout": mse_rollout,
        "mse_rollout_oracle": mse_rollout_oracle,
        "mse_rollout_ols": mse_rollout_ols,
        "mse_rollout_last": mse_rollout_last,
        "rel_error_rollout": rel_error_rollout,
        "rel_error_rollout_last": rel_error_rollout_last,
    }

    # ---------- SPD / ILWD on a representative context ----------
    if p > 0 and test_weights and len(test_weights) > 0:
        true_weights = [np.asarray(w) for w in test_weights[0][:p]]

        single_context = context[0].detach().cpu().numpy()
        ols_weights = fit_ols_weights_from_context(single_context, p)

        implicit_weights = extract_implicit_weights(
            model,
            context=context[0:1],
            p=p,
        )

        metrics["spd"] = compute_spd(true_weights, implicit_weights)
        metrics["ilwd"] = compute_ilwd(implicit_weights, ols_weights)
    else:
        metrics["spd"] = None
        metrics["ilwd"] = None

    return metrics


def bootstrap_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.
    """
    if seed is not None:
        np.random.seed(seed)

    values = np.asarray(values)
    n = len(values)
    if n == 0:
        raise ValueError("values must be non-empty for bootstrap CI.")

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    alpha = 1.0 - confidence
    lower = np.percentile(bootstrap_means, (alpha / 2.0) * 100.0)
    upper = np.percentile(bootstrap_means, (1.0 - alpha / 2.0) * 100.0)

    return float(lower), float(upper)


if __name__ == "__main__":
    # Simple smoke tests
    print("Testing evaluation metrics...")

    batch_size = 10
    d = 5
    p = 3

    predictions = torch.randn(batch_size, d)
    targets = torch.randn(batch_size, d)

    mse = compute_mse(predictions, targets)
    print(f"MSE: {mse:.6f}")

    oracle_mse = 0.5
    rel_error = compute_relative_error(mse, oracle_mse)
    print(f"Relative error: {rel_error:.6f}")

    true_weights = [np.random.randn(d, d) for _ in range(p)]
    learned_weights = [np.random.randn(d, d) for _ in range(p)]
    spd = compute_spd(true_weights, learned_weights)
    print(f"SPD: {spd:.6f}")

    implicit_weights = [np.random.randn(d, d) for _ in range(p)]
    ols_weights = [np.random.randn(d, d) for _ in range(p)]
    ilwd = compute_ilwd(implicit_weights, ols_weights)
    print(f"ILWD: {ilwd:.6f}")

    values = np.random.randn(100)
    lower, upper = bootstrap_confidence_interval(values)
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
