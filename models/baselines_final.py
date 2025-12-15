"""
Baseline models for AR(p) prediction:
1. Oracle: Uses ground-truth AR weights
2. OLS: Ordinary least squares on context window
3. Last-value: Naive predictor that repeats the last value
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class OraclePredictor:
    """
    Oracle predictor that uses ground-truth AR(p) weights.
    Updated for Zero-Mean AR process.
    """

    def __init__(self, weights: List[np.ndarray]):
        """
        Args:
            weights: List of weight matrices [W1, W2, ..., Wp], each of shape (d, d)
        """
        self.weights = [torch.tensor(w, dtype=torch.float32) for w in weights]
        self.p = len(weights)
        self.d = weights[0].shape[0]


    def predict_next(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict next state given context.
        """
        batch_size, context_len, d = context.shape
        device = context.device

        assert context_len >= self.p, f"Context length {context_len} must be >= p={self.p}"

        # Move weights to same device
        weights = [w.to(device) for w in self.weights]

        prediction = torch.zeros(batch_size, d, device=device)

        # Compute s_t = W1*s_{t-1} + W2*s_{t-2} + ... + Wp*s_{t-p}
        for i, W in enumerate(weights):
            # W @ s_{t-i-1}
            s_lag = context[:, -(i+1), :]  # (batch, d)
            prediction += torch.matmul(s_lag, W.T)  # (batch, d)

        return prediction

    def autoregressive_predict(self, context: torch.Tensor, n_steps: int) -> torch.Tensor:
        batch_size = context.shape[0]
        predictions = []
        current_seq = context.clone()

        for _ in range(n_steps):
            next_state = self.predict_next(current_seq)
            predictions.append(next_state)
            next_state = next_state.unsqueeze(1)
            current_seq = torch.cat([current_seq, next_state], dim=1)

        predictions = torch.stack(predictions, dim=1)
        return predictions


class OLSPredictor:
    """
    OLS AR(p) estimator using Ridge Regression, with fixed alphas per p.
    """

    BEST_ALPHA = {
        1: 20.0,
        2: 25.0,
        5: 25.0,
        10: 25.0,
    }

    def __init__(self, p: int, ridge_alpha: float = None):
        """
        If ridge_alpha is None, automatically assign the best one for the given p.
        """
        self.p = p

        if ridge_alpha is None:
            if p in self.BEST_ALPHA:
                self.ridge_alpha = self.BEST_ALPHA[p]
            else:
                raise ValueError(f"No default alpha known for p={p}. "
                                 f"Provide ridge_alpha manually.")
        else:
            self.ridge_alpha = ridge_alpha


    def _fit_ols(self, sequence: torch.Tensor) -> List[torch.Tensor]:
        """
        Fit AR(p) model using Ridge Regression (Closed form solution).
        
        Solving for W in: (X^T X + alpha * I) W^T = X^T Y

        Args:
            sequence: Sequence of shape (seq_len, d)

        Returns:
            List of fitted weight matrices [W1, ..., Wp]
        """
        seq_len, d = sequence.shape
        device = sequence.device

        # Not enough data
        if seq_len <= self.p:
            return [torch.zeros(d, d, device=device) for _ in range(self.p)]

        # 1. Construct design matrix X and target Y
        X_list = []
        Y_list = []

        for t in range(self.p, seq_len):
            # Flatten past p states: [s_{t-1}, ..., s_{t-p}]
            # Shape: (p * d)
            x_t = torch.cat([sequence[t - i - 1] for i in range(self.p)])
            X_list.append(x_t)
            Y_list.append(sequence[t])

        X = torch.stack(X_list)  # Shape: (N_samples, p*d)
        Y = torch.stack(Y_list)  # Shape: (N_samples, d)

        # 2. Ridge Regression: W^T = (X^T X + alpha I)^-1 X^T Y
        # We need to solve for weights. 
        # X: (N, K), Y: (N, d) -> Weights should be (d, K)
        
        N, K = X.shape # K = p*d
        
        # Compute X^T X (covariance matrix of features)
        XTX = torch.matmul(X.T, X)  # (K, K)
        
        # Add Ridge Regularization (alpha * Identity)
        I_mat = torch.eye(K, device=device)
        XTX_reg = XTX + self.ridge_alpha * I_mat

        # Compute X^T Y
        XTY = torch.matmul(X.T, Y)  # (K, d)

        try:
            # Solve linear system: A * Z = B  =>  Z = A^-1 B
            # Where Z is W^T (shape K, d)
            W_transposed = torch.linalg.solve(XTX_reg, XTY)
            W_flat = W_transposed.T  # Shape: (d, p*d)
        except RuntimeError:
            # Fallback if solver fails (very rare with Ridge)
            W_flat = torch.zeros(d, self.p * d, device=device)

        # 3. Reshape to list of weight matrices [W1, ..., Wp]
        fitted_weights = []
        for i in range(self.p):
            # Extract W_i corresponding to lag i+1
            W_i = W_flat[:, i*d:(i+1)*d]
            fitted_weights.append(W_i)

        return fitted_weights

    def predict_next(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict next state given context.

        Args:
            context: Context of shape (batch, context_len, d)

        Returns:
            Next state prediction of shape (batch, d)
        """
        batch_size, context_len, d = context.shape
        device = context.device

        predictions = []

        for b in range(batch_size):
            # Fit OLS (actually Ridge) on this sequence
            weights = self._fit_ols(context[b])

            # Predict next state
            prediction = torch.zeros(d, device=device)
            for i, W in enumerate(weights):
                if context_len > i:
                    s_lag = context[b, -(i+1), :]
                    # Model: s_t = Sum(W_i @ s_{t-i})
                    # Implementation detail: 
                    # If s_lag is 1D (d,), W is (d,d), we want output (d,)
                    # torch.matmul(W, s_lag) is correct.
                    # Previous code used matmul(s_lag, W.T) which is mathematically equivalent
                    # if s_lag is treated as row vector. We keep consistent with W shape.
                    prediction += torch.matmul(s_lag, W.T)

            predictions.append(prediction)

        return torch.stack(predictions)  # (batch, d)

    def autoregressive_predict(self, context: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Generate n_steps predictions autoregressively.
        """
        batch_size = context.shape[0]
        predictions = []

        # Start with context
        current_seq = context.clone()

        for _ in range(n_steps):
            # Predict next state
            next_state = self.predict_next(current_seq)  # (batch, d)
            predictions.append(next_state)

            # Append to sequence
            next_state = next_state.unsqueeze(1)  # (batch, 1, d)
            current_seq = torch.cat([current_seq, next_state], dim=1)

        predictions = torch.stack(predictions, dim=1)  # (batch, n_steps, d)
        return predictions


class LastValuePredictor:
    """
    Naive baseline that always predicts the last observed value.
    """

    def predict_next(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict next state as the last value in context.

        Args:
            context: Context of shape (batch, context_len, d)

        Returns:
            Next state prediction of shape (batch, d)
        """
        return context[:, -1, :]  # Return last value

    def autoregressive_predict(self, context: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Generate n_steps predictions autoregressively.
        Note: For last-value predictor, each prediction is just the previous value.

        Args:
            context: Initial context of shape (batch, context_len, d)
            n_steps: Number of steps to predict

        Returns:
            Predictions of shape (batch, n_steps, d)
        """
        batch_size, _, d = context.shape
        device = context.device

        # Get last value
        last_value = context[:, -1, :]  # (batch, d)

        # Repeat for n_steps
        predictions = last_value.unsqueeze(1).repeat(1, n_steps, 1)  # (batch, n_steps, d)

        return predictions


if __name__ == "__main__":
    # Test baselines
    print("Testing baseline models...")

    # Generate synthetic data
    from data.ar_process import generate_stable_ar_weights, generate_ar_sequence

    p, d, T = 3, 5, 100
    batch_size = 4

    # Generate weights and sequences
    weights = generate_stable_ar_weights(p, d, seed=42)
    sequences = []
    for i in range(batch_size):
        seq = generate_ar_sequence(weights, T, noise_std=0.1, seed=42+i)
        sequences.append(seq)
    sequences = torch.tensor(np.array(sequences), dtype=torch.float32)  # (batch, T, d)

    # Split into context and target
    context_len = 70
    context = sequences[:, :context_len, :]
    target = sequences[:, context_len:, :]

    print(f"Context shape: {context.shape}")
    print(f"Target shape: {target.shape}")

    # Test Oracle
    print("\nTesting Oracle predictor...")
    oracle = OraclePredictor(weights)
    next_pred = oracle.predict_next(context)
    print(f"Next prediction shape: {next_pred.shape}")
    print(f"Error vs. target: {torch.mean((next_pred - target[:, 0, :])**2).item():.6f}")

    rollout = oracle.autoregressive_predict(context, n_steps=10)
    print(f"Rollout shape: {rollout.shape}")
    print(f"Rollout error: {torch.mean((rollout - target[:, :10, :])**2).item():.6f}")

    # Test OLS
    print("\nTesting OLS predictor...")
    ols = OLSPredictor(p)
    next_pred = ols.predict_next(context)
    print(f"Next prediction shape: {next_pred.shape}")
    print(f"Error vs. target: {torch.mean((next_pred - target[:, 0, :])**2).item():.6f}")

    rollout = ols.autoregressive_predict(context, n_steps=10)
    print(f"Rollout shape: {rollout.shape}")
    print(f"Rollout error: {torch.mean((rollout - target[:, :10, :])**2).item():.6f}")

    # Test Last-value
    print("\nTesting Last-value predictor...")
    last_val = LastValuePredictor()
    next_pred = last_val.predict_next(context)
    print(f"Next prediction shape: {next_pred.shape}")
    print(f"Error vs. target: {torch.mean((next_pred - target[:, 0, :])**2).item():.6f}")

    rollout = last_val.autoregressive_predict(context, n_steps=10)
    print(f"Rollout shape: {rollout.shape}")
    print(f"Rollout error: {torch.mean((rollout - target[:, :10, :])**2).item():.6f}")
