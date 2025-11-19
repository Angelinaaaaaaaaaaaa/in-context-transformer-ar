"""
Decoder-only GPT architecture for in-context learning.

Following Sander et al. (2024):
- 6 layers, 8 heads, d_model=256, FFN=1024
- Learned positional encodings
- Causal mask
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention with causal masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Causal mask of shape (seq_len, seq_len)
            return_attention: If True, return attention weights

        Returns:
            - output: Tensor of shape (batch, seq_len, d_model)
            - attention_weights: If return_attention, tensor of shape (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)  # (batch, n_heads, seq_len, d_head)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, seq_len, seq_len)

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, d_head)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Output projection
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output, None


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer decoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual
        attn_output, attn_weights = self.attention(self.norm1(x), mask, return_attention)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x, attn_weights


class GPTModel(nn.Module):
    """
    Decoder-only GPT model for in-context learning of AR(p) processes.

    Args:
        d_input: Input dimension (state dimension)
        d_model: Model dimension
        n_layers: Number of Transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(self,
                 d_input: int,
                 d_model: int = 256,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Learned positional encoding
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, d_input)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 style."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive attention."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask

    def forward(self,
                x: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_input)
            return_attention: If True, return attention weights from all layers

        Returns:
            - output: Predictions of shape (batch, seq_len, d_input)
            - attention_weights: If return_attention, list of attention weights from each layer
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.pos_encoding(positions)  # (1, seq_len, d_model)
        x = x + pos_emb
        x = self.dropout(x)

        # Create causal mask
        mask = self._create_causal_mask(seq_len, device)

        # Apply Transformer blocks
        all_attention_weights = [] if return_attention else None
        for block in self.blocks:
            x, attn_weights = block(x, mask, return_attention)
            if return_attention:
                all_attention_weights.append(attn_weights)

        # Output projection
        output = self.output_proj(x)  # (batch, seq_len, d_input)

        return output, all_attention_weights

    def predict_next(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict next state given context.

        Args:
            context: Context of shape (batch, context_len, d_input)

        Returns:
            Prediction for next state of shape (batch, d_input)
        """
        with torch.no_grad():
            output, _ = self.forward(context)
            # Return prediction at last position
            return output[:, -1, :]

    def autoregressive_predict(self,
                              context: torch.Tensor,
                              n_steps: int) -> torch.Tensor:
        """
        Generate n_steps predictions autoregressively.

        Args:
            context: Initial context of shape (batch, context_len, d_input)
            n_steps: Number of steps to predict

        Returns:
            Predictions of shape (batch, n_steps, d_input)
        """
        batch_size = context.shape[0]
        predictions = []

        # Start with context
        current_seq = context

        for _ in range(n_steps):
            # Predict next state
            next_state = self.predict_next(current_seq)  # (batch, d_input)
            predictions.append(next_state)

            # Append to sequence (keep only last max_seq_len-1 to make room for new prediction)
            next_state = next_state.unsqueeze(1)  # (batch, 1, d_input)
            current_seq = torch.cat([current_seq, next_state], dim=1)

            if current_seq.shape[1] > self.max_seq_len:
                current_seq = current_seq[:, -self.max_seq_len:, :]

        predictions = torch.stack(predictions, dim=1)  # (batch, n_steps, d_input)
        return predictions


if __name__ == "__main__":
    # Test model
    print("Testing model...")

    batch_size = 4
    seq_len = 50
    d_input = 5

    # Create model
    model = GPTModel(
        d_input=d_input,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_input)
    output, attention_weights = model(x, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
    print(f"Attention weight shape: {attention_weights[0].shape}")

    # Test prediction
    context = torch.randn(batch_size, 20, d_input)
    next_state = model.predict_next(context)
    print(f"\nNext state prediction shape: {next_state.shape}")

    # Test autoregressive prediction
    predictions = model.autoregressive_predict(context, n_steps=10)
    print(f"Autoregressive predictions shape: {predictions.shape}")
