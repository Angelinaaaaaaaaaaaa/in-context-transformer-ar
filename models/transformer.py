"""
Decoder-only GPT architecture for in-context learning.

Config is flexible, but in our main experiments we often use:
- 1 layer, 4 heads, d_model=256, FFN=1024
- Learned positional encodings
- Causal mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer，不作为模型参数更新，但随 state_dict 保存
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention with causal masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Causal mask of shape (seq_len, seq_len)
            return_attention: If True, return attention weights

        Returns:
            - output: Tensor of shape (batch, seq_len, d_model)
            - attention_weights: If return_attention, tensor of shape
              (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )  # (batch, n_heads, seq_len, d_head)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, seq_len, seq_len)

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax (dropout currently disabled)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)
        # attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, d_head)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.d_head
        )

        # Output projection
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output, None


class FeedForward(nn.Module):
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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        x_norm1 = self.norm1(x)
        attn_output, attn_weights = self.attention(x_norm1, mask, return_attention)
        
        # 2. Residual Connection 
        x = x + self.dropout(attn_output)

        x_norm2 = self.norm2(x)
        ff_output = self.ff(x_norm2)
        
        # 4. Residual Connection
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

    def __init__(
        self,
        d_input: int,
        d_model: int = 256,
        n_layers: int = 1,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Learned positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        # Output projection
        self.output_proj = nn.Linear(d_model, d_input)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights:
        - Linear: Xavier Uniform, Bias=0
        - LayerNorm: Weight=1, Bias=0
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive attention."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
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
        x = self.pos_encoding(x)
        # x = self.dropout(x)

        # Create causal mask
        mask = self._create_causal_mask(seq_len, device)

        # Apply Transformer blocks
        all_attention_weights = [] if return_attention else None
        for block in self.blocks:
            x, attn_weights = block(x, mask, return_attention)
            if return_attention:
                all_attention_weights.append(attn_weights)

        x = self.ln_f(x)
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
            return output[:, -1, :]

    def autoregressive_predict(self, context: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Generate n_steps predictions autoregressively.

        Args:
            context: Initial context of shape (batch, context_len, d_input)
            n_steps: Number of steps to predict

        Returns:
            Predictions of shape (batch, n_steps, d_input)
        """
        predictions = []
        current_seq = context

        for _ in range(n_steps):
            # Predict next state
            next_state = self.predict_next(current_seq)  # (batch, d_input)
            predictions.append(next_state)

            next_state = next_state.unsqueeze(1)  # (batch, 1, d_input)
            current_seq = torch.cat([current_seq, next_state], dim=1)

            if current_seq.shape[1] > self.max_seq_len:
                current_seq = current_seq[:, -self.max_seq_len :, :]

        predictions = torch.stack(predictions, dim=1)
        return predictions