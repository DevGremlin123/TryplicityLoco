"""
Grouped Query Attention (GQA) Layer
4 layers out of 32 use this (at positions 8, 16, 24, 32).
GQA: 16 query heads, 4 KV heads (4:1 ratio). Saves memory.
Handles the tasks Mamba struggles with: precise retrieval, copying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.bitnet import BitLinear


class GQAAttention(nn.Module):
    """
    Grouped Query Attention with RoPE positional encoding.
    16 query heads, 4 KV heads. Each KV head is shared by 4 query heads.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        max_seq_len: int = 4096,
        use_bitnet: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads  # 4 queries per KV head
        self.dropout = dropout

        Linear = BitLinear if use_bitnet else nn.Linear

        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

        # RoPE frequencies (precomputed)
        self._init_rope(max_seq_len)

    def _init_rope(self, max_seq_len: int):
        """Precompute rotary positional embedding frequencies."""
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [max_seq, head_dim//2]
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def _apply_rope(self, x: torch.Tensor, seq_offset: int = 0) -> torch.Tensor:
        """Apply rotary positional encoding."""
        seq_len = x.shape[2]
        cos = self.cos_cache[seq_offset:seq_offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[seq_offset:seq_offset + seq_len].unsqueeze(0).unsqueeze(0)

        # Split into pairs and rotate
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1
        ).flatten(-2)
        return rotated

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            mask: optional causal mask
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q: [B, num_heads, S, head_dim], k/v: [B, num_kv_heads, S, head_dim]

        # Apply RoPE
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # Expand KV heads for GQA: repeat each KV head for its group of query heads
        k = k.repeat_interleave(self.num_groups, dim=1)  # [B, num_heads, S, head_dim]
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]

        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_weights.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            attn_weights.masked_fill_(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, S, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)

        return self.o_proj(attn_output)


if __name__ == "__main__":
    print("GQA Attention smoke test...")
    attn = GQAAttention(hidden_size=256, num_heads=8, num_kv_heads=2, head_dim=32)
    x = torch.randn(2, 32, 256)
    out = attn(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    # Check causal: future shouldn't affect past
    x2 = x.clone()
    x2[:, 16:] = torch.randn(2, 16, 256)
    out2 = attn(x2)
    past_diff = (out[:, :16] - out2[:, :16]).abs().max().item()
    print(f"  Causal check: {past_diff:.6f} (should be ~0)")

    # Parameter count
    total_params = sum(p.numel() for p in attn.parameters())
    print(f"  Parameters: {total_params:,}")
    print("  GQA Attention OK.")
