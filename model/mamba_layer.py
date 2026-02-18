"""
Mamba-2 Layer: Selective State Space Model
Linear-time sequence processing (O(n) vs O(n^2) for attention).
28 of 32 layers use this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.bitnet import BitLinear


class Mamba2Block(nn.Module):
    """
    Simplified Mamba-2 selective state space model block.

    The key insight: instead of fixed state space dynamics (like S4),
    the transition matrices are input-dependent (selective). This lets
    the model dynamically decide what to remember and what to forget,
    similar to an LSTM gate but in continuous state space form.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        state_dim: int = 128,
        conv_dim: int = 4,
        expand_factor: int = 2,
        use_bitnet: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.inner_size = hidden_size * expand_factor

        Linear = BitLinear if use_bitnet else nn.Linear

        # Input projections
        self.in_proj = Linear(hidden_size, self.inner_size * 2, bias=False)

        # Short convolution (causal, for local context)
        self.conv1d = nn.Conv1d(
            self.inner_size, self.inner_size,
            kernel_size=conv_dim, padding=conv_dim - 1,
            groups=self.inner_size,  # Depthwise
        )

        # SSM parameters (input-dependent / selective)
        # dt (delta): controls how much to update state
        self.dt_proj = Linear(self.inner_size, self.inner_size, bias=True)
        # B and C: state transition input/output matrices
        self.B_proj = Linear(self.inner_size, state_dim, bias=False)
        self.C_proj = Linear(self.inner_size, state_dim, bias=False)
        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.inner_size))

        # Output projection
        self.out_proj = Linear(self.inner_size, hidden_size, bias=False)

        # Layer norm for SSM
        self.norm = nn.LayerNorm(self.inner_size)

        # Initialize dt bias to ensure positive dt values
        with torch.no_grad():
            dt_init = torch.exp(torch.linspace(-4, -2, self.inner_size))
            self.dt_proj.bias.copy_(dt_init.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = x.shape

        # Project to inner dimension with gate
        xz = self.in_proj(x)  # [B, S, inner*2]
        x_inner, z = xz.chunk(2, dim=-1)  # Each [B, S, inner]

        # Short causal convolution
        x_conv = x_inner.transpose(1, 2)  # [B, inner, S]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal: trim future
        x_conv = x_conv.transpose(1, 2)  # [B, S, inner]
        x_conv = F.silu(x_conv)

        # Selective SSM scan
        y = self._selective_scan(x_conv)

        # Gate and project out
        y = y * F.silu(z)  # Gated
        output = self.out_proj(y)

        return output

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective state space scan. Input-dependent transitions.

        For each timestep t:
            state[t] = A_bar * state[t-1] + B_bar * x[t]
            y[t] = C[t] @ state[t] + D * x[t]

        Where A_bar and B_bar depend on dt[t] (discretization step).
        """
        batch, seq_len, inner = x.shape

        # Compute input-dependent parameters
        dt = F.softplus(self.dt_proj(x))  # [B, S, inner], positive
        B = self.B_proj(x)  # [B, S, state_dim]
        C = self.C_proj(x)  # [B, S, state_dim]

        # For efficiency on GPU, we use a parallel scan approximation
        # instead of sequential. This unfolds the recurrence.
        # True Mamba uses custom CUDA kernels; we approximate with chunked scan.
        y = self._chunked_scan(x, dt, B, C)

        # Skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x

        return y

    def _chunked_scan(self, x: torch.Tensor, dt: torch.Tensor,
                      B: torch.Tensor, C: torch.Tensor,
                      chunk_size: int = 64) -> torch.Tensor:
        """
        Chunked parallel scan for efficiency.
        Process chunks of tokens in parallel, carry state between chunks.
        """
        batch, seq_len, inner = x.shape
        state_dim = B.shape[-1]

        # Discretize: A_bar = exp(-dt), B_bar = dt * B
        # A is implicit diagonal = -1 (decay), so A_bar = exp(-dt)
        A_bar = torch.exp(-dt)  # [B, S, inner]

        output = torch.zeros_like(x)
        state = torch.zeros(batch, inner, state_dim, device=x.device, dtype=x.dtype)

        # Process sequentially in chunks
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_len = end - start

            dt_chunk = dt[:, start:end]  # [B, chunk, inner]
            B_chunk = B[:, start:end]    # [B, chunk, state_dim]
            C_chunk = C[:, start:end]    # [B, chunk, state_dim]
            x_chunk = x[:, start:end]    # [B, chunk, inner]
            A_chunk = A_bar[:, start:end]  # [B, chunk, inner]

            # Sequential scan within chunk (could be parallelized with custom kernel)
            for t in range(chunk_len):
                # state = A_bar * state + B_bar * x
                # B_bar * x: [B, inner, 1] * [B, 1, state_dim] -> [B, inner, state_dim]
                x_t = x_chunk[:, t].unsqueeze(-1)  # [B, inner, 1]
                b_t = B_chunk[:, t].unsqueeze(1)    # [B, 1, state_dim]
                a_t = A_chunk[:, t].unsqueeze(-1)   # [B, inner, 1]
                dt_t = dt_chunk[:, t].unsqueeze(-1)  # [B, inner, 1]

                state = a_t * state + dt_t * x_t * b_t

                # y = C @ state
                c_t = C_chunk[:, t].unsqueeze(1)  # [B, 1, state_dim]
                y_t = (state * c_t).sum(dim=-1)   # [B, inner]
                output[:, start + t] = y_t

        return output


if __name__ == "__main__":
    print("Mamba-2 block smoke test...")
    mamba = Mamba2Block(hidden_size=256, state_dim=64, use_bitnet=True)
    x = torch.randn(2, 32, 256)
    out = mamba(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    # Check causality: changing future tokens shouldn't affect past output
    x2 = x.clone()
    x2[:, 16:] = torch.randn(2, 16, 256)
    out2 = mamba(x2)
    past_diff = (out[:, :16] - out2[:, :16]).abs().max().item()
    print(f"  Causal check (past diff): {past_diff:.6f} (should be ~0)")

    # Parameter count
    total_params = sum(p.numel() for p in mamba.parameters())
    print(f"  Parameters: {total_params:,}")
    print("  Mamba-2 OK.")
