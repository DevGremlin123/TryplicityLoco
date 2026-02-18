"""
Mixture of Experts Layer
DeepSeekMoE style: 64 fine-grained experts + 2 shared experts.
Dynamic sparsity routing with lateral inhibition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from model.bitnet import BitLinear
from brain.spike_activation import SpikeGatedMLP
from brain.dynamic_sparsity import LateralInhibitionRouter, DynamicSparsityRouter


class SharedExpert(nn.Module):
    """Shared expert that processes all tokens (always active)."""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 use_bitnet: bool = True, use_spike: bool = True):
        super().__init__()
        self.mlp = SpikeGatedMLP(hidden_size, intermediate_size,
                                 use_bitnet=use_bitnet, use_spike=use_spike)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MoELayer(nn.Module):
    """
    Mixture of Experts with:
    - 64 fine-grained experts (small, specialized)
    - 2 shared experts (always active, handle common patterns)
    - Dynamic sparsity routing (4-12 active per token)
    - Lateral inhibition (force diverse expert selection)
    - Spike activations inside expert FFNs
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expert_hidden_size: int = 512,
        num_experts: int = 64,
        num_shared_experts: int = 2,
        base_active: int = 8,
        min_active: int = 4,
        max_active: int = 12,
        use_bitnet: bool = True,
        use_spike: bool = True,
        use_dynamic_sparsity: bool = True,
        use_lateral_inhibition: bool = True,
        inhibition_strength: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts

        # Routed experts
        self.experts = nn.ModuleList([
            SpikeGatedMLP(hidden_size, expert_hidden_size,
                         use_bitnet=use_bitnet, use_spike=use_spike)
            for _ in range(num_experts)
        ])

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            SharedExpert(hidden_size, expert_hidden_size * 2,  # Shared experts are larger
                        use_bitnet=use_bitnet, use_spike=use_spike)
            for _ in range(num_shared_experts)
        ])

        # Router
        if use_lateral_inhibition and use_dynamic_sparsity:
            self.router = LateralInhibitionRouter(
                hidden_size, num_experts, base_active, min_active, max_active,
                inhibition_strength=inhibition_strength,
            )
        elif use_dynamic_sparsity:
            self.router = DynamicSparsityRouter(
                hidden_size, num_experts, base_active, min_active, max_active,
            )
        else:
            self.router = DynamicSparsityRouter(
                hidden_size, num_experts, base_active, min_active, max_active,
                enabled=False,  # Falls back to fixed top-k
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
            aux_loss: router auxiliary loss
            stats: monitoring dict
        """
        batch, seq_len, hidden = x.shape

        # Get routing weights
        routing_weights, aux_loss, stats = self.router(x)
        # routing_weights: [B, S, num_experts] (sparse)

        # Shared experts (process all tokens)
        shared_output = sum(se(x) for se in self.shared_experts)

        # Routed experts (sparse computation)
        # Flatten batch and seq for expert dispatch
        x_flat = x.reshape(-1, hidden)  # [B*S, hidden]
        routing_flat = routing_weights.reshape(-1, self.num_experts)  # [B*S, num_experts]

        # Find which tokens go to which experts
        expert_output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            # Get tokens routed to this expert
            token_weights = routing_flat[:, i]  # [B*S]
            active_mask = token_weights > 0

            if active_mask.any():
                # Only compute for tokens that are actually routed here
                active_tokens = x_flat[active_mask]  # [num_active, hidden]
                active_weights = token_weights[active_mask].unsqueeze(-1)  # [num_active, 1]
                expert_result = expert(active_tokens) * active_weights
                expert_output[active_mask] += expert_result

        expert_output = expert_output.reshape(batch, seq_len, hidden)

        # Combine shared + routed
        output = shared_output + expert_output

        # Add sparsity stats
        total_sparsity = sum(e.get_sparsity() for e in self.experts) / len(self.experts)
        stats["expert_activation_sparsity"] = total_sparsity

        return output, aux_loss, stats

    def get_expert_weights_for_similarity(self):
        """Get expert gate weights for lateral inhibition similarity update."""
        return [e.gate_proj.weight.data for e in self.experts]

    def update_lateral_inhibition(self):
        """Update expert similarity matrix if using lateral inhibition."""
        if isinstance(self.router, LateralInhibitionRouter) and self.router.should_update_similarity():
            weights = self.get_expert_weights_for_similarity()
            self.router.update_similarity_matrix(weights)


if __name__ == "__main__":
    print("MoE layer smoke test...")
    moe = MoELayer(
        hidden_size=256,
        expert_hidden_size=128,
        num_experts=16,
        num_shared_experts=2,
        base_active=4,
        min_active=2,
        max_active=6,
        use_bitnet=True,
        use_spike=True,
        use_dynamic_sparsity=True,
        use_lateral_inhibition=True,
    )

    x = torch.randn(2, 16, 256)
    moe.train()
    out, aux_loss, stats = moe(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print(f"  Aux loss: {aux_loss.item():.4f}")
    print(f"  Active experts: {stats['avg_active_experts']:.1f} "
          f"(range: {stats['min_active_experts']}-{stats['max_active_experts']})")
    print(f"  Expert activation sparsity: {stats['expert_activation_sparsity']:.1%}")

    # Parameter count
    total = sum(p.numel() for p in moe.parameters())
    print(f"  Total parameters: {total:,}")
    print("  MoE layer OK.")
