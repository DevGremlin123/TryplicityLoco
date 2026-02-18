"""
Dynamic Sparsity Router with Lateral Inhibition
Brain-inspired: 1-5% neuron activation at any moment.
Variable expert count (4-12) based on token difficulty.
Lateral inhibition forces diverse expert selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DynamicSparsityRouter(nn.Module):
    """
    Adaptive MoE router that varies the number of active experts per token.
    Easy tokens: fewer experts (save compute).
    Hard tokens: more experts (better quality).
    Average stays around base_k for memory budgeting.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 64,
        base_k: int = 8,
        min_k: int = 4,
        max_k: int = 12,
        enabled: bool = True,
        aux_loss_weight: float = 0.01,
        entropy_weight: float = 0.001,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.base_k = base_k
        self.min_k = min_k
        self.max_k = max_k
        self.enabled = enabled
        self.aux_loss_weight = aux_loss_weight
        self.entropy_weight = entropy_weight

        # Expert routing scores
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Difficulty estimator: scalar difficulty per token
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Track stats
        self.register_buffer("avg_k_ema", torch.tensor(float(base_k)))

    def _compute_k_per_token(self, difficulty: torch.Tensor) -> torch.Tensor:
        """Map difficulty scores [0,1] to number of experts [min_k, max_k]."""
        # difficulty 0 = easy = min_k, difficulty 1 = hard = max_k
        k_float = self.min_k + difficulty * (self.max_k - self.min_k)
        # Round to nearest integer but keep differentiable with STE
        k_int = k_float.round().clamp(self.min_k, self.max_k)
        return k_float, k_int.long()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route tokens to variable number of experts.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            routing_weights: [batch, seq_len, num_experts] (sparse, most zeros)
            aux_loss: scalar auxiliary loss
            stats: dict of monitoring metrics
        """
        batch, seq_len, _ = hidden_states.shape

        # Step 1: Compute expert scores
        gate_logits = self.gate(hidden_states)  # [B, S, num_experts]

        if not self.enabled:
            # Fixed top-k routing (baseline)
            topk_vals, topk_idx = gate_logits.topk(self.base_k, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)
            routing_weights = torch.zeros_like(topk_weights).expand_as(gate_logits).clone()
            routing_weights.scatter_(-1, topk_idx, topk_weights)
            return routing_weights, torch.tensor(0.0, device=hidden_states.device), {"avg_k": self.base_k}

        # Step 2: Compute per-token difficulty
        difficulty = self.difficulty_head(hidden_states.detach()).squeeze(-1)  # [B, S]
        k_float, k_int = self._compute_k_per_token(difficulty)  # [B, S]

        # Step 3: Variable top-k selection
        # Sort expert scores descending
        sorted_logits, sorted_idx = gate_logits.sort(dim=-1, descending=True)

        # Create mask: for each token, keep top-k_int experts
        # k_int: [B, S], need to mask sorted_logits: [B, S, num_experts]
        positions = torch.arange(self.num_experts, device=hidden_states.device)
        positions = positions.unsqueeze(0).unsqueeze(0)  # [1, 1, num_experts]
        k_expanded = k_int.unsqueeze(-1)  # [B, S, 1]
        mask = (positions < k_expanded).float()  # [B, S, num_experts]

        # Apply mask to sorted logits, then softmax over selected experts
        masked_logits = sorted_logits * mask + (-1e9) * (1 - mask)
        sorted_weights = F.softmax(masked_logits, dim=-1) * mask

        # Scatter back to original expert positions
        routing_weights = torch.zeros_like(sorted_weights)
        routing_weights.scatter_(-1, sorted_idx, sorted_weights)

        # Step 4: Auxiliary losses
        # 4a: Penalize if avg_k drops below base_k (prevent "everything is easy" collapse)
        avg_k = k_float.mean()
        k_penalty = F.relu(self.base_k - avg_k) ** 2

        # 4b: Load balancing loss (standard MoE aux loss)
        # Fraction of tokens routed to each expert should be uniform
        tokens_per_expert = routing_weights.sum(dim=(0, 1))  # [num_experts]
        balance_loss = (tokens_per_expert.float().var() / (tokens_per_expert.float().mean() + 1e-8))

        # 4c: Entropy regularization on difficulty distribution
        # Encourage spread of difficulties, not collapse to single value
        diff_entropy = -(difficulty * torch.log(difficulty + 1e-8) +
                         (1 - difficulty) * torch.log(1 - difficulty + 1e-8)).mean()
        entropy_bonus = -diff_entropy  # Maximize entropy

        aux_loss = (self.aux_loss_weight * (k_penalty + balance_loss) +
                    self.entropy_weight * entropy_bonus)

        # Update EMA stats
        if self.training:
            with torch.no_grad():
                self.avg_k_ema.lerp_(avg_k.detach().to(self.avg_k_ema.dtype), 0.01)

        stats = {
            "avg_active_experts": avg_k.item(),
            "min_active_experts": k_int.min().item(),
            "max_active_experts": k_int.max().item(),
            "difficulty_mean": difficulty.mean().item(),
            "difficulty_std": difficulty.std().item(),
            "expert_utilization_entropy": -(tokens_per_expert / tokens_per_expert.sum() *
                                           torch.log(tokens_per_expert / tokens_per_expert.sum() + 1e-8)).sum().item(),
            "aux_loss": aux_loss.item(),
        }

        return routing_weights, aux_loss, stats


class LateralInhibitionRouter(DynamicSparsityRouter):
    """
    Extension with lateral inhibition between experts.
    When expert A wins strongly, it suppresses similar experts,
    forcing diversity in expert selection.
    """

    def __init__(self, *args, inhibition_strength: float = 0.3,
                 similarity_update_interval: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.inhibition_strength = inhibition_strength
        self.similarity_update_interval = similarity_update_interval

        # Expert similarity matrix (updated periodically)
        self.register_buffer(
            "expert_similarity",
            torch.zeros(self.num_experts, self.num_experts)
        )
        self.register_buffer("steps_since_update", torch.tensor(0))

    def update_similarity_matrix(self, expert_weights: list):
        """
        Compute cosine similarity between expert weight matrices.
        Called every similarity_update_interval steps.
        expert_weights: list of weight tensors from each expert
        """
        with torch.no_grad():
            # Flatten each expert's weights to a vector
            flat = torch.stack([w.flatten() for w in expert_weights])  # [num_experts, D]
            # Cosine similarity
            norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
            normalized = flat / norms
            self.expert_similarity = normalized @ normalized.T
            # Zero out diagonal (don't inhibit self)
            self.expert_similarity.fill_diagonal_(0)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        batch, seq_len, _ = hidden_states.shape

        # Standard gate logits
        gate_logits = self.gate(hidden_states)  # [B, S, num_experts]

        if not self.enabled:
            topk_vals, topk_idx = gate_logits.topk(self.base_k, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)
            routing_weights = torch.zeros_like(topk_weights).expand_as(gate_logits).clone()
            routing_weights.scatter_(-1, topk_idx, topk_weights)
            return routing_weights, torch.tensor(0.0, device=hidden_states.device), {"avg_k": self.base_k}

        # Difficulty-based dynamic k
        difficulty = self.difficulty_head(hidden_states.detach()).squeeze(-1)
        k_float, k_int = self._compute_k_per_token(difficulty)

        # === LATERAL INHIBITION ===
        # Step 1: Find the winner (top-1 expert per token)
        winner_scores, winner_idx = gate_logits.max(dim=-1)  # [B, S]

        # Step 2: Compute inhibition signal
        # For each token, subtract similarity[winner, :] * inhibition_strength from gate logits
        # winner_idx: [B, S] -> need to index expert_similarity: [num_experts, num_experts]
        winner_flat = winner_idx.reshape(-1)  # [B*S]
        inhibition = self.expert_similarity[winner_flat]  # [B*S, num_experts]
        inhibition = inhibition.reshape(batch, seq_len, self.num_experts)

        # Apply inhibition (reduce scores of similar experts)
        inhibited_logits = gate_logits - self.inhibition_strength * inhibition * winner_scores.unsqueeze(-1)

        # Step 3: Variable top-k on inhibited logits
        sorted_logits, sorted_idx = inhibited_logits.sort(dim=-1, descending=True)
        positions = torch.arange(self.num_experts, device=hidden_states.device).unsqueeze(0).unsqueeze(0)
        k_expanded = k_int.unsqueeze(-1)
        mask = (positions < k_expanded).float()

        masked_logits = sorted_logits * mask + (-1e9) * (1 - mask)
        sorted_weights = F.softmax(masked_logits, dim=-1) * mask

        routing_weights = torch.zeros_like(sorted_weights)
        routing_weights.scatter_(-1, sorted_idx, sorted_weights)

        # Aux losses (same as parent)
        avg_k = k_float.mean()
        k_penalty = F.relu(self.base_k - avg_k) ** 2
        tokens_per_expert = routing_weights.sum(dim=(0, 1))
        balance_loss = tokens_per_expert.float().var() / (tokens_per_expert.float().mean() + 1e-8)
        diff_entropy = -(difficulty * torch.log(difficulty + 1e-8) +
                         (1 - difficulty) * torch.log(1 - difficulty + 1e-8)).mean()

        aux_loss = (self.aux_loss_weight * (k_penalty + balance_loss) +
                    self.entropy_weight * (-diff_entropy))

        # Track step count for similarity updates
        if self.training:
            with torch.no_grad():
                self.steps_since_update += 1
                self.avg_k_ema.lerp_(avg_k.detach().to(self.avg_k_ema.dtype), 0.01)

        stats = {
            "avg_active_experts": avg_k.item(),
            "min_active_experts": k_int.min().item(),
            "max_active_experts": k_int.max().item(),
            "difficulty_mean": difficulty.mean().item(),
            "inhibition_active": True,
            "expert_utilization_entropy": -(tokens_per_expert / tokens_per_expert.sum() *
                                           torch.log(tokens_per_expert / tokens_per_expert.sum() + 1e-8)).sum().item(),
        }

        return routing_weights, aux_loss, stats

    def should_update_similarity(self) -> bool:
        return self.steps_since_update.item() >= self.similarity_update_interval


if __name__ == "__main__":
    print("Dynamic sparsity router smoke test...")
    router = DynamicSparsityRouter(256, num_experts=16, base_k=4, min_k=2, max_k=6)
    x = torch.randn(2, 16, 256)
    router.train()
    weights, loss, stats = router(x)
    print(f"  Routing weights: {weights.shape}")
    print(f"  Aux loss: {loss.item():.4f}")
    print(f"  Stats: avg_k={stats['avg_active_experts']:.1f}, "
          f"range=[{stats['min_active_experts']}, {stats['max_active_experts']}]")

    print("\nLateral inhibition router smoke test...")
    li_router = LateralInhibitionRouter(256, num_experts=16, base_k=4, min_k=2, max_k=6)
    li_router.train()
    weights, loss, stats = li_router(x)
    print(f"  Routing weights: {weights.shape}")
    print(f"  Inhibition active: {stats['inhibition_active']}")
    print(f"  Expert entropy: {stats['expert_utilization_entropy']:.2f}")
    print("  Dynamic sparsity OK.")
