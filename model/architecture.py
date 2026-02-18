"""
Tryplicity Model Architecture
Hybrid Mamba-2 + GQA Attention + MoE + BitNet + Hyper-Connections
3B total params, ~600M active per token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from model.bitnet import BitLinear
from model.mamba_layer import Mamba2Block
from model.attention_layer import GQAAttention
from model.moe_layer import MoELayer
from model.hyper_connections import HyperConnection, HyperConnectionManager


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class TryplicityBlock(nn.Module):
    """
    Single transformer block.
    Pre-norm architecture: norm -> sequence_mixer -> residual -> norm -> moe -> residual
    sequence_mixer is either Mamba-2 or GQA Attention depending on layer index.
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int = 2048,
        num_layers: int = 32,
        attention_layer_indices: list = None,
        # Mamba params
        mamba_state_dim: int = 128,
        mamba_conv_dim: int = 4,
        mamba_expand_factor: int = 2,
        # Attention params
        num_attention_heads: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        max_seq_len: int = 4096,
        # MoE params
        num_experts: int = 64,
        num_shared_experts: int = 2,
        expert_hidden_size: int = 512,
        base_active_experts: int = 8,
        min_active_experts: int = 4,
        max_active_experts: int = 12,
        # Feature flags
        use_bitnet: bool = True,
        use_spike: bool = True,
        use_dynamic_sparsity: bool = True,
        use_lateral_inhibition: bool = True,
        use_hyper_connections: bool = True,
        hyper_num_streams: int = 4,
        # Other
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_hyper_connections = use_hyper_connections

        if attention_layer_indices is None:
            attention_layer_indices = [7, 15, 23, 31]

        self.is_attention_layer = layer_idx in attention_layer_indices

        # Pre-norm
        self.norm1 = RMSNorm(hidden_size, rms_norm_eps)
        self.norm2 = RMSNorm(hidden_size, rms_norm_eps)

        # Sequence mixer: Mamba-2 or GQA Attention
        if self.is_attention_layer:
            self.sequence_mixer = GQAAttention(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                use_bitnet=use_bitnet,
                dropout=dropout,
            )
        else:
            self.sequence_mixer = Mamba2Block(
                hidden_size=hidden_size,
                state_dim=mamba_state_dim,
                conv_dim=mamba_conv_dim,
                expand_factor=mamba_expand_factor,
                use_bitnet=use_bitnet,
            )

        # MoE feedforward
        self.moe = MoELayer(
            hidden_size=hidden_size,
            expert_hidden_size=expert_hidden_size,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            base_active=base_active_experts,
            min_active=min_active_experts,
            max_active=max_active_experts,
            use_bitnet=use_bitnet,
            use_spike=use_spike,
            use_dynamic_sparsity=use_dynamic_sparsity,
            use_lateral_inhibition=use_lateral_inhibition,
        )

        # Hyper-connection for this layer
        if use_hyper_connections:
            self.hyper_conn = HyperConnection(
                hidden_size, hyper_num_streams, layer_idx, num_layers
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        streams: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, dict]:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            streams: [num_streams, batch, seq_len, hidden] if using hyper-connections

        Returns:
            output: [batch, seq_len, hidden_size]
            updated_streams: updated hyper-connection streams
            aux_loss: MoE routing loss
            stats: monitoring dict
        """
        # Get input from hyper-connections if enabled
        if self.use_hyper_connections and streams is not None:
            layer_input = self.hyper_conn.select_input(streams)
        else:
            layer_input = x

        # Pre-norm + sequence mixer + residual
        normed = self.norm1(layer_input)
        mixed = self.sequence_mixer(normed)
        mixed = self.dropout(mixed)
        h = layer_input + mixed

        # Pre-norm + MoE + residual
        normed2 = self.norm2(h)
        moe_out, aux_loss, stats = self.moe(normed2)
        moe_out = self.dropout(moe_out)
        output = h + moe_out

        # Update hyper-connection streams
        updated_streams = streams
        if self.use_hyper_connections and streams is not None:
            updated_streams = self.hyper_conn.update_streams(streams, layer_input, output)

        stats["layer_idx"] = self.layer_idx
        stats["layer_type"] = "attention" if self.is_attention_layer else "mamba"

        return output, updated_streams, aux_loss, stats


class TryplicityModel(nn.Module):
    """
    Full Tryplicity language model.
    Embedding -> 32 TryplicityBlocks -> RMSNorm -> LM Head
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_layers: int = 32,
        max_seq_len: int = 4096,
        # Mamba
        mamba_state_dim: int = 128,
        mamba_conv_dim: int = 4,
        mamba_expand_factor: int = 2,
        # Attention
        num_attention_heads: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        # MoE
        num_experts: int = 64,
        num_shared_experts: int = 2,
        expert_hidden_size: int = 512,
        base_active_experts: int = 8,
        min_active_experts: int = 4,
        max_active_experts: int = 12,
        # Features
        use_bitnet: bool = True,
        use_spike: bool = True,
        use_dynamic_sparsity: bool = True,
        use_lateral_inhibition: bool = True,
        use_hyper_connections: bool = True,
        hyper_num_streams: int = 4,
        # Multi-token prediction
        num_predict_tokens: int = 4,
        # Other
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_hyper_connections = use_hyper_connections
        self.gradient_checkpointing = gradient_checkpointing
        self.num_predict_tokens = num_predict_tokens

        attention_layer_indices = [7, 15, 23, 31][:num_layers // 8 or 1]
        # For small models, ensure at least 1 attention layer
        if num_layers < 8:
            attention_layer_indices = [num_layers - 1]

        # Token embedding (not BitNet -- embeddings need full precision)
        self.embed = nn.Embedding(vocab_size, hidden_size)

        # Hyper-connection manager
        if use_hyper_connections:
            self.hyper_manager = HyperConnectionManager(hidden_size, hyper_num_streams, num_layers)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TryplicityBlock(
                layer_idx=i,
                hidden_size=hidden_size,
                num_layers=num_layers,
                attention_layer_indices=attention_layer_indices,
                mamba_state_dim=mamba_state_dim,
                mamba_conv_dim=mamba_conv_dim,
                mamba_expand_factor=mamba_expand_factor,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                expert_hidden_size=expert_hidden_size,
                base_active_experts=base_active_experts,
                min_active_experts=min_active_experts,
                max_active_experts=max_active_experts,
                use_bitnet=use_bitnet,
                use_spike=use_spike,
                use_dynamic_sparsity=use_dynamic_sparsity,
                use_lateral_inhibition=use_lateral_inhibition,
                use_hyper_connections=use_hyper_connections,
                hyper_num_streams=hyper_num_streams,
                dropout=dropout,
                rms_norm_eps=rms_norm_eps,
            )
            for i in range(num_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(hidden_size, rms_norm_eps)

        # LM head (weight-tied with embedding)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed.weight

        # Multi-token prediction heads (additional heads for tokens 2, 3, 4)
        if num_predict_tokens > 1:
            self.multi_token_heads = nn.ModuleList([
                nn.Sequential(
                    RMSNorm(hidden_size, rms_norm_eps),
                    nn.Linear(hidden_size, vocab_size, bias=False),
                )
                for _ in range(num_predict_tokens - 1)
            ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] (shifted targets for LM loss)

        Returns:
            dict with 'logits', 'loss', 'aux_loss', 'stats'
        """
        batch, seq_len = input_ids.shape

        # Embed tokens
        x = self.embed(input_ids)  # [B, S, hidden]

        # Init hyper-connection streams
        streams = None
        if self.use_hyper_connections:
            streams = self.hyper_manager.init_streams(x)

        # Pass through all layers
        total_aux_loss = 0.0
        all_stats = []

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, streams, aux_loss, stats = torch.utils.checkpoint.checkpoint(
                    layer, x, streams, use_reentrant=False,
                )
            else:
                x, streams, aux_loss, stats = layer(x, streams)
            total_aux_loss = total_aux_loss + aux_loss
            all_stats.append(stats)

        # Collapse hyper-connection streams
        if self.use_hyper_connections and streams is not None:
            x = self.hyper_manager.collapse_streams(streams)

        # Final norm
        x = self.final_norm(x)

        # LM head (next token prediction)
        logits = self.lm_head(x)  # [B, S, vocab]

        result = {
            "logits": logits,
            "hidden_states": x,
            "aux_loss": total_aux_loss,
            "stats": all_stats,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Standard next-token prediction loss
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Multi-token prediction losses
            multi_token_loss = 0.0
            if self.num_predict_tokens > 1 and hasattr(self, "multi_token_heads"):
                for i, head in enumerate(self.multi_token_heads):
                    offset = i + 2  # Predict token at position +2, +3, +4
                    if seq_len > offset:
                        mt_logits = head(x[:, :-offset])
                        mt_labels = labels[:, offset:]
                        mt_loss = F.cross_entropy(
                            mt_logits.reshape(-1, self.vocab_size),
                            mt_labels.reshape(-1),
                            ignore_index=-100,
                        )
                        multi_token_loss = multi_token_loss + mt_loss

                multi_token_loss = multi_token_loss / max(1, len(self.multi_token_heads))

            # Total loss = main + multi-token + MoE aux
            total_loss = loss + 0.5 * multi_token_loss + total_aux_loss
            result["loss"] = total_loss
            result["lm_loss"] = loss
            result["multi_token_loss"] = multi_token_loss

        return result

    def count_parameters(self) -> dict:
        """Count parameters breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embed_params = sum(p.numel() for p in self.embed.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "embedding": embed_params,
            "per_layer_avg": (total - embed_params) // max(1, self.num_layers),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            context = generated[:, -4096:]

            result = self(context)
            logits = result["logits"][:, -1, :]  # Last position

            # Temperature
            logits = logits / temperature

            # Top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k).values[:, -1:]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

        return generated


def create_tiny_model() -> TryplicityModel:
    """Create a tiny model for 1-minute test training."""
    return TryplicityModel(
        vocab_size=32000,
        hidden_size=256,
        num_layers=4,
        max_seq_len=512,
        mamba_state_dim=32,
        mamba_conv_dim=4,
        mamba_expand_factor=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=64,
        num_experts=8,
        num_shared_experts=1,
        expert_hidden_size=128,
        base_active_experts=4,
        min_active_experts=2,
        max_active_experts=6,
        use_bitnet=True,
        use_spike=True,
        use_dynamic_sparsity=True,
        use_lateral_inhibition=True,
        use_hyper_connections=True,
        hyper_num_streams=2,
        num_predict_tokens=2,
        gradient_checkpointing=False,
    )


def create_5090_model() -> TryplicityModel:
    """Create a medium model (~1B total params) that fits in 32 GB VRAM for RTX 5090.
    Same architecture as full model (Mamba+GQA+MoE+BitNet+Spike+HyperConn)
    but halved dimensions. Model ~2 GB bf16, gradients ~2 GB, optimizer ~4 GB = ~8 GB.
    Leaves ~24 GB headroom for activations. Trains the full pipeline correctly."""
    return TryplicityModel(
        vocab_size=32000,
        hidden_size=1024,
        num_layers=24,
        max_seq_len=2048,
        mamba_state_dim=64,
        mamba_conv_dim=4,
        mamba_expand_factor=2,
        num_attention_heads=8,
        num_kv_heads=2,
        head_dim=128,
        num_experts=32,
        num_shared_experts=2,
        expert_hidden_size=384,
        base_active_experts=6,
        min_active_experts=4,
        max_active_experts=10,
        use_bitnet=True,
        use_spike=True,
        use_dynamic_sparsity=True,
        use_lateral_inhibition=True,
        use_hyper_connections=True,
        hyper_num_streams=4,
        num_predict_tokens=2,
        gradient_checkpointing=True,
    )


def create_full_model() -> TryplicityModel:
    """Create the full 9.4B parameter model (3B active) for cloud GPU training."""
    return TryplicityModel(
        vocab_size=32000,
        hidden_size=2048,
        num_layers=32,
        max_seq_len=4096,
        mamba_state_dim=128,
        mamba_conv_dim=4,
        mamba_expand_factor=2,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        num_experts=64,
        num_shared_experts=2,
        expert_hidden_size=512,
        base_active_experts=8,
        min_active_experts=4,
        max_active_experts=12,
        use_bitnet=True,
        use_spike=True,
        use_dynamic_sparsity=True,
        use_lateral_inhibition=True,
        use_hyper_connections=True,
        hyper_num_streams=4,
        num_predict_tokens=4,
        gradient_checkpointing=True,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TRYPLICITY MODEL SMOKE TEST")
    print("=" * 60)

    # Test tiny model (fits easily on any GPU or CPU)
    print("\nCreating tiny model...")
    model = create_tiny_model()
    params = model.count_parameters()
    print(f"  Total params: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Per layer avg: {params['per_layer_avg']:,}")

    # Forward pass
    x = torch.randint(0, 32000, (2, 64))
    labels = torch.randint(0, 32000, (2, 64))

    model.train()
    result = model(x, labels=labels)
    print(f"\n  Input: {x.shape}")
    print(f"  Logits: {result['logits'].shape}")
    print(f"  Loss: {result['loss'].item():.4f}")
    print(f"  LM loss: {result['lm_loss'].item():.4f}")
    print(f"  Multi-token loss: {result['multi_token_loss'].item():.4f}")
    print(f"  Aux loss: {result['aux_loss'].item():.4f}")

    # Backward pass
    result["loss"].backward()
    print("  Backward pass: OK")

    # Generation
    prompt = torch.randint(0, 32000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"\n  Generation: {prompt.shape} -> {generated.shape}")

    print("\n  Model smoke test PASSED.")
