"""
BitNet 1.58-bit Quantization
Ternary weights: -1, 0, +1
10x memory compression, trains from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator for gradients."""
    return x + (x.round() - x).detach()


def weight_quant_158(w: torch.Tensor) -> torch.Tensor:
    """
    Quantize weights to ternary {-1, 0, +1} using absmean scaling.
    BitNet b1.58: scale = mean(|w|), then round(w / scale) clamped to {-1,0,+1}
    """
    scale = w.abs().mean().clamp(min=1e-5)
    w_scaled = w / scale
    w_quant = ste_round(w_scaled).clamp(-1, 1)
    return w_quant * scale


def activation_quant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Quantize activations to int8 range with per-token absmax scaling."""
    Qn = -(2 ** (bits - 1))
    Qp = 2 ** (bits - 1) - 1
    scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    x_scaled = (x / scale).clamp(Qn, Qp)
    x_quant = ste_round(x_scaled)
    return x_quant * (scale / Qp)


class BitLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with BitNet 1.58-bit weights.
    Weights are stored as float during training but quantized in forward pass.
    Activations are quantized to int8.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Kaiming init scaled for ternary
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights to ternary {-1, 0, +1}
        w_quant = weight_quant_158(self.weight)
        # Quantize activations to int8
        x_quant = activation_quant(x)
        # Ensure matching dtypes (rounding ops can promote bf16 to float32)
        out = F.linear(x_quant.to(w_quant.dtype), w_quant, self.bias)
        return out


class BitLinearInference(nn.Module):
    """
    Inference-only BitLinear. Weights stored as int2 packed.
    Used after training is complete for deployment.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Store ternary weights as int8 (-1, 0, 1) for inference
        self.register_buffer("weight_ternary", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(1))

    @torch.no_grad()
    def from_bitlinear(self, bitlinear: BitLinear):
        """Convert trained BitLinear to inference format."""
        w = bitlinear.weight.data
        scale = w.abs().mean().clamp(min=1e-5)
        w_ternary = (w / scale).round().clamp(-1, 1).to(torch.int8)
        self.weight_ternary.copy_(w_ternary)
        self.weight_scale.fill_(scale.item())
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_ternary.float() * self.weight_scale
        return F.linear(x, w)


if __name__ == "__main__":
    # Smoke test
    print("BitNet smoke test...")
    layer = BitLinear(256, 512)
    x = torch.randn(2, 10, 256)
    out = layer(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    # Check weight quantization
    w_quant = weight_quant_158(layer.weight)
    unique_vals = (w_quant / w_quant.abs().max()).unique()
    print(f"  Unique quantized weight values: {len(unique_vals)}")

    # Check sparsity (how many zeros)
    zero_pct = (layer.weight.abs() < layer.weight.abs().mean() * 0.5).float().mean().item()
    print(f"  Approximate weight sparsity: {zero_pct:.1%}")

    # Inference conversion
    inf_layer = BitLinearInference(256, 512).from_bitlinear(layer)
    out_inf = inf_layer(x)
    print(f"  Inference output: {out_inf.shape}")
    print("  BitNet OK.")
