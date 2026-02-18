"""
Hyper-Connections: Replace residual connections for 1.8x faster convergence.
ICLR 2025. Instead of x + f(x), we use a multi-stream approach where
multiple parallel streams carry information through the network.
"""

import torch
import torch.nn as nn


class HyperConnection(nn.Module):
    """
    Hyper-connection layer that replaces standard residual connections.

    Instead of: output = x + layer(x)
    We use: output = alpha * x + beta * layer(x)

    Where alpha and beta are learned per-stream mixing coefficients.
    Multiple streams allow different information pathways through the network.
    """

    def __init__(self, hidden_size: int, num_streams: int = 4, layer_idx: int = 0, num_layers: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.layer_idx = layer_idx

        # Per-stream mixing weights (initialized for stable training)
        # Alpha: how much of the input to keep (starts near 1)
        # Beta: how much of the layer output to add (starts near 0.1)
        self.alpha = nn.Parameter(torch.ones(num_streams, 1, 1, hidden_size))
        self.beta = nn.Parameter(torch.full((num_streams, 1, 1, hidden_size), 0.1))

        # Stream selection: which stream feeds into this layer
        # and which stream receives the output
        self.input_stream_proj = nn.Linear(hidden_size * num_streams, hidden_size, bias=False)
        self.output_stream_proj = nn.Linear(hidden_size, hidden_size * num_streams, bias=False)

        # Initialize projections close to identity-like behavior
        nn.init.zeros_(self.output_stream_proj.weight)
        # Input proj: average across streams initially
        with torch.no_grad():
            w = torch.zeros(hidden_size, hidden_size * num_streams)
            for s in range(num_streams):
                w[:, s * hidden_size:(s + 1) * hidden_size] = torch.eye(hidden_size) / num_streams
            self.input_stream_proj.weight.copy_(w)

    def select_input(self, streams: torch.Tensor) -> torch.Tensor:
        """
        Select and mix input from all streams for the current layer.
        streams: [num_streams, batch, seq_len, hidden]
        returns: [batch, seq_len, hidden]
        """
        batch, seq_len = streams.shape[1], streams.shape[2]
        # Concatenate all streams
        concat = streams.permute(1, 2, 0, 3).reshape(batch, seq_len, -1)
        return self.input_stream_proj(concat)

    def update_streams(self, streams: torch.Tensor, layer_input: torch.Tensor,
                       layer_output: torch.Tensor) -> torch.Tensor:
        """
        Update all streams after the layer processes input.
        streams: [num_streams, batch, seq_len, hidden]
        layer_input: [batch, seq_len, hidden]
        layer_output: [batch, seq_len, hidden]
        returns: updated streams [num_streams, batch, seq_len, hidden]
        """
        # Compute stream updates from layer output
        batch, seq_len, _ = layer_output.shape
        delta = self.output_stream_proj(layer_output)
        delta = delta.reshape(batch, seq_len, self.num_streams, self.hidden_size)
        delta = delta.permute(2, 0, 1, 3)  # [num_streams, batch, seq_len, hidden]

        # Mix: alpha * old_stream + beta * delta
        new_streams = self.alpha * streams + self.beta * delta
        return new_streams


class HyperConnectionManager:
    """
    Manages the multi-stream state across all layers.
    Initialize once, then call at each layer.
    """

    def __init__(self, hidden_size: int, num_streams: int, num_layers: int):
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.num_layers = num_layers

    def init_streams(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initialize streams from input embedding.
        x: [batch, seq_len, hidden]
        returns: [num_streams, batch, seq_len, hidden]
        """
        # All streams start as copies of the input
        return x.unsqueeze(0).expand(self.num_streams, -1, -1, -1).clone()

    def collapse_streams(self, streams: torch.Tensor) -> torch.Tensor:
        """
        Collapse streams back to single hidden state for the output head.
        streams: [num_streams, batch, seq_len, hidden]
        returns: [batch, seq_len, hidden]
        """
        # Simple mean across streams
        return streams.mean(dim=0)


if __name__ == "__main__":
    print("Hyper-connections smoke test...")
    hidden = 256
    num_streams = 4
    num_layers = 4
    batch, seq = 2, 16

    manager = HyperConnectionManager(hidden, num_streams, num_layers)

    x = torch.randn(batch, seq, hidden)
    streams = manager.init_streams(x)
    print(f"  Input: {x.shape}")
    print(f"  Streams initialized: {streams.shape}")

    # Simulate passing through layers
    for i in range(num_layers):
        hc = HyperConnection(hidden, num_streams, layer_idx=i, num_layers=num_layers)
        layer_in = hc.select_input(streams)
        # Simulate a layer
        layer_out = layer_in * 0.9 + torch.randn_like(layer_in) * 0.1
        streams = hc.update_streams(streams, layer_in, layer_out)
        print(f"  Layer {i}: input {layer_in.shape}, output {layer_out.shape}, streams {streams.shape}")

    output = manager.collapse_streams(streams)
    print(f"  Final output: {output.shape}")
    print("  Hyper-connections OK.")
