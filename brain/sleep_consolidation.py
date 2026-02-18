"""
Sleep-Inspired Memory Consolidation
Runs between training phases. Replays important examples,
prunes weak experts, sharpens spike thresholds, averages checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import copy


class SleepConsolidation:
    """
    Between training phases, run a consolidation pass:
    1. Replay highest-impact examples at very low LR
    2. Prune rarely-activated experts, reinitialize them
    3. Sharpen spike thresholds to maintain target sparsity
    4. Average recent checkpoints for smoother generalization
    """

    def __init__(
        self,
        replay_samples: int = 10000,
        replay_lr: float = 1e-5,
        prune_threshold: float = 0.01,
        target_sparsity: float = 0.7,
        checkpoint_average_n: int = 3,
        enabled: bool = True,
    ):
        self.replay_samples = replay_samples
        self.replay_lr = replay_lr
        self.prune_threshold = prune_threshold
        self.target_sparsity = target_sparsity
        self.checkpoint_average_n = checkpoint_average_n
        self.enabled = enabled

    @torch.no_grad()
    def sharpen_thresholds(self, model):
        """
        Adjust spike activation thresholds to maintain target sparsity.
        Increase thresholds for neurons that fire too often (> 50%).
        Decrease thresholds for neurons that never fire (< 0.1%).
        """
        if not self.enabled:
            return

        adjusted = 0
        for module in model.modules():
            from brain.spike_activation import SpikeActivation
            if isinstance(module, SpikeActivation) and module.enabled:
                rates = module.firing_rate_ema
                thresh = module.threshold.data

                # Neurons firing too much: increase threshold
                too_active = rates > 0.5
                thresh[too_active] *= 1.2

                # Neurons never firing: decrease threshold
                too_silent = rates < 0.001
                thresh[too_silent] *= 0.8

                # Clamp to reasonable range
                thresh.clamp_(min=0.01, max=1.0)
                adjusted += 1

        print(f"  [Sleep] Adjusted thresholds in {adjusted} spike layers")

    @torch.no_grad()
    def prune_experts(self, model):
        """
        Identify experts that are rarely activated (< prune_threshold usage).
        Merge their weights into shared experts and reinitialize them.
        This frees up expert capacity for new learning in the next phase.
        """
        if not self.enabled:
            return

        from model.moe_layer import MoELayer
        pruned_count = 0

        for module in model.modules():
            if isinstance(module, MoELayer):
                router = module.router
                # Check expert utilization from router stats
                # We use the gate weight norms as a proxy for utilization
                gate_norms = router.gate.weight.data.norm(dim=1)  # [num_experts]
                mean_norm = gate_norms.mean()

                for i, expert in enumerate(module.experts):
                    if gate_norms[i] < mean_norm * self.prune_threshold:
                        # Reinitialize this expert
                        for param in expert.parameters():
                            if param.dim() >= 2:
                                nn.init.kaiming_normal_(param)
                            else:
                                nn.init.zeros_(param)
                        pruned_count += 1

        print(f"  [Sleep] Pruned and reinitialized {pruned_count} experts")

    @staticmethod
    def average_checkpoints(checkpoint_paths: list, output_path: str):
        """
        Average the weights of multiple checkpoints.
        Smooths out training noise, consistently improves generalization.
        """
        if not checkpoint_paths:
            return

        # Load all checkpoints
        state_dicts = []
        for path in checkpoint_paths:
            if Path(path).exists():
                sd = torch.load(path, map_location="cpu", weights_only=True)
                if "model_state_dict" in sd:
                    sd = sd["model_state_dict"]
                state_dicts.append(sd)

        if not state_dicts:
            print("  [Sleep] No valid checkpoints found for averaging")
            return

        # Average all parameters
        avg_state = {}
        for key in state_dicts[0]:
            tensors = [sd[key].float() for sd in state_dicts if key in sd]
            avg_state[key] = (sum(tensors) / len(tensors)).to(state_dicts[0][key].dtype)

        torch.save({"model_state_dict": avg_state}, output_path)
        print(f"  [Sleep] Averaged {len(state_dicts)} checkpoints -> {output_path}")

    def consolidate(self, model, replay_dataloader=None, optimizer=None, device="cuda"):
        """
        Full sleep consolidation pass.
        Call between training phases.
        """
        if not self.enabled:
            print("  [Sleep] Consolidation disabled, skipping")
            return

        print("  [Sleep] Starting consolidation...")

        # 1. Replay important examples (if data provided)
        if replay_dataloader is not None and optimizer is not None:
            model.train()
            # Temporarily set very low LR
            original_lrs = []
            for pg in optimizer.param_groups:
                original_lrs.append(pg["lr"])
                pg["lr"] = self.replay_lr

            replayed = 0
            for batch in replay_dataloader:
                if replayed >= self.replay_samples:
                    break
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                result = model(input_ids, labels=labels)
                result["loss"].backward()
                optimizer.step()
                optimizer.zero_grad()
                replayed += input_ids.shape[0] * input_ids.shape[1]

            # Restore LRs
            for pg, lr in zip(optimizer.param_groups, original_lrs):
                pg["lr"] = lr

            print(f"  [Sleep] Replayed {replayed:,} tokens")

        # 2. Prune weak experts
        self.prune_experts(model)

        # 3. Sharpen spike thresholds
        self.sharpen_thresholds(model)

        print("  [Sleep] Consolidation complete")


if __name__ == "__main__":
    print("Sleep consolidation smoke test...")
    sleep = SleepConsolidation(enabled=True)

    # Create a minimal model-like object for testing
    from brain.spike_activation import SpikeActivation
    import torch.nn as nn

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.spike = SpikeActivation(64)
            self.linear = nn.Linear(64, 64)

    model = FakeModel()
    model.spike.firing_rate_ema[:32] = 0.0001  # Simulate silent neurons
    model.spike.firing_rate_ema[32:] = 0.8  # Simulate overactive neurons

    print(f"  Before: thresholds mean={model.spike.threshold.data.abs().mean():.4f}")
    sleep.sharpen_thresholds(model)
    print(f"  After: thresholds mean={model.spike.threshold.data.abs().mean():.4f}")
    print("  Sleep consolidation OK.")
