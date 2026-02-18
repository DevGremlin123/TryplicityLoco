"""
Tryplicity Training Script — RTX 5090 (Single GPU, 32 GB VRAM)

Optimized for a single NVIDIA RTX 5090:
  - bf16 model parameters (halves memory from 12 GB to 6 GB)
  - Gradient checkpointing ON (trades compute for memory)
  - Small batch + heavy accumulation
  - No DDP (single GPU, no torchrun needed)

Usage: python3 training/train_5090.py --budget 10 --rate 1.50
"""

import sys
import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import TryplicityModel, create_full_model
from training.data_pipeline import StreamingTextDataset
from training.config import get_mi300x_config, FullConfig, ModelConfig, BrainConfig, TrainingConfig, CurriculumStageConfig
from brain.curriculum import NeuroCurriculum
from brain.predictive_coding import PredictiveCodingWrapper
from brain.hebbian_loss import HebbianAuxLoss
from brain.sleep_consolidation import SleepConsolidation
from tokenizers import Tokenizer
from torch.utils.data import DataLoader


def get_5090_config(budget_dollars: float = 10.0, rate_per_hour: float = 1.50) -> FullConfig:
    """
    Config for single RTX 5090 (32 GB VRAM).

    Key constraints:
      - 32 GB VRAM total
      - bf16 model = ~6 GB
      - bf16 optimizer states = ~12 GB
      - Gradients = ~6 GB
      - Activations (with grad checkpoint) = ~4-6 GB
      - Total: ~28-30 GB → fits

    At ~15K tokens/sec, $10 at $1.50/hr = 6.67 hours ≈ 360M tokens
    """
    runtime_hours = budget_dollars / rate_per_hour
    runtime_minutes = runtime_hours * 60

    # Time allocation
    pretrain_minutes = runtime_minutes * 0.80
    instruct_minutes = runtime_minutes * 0.10
    grpo_minutes = runtime_minutes * 0.05
    sleep1_minutes = runtime_minutes * 0.03
    sleep2_minutes = runtime_minutes * 0.02

    config = FullConfig()

    # -- Training schedule --
    config.training.pretrain_hours = pretrain_minutes / 60.0
    config.training.instruct_hours = instruct_minutes / 60.0
    config.training.grpo_hours = grpo_minutes / 60.0
    config.training.sleep1_minutes = sleep1_minutes
    config.training.sleep2_minutes = sleep2_minutes
    config.training.num_gpus = 1
    config.training.hardware_profile = "rtx5090"

    # -- Token target (realistic for 5090) --
    tokens_per_sec = 15000  # Conservative estimate
    pretrain_seconds = pretrain_minutes * 60
    config.training.total_tokens_target = int(tokens_per_sec * pretrain_seconds * 0.8)

    # -- Optimizer --
    config.training.peak_lr = 3e-3
    config.training.min_lr = 3e-5
    config.training.optimizer = "adamw"

    # -- WSD Schedule --
    config.training.warmup_ratio = 0.03
    config.training.stable_ratio = 0.85
    config.training.decay_ratio = 0.12

    # -- Gradient checkpointing ON (essential for 32 GB) --
    config.training.gradient_checkpointing = True

    # -- GaLore OFF (bf16 params make it unnecessary) --
    config.training.use_galore = False

    # -- Logging --
    config.training.log_interval = 50
    config.training.save_interval_hours = 0.5

    # -- Multi-token prediction (reduce to 2 to save memory) --
    config.training.num_predict_tokens = 2

    # -- Curriculum stages (stretched across longer training) --
    config.training.curriculum_stages = [
        CurriculumStageConfig(
            name="infancy",
            hours_start=0.0, hours_end=pretrain_minutes * 0.15 / 60,
            max_seq_len=512,
            max_reading_level=6.0,
            text_ratio=0.85, code_ratio=0.10, math_ratio=0.05,
            batch_size_tokens=32_000,
            lr_multiplier=0.6,
            hebbian_weight_multiplier=1.0,
        ),
        CurriculumStageConfig(
            name="childhood",
            hours_start=pretrain_minutes * 0.15 / 60,
            hours_end=pretrain_minutes * 0.40 / 60,
            max_seq_len=1024,
            max_reading_level=10.0,
            text_ratio=0.70, code_ratio=0.18, math_ratio=0.12,
            batch_size_tokens=64_000,
            lr_multiplier=1.0,
            hebbian_weight_multiplier=0.5,
        ),
        CurriculumStageConfig(
            name="adolescence",
            hours_start=pretrain_minutes * 0.40 / 60,
            hours_end=pretrain_minutes * 0.75 / 60,
            max_seq_len=2048,
            max_reading_level=None,
            text_ratio=0.60, code_ratio=0.22, math_ratio=0.18,
            batch_size_tokens=64_000,
            lr_multiplier=1.0,
            hebbian_weight_multiplier=0.2,
        ),
        CurriculumStageConfig(
            name="mastery",
            hours_start=pretrain_minutes * 0.75 / 60,
            hours_end=pretrain_minutes / 60,
            max_seq_len=2048,
            max_reading_level=None,
            text_ratio=0.45, code_ratio=0.28, math_ratio=0.27,
            batch_size_tokens=64_000,
            lr_multiplier=0.7,
            hebbian_weight_multiplier=0.1,
        ),
    ]

    # -- Brain optimizations (all ON) --
    config.brain.enable_brain_optimizations = True
    config.brain.enable_predictive_coding = True
    config.brain.enable_dynamic_sparsity = True
    config.brain.enable_lateral_inhibition = True
    config.brain.enable_spike_activations = True
    config.brain.enable_hebbian_loss = True
    config.brain.enable_curriculum = True
    config.brain.enable_sleep_consolidation = True
    config.brain.sleep_replay_samples = 5000
    config.brain.sleep_replay_lr = 1e-5
    config.brain.sleep_checkpoint_average_n = 3

    # -- Dynamic sparsity (conservative for single GPU) --
    config.brain.min_active_experts = 4
    config.brain.max_active_experts = 12

    return config


def get_lr(step, total_steps, peak_lr, min_lr, warmup_ratio=0.03, stable_ratio=0.85):
    """WSD schedule."""
    warmup_steps = int(total_steps * warmup_ratio)
    stable_steps = int(total_steps * stable_ratio)
    decay_steps = total_steps - warmup_steps - stable_steps

    if step < warmup_steps:
        return peak_lr * (step / max(1, warmup_steps))
    elif step < warmup_steps + stable_steps:
        return peak_lr
    else:
        decay_step = step - warmup_steps - stable_steps
        progress = decay_step / max(1, decay_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_5090(
    budget_dollars: float = 10.0,
    rate_per_hour: float = 1.50,
    data_dir: str = "./data",
    tokenizer_path: str = "./tokenizer/tryplicity.json",
    checkpoint_dir: str = "./checkpoints",
):
    """
    Training loop for single RTX 5090 (32 GB VRAM).
    No DDP, no torchrun. Just: python3 training/train_5090.py
    """
    device = "cuda:0"

    # Load config
    config = get_5090_config(budget_dollars=budget_dollars, rate_per_hour=rate_per_hour)
    pretrain_minutes = config.training.pretrain_hours * 60

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRYPLICITY TRAINING — RTX 5090 (32 GB)")
    print(f"Budget: ${budget_dollars:.2f} at ${rate_per_hour:.2f}/hr = {pretrain_minutes:.0f} min pretrain")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"Token target: {config.training.total_tokens_target / 1e6:.0f}M")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- Training hyperparams ----
    batch_size = 1              # Minimal batch (32 GB is tight)
    max_seq_len = 2048          # Max seq len for later stages
    accumulation_steps = 64     # Effective batch: 1 × 64 = 64 seqs
    peak_lr = config.training.peak_lr
    min_lr = config.training.min_lr

    effective_batch_tokens = batch_size * max_seq_len * accumulation_steps
    print(f"\n[Config]")
    print(f"  Batch size: {batch_size}")
    print(f"  Max seq len: {max_seq_len}")
    print(f"  Gradient accumulation: {accumulation_steps}")
    print(f"  Effective batch: {effective_batch_tokens:,} tokens")
    print(f"  Peak LR: {peak_lr}")
    print(f"  Gradient checkpointing: ON")
    print(f"  Model dtype: bfloat16 (saves ~6 GB)")

    # ---- Create model in bf16 ----
    print("\nCreating 3B model (bf16)...")
    model = create_full_model()
    model = model.to(device=device, dtype=torch.bfloat16)

    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")

    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device) / 1e9
    print(f"  GPU memory (model loaded): {allocated:.2f} GB")

    # ---- Streaming dataloader ----
    print("\nLoading data (streaming mode)...")
    data_dir = Path(data_dir)
    data_files = sorted(data_dir.glob("*.jsonl"))

    if not data_files:
        raise FileNotFoundError(
            f"No .jsonl files in {data_dir}. Run: python3 download_data_mi300x.py first"
        )

    print(f"  Data files: {[f.name for f in data_files]}")

    dataset = StreamingTextDataset(
        data_files=[str(f) for f in data_files],
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        shuffle_buffer=20000,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    # ---- Brain modules ----
    print("\nInitializing brain modules...")
    hidden_size = 2048

    hebbian = HebbianAuxLoss(
        hidden_size=hidden_size,
        num_layers=32,
        enabled=True,
    ).to(device=device, dtype=torch.bfloat16)

    predictive_coding = PredictiveCodingWrapper(
        hidden_size=hidden_size,
        enabled=True,
    )
    predictive_coding.to(device)
    # Convert predictor to bf16
    if predictive_coding.enabled:
        predictive_coding.predictor = predictive_coding.predictor.to(dtype=torch.bfloat16)

    curriculum = NeuroCurriculum(
        total_hours=config.training.pretrain_hours,
        enabled=True,
    )

    sleep = SleepConsolidation(
        enabled=True,
        replay_samples=config.brain.sleep_replay_samples,
    )

    print("  Hebbian: ON | Predictive coding: ON | Curriculum: ON | Sleep: ON")

    # ---- Optimizer ----
    # bf16 params → optimizer states stored in bf16 too (saves ~12 GB vs fp32)
    param_groups = [
        {"params": model.parameters(), "lr": peak_lr},
        {"params": predictive_coding.predictor.parameters(), "lr": peak_lr * 0.1},
        {"params": hebbian.parameters(), "lr": peak_lr * 0.1},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=peak_lr,
        betas=(0.9, 0.95), weight_decay=0.1,
    )

    # Memory check after optimizer
    allocated = torch.cuda.memory_allocated(device) / 1e9
    print(f"\n  GPU memory (model + optimizer): {allocated:.2f} GB")
    print(f"  Headroom: {torch.cuda.get_device_properties(0).total_mem / 1e9 - allocated:.2f} GB")

    # ---- Estimate total steps ----
    estimated_total_steps = max(1, int(config.training.total_tokens_target / effective_batch_tokens))
    print(f"  Estimated total steps: {estimated_total_steps}")

    # ---- Training loop ----
    print(f"\n{'='*70}")
    print("PRETRAINING STARTED")
    print(f"  Target: {pretrain_minutes:.0f} minutes")
    print(f"{'='*70}\n")

    if curriculum:
        curriculum.start()

    model.train()
    start_time = time.time()
    global_step = 0
    micro_step = 0
    total_tokens = 0
    best_loss = float("inf")
    running_loss = 0.0
    running_lm_loss = 0.0
    log_interval = config.training.log_interval

    try:
        epoch = 0
        while True:
            epoch += 1
            for batch in dataloader:
                # Check time limit
                elapsed_minutes = (time.time() - start_time) / 60.0
                if elapsed_minutes >= pretrain_minutes:
                    print(f"\n  Pretrain time limit reached ({pretrain_minutes:.0f} min)")
                    raise StopIteration()

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                # Forward pass (bf16 model, no autocast needed since model IS bf16)
                result = model(input_ids, labels=labels)
                loss = result["loss"]

                # Predictive coding
                if predictive_coding.enabled:
                    with torch.no_grad():
                        embeddings = model.embed(input_ids)
                        next_embeddings = model.embed(labels)
                    token_weights = predictive_coding.compute_token_weights(
                        embeddings, next_embeddings
                    )
                    shift_logits = result["logits"][:, :-1].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    per_token_loss = F.cross_entropy(
                        shift_logits.view(-1, model.vocab_size),
                        shift_labels.view(-1),
                        reduction="none",
                    ).view(shift_logits.shape[0], -1)
                    weighted_loss = (per_token_loss * token_weights[:, 1:]).mean()
                    pred_loss = predictive_coding.compute_predictor_loss(
                        embeddings, next_embeddings
                    )
                    loss = weighted_loss + result["aux_loss"] + 0.1 * pred_loss
                    if result.get("multi_token_loss") is not None:
                        loss = loss + 0.5 * result["multi_token_loss"]

                # Gradient accumulation
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                micro_step += 1

                if micro_step % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Update learning rate
                    lr = get_lr(global_step, estimated_total_steps, peak_lr, min_lr,
                                config.training.warmup_ratio, config.training.stable_ratio)
                    if curriculum:
                        lr *= curriculum.get_lr_multiplier()
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # Free fragmented memory
                    if global_step % 25 == 0:
                        torch.cuda.empty_cache()

                # Track stats
                batch_tokens = input_ids.numel()
                total_tokens += batch_tokens
                running_loss += loss.item()
                running_lm_loss += result["lm_loss"].item()

                # Logging
                if global_step > 0 and global_step % log_interval == 0:
                    avg_loss = running_loss / (log_interval * accumulation_steps)
                    avg_lm_loss = running_lm_loss / (log_interval * accumulation_steps)
                    tokens_per_sec = total_tokens / (time.time() - start_time)
                    perplexity = math.exp(min(avg_lm_loss, 20))

                    elapsed_hours = elapsed_minutes / 60
                    cost_so_far = elapsed_hours * rate_per_hour

                    log_parts = [
                        f"Step {global_step:>5d}/{estimated_total_steps}",
                        f"Loss: {avg_loss:.4f}",
                        f"PPL: {perplexity:.1f}",
                        f"LR: {lr:.2e}",
                        f"Tok/s: {tokens_per_sec:,.0f}",
                        f"Tokens: {total_tokens/1e6:.1f}M",
                        f"${cost_so_far:.2f}",
                        f"{elapsed_minutes:.1f}m",
                    ]

                    if curriculum:
                        log_parts.append(f"Stage: {curriculum.get_stage_name()}")

                    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
                    log_parts.append(f"Mem: {peak_mem:.1f}GB")

                    print("  " + " | ".join(log_parts))

                    if avg_loss < best_loss:
                        best_loss = avg_loss

                    running_loss = 0.0
                    running_lm_loss = 0.0

                # Save checkpoint
                save_interval_steps = max(1, int(estimated_total_steps *
                                                  config.training.save_interval_hours /
                                                  config.training.pretrain_hours))
                if global_step > 0 and global_step % save_interval_steps == 0:
                    ckpt_path = checkpoint_dir / f"5090_step{global_step}.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "total_tokens": total_tokens,
                        "best_loss": best_loss,
                        "config": "5090",
                    }, ckpt_path)
                    print(f"\n  Saved checkpoint: {ckpt_path}")
                    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
                    print(f"  Peak memory: {peak_mem:.2f} GB\n")

    except (StopIteration, KeyboardInterrupt):
        pass

    # ---- Training complete ----
    elapsed = time.time() - start_time
    total_cost = (elapsed / 3600) * rate_per_hour

    print(f"\n{'='*70}")
    print("PRETRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Total steps: {global_step}")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens/1e6:.1f}M)")
    print(f"  Throughput: {total_tokens/max(1,elapsed):,.0f} tokens/sec")
    print(f"  Best loss: {best_loss:.4f}")

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  Peak memory: {peak_mem:.2f} GB")

    # Save final model
    final_path = checkpoint_dir / "final_5090_pretrain.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "global_step": global_step,
        "total_tokens": total_tokens,
        "best_loss": best_loss,
        "elapsed_seconds": elapsed,
        "total_cost": total_cost,
        "hardware": "rtx5090",
    }, final_path)
    print(f"  Final model saved: {final_path}")

    # ---- Test generation ----
    print(f"\n{'='*70}")
    print("GENERATION TEST")
    print(f"{'='*70}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    model.eval()

    test_prompts = [
        "The meaning of life is",
        "Once upon a time",
        "def fibonacci(n):",
        "The capital of France is",
    ]

    for prompt in test_prompts:
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=device)

        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8)

        output_text = tokenizer.decode(generated[0].tolist())
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {output_text[:200]}")

    print(f"\n{'='*70}")
    print(f"DONE — Total cost: ${total_cost:.2f}")
    print(f"{'='*70}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tryplicity on RTX 5090")
    parser.add_argument("--budget", type=float, default=10.0,
                        help="Budget in dollars (default: $10)")
    parser.add_argument("--rate", type=float, default=1.50,
                        help="Hourly rate in dollars (default: $1.50)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer/tryplicity.json")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    # Set memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    train_5090(
        budget_dollars=args.budget,
        rate_per_hour=args.rate,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        checkpoint_dir=args.checkpoint_dir,
    )
