"""
Tryplicity Training Script — 8x MI300X Optimized
Distributed training for AMD Instinct MI300X cluster.

Budget: ~$10 at $12.08/hr = ~50 min total runtime
  - 36 min pretraining (~2B tokens)
  - 6 min instruct-tuning
  - 5 min GRPO alignment
  - 3 min sleep consolidation

Launch with: torchrun --nproc_per_node=8 training/train_mi300x.py
"""

import sys
import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import TryplicityModel, create_full_model
from training.data_pipeline import create_dataloader, StreamingTextDataset
from training.config import get_mi300x_config
from brain.curriculum import NeuroCurriculum
from brain.predictive_coding import PredictiveCodingWrapper
from brain.hebbian_loss import HebbianAuxLoss
from brain.sleep_consolidation import SleepConsolidation
from tokenizers import Tokenizer
from torch.utils.data import DataLoader


def setup_distributed():
    """Initialize distributed training (DDP) for 8x MI300X."""
    dist.init_process_group(backend="nccl")  # ROCm supports NCCL via RCCL
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is rank 0."""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_main(msg):
    """Only print from rank 0."""
    if is_main_process():
        print(msg)


def get_lr(step: int, total_steps: int, peak_lr: float, min_lr: float,
           warmup_ratio: float = 0.03, stable_ratio: float = 0.82):
    """WSD (Warmup-Stable-Decay) schedule, tuned for short MI300X runs."""
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


def train_mi300x(
    budget_dollars: float = 10.0,
    rate_per_hour: float = 12.08,
    data_dir: str = "./data",
    tokenizer_path: str = "./tokenizer/tryplicity.json",
    checkpoint_dir: str = "./checkpoints",
):
    """
    Main training loop for 8x MI300X.

    Key differences from RTX 5070 train.py:
    - DDP across 8 GPUs
    - No gradient checkpointing (1.5 TB VRAM)
    - No GaLore (full optimizer states fit)
    - Batch sizes: 8/GPU × 8 GPUs × 16 accum = 1024 seqs effective
    - Streaming dataloader (2B tokens too big for RAM)
    - Compressed curriculum (36 min instead of 10 hrs)
    - Higher peak LR (6e-3) for faster convergence
    """
    # Setup distributed
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    # Load config
    config = get_mi300x_config(budget_dollars=budget_dollars, rate_per_hour=rate_per_hour)
    pretrain_minutes = config.training.pretrain_hours * 60
    total_minutes = (config.training.pretrain_hours + config.training.instruct_hours +
                     config.training.grpo_hours) * 60 + config.training.sleep1_minutes + config.training.sleep2_minutes

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print_main("=" * 70)
    print_main(f"TRYPLICITY TRAINING — 8x MI300X CLUSTER")
    print_main(f"Budget: ${budget_dollars:.2f} at ${rate_per_hour:.2f}/hr = {total_minutes:.0f} min")
    print_main(f"GPUs: {world_size}x MI300X (1536 GB HBM3 total)")
    print_main(f"Token target: {config.training.total_tokens_target / 1e9:.1f}B")
    print_main(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_main("=" * 70)

    # ---- Training hyperparams ----
    batch_size_per_gpu = 8      # 8 seqs × 4096 tokens = 33K tokens/GPU (fits in 192GB)
    max_seq_len = 4096
    accumulation_steps = 16     # Effective batch: 8 × 8 × 16 = 1024 seqs = 4.2M tokens
    peak_lr = config.training.peak_lr
    min_lr = config.training.min_lr

    effective_batch_seqs = batch_size_per_gpu * world_size * accumulation_steps
    effective_batch_tokens = effective_batch_seqs * max_seq_len
    print_main(f"\n[Config]")
    print_main(f"  Per-GPU batch: {batch_size_per_gpu} seqs × {max_seq_len} tokens = {batch_size_per_gpu * max_seq_len:,} tokens")
    print_main(f"  World size: {world_size}")
    print_main(f"  Gradient accumulation: {accumulation_steps}")
    print_main(f"  Effective batch: {effective_batch_seqs} seqs = {effective_batch_tokens:,} tokens")
    print_main(f"  Peak LR: {peak_lr}")
    print_main(f"  Gradient checkpointing: OFF (1.5TB VRAM)")
    print_main(f"  GaLore: OFF (full optimizer states)")

    # ---- Create model ----
    print_main("\nCreating 3B model...")
    model = create_full_model()
    model = model.to(device)

    if is_main_process():
        params = model.count_parameters()
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device) / 1e9
        print(f"  GPU {local_rank} memory (model loaded): {allocated:.2f} GB")

    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    # ---- Create streaming dataloader ----
    print_main("\nLoading data (streaming mode)...")
    data_dir = Path(data_dir)
    data_files = sorted(data_dir.glob("*.jsonl"))

    if not data_files:
        raise FileNotFoundError(
            f"No .jsonl files in {data_dir}. Run: python download_data_mi300x.py first"
        )

    print_main(f"  Data files: {[f.name for f in data_files]}")

    # Streaming dataset — each GPU reads different data via file sharding
    dataset = StreamingTextDataset(
        data_files=[str(f) for f in data_files],
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        shuffle_buffer=50000,  # Larger shuffle buffer for better mixing
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
    )

    # ---- Brain modules ----
    print_main("\nInitializing brain modules...")
    hidden_size = 2048
    num_layers = 32

    hebbian = HebbianAuxLoss(
        hidden_size=hidden_size,
        num_layers=num_layers,
        enabled=True,
    ).to(device)

    predictive_coding = PredictiveCodingWrapper(
        hidden_size=hidden_size,
        enabled=True,
    ).to(device)

    curriculum = NeuroCurriculum(
        total_hours=config.training.pretrain_hours,
        enabled=True,
    )

    sleep = SleepConsolidation(
        enabled=True,
        replay_samples=config.brain.sleep_replay_samples,
    )

    print_main("  Hebbian: ON | Predictive coding: ON | Curriculum: ON | Sleep: ON")
    print_main(f"  Dynamic sparsity: {config.brain.min_active_experts}-{config.brain.max_active_experts} active experts")

    # ---- Optimizer ----
    param_groups = [
        {"params": model.parameters(), "lr": peak_lr},
        {"params": predictive_coding.parameters(), "lr": peak_lr * 0.1},
        {"params": hebbian.parameters(), "lr": peak_lr * 0.1},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=peak_lr,
        betas=(0.9, 0.95), weight_decay=0.1,
        fused=True,  # Fused AdamW for AMD GPUs (ROCm 6+)
    )

    # ---- Estimate total steps ----
    # ~2B tokens / 4.2M tokens per effective step = ~476 steps
    estimated_total_steps = int(config.training.total_tokens_target / effective_batch_tokens)
    print_main(f"\n  Estimated total optimizer steps: {estimated_total_steps}")
    print_main(f"  Warmup steps: {int(estimated_total_steps * config.training.warmup_ratio)}")

    # ---- AMP scaler for mixed precision ----
    # bfloat16 training — no scaler needed (bf16 doesn't underflow like fp16)
    use_amp = True

    # ---- Training loop ----
    print_main(f"\n{'='*70}")
    print_main("PRETRAINING STARTED")
    print_main(f"  Target: {pretrain_minutes:.0f} minutes, ~{config.training.total_tokens_target/1e9:.1f}B tokens")
    print_main(f"{'='*70}\n")

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
                    print_main(f"\n  Pretrain time limit reached ({pretrain_minutes:.0f} min)")
                    raise StopIteration()

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                # Forward pass with bfloat16 autocast
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    # Access the unwrapped model for methods DDP doesn't expose
                    raw_model = model.module

                    result = model(input_ids, labels=labels)
                    loss = result["loss"]

                    # Predictive coding: weight tokens by surprise
                    if predictive_coding.enabled:
                        with torch.no_grad():
                            embeddings = raw_model.embed(input_ids)
                            next_embeddings = raw_model.embed(labels)
                        token_weights = predictive_coding.compute_token_weights(
                            embeddings, next_embeddings
                        )
                        shift_logits = result["logits"][:, :-1].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        per_token_loss = F.cross_entropy(
                            shift_logits.view(-1, raw_model.vocab_size),
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
                    optimizer.zero_grad(set_to_none=True)  # More memory efficient
                    global_step += 1

                    # Free fragmented GPU memory periodically
                    if global_step % 50 == 0:
                        torch.cuda.empty_cache()

                # Track stats
                batch_tokens = input_ids.numel()
                total_tokens += batch_tokens * world_size  # All GPUs process data
                running_loss += loss.item()
                running_lm_loss += result["lm_loss"].item()

                # Logging (rank 0 only)
                if global_step > 0 and global_step % log_interval == 0 and is_main_process():
                    avg_loss = running_loss / (log_interval * accumulation_steps)
                    avg_lm_loss = running_lm_loss / (log_interval * accumulation_steps)
                    tokens_per_sec = total_tokens / (time.time() - start_time)
                    perplexity = math.exp(min(avg_lm_loss, 20))

                    # Cost tracking
                    elapsed_hours = elapsed_minutes / 60
                    cost_so_far = elapsed_hours * rate_per_hour

                    log_parts = [
                        f"Step {global_step:>5d}/{estimated_total_steps}",
                        f"Loss: {avg_loss:.4f}",
                        f"PPL: {perplexity:.1f}",
                        f"LR: {lr:.2e}",
                        f"Tok/s: {tokens_per_sec:,.0f}",
                        f"Tokens: {total_tokens/1e9:.2f}B",
                        f"${cost_so_far:.2f}",
                        f"{elapsed_minutes:.1f}m",
                    ]

                    if curriculum:
                        log_parts.append(f"Stage: {curriculum.get_stage_name()}")

                    if result.get("stats"):
                        last_stats = result["stats"][-1]
                        log_parts.append(f"Experts: {last_stats.get('avg_active_experts', 0):.1f}")

                    print("  " + " | ".join(log_parts))

                    if avg_loss < best_loss:
                        best_loss = avg_loss

                    running_loss = 0.0
                    running_lm_loss = 0.0

                # Save checkpoint (rank 0 only)
                save_interval_steps = max(1, int(estimated_total_steps * config.training.save_interval_hours /
                                                  config.training.pretrain_hours))
                if global_step > 0 and global_step % save_interval_steps == 0 and is_main_process():
                    ckpt_path = checkpoint_dir / f"mi300x_step{global_step}.pt"
                    torch.save({
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "total_tokens": total_tokens,
                        "best_loss": best_loss,
                        "config": "mi300x",
                    }, ckpt_path)
                    print(f"\n  Saved checkpoint: {ckpt_path}")
                    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
                    print(f"  GPU {local_rank} peak memory: {peak_mem:.2f} GB\n")

    except (StopIteration, KeyboardInterrupt):
        pass

    # Synchronize all GPUs
    if dist.is_initialized():
        dist.barrier()

    # ---- Training complete ----
    elapsed = time.time() - start_time
    total_cost = (elapsed / 3600) * rate_per_hour

    print_main(f"\n{'='*70}")
    print_main("PRETRAINING COMPLETE")
    print_main(f"{'='*70}")
    print_main(f"  Total time: {elapsed/60:.1f} minutes")
    print_main(f"  Total cost: ${total_cost:.2f}")
    print_main(f"  Total steps: {global_step}")
    print_main(f"  Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print_main(f"  Throughput: {total_tokens/elapsed:,.0f} tokens/sec")
    print_main(f"  Best loss: {best_loss:.4f}")
    print_main(f"  Final perplexity: {math.exp(min(best_loss, 20)):.1f}")

    if device.startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        print_main(f"  GPU {local_rank} peak memory: {peak_mem:.2f} GB")

    # Save final model (rank 0 only)
    if is_main_process():
        final_path = checkpoint_dir / "final_mi300x_pretrain.pt"
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "global_step": global_step,
            "total_tokens": total_tokens,
            "best_loss": best_loss,
            "elapsed_seconds": elapsed,
            "total_cost": total_cost,
            "hardware": "8x_mi300x",
        }, final_path)
        print_main(f"  Final model saved: {final_path}")

    # ---- Test generation (rank 0 only) ----
    if is_main_process():
        print(f"\n{'='*70}")
        print("GENERATION TEST")
        print(f"{'='*70}")

        tokenizer = Tokenizer.from_file(tokenizer_path)
        raw_model = model.module
        raw_model.eval()

        test_prompts = [
            "The meaning of life is",
            "Once upon a time",
            "In mathematics, we can prove that",
            "def fibonacci(n):",
            "The capital of France is",
            "To solve this equation, first we need to",
            "import torch\n\nclass NeuralNetwork(nn.Module):",
        ]

        for prompt in test_prompts:
            encoded = tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded.ids], device=device)

            with torch.no_grad():
                generated = raw_model.generate(input_ids, max_new_tokens=80, temperature=0.8)

            output_text = tokenizer.decode(generated[0].tolist())
            print(f"\n  Prompt: {prompt}")
            print(f"  Output: {output_text[:250]}")

    print_main(f"\n{'='*70}")
    print_main(f"DONE — Total cost: ${total_cost:.2f}")
    print_main(f"{'='*70}")

    cleanup_distributed()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tryplicity on 8x MI300X")
    parser.add_argument("--budget", type=float, default=10.0,
                        help="Budget in dollars (default: $10)")
    parser.add_argument("--rate", type=float, default=12.08,
                        help="Hourly rate in dollars (default: $12.08 spot)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer/tryplicity.json")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    train_mi300x(
        budget_dollars=args.budget,
        rate_per_hour=args.rate,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        checkpoint_dir=args.checkpoint_dir,
    )
