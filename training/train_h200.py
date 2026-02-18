"""
Tryplicity Training Script -- 8x NVIDIA H200 SXM (141 GB HBM3e each, 1,128 GB total)
Fully Sharded Data Parallel (FSDP) training for the full 9.4B model.

Launch: torchrun --nproc_per_node=8 training/train_h200.py
Default: 11 minutes at $28.72/hr (8x $3.59/GPU) = ~$5.26

Hardware per H200 SXM:
    BF16 tensor:   ~989 TFLOPS
    Mem bandwidth: 4.8 TB/s HBM3e
    NVLink:        900 GB/s GPU-to-GPU
    8x total:      ~7.9 PFLOPS BF16, 38.4 TB/s bandwidth

WHY FSDP (not DDP):
    H200 = 141 GB per GPU.
    DDP needs full model + optimizer + gradients PER GPU = 150.4 GB -> OOM!
    FSDP shards everything across 8 GPUs:

      Sharded params (fp32):     9.4B x 4 / 8 =  4.7 GB per GPU
      Sharded optimizer (fp32):  9.4B x 8 / 8 =  9.4 GB per GPU
      Sharded gradients (fp32):  9.4B x 4 / 8 =  4.7 GB per GPU
      Base per GPU:                              18.8 GB
      Activations (batch=4, grad ckpt):         ~15   GB
      FSDP unsharding buffer (1 block):          ~2   GB
      Overhead:                                   ~5   GB
      ---------------------------------------------------
      Total per GPU:                             ~41   GB / 141 GB
      Headroom:                                 ~100   GB  (VERY SAFE)
"""

import sys
import os
import time
import math
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import create_full_model, TryplicityBlock
from training.data_pipeline import StreamingTextDataset
from brain.curriculum import NeuroCurriculum
from brain.hebbian_loss import HebbianAuxLoss
from brain.sleep_consolidation import SleepConsolidation
from tokenizers import Tokenizer


def setup_distributed():
    """Initialize distributed process group with NCCL backend."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def get_lr(step, total_steps, peak_lr, min_lr,
           warmup_ratio=0.03, stable_ratio=0.82):
    """WSD schedule tuned for short runs (faster warmup)."""
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


def train_h200(
    minutes: float = 11.0,
    rate_per_hour: float = 28.72,
    data_dir: str = "./data",
    tokenizer_path: str = "./tokenizer/tryplicity.json",
    checkpoint_dir: str = "./checkpoints",
):
    """
    Main training loop for 8x NVIDIA H200 SXM using FSDP.

    FSDP (Fully Sharded Data Parallel) is REQUIRED because:
    - H200 has 141 GB per GPU
    - Full model + optimizer + gradients = 150.4 GB (DDP would OOM)
    - FSDP shards everything: only ~18.8 GB base per GPU

    Key differences from DDP scripts (B200/MI300X):
    - Uses FSDP instead of DDP
    - FSDP handles mixed precision (param_dtype=bf16)
    - Checkpoint saving uses FSDP state_dict_type context
    - Gradient clipping uses model.clip_grad_norm_ (FSDP method)
    - Predictive coding uses logit confidence (no model.module access)
    - Generation test uses FSDP.summon_full_params
    """
    # ---- Distributed setup ----
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    # ---- Config ----
    # FSDP shards model across GPUs: ~18.8 GB base per GPU
    # With batch=4 + grad ckpt: ~41 GB / 141 GB = ~100 GB headroom
    batch_per_gpu = 4         # 4 seqs x 4096 tokens = 16K tokens/GPU/micro-step
    max_seq_len = 4096
    accumulation_steps = 4    # Effective batch = 4 * 8 * 4 = 128 seqs = 524K tokens
    peak_lr = 3e-3
    min_lr = 3e-5
    log_interval = 10
    vocab_size = 32000

    effective_batch_tokens = batch_per_gpu * world_size * accumulation_steps * max_seq_len
    cost_estimate = (minutes / 60.0) * rate_per_hour

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"TRYPLICITY TRAINING -- {world_size}x NVIDIA H200 SXM (FSDP)")
    log(f"Time: {minutes:.0f} minutes | Cost: ~${cost_estimate:.2f} at ${rate_per_hour:.2f}/hr")
    log(f"GPUs: {world_size}x H200 SXM (141 GB HBM3e each = {world_size * 141} GB total)")
    log(f"Strategy: FSDP FULL_SHARD (ZeRO Stage 3 — model sharded across all GPUs)")
    log(f"Batch: {batch_per_gpu}/GPU x {world_size} GPUs x {accumulation_steps} accum = {effective_batch_tokens:,} tokens/step")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ---- Create model ----
    log("\nCreating full 9.4B parameter model...")
    # Same seed on all ranks for consistent initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = create_full_model()
    model.gradient_checkpointing = True

    if is_main():
        params = model.count_parameters()
        log(f"  Total parameters: {params['total']:,}")
        log(f"  Trainable: {params['trainable']:,}")

    # Move to GPU (FSDP will reshard)
    model = model.to(device)

    # ---- FSDP wrapping ----
    log("\nWrapping model with FSDP (FULL_SHARD)...")

    # Each TryplicityBlock becomes a separate FSDP unit
    # embed, lm_head, final_norm, multi_token_heads stay in root FSDP unit
    # Weight tying (embed <-> lm_head) preserved with use_orig_params=True
    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TryplicityBlock},
    )

    # Mixed precision: fp32 storage, bf16 forward/reduce
    fsdp_mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,    # Forward pass in bf16
        reduce_dtype=torch.bfloat16,   # Gradient allreduce in bf16
        buffer_dtype=torch.bfloat16,   # Buffers (RoPE, firing rates) in bf16
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=fsdp_mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        use_orig_params=True,          # Required for weight tying (embed <-> lm_head)
        sync_module_states=True,       # Broadcast rank 0 weights to all ranks
    )

    if is_main():
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device) / 1e9
        total_gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        log(f"  GPU {local_rank} memory after FSDP wrap: {allocated:.2f} GB / {total_gpu_mem:.0f} GB")
        log(f"  Headroom: {total_gpu_mem - allocated:.1f} GB")
        log(f"  FSDP sharding: each GPU holds 1/{world_size} of model")

    # ---- Load data ----
    log("\nLoading data...")
    data_dir_path = Path(data_dir)
    data_files = sorted(data_dir_path.glob("*.jsonl"))

    if not data_files:
        log(f"ERROR: No .jsonl files in {data_dir}")
        log("Run: bash run_h200.sh prep")
        cleanup_distributed()
        return

    log(f"  Data files: {[f.name for f in data_files]}")

    dataset = StreamingTextDataset(
        data_files=[str(f) for f in data_files],
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        shuffle_buffer=50000,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_per_gpu,
        num_workers=4,
        pin_memory=True,
    )

    # ---- Brain modules (non-FSDP, local to each GPU) ----
    log("\nBrain modules: Curriculum ON, Predictive Coding ON (logit-based for FSDP)")
    hidden_size = 2048

    hebbian = HebbianAuxLoss(hidden_size=hidden_size, num_layers=32, enabled=True).to(device)
    curriculum = NeuroCurriculum(total_hours=minutes / 60.0, enabled=True)
    sleep = SleepConsolidation(enabled=True, replay_samples=5000)

    # ---- Optimizer ----
    # Note: predictive coding uses logit-based weights (no separate predictor params)
    param_groups = [
        {"params": model.parameters(), "lr": peak_lr},
        {"params": hebbian.parameters(), "lr": peak_lr * 0.1},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)

    # ---- Memory verification ----
    if is_main():
        after_optim = torch.cuda.memory_allocated(device) / 1e9
        total_gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        log(f"\n  GPU memory after optimizer: {after_optim:.2f} GB / {total_gpu_mem:.0f} GB")
        log(f"  Headroom: {total_gpu_mem - after_optim:.1f} GB (need ~20 GB for batch=4 activations)")
        if after_optim > total_gpu_mem * 0.5:
            log("  CAUTION: Using >50% of VRAM at baseline. Monitor during training.")
        else:
            log("  STATUS: Memory is SAFE (FSDP sharding is working)")

    # Estimate total optimizer steps
    # 8x H200 with FSDP: ~150K-250K tokens/sec (conservative)
    estimated_tok_per_sec = 200000
    estimated_total_steps = max(50, int((minutes * 60 * estimated_tok_per_sec) / effective_batch_tokens))
    log(f"\n  Estimated throughput: ~{estimated_tok_per_sec // 1000}K tokens/sec")
    log(f"  Estimated optimizer steps: {estimated_total_steps}")
    log(f"  Effective batch: {effective_batch_tokens:,} tokens/step")

    # ---- Training loop ----
    log(f"\n{'='*70}")
    log("TRAINING STARTED")
    log(f"{'='*70}\n")

    curriculum.start()
    model.train()
    start_time = time.time()
    global_step = 0
    micro_step = 0
    total_tokens = 0
    best_loss = float("inf")
    running_loss = 0.0
    running_lm_loss = 0.0

    try:
        epoch = 0
        while True:
            epoch += 1
            for batch in dataloader:
                # Time check
                elapsed_min = (time.time() - start_time) / 60.0
                if elapsed_min >= minutes:
                    log(f"\n  Time limit reached ({minutes:.0f} min)")
                    raise StopIteration()

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                # Forward with bf16 autocast (complements FSDP mixed precision)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = model(input_ids, labels=labels)
                    loss = result["loss"]

                    # Predictive coding (FSDP-compatible: logit confidence weighting)
                    # Instead of calling raw_model.embed() (breaks with FSDP sharding),
                    # we weight tokens by the model's own prediction difficulty.
                    # Same effect: surprising tokens get more gradient signal.
                    with torch.no_grad():
                        shift_logits_pc = result["logits"][:, :-1].contiguous()
                        shift_labels_pc = labels[:, 1:].contiguous()
                        # Softmax in fp32 for numerical stability
                        probs = F.softmax(shift_logits_pc.float(), dim=-1)
                        correct_probs = probs.gather(-1, shift_labels_pc.unsqueeze(-1)).squeeze(-1)
                        # Low probability = surprising = higher training weight
                        difficulty = 1.0 - correct_probs.clamp(0, 1)
                        token_weights = 0.5 + difficulty  # Range [0.5, 1.5]
                        token_weights = token_weights / token_weights.mean().clamp(min=1e-8)

                    # Recompute loss with predictive coding weights
                    shift_logits = result["logits"][:, :-1].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    per_token_loss = F.cross_entropy(
                        shift_logits.reshape(-1, vocab_size),
                        shift_labels.reshape(-1),
                        reduction="none",
                    ).reshape(shift_logits.shape[0], -1)
                    weighted_loss = (per_token_loss * token_weights).mean()
                    loss = weighted_loss + result["aux_loss"]
                    if result.get("multi_token_loss") is not None:
                        loss = loss + 0.5 * result["multi_token_loss"]

                # Gradient accumulation
                (loss / accumulation_steps).backward()
                micro_step += 1

                if micro_step % accumulation_steps == 0:
                    # FSDP-compatible gradient clipping (handles sharded gradients)
                    model.clip_grad_norm_(1.0)

                    lr = get_lr(global_step, estimated_total_steps, peak_lr, min_lr)
                    lr *= curriculum.get_lr_multiplier()
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                # Stats
                batch_tokens = input_ids.numel() * world_size
                total_tokens += batch_tokens
                running_loss += loss.item()
                running_lm_loss += result["lm_loss"].item()

                # Log
                if global_step > 0 and global_step % log_interval == 0 and is_main():
                    denom = log_interval * accumulation_steps
                    avg_loss = running_loss / denom
                    avg_lm = running_lm_loss / denom
                    tok_s = total_tokens / (time.time() - start_time)
                    ppl = math.exp(min(avg_lm, 20))
                    cost = (elapsed_min / 60) * rate_per_hour

                    parts = [
                        f"Step {global_step:>5d}",
                        f"Loss: {avg_loss:.4f}",
                        f"PPL: {ppl:.1f}",
                        f"LR: {lr:.2e}",
                        f"Tok/s: {tok_s:,.0f}",
                        f"Total: {total_tokens/1e6:.0f}M",
                        f"${cost:.2f}",
                        f"{elapsed_min:.1f}m",
                        f"Stage: {curriculum.get_stage_name()}",
                    ]
                    if result.get("stats"):
                        parts.append(f"Exp: {result['stats'][-1].get('avg_active_experts', 0):.1f}")

                    log("  " + " | ".join(parts))

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                    running_loss = 0.0
                    running_lm_loss = 0.0

                # Checkpoint every ~3 minutes (ALL ranks must participate in FSDP state_dict)
                if global_step > 0 and global_step % max(1, estimated_total_steps // 4) == 0:
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                        state_dict = model.state_dict()
                        if is_main():
                            ckpt = checkpoint_dir / f"h200_step{global_step}.pt"
                            torch.save({
                                "model_state_dict": state_dict,
                                "global_step": global_step,
                                "total_tokens": total_tokens,
                                "best_loss": best_loss,
                            }, ckpt)
                            peak = torch.cuda.max_memory_allocated(device) / 1e9
                            log(f"\n  Checkpoint: {ckpt} | Peak GPU: {peak:.1f} GB\n")

    except (StopIteration, KeyboardInterrupt):
        pass

    # Sync all GPUs
    if dist.is_initialized():
        dist.barrier()

    # ---- Done ----
    elapsed = time.time() - start_time
    total_cost = (elapsed / 3600) * rate_per_hour

    log(f"\n{'='*70}")
    log("TRAINING COMPLETE")
    log(f"{'='*70}")
    log(f"  Time: {elapsed/60:.1f} minutes")
    log(f"  Cost: ${total_cost:.2f}")
    log(f"  Steps: {global_step}")
    log(f"  Tokens: {total_tokens:,} ({total_tokens/1e6:.0f}M)")
    log(f"  Throughput: {total_tokens/max(1,elapsed):,.0f} tokens/sec")
    log(f"  Best loss: {best_loss:.4f}")
    log(f"  Perplexity: {math.exp(min(best_loss, 20)):.1f}")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        total_gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        log(f"  Peak GPU memory: {peak:.1f} GB / {total_gpu_mem:.0f} GB")

    # Save final model (FSDP full state dict — all ranks participate)
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
        if is_main():
            final = checkpoint_dir / "final_h200.pt"
            torch.save({
                "model_state_dict": state_dict,
                "global_step": global_step,
                "total_tokens": total_tokens,
                "best_loss": best_loss,
                "elapsed_seconds": elapsed,
                "cost_dollars": total_cost,
                "hardware": f"{world_size}x_h200_sxm",
            }, final)
            log(f"  Model saved: {final}")

    # Generation test (gather full params temporarily on all GPUs)
    log(f"\n{'='*70}")
    log("GENERATION TEST")
    log(f"{'='*70}")

    with FSDP.summon_full_params(model, writeback=False):
        if is_main():
            tokenizer = Tokenizer.from_file(tokenizer_path)
            model.eval()

            prompts = [
                "The meaning of life is",
                "Once upon a time",
                "In mathematics, we can prove that",
                "def fibonacci(n):",
                "The capital of France is",
            ]

            for prompt in prompts:
                encoded = tokenizer.encode(prompt)
                ids = torch.tensor([encoded.ids], device=device)
                with torch.no_grad():
                    gen = model.module.generate(ids, max_new_tokens=80, temperature=0.8)
                text = tokenizer.decode(gen[0].tolist())
                log(f"\n  > {prompt}")
                log(f"    {text[:250]}")

    log(f"\n{'='*70}")
    log(f"DONE -- ${total_cost:.2f} spent")
    log(f"IMPORTANT: Download checkpoint before shutting down!")
    log(f"  scp <pod>:./checkpoints/final_h200.pt ./")
    log(f"{'='*70}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tryplicity on 8x NVIDIA H200 SXM")
    parser.add_argument("--minutes", type=float, default=11.0,
                        help="Training time in minutes (default: 11)")
    parser.add_argument("--rate", type=float, default=28.72,
                        help="Hourly rate in dollars for all 8 GPUs (default: 28.72)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer/tryplicity.json")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    train_h200(
        minutes=args.minutes,
        rate_per_hour=args.rate,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        checkpoint_dir=args.checkpoint_dir,
    )
