"""
Tryplicity Training Script -- 4x NVIDIA B200 (192 GB HBM3e each, 768 GB total)
Distributed data-parallel training for the full 9.4B model.

Launch: torchrun --nproc_per_node=4 training/train_b200.py
Default: 11 minutes at $20.00/hr (4x $5.00/GPU) = ~$3.67

Hardware per B200:
    BF16 tensor:   ~2,250 TFLOPS
    Mem bandwidth: 8 TB/s HBM3e
    NVLink 5:      1.8 TB/s GPU-to-GPU
    4x total:      ~9 PFLOPS BF16, 32 TB/s bandwidth

Memory budget per B200 (192 GB HBM3e):
    Model weights (fp32 + autocast):  ~37.6 GB  (9.4B x 4 bytes)
    Optimizer (momentum + variance):  ~75.2 GB  (9.4B x 8 bytes)
    Gradients (fp32):                 ~37.6 GB  (9.4B x 4 bytes)
    Activations (batch=4, grad ckpt): ~20   GB
    DDP buffers + overhead:           ~5    GB
    -----------------------------------------------
    Total per GPU:                    ~175  GB
    Headroom:                         ~17   GB  (SAFE)

    Note: fp32 weights + autocast bf16 forward uses the SAME total memory
    as bf16 weights + fp32 optimizer master copy (150 GB base either way),
    but avoids the bf16 buffer dtype bugs documented in CLAUDE.md.
"""

import sys
import os
import time
import math
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import create_full_model
from training.data_pipeline import StreamingTextDataset
from brain.curriculum import NeuroCurriculum
from brain.predictive_coding import PredictiveCodingWrapper
from brain.hebbian_loss import HebbianAuxLoss
from brain.sleep_consolidation import SleepConsolidation
from tokenizers import Tokenizer


def setup_distributed():
    """Initialize DDP with NCCL backend (NVIDIA)."""
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


def train_b200(
    minutes: float = 11.0,
    rate_per_hour: float = 20.00,
    data_dir: str = "./data",
    tokenizer_path: str = "./tokenizer/tryplicity.json",
    checkpoint_dir: str = "./checkpoints",
):
    """
    Main training loop for 4x NVIDIA B200.

    With 4x B200 and 768 GB total VRAM:
    - Full 9.4B model fits on each GPU via DDP (no sharding needed)
    - Gradient checkpointing ON to keep batch=4 under 192 GB
    - Batch size 4 per GPU (B200 has 8 TB/s bandwidth — feeds big batches fast)
    - Gradient accumulation 8 steps
    - Effective batch = 4 seqs * 4 GPUs * 8 accum = 128 seqs = 524K tokens
    """
    # ---- Distributed setup ----
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    # ---- Config ----
    # B200: 192 GB HBM3e per GPU. With fp32 model + autocast:
    #   Base (weights + optimizer + grads) = 150.4 GB
    #   batch=4 + grad ckpt activations    = ~20 GB
    #   DDP buffers                        = ~5 GB
    #   Total                              = ~175 GB / 192 GB (17 GB headroom)
    batch_per_gpu = 4         # 4 seqs * 4096 tokens = 16K tokens/GPU/micro-step
    max_seq_len = 4096
    accumulation_steps = 8    # Effective batch = 4 * 4 * 8 = 128 seqs = 524K tokens
    peak_lr = 3e-3
    min_lr = 3e-5
    log_interval = 10

    effective_batch_tokens = batch_per_gpu * world_size * accumulation_steps * max_seq_len
    cost_estimate = (minutes / 60.0) * rate_per_hour

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"TRYPLICITY TRAINING -- {world_size}x NVIDIA B200")
    log(f"Time: {minutes:.0f} minutes | Cost: ~${cost_estimate:.2f} at ${rate_per_hour:.2f}/hr")
    log(f"GPUs: {world_size}x B200 (192 GB HBM3e each = {world_size * 192} GB total)")
    log(f"Batch: {batch_per_gpu}/GPU x {world_size} GPUs x {accumulation_steps} accum = {effective_batch_tokens:,} tokens/step")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ---- Create model ----
    log("\nCreating full 9.4B parameter model...")
    model = create_full_model()
    # Gradient checkpointing ON -- saves ~30 GB activation memory per GPU
    # Required: base memory is 150 GB + batch=4 activations need grad ckpt to fit in 192 GB
    model.gradient_checkpointing = True
    model = model.to(device)

    if is_main():
        params = model.count_parameters()
        log(f"  Total parameters: {params['total']:,}")
        log(f"  Trainable: {params['trainable']:,}")
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device) / 1e9
        total_gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        log(f"  GPU {local_rank} memory after model load: {allocated:.2f} GB / {total_gpu_mem:.0f} GB")

    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ---- Load data ----
    log("\nLoading data...")
    data_dir_path = Path(data_dir)
    data_files = sorted(data_dir_path.glob("*.jsonl"))

    if not data_files:
        log(f"ERROR: No .jsonl files in {data_dir}")
        log("Run: python download_data_mi300x.py")
        cleanup_distributed()
        return

    log(f"  Data files: {[f.name for f in data_files]}")

    # Use streaming dataset for large data
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

    # ---- Brain modules ----
    log("\nBrain modules: ALL ON")
    hidden_size = 2048

    hebbian = HebbianAuxLoss(hidden_size=hidden_size, num_layers=32, enabled=True).to(device)
    predictive_coding = PredictiveCodingWrapper(hidden_size=hidden_size, enabled=True)
    predictive_coding.to(device)
    curriculum = NeuroCurriculum(total_hours=minutes / 60.0, enabled=True)
    sleep = SleepConsolidation(enabled=True, replay_samples=5000)

    # ---- Optimizer ----
    param_groups = [
        {"params": model.parameters(), "lr": peak_lr},
        {"params": predictive_coding.parameters(), "lr": peak_lr * 0.1},
        {"params": hebbian.parameters(), "lr": peak_lr * 0.1},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)

    # ---- Memory verification ----
    if is_main():
        after_optim = torch.cuda.memory_allocated(device) / 1e9
        total_gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        log(f"\n  GPU memory after optimizer: {after_optim:.2f} GB / {total_gpu_mem:.0f} GB")
        log(f"  Headroom: {total_gpu_mem - after_optim:.1f} GB (need ~25 GB for batch=4 activations w/ grad ckpt)")
        if after_optim > total_gpu_mem * 0.9:
            log("  WARNING: Memory very tight! May OOM during forward pass")
            log("  Consider reducing batch_per_gpu to 2")
        elif after_optim > total_gpu_mem * 0.8:
            log("  CAUTION: Memory is usable but tight. Gradient checkpointing is ON.")
        else:
            log("  STATUS: Memory is SAFE")

    # Estimate total optimizer steps based on throughput
    # 4x B200 at ~2,250 TFLOPS each = ~9 PFLOPS total
    # Expected throughput: ~200K-400K tokens/sec for our MoE+Mamba model
    # Conservative 300K tok/s: 11 min = 660 sec = ~198M tokens
    # At 524K tokens/step = ~378 optimizer steps
    estimated_tok_per_sec = 300000
    estimated_total_steps = max(50, int((minutes * 60 * estimated_tok_per_sec) / effective_batch_tokens))
    log(f"\n  Estimated throughput: ~{estimated_tok_per_sec // 1000}K tokens/sec")
    log(f"  Estimated optimizer steps: {estimated_total_steps}")

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

                # Forward with bf16 autocast (model stays fp32, autocast handles bf16 matmuls)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw_model = model.module
                    result = model(input_ids, labels=labels)
                    loss = result["loss"]

                    # Predictive coding
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
                (loss / accumulation_steps).backward()
                micro_step += 1

                if micro_step % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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

                # Checkpoint every ~3 minutes
                if global_step > 0 and global_step % max(1, estimated_total_steps // 4) == 0 and is_main():
                    ckpt = checkpoint_dir / f"b200_step{global_step}.pt"
                    torch.save({
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
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

    # Save final model
    if is_main():
        final = checkpoint_dir / "final_b200.pt"
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "global_step": global_step,
            "total_tokens": total_tokens,
            "best_loss": best_loss,
            "elapsed_seconds": elapsed,
            "cost_dollars": total_cost,
            "hardware": f"{world_size}x_b200",
        }, final)
        log(f"  Model saved: {final}")

        # Generation test
        log(f"\n{'='*70}")
        log("GENERATION TEST")
        log(f"{'='*70}")

        tokenizer = Tokenizer.from_file(tokenizer_path)
        raw_model = model.module
        raw_model.eval()

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
                gen = raw_model.generate(ids, max_new_tokens=80, temperature=0.8)
            text = tokenizer.decode(gen[0].tolist())
            log(f"\n  > {prompt}")
            log(f"    {text[:250]}")

    log(f"\n{'='*70}")
    log(f"DONE -- ${total_cost:.2f} spent")
    log(f"IMPORTANT: Download checkpoint before shutting down!")
    log(f"  scp <pod>:./checkpoints/final_b200.pt ./")
    log(f"{'='*70}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tryplicity on 4x NVIDIA B200")
    parser.add_argument("--minutes", type=float, default=11.0,
                        help="Training time in minutes (default: 11)")
    parser.add_argument("--rate", type=float, default=20.00,
                        help="Hourly rate in dollars for all GPUs (default: 20.00)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer/tryplicity.json")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    train_b200(
        minutes=args.minutes,
        rate_per_hour=args.rate,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        checkpoint_dir=args.checkpoint_dir,
    )
