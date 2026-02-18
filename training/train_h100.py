"""
Tryplicity Training Script — 16x H100 SXM Cluster
Distributed training across 2 RunPod pods (8 GPUs each) via NCCL.

Budget: ~$13-17 at $51.65/hr = ~15-20 min total runtime
  - 11 min pretraining (~2B tokens)
  - 2 min instruct-tuning
  - 2 min GRPO alignment
  - 1 min sleep consolidation

Launch with: torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=29500 training/train_h100.py
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
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import TryplicityModel, create_full_model
from training.data_pipeline import StreamingTextDataset
from training.config import get_h100_config
from brain.curriculum import NeuroCurriculum
from brain.predictive_coding import PredictiveCodingWrapper
from brain.hebbian_loss import HebbianAuxLoss
from brain.sleep_consolidation import SleepConsolidation
from tokenizers import Tokenizer
from torch.utils.data import DataLoader


def setup_distributed():
    """Initialize DDP for multi-node H100 cluster."""
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
        print(msg)


def get_lr(step, total_steps, peak_lr, min_lr, warmup_ratio=0.03, stable_ratio=0.82):
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


def train_h100(
    budget_dollars: float = 17.0,
    rate_per_hour: float = 51.65,
    data_dir: str = "./data",
    tokenizer_path: str = "./tokenizer/tryplicity.json",
    checkpoint_dir: str = "./checkpoints",
):
    """Main training loop for 16x H100 SXM cluster."""
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"

    config = get_h100_config(budget_dollars=budget_dollars, rate_per_hour=rate_per_hour)
    pretrain_minutes = config.training.pretrain_hours * 60
    total_minutes = (config.training.pretrain_hours + config.training.instruct_hours +
                     config.training.grpo_hours) * 60 + config.training.sleep1_minutes + config.training.sleep2_minutes

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"TRYPLICITY — 16x H100 SXM CLUSTER (2 pods)")
    log(f"Budget: ${budget_dollars:.2f} at ${rate_per_hour:.2f}/hr = {total_minutes:.0f} min")
    log(f"GPUs: {world_size}x H100 SXM (1280 GB HBM3 total)")
    log(f"Token target: {config.training.total_tokens_target / 1e9:.1f}B")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # -- Batch config --
    # H100: 80 GB VRAM per GPU. Our 3B BitNet model ~6 GB.
    # Plenty of room for large batches.
    batch_size_per_gpu = 24     # 24 seqs × 4096 = 98K tokens/GPU
    max_seq_len = 4096
    accumulation_steps = 2      # Effective: 24 × 16 × 2 = 768 seqs = 3.1M tokens/step

    effective_batch_seqs = batch_size_per_gpu * world_size * accumulation_steps
    effective_batch_tokens = effective_batch_seqs * max_seq_len
    peak_lr = config.training.peak_lr
    min_lr = config.training.min_lr

    log(f"\n[Config]")
    log(f"  Per-GPU batch: {batch_size_per_gpu} × {max_seq_len} = {batch_size_per_gpu * max_seq_len:,} tok")
    log(f"  World size: {world_size} GPUs across {world_size // 8} pods")
    log(f"  Grad accumulation: {accumulation_steps}")
    log(f"  Effective batch: {effective_batch_seqs} seqs = {effective_batch_tokens:,} tokens")
    log(f"  Peak LR: {peak_lr}")
    log(f"  Grad checkpointing: OFF")
    log(f"  GaLore: OFF")

    # -- Model --
    log("\nCreating 3B model...")
    model = create_full_model()
    model = model.to(device)

    if is_main():
        params = model.count_parameters()
        print(f"  Total params: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        mem = torch.cuda.memory_allocated(device) / 1e9
        print(f"  GPU {local_rank} memory (model): {mem:.2f} GB / 80 GB")

    # DDP wrap
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    # -- Data --
    log("\nLoading data (streaming)...")
    data_dir = Path(data_dir)
    data_files = sorted(data_dir.glob("*.jsonl"))
    if not data_files:
        raise FileNotFoundError(f"No .jsonl files in {data_dir}")
    log(f"  Files: {[f.name for f in data_files]}")

    dataset = StreamingTextDataset(
        data_files=[str(f) for f in data_files],
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        shuffle_buffer=50000,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size_per_gpu,
                            num_workers=4, pin_memory=True, prefetch_factor=4)

    # -- Brain modules --
    log("\nBrain modules: ALL ON")
    hidden_size, num_layers = 2048, 32
    hebbian = HebbianAuxLoss(hidden_size=hidden_size, num_layers=num_layers, enabled=True).to(device)
    predictive_coding = PredictiveCodingWrapper(hidden_size=hidden_size, enabled=True).to(device)
    curriculum = NeuroCurriculum(total_hours=config.training.pretrain_hours, enabled=True)
    sleep = SleepConsolidation(enabled=True, replay_samples=config.brain.sleep_replay_samples)

    # -- Optimizer --
    param_groups = [
        {"params": model.parameters(), "lr": peak_lr},
        {"params": predictive_coding.parameters(), "lr": peak_lr * 0.1},
        {"params": hebbian.parameters(), "lr": peak_lr * 0.1},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=peak_lr, betas=(0.9, 0.95),
                                   weight_decay=0.1, fused=True)

    # -- Step estimate --
    estimated_total_steps = int(config.training.total_tokens_target / effective_batch_tokens)
    log(f"\n  Estimated optimizer steps: {estimated_total_steps}")
    log(f"  Warmup: {int(estimated_total_steps * 0.03)} steps")

    # -- Training loop --
    log(f"\n{'='*70}")
    log(f"PRETRAINING: {pretrain_minutes:.0f} min, ~{config.training.total_tokens_target/1e9:.1f}B tokens")
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
                elapsed_min = (time.time() - start_time) / 60.0
                if elapsed_min >= pretrain_minutes:
                    log(f"\n  Time limit ({pretrain_minutes:.0f} min)")
                    raise StopIteration()

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw = model.module
                    result = model(input_ids, labels=labels)
                    loss = result["loss"]

                    if predictive_coding.enabled:
                        with torch.no_grad():
                            emb = raw.embed(input_ids)
                            next_emb = raw.embed(labels)
                        weights = predictive_coding.compute_token_weights(emb, next_emb)
                        shift_logits = result["logits"][:, :-1].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        ptl = F.cross_entropy(
                            shift_logits.view(-1, raw.vocab_size),
                            shift_labels.view(-1), reduction="none",
                        ).view(shift_logits.shape[0], -1)
                        wl = (ptl * weights[:, 1:]).mean()
                        pl = predictive_coding.compute_predictor_loss(emb, next_emb)
                        loss = wl + result["aux_loss"] + 0.1 * pl
                        if result.get("multi_token_loss") is not None:
                            loss = loss + 0.5 * result["multi_token_loss"]

                (loss / accumulation_steps).backward()
                micro_step += 1

                if micro_step % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    lr = get_lr(global_step, estimated_total_steps, peak_lr, min_lr)
                    if curriculum:
                        lr *= curriculum.get_lr_multiplier()
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                batch_tokens = input_ids.numel()
                total_tokens += batch_tokens * world_size
                running_loss += loss.item()
                running_lm_loss += result["lm_loss"].item()

                if global_step > 0 and global_step % config.training.log_interval == 0 and is_main():
                    n = config.training.log_interval * accumulation_steps
                    avg_loss = running_loss / n
                    avg_lm = running_lm_loss / n
                    tps = total_tokens / (time.time() - start_time)
                    ppl = math.exp(min(avg_lm, 20))
                    cost = (elapsed_min / 60) * rate_per_hour

                    parts = [
                        f"Step {global_step:>4d}/{estimated_total_steps}",
                        f"Loss:{avg_loss:.4f}",
                        f"PPL:{ppl:.1f}",
                        f"LR:{lr:.1e}",
                        f"Tok/s:{tps:,.0f}",
                        f"{total_tokens/1e9:.2f}B",
                        f"${cost:.2f}",
                        f"{elapsed_min:.1f}m",
                    ]
                    if curriculum:
                        parts.append(curriculum.get_stage_name())
                    if result.get("stats"):
                        parts.append(f"E:{result['stats'][-1].get('avg_active_experts',0):.0f}")
                    print("  " + " | ".join(parts))

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                    running_loss = 0.0
                    running_lm_loss = 0.0

                # Checkpoint
                save_every = max(1, int(estimated_total_steps * 0.25))
                if global_step > 0 and global_step % save_every == 0 and is_main():
                    p = checkpoint_dir / f"h100_step{global_step}.pt"
                    torch.save({"model_state_dict": model.module.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "global_step": global_step, "total_tokens": total_tokens,
                                "best_loss": best_loss}, p)
                    print(f"\n  Saved: {p} | GPU peak: {torch.cuda.max_memory_allocated(device)/1e9:.1f} GB\n")

    except (StopIteration, KeyboardInterrupt):
        pass

    if dist.is_initialized():
        dist.barrier()

    elapsed = time.time() - start_time
    cost = (elapsed / 3600) * rate_per_hour

    log(f"\n{'='*70}")
    log(f"PRETRAINING COMPLETE")
    log(f"  Time: {elapsed/60:.1f} min | Cost: ${cost:.2f}")
    log(f"  Steps: {global_step} | Tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    log(f"  Throughput: {total_tokens/elapsed:,.0f} tok/s")
    log(f"  Best loss: {best_loss:.4f} | PPL: {math.exp(min(best_loss, 20)):.1f}")
    if device.startswith("cuda"):
        log(f"  GPU peak mem: {torch.cuda.max_memory_allocated(device)/1e9:.1f} GB / 80 GB")
    log(f"{'='*70}")

    # Save final
    if is_main():
        fp = checkpoint_dir / "final_h100_pretrain.pt"
        torch.save({"model_state_dict": model.module.state_dict(),
                     "global_step": global_step, "total_tokens": total_tokens,
                     "best_loss": best_loss, "elapsed_seconds": elapsed,
                     "total_cost": cost, "hardware": "16x_h100_sxm"}, fp)
        log(f"  Saved: {fp}")

        # Generation test
        log(f"\n{'='*70}\nGENERATION TEST\n{'='*70}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        raw = model.module
        raw.eval()
        for prompt in ["The meaning of life is", "def fibonacci(n):",
                       "In mathematics,", "The capital of France is",
                       "import torch\nclass Model(nn.Module):"]:
            ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
            with torch.no_grad():
                gen = raw.generate(ids, max_new_tokens=80, temperature=0.8)
            print(f"\n  > {prompt}")
            print(f"    {tokenizer.decode(gen[0].tolist())[:250]}")

    log(f"\n{'='*70}\nDONE — ${cost:.2f} total\n{'='*70}")
    cleanup_distributed()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tryplicity on 16x H100 SXM")
    parser.add_argument("--budget", type=float, default=17.0)
    parser.add_argument("--rate", type=float, default=51.65)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer/tryplicity.json")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    train_h100(budget_dollars=args.budget, rate_per_hour=args.rate,
               data_dir=args.data_dir, tokenizer_path=args.tokenizer,
               checkpoint_dir=args.checkpoint_dir)
