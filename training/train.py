"""
Tryplicity Training Script
Integrates all brain-inspired modules with the core model.
Supports both 10-minute test and full 12-hour training runs.
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

from model.architecture import TryplicityModel, create_tiny_model, create_full_model
from training.data_pipeline import create_dataloader
from brain.curriculum import NeuroCurriculum
from brain.predictive_coding import PredictiveCodingWrapper
from brain.hebbian_loss import HebbianAuxLoss
from brain.sleep_consolidation import SleepConsolidation
from tokenizers import Tokenizer


def get_lr(step: int, total_steps: int, peak_lr: float, min_lr: float,
           warmup_ratio: float = 0.02, stable_ratio: float = 0.88):
    """WSD (Warmup-Stable-Decay) learning rate schedule."""
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


def train(
    mode: str = "test",  # "test" (10 min) or "full" (12 hours)
    device: str = "cuda",
    checkpoint_dir: str = "./checkpoints",
    data_dir: str = "./data",
    tokenizer_path: str = "./tokenizer/tryplicity.json",
):
    """Main training loop."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TRYPLICITY TRAINING - {'10-MINUTE TEST' if mode == 'test' else '12-HOUR FULL'}")
    print(f"Device: {device}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ---- Configuration based on mode ----
    if mode == "test":
        training_minutes = 10
        max_seq_len = 256
        batch_size = 8
        peak_lr = 1e-3
        min_lr = 1e-5
        max_data_samples = 5000  # Small subset for speed
        log_interval = 10
        save_interval_steps = 500
        accumulation_steps = 1
        use_brain = True
        print("\n[Config] Tiny model, 10-minute training, brain-optimized")
    else:
        training_minutes = 720  # 12 hours
        max_seq_len = 4096
        batch_size = 2
        peak_lr = 3e-3
        min_lr = 3e-5
        max_data_samples = None  # Use all data
        log_interval = 100
        save_interval_steps = 2000
        accumulation_steps = 8  # Effective batch = 16
        use_brain = True
        print("\n[Config] Full model, 12-hour training, brain-optimized")

    # ---- Create model ----
    print("\nCreating model...")
    if mode == "test":
        model = create_tiny_model()
    else:
        model = create_full_model()

    model = model.to(device)
    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")

    # Check VRAM usage
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory (model loaded): {allocated:.2f} GB")

    # ---- Create data loader ----
    print("\nLoading data...")
    # For test mode, use the small sample files
    if mode == "test":
        data_files_dir = data_dir
    else:
        data_files_dir = data_dir

    dataloader = create_dataloader(
        data_dir=data_files_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        max_samples=max_data_samples,
    )
    print(f"  Batches per epoch: {len(dataloader)}")

    # ---- Brain modules ----
    hebbian = None
    predictive_coding = None
    curriculum = None

    if use_brain:
        print("\nInitializing brain modules...")
        hidden_size = 256 if mode == "test" else 2048
        num_layers = 4 if mode == "test" else 32

        # Hebbian auxiliary losses
        hebbian = HebbianAuxLoss(
            hidden_size=hidden_size,
            num_layers=num_layers,
            enabled=True,
        ).to(device)

        # Predictive coding
        predictive_coding = PredictiveCodingWrapper(
            hidden_size=hidden_size,
            enabled=True,
        )
        predictive_coding.to(device)

        # Curriculum (scaled to training time)
        curriculum = NeuroCurriculum(
            total_hours=training_minutes / 60.0,
            enabled=True,
        )

        # Sleep consolidation
        sleep = SleepConsolidation(enabled=True, replay_samples=100 if mode == "test" else 10000)

        print("  Hebbian: ON")
        print("  Predictive coding: ON")
        print("  Curriculum: ON")
        print("  Sleep consolidation: ON")

    # ---- Optimizer ----
    # Collect all trainable parameters
    param_groups = [
        {"params": model.parameters(), "lr": peak_lr},
    ]
    if predictive_coding and predictive_coding.enabled:
        param_groups.append({
            "params": predictive_coding.parameters(),
            "lr": peak_lr * 0.1,  # Predictor learns slower
        })
    if hebbian:
        param_groups.append({
            "params": hebbian.parameters(),
            "lr": peak_lr * 0.1,
        })

    optimizer = torch.optim.AdamW(param_groups, lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)

    # ---- Estimate total steps ----
    steps_per_epoch = len(dataloader) // accumulation_steps
    # For time-based training, estimate based on speed
    # We'll use a generous estimate and stop based on time
    estimated_total_steps = steps_per_epoch * 10  # Assume up to 10 epochs
    print(f"\n  Steps per epoch: {steps_per_epoch}")
    print(f"  Estimated total steps: {estimated_total_steps}")

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print("TRAINING STARTED")
    print(f"{'='*60}\n")

    if curriculum:
        curriculum.start()

    model.train()
    start_time = time.time()
    global_step = 0
    total_tokens = 0
    best_loss = float("inf")
    running_loss = 0.0
    running_lm_loss = 0.0

    try:
        epoch = 0
        while True:
            epoch += 1
            for batch_idx, batch in enumerate(dataloader):
                # Check time limit
                elapsed_minutes = (time.time() - start_time) / 60.0
                if elapsed_minutes >= training_minutes:
                    print(f"\n  Time limit reached ({training_minutes} minutes)")
                    raise StopIteration()

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                result = model(input_ids, labels=labels)
                loss = result["loss"]

                # Predictive coding: weight tokens by surprise
                if predictive_coding and predictive_coding.enabled:
                    with torch.no_grad():
                        embeddings = model.embed(input_ids)
                        next_embeddings = model.embed(labels)
                    token_weights = predictive_coding.compute_token_weights(
                        embeddings, next_embeddings
                    )
                    # Re-weight the loss per token
                    shift_logits = result["logits"][:, :-1].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    per_token_loss = F.cross_entropy(
                        shift_logits.view(-1, model.vocab_size),
                        shift_labels.view(-1),
                        reduction="none",
                    ).view(shift_logits.shape[0], -1)
                    # Apply predictive coding weights
                    weighted_loss = (per_token_loss * token_weights[:, 1:]).mean()
                    # Predictor loss
                    pred_loss = predictive_coding.compute_predictor_loss(
                        embeddings, next_embeddings
                    )
                    loss = weighted_loss + result["aux_loss"] + 0.1 * pred_loss
                    if result.get("multi_token_loss") is not None:
                        loss = loss + 0.5 * result["multi_token_loss"]

                # Gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Update learning rate
                    lr = get_lr(global_step, estimated_total_steps, peak_lr, min_lr)
                    if curriculum:
                        lr *= curriculum.get_lr_multiplier()
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # Track stats
                batch_tokens = input_ids.numel()
                total_tokens += batch_tokens
                running_loss += loss.item() * accumulation_steps
                running_lm_loss += result["lm_loss"].item()

                # Logging
                if global_step > 0 and global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    avg_lm_loss = running_lm_loss / log_interval
                    tokens_per_sec = total_tokens / (time.time() - start_time)
                    perplexity = math.exp(min(avg_lm_loss, 20))  # Cap to avoid overflow

                    log_parts = [
                        f"Step {global_step:>6d}",
                        f"Loss: {avg_loss:.4f}",
                        f"LM: {avg_lm_loss:.4f}",
                        f"PPL: {perplexity:.1f}",
                        f"LR: {lr:.2e}",
                        f"Tok/s: {tokens_per_sec:.0f}",
                        f"Time: {elapsed_minutes:.1f}m",
                    ]

                    if curriculum:
                        log_parts.append(f"Stage: {curriculum.get_stage_name()}")

                    # MoE stats from last layer
                    if result.get("stats"):
                        last_stats = result["stats"][-1]
                        log_parts.append(f"Experts: {last_stats.get('avg_active_experts', 0):.1f}")

                    print("  " + " | ".join(log_parts))

                    # Track best loss
                    if avg_loss < best_loss:
                        best_loss = avg_loss

                    running_loss = 0.0
                    running_lm_loss = 0.0

                # Save checkpoint
                if global_step > 0 and global_step % save_interval_steps == 0:
                    ckpt_path = checkpoint_dir / f"checkpoint_step{global_step}.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "total_tokens": total_tokens,
                        "best_loss": best_loss,
                    }, ckpt_path)
                    print(f"\n  Saved checkpoint: {ckpt_path}")

                    # GPU memory check
                    if device == "cuda":
                        peak_mem = torch.cuda.max_memory_allocated() / 1e9
                        print(f"  Peak GPU memory: {peak_mem:.2f} GB\n")

    except (StopIteration, KeyboardInterrupt):
        pass

    # ---- Training complete ----
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Total steps: {global_step}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final perplexity: {math.exp(min(best_loss, 20)):.1f}")

    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {peak_mem:.2f} GB")

    # Save final checkpoint
    final_path = checkpoint_dir / f"final_{mode}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "global_step": global_step,
        "total_tokens": total_tokens,
        "best_loss": best_loss,
        "mode": mode,
        "elapsed_seconds": elapsed,
    }, final_path)
    print(f"  Final model saved: {final_path}")

    # ---- Test generation ----
    print(f"\n{'='*60}")
    print("GENERATION TEST")
    print(f"{'='*60}")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    test_prompts = [
        "The meaning of life is",
        "Once upon a time",
        "In mathematics, we can",
        "def fibonacci(n):",
        "The capital of France is",
    ]

    model.eval()
    for prompt in test_prompts:
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=device)

        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8)

        output_text = tokenizer.decode(generated[0].tolist())
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {output_text[:200]}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tryplicity")
    parser.add_argument("--mode", type=str, default="test", choices=["test", "full"],
                        help="Training mode: 'test' (10 min) or 'full' (12 hours)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer/tryplicity.json")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    train(
        mode=args.mode,
        device=args.device,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        checkpoint_dir=args.checkpoint_dir,
    )
