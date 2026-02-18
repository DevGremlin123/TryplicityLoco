"""
Tryplicity Universal Training Script — Auto-detects hardware, picks best strategy.

Single GPU (<= 48 GB):  Small model (~1B), no distributed
Multi-GPU (>= 150 GB):  Full 9.4B, DDP (each GPU holds full model)
Multi-GPU (< 150 GB):   Full 9.4B, FSDP (shards across GPUs)

Launch:
  Single GPU:  python training/train.py --minutes 30
  Multi-GPU:   torchrun --nproc_per_node=N training/train.py --minutes 11
"""

import sys, os, time, math, argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.architecture import create_full_model, create_5090_model, TryplicityBlock
from training.data_pipeline import StreamingTextDataset
from brain.curriculum import NeuroCurriculum
from brain.hebbian_loss import HebbianAuxLoss
from tokenizers import Tokenizer


def detect_hardware():
    if not torch.cuda.is_available(): raise RuntimeError("No CUDA GPUs")
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if n == 1:
        return {"n": n, "name": name, "vram": vram, "total": vram,
                "strategy": "single", "model": "small" if vram <= 48 else "full"}
    elif vram >= 150:
        return {"n": n, "name": name, "vram": vram, "total": vram*n, "strategy": "ddp", "model": "full"}
    else:
        return {"n": n, "name": name, "vram": vram, "total": vram*n, "strategy": "fsdp", "model": "full"}


def batch_cfg(hw):
    v, n, s = hw["vram"], hw["n"], hw["strategy"]
    if s == "single":
        if v <= 32: return 1, 64, 2048
        elif v <= 48: return 2, 32, 2048
        else: return 4, 16, 4096
    return 4, max(1, 32 // n), 4096


def setup(s):
    if s == "single": return 0, 1
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr)
    return lr, dist.get_world_size()

def cleanup(s):
    if s != "single":
        import torch.distributed as dist
        if dist.is_initialized(): dist.destroy_process_group()

def main_rank(s):
    if s == "single": return True
    import torch.distributed as dist
    return not dist.is_initialized() or dist.get_rank() == 0

def log(m, s="single"):
    if main_rank(s): print(m, flush=True)

def lr_sched(step, total, peak, mn, wu=0.03, st=0.82):
    w = int(total * wu); s = int(total * st)
    if step < w: return peak * (step / max(1, w))
    elif step < w + s: return peak
    else:
        p = (step - w - s) / max(1, total - w - s)
        return mn + 0.5 * (peak - mn) * (1 + math.cos(math.pi * p))


def train(minutes=11.0, rate=0.0, data_dir="./data/filtered",
          tok_path="./tokenizer/tryplicity.json", ckpt_dir="./checkpoints"):

    hw = detect_hardware()
    s = hw["strategy"]
    bpg, accum, seq = batch_cfg(hw)
    lr0, world = setup(s)
    dev = f"cuda:{lr0}"
    V = 32000
    eff = bpg * world * accum * seq

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    log("=" * 70, s)
    log(f"TRYPLICITY TRAINING", s)
    log(f"  {world}x {hw['name']} ({hw['vram']:.0f} GB) | {hw['total']:.0f} GB total", s)
    log(f"  Strategy: {s.upper()} | Model: {'9.4B' if hw['model']=='full' else '~1B'}", s)
    log(f"  {eff:,} tokens/step | {minutes:.0f} min" + (f" | ~${(minutes/60)*rate:.2f}" if rate > 0 else ""), s)
    log("=" * 70, s)

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model = create_full_model() if hw["model"] == "full" else create_5090_model()
    model.gradient_checkpointing = True
    model = model.to(dev)
    H, L = model.hidden_size, model.num_layers

    if main_rank(s):
        p = model.count_parameters()
        log(f"  Params: {p['total']:,}", s)

    use_embed_pc = False
    if s == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[lr0], find_unused_parameters=False)
        use_embed_pc = True
    elif s == "fsdp":
        from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, ShardingStrategy,
            MixedPrecision, FullStateDictConfig, StateDictType)
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from functools import partial
        model = FSDP(model,
            auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={TryplicityBlock}),
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
            sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=lr0,
            use_orig_params=True, sync_module_states=True)

    data_files = sorted(Path(data_dir).glob("*.jsonl"))
    if not data_files:
        log(f"ERROR: No .jsonl in {data_dir}. Run: bash run.sh prep", s); cleanup(s); return
    log(f"  Data: {[f.name for f in data_files]}", s)

    ds = StreamingTextDataset([str(f) for f in data_files], tok_path, seq, 50000)
    dl = DataLoader(ds, batch_size=bpg, num_workers=4, pin_memory=True)

    heb = HebbianAuxLoss(hidden_size=H, num_layers=L, enabled=True).to(dev)
    cur = NeuroCurriculum(total_hours=minutes / 60.0, enabled=True)

    pc = None
    if use_embed_pc:
        from brain.predictive_coding import PredictiveCodingWrapper
        pc = PredictiveCodingWrapper(hidden_size=H, enabled=True); pc.to(dev)

    pgs = [{"params": model.parameters(), "lr": 3e-3}, {"params": heb.parameters(), "lr": 3e-4}]
    if pc: pgs.append({"params": pc.parameters(), "lr": 3e-4})
    opt = torch.optim.AdamW(pgs, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)

    est = max(50, int((minutes * 60 * (200000 if s != "single" else 50000)) / eff))

    log(f"\nTRAINING STARTED\n", s); cur.start(); model.train()
    t0 = time.time()
    step, micro, tok, best = 0, 0, 0, float("inf")
    rl, rlm = 0.0, 0.0

    try:
        while True:
            for batch in dl:
                if (time.time() - t0) / 60 >= minutes: raise StopIteration()
                ids = batch["input_ids"].to(dev, non_blocking=True)
                lab = batch["labels"].to(dev, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    r = model(ids, labels=lab)
                    loss = r["loss"]

                    if use_embed_pc and pc:
                        raw = model.module if s == "ddp" else model
                        with torch.no_grad():
                            e1, e2 = raw.embed(ids), raw.embed(lab)
                        tw = pc.compute_token_weights(e1, e2)
                        sl, tl = r["logits"][:, :-1].contiguous(), lab[:, 1:].contiguous()
                        ptl = F.cross_entropy(sl.view(-1, V), tl.view(-1), reduction="none").view(sl.shape[0], -1)
                        loss = (ptl * tw[:, 1:]).mean() + r["aux_loss"] + 0.1 * pc.compute_predictor_loss(e1, e2)
                        if r.get("multi_token_loss") is not None: loss = loss + 0.5 * r["multi_token_loss"]
                    else:
                        with torch.no_grad():
                            sp = r["logits"][:, :-1].contiguous()
                            tp = lab[:, 1:].contiguous()
                            pr = F.softmax(sp.float(), dim=-1)
                            tw = 0.5 + (1.0 - pr.gather(-1, tp.unsqueeze(-1)).squeeze(-1).clamp(0, 1))
                            tw = tw / tw.mean().clamp(min=1e-8)
                        sl, tl = r["logits"][:, :-1].contiguous(), lab[:, 1:].contiguous()
                        ptl = F.cross_entropy(sl.reshape(-1, V), tl.reshape(-1), reduction="none").reshape(sl.shape[0], -1)
                        loss = (ptl * tw).mean() + r["aux_loss"]
                        if r.get("multi_token_loss") is not None: loss = loss + 0.5 * r["multi_token_loss"]

                (loss / accum).backward(); micro += 1
                if micro % accum == 0:
                    if s == "fsdp": model.clip_grad_norm_(1.0)
                    else: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    lr = lr_sched(step, est, 3e-3, 3e-5) * cur.get_lr_multiplier()
                    for pg in opt.param_groups: pg["lr"] = lr
                    opt.step(); opt.zero_grad(set_to_none=True); step += 1

                tok += ids.numel() * world
                rl += loss.item(); rlm += r["lm_loss"].item()

                if step > 0 and step % 10 == 0 and main_rank(s):
                    d = 10 * accum; al, am = rl/d, rlm/d; em = (time.time()-t0)/60
                    parts = [f"Step {step:>5d}", f"Loss: {al:.4f}", f"PPL: {math.exp(min(am,20)):.1f}",
                             f"Tok/s: {tok/(time.time()-t0):,.0f}", f"{tok/1e6:.0f}M tok", f"{em:.1f}m"]
                    if rate > 0: parts.append(f"${(em/60)*rate:.2f}")
                    log("  " + " | ".join(parts), s)
                    if al < best: best = al
                    rl, rlm = 0.0, 0.0

                if step > 0 and step % max(1, est // 4) == 0:
                    save(model, s, ckpt_dir, step, tok, best, dev)
    except (StopIteration, KeyboardInterrupt): pass

    if s != "single":
        import torch.distributed as dist
        if dist.is_initialized(): dist.barrier()

    el = time.time() - t0
    log(f"\n{'='*70}", s)
    log(f"DONE | {el/60:.1f}m | {step} steps | {tok/1e6:.0f}M tokens | {tok/max(1,el):,.0f} tok/s | Loss: {best:.4f}", s)
    if rate > 0: log(f"Cost: ${(el/3600)*rate:.2f}", s)

    save(model, s, ckpt_dir, step, tok, best, dev, final=True)
    gen_test(model, s, tok_path, dev)
    log(f"\nDownload: scp <pod>:~/Tryplicity/checkpoints/final.pt ./", s)
    cleanup(s)


def save(model, s, d, step, tok, loss, dev, final=False):
    name = "final.pt" if final else f"step{step}.pt"
    dat = {"global_step": step, "total_tokens": tok, "best_loss": loss}
    if s == "fsdp":
        from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            dat["model_state_dict"] = model.state_dict()
            if main_rank(s): torch.save(dat, Path(d) / name)
    elif s == "ddp":
        if main_rank(s): dat["model_state_dict"] = model.module.state_dict(); torch.save(dat, Path(d) / name)
    else:
        dat["model_state_dict"] = model.state_dict(); torch.save(dat, Path(d) / name)
    if main_rank(s): print(f"  Saved: {name} | Peak: {torch.cuda.max_memory_allocated(dev)/1e9:.1f} GB", flush=True)


def gen_test(model, s, tp, dev):
    if not main_rank(s): return
    log(f"\nGENERATION TEST", s)
    t = Tokenizer.from_file(tp)
    prompts = ["The meaning of life is", "Once upon a time", "def fibonacci(n):"]
    def g(m, i):
        x = i
        with torch.no_grad():
            for _ in range(60):
                r = m(x[:, -4096:]); p = F.softmax(r["logits"][:, -1, :] / 0.8, dim=-1)
                x = torch.cat([x, torch.multinomial(p, 1)], dim=-1)
        return x
    if s == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(model, writeback=False):
            model.eval()
            for p in prompts:
                i = torch.tensor([t.encode(p).ids], device=dev)
                log(f"  > {p}\n    {t.decode(g(model, i)[0].tolist())[:200]}", s)
    else:
        raw = model.module if s == "ddp" else model; raw.eval()
        for p in prompts:
            i = torch.tensor([t.encode(p).ids], device=dev)
            log(f"  > {p}\n    {t.decode(g(raw, i)[0].tolist())[:200]}", s)


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--minutes", type=float, default=11.0)
    pa.add_argument("--rate", type=float, default=0.0)
    pa.add_argument("--data-dir", type=str, default="./data/filtered")
    pa.add_argument("--tokenizer", type=str, default="./tokenizer/tryplicity.json")
    pa.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    a = pa.parse_args()
    train(a.minutes, a.rate, a.data_dir, a.tokenizer, a.checkpoint_dir)
