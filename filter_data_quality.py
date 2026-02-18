"""
Tryplicity Data Quality Filter — Multi-Model Ensemble on 4x B200
================================================================

Runs 5 diverse AI quality judges across 4 GPUs to filter training data.
A document only survives if 3/5 models agree it's high quality.

Models (diverse backgrounds = no single bias):
  1. FineWeb-Edu Classifier   — HuggingFace's educational quality scorer (RoBERTa)
  2. Qwen2.5-0.5B-Instruct    — Alibaba, different training data perspective
  3. SmolLM2-360M-Instruct    — HuggingFace, fast and capable
  4. Llama-3.2-1B-Instruct    — Meta, web-scale perspective
  5. Statistical quality       — Heuristic ensemble (repetition, info density, structure)

Pipeline:
  1. Load all models across 4 GPUs
  2. Each document scored by ALL 5 judges
  3. Majority vote (>= 3/5 pass) -> document kept
  4. Write filtered data to data/filtered/
  5. Training script reads from data/filtered/

Speed: ~30-60 min for 40 GB on 4x B200
Usage: python filter_data_quality.py [--data-dir ./data] [--threshold 3]
"""

import json
import time
import argparse
import threading
from pathlib import Path
from collections import Counter

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)


# ================================================================
# Pretty progress display
# ================================================================

def progress_bar(current, total, width=40):
    """ASCII progress bar: [========>           ] 42%"""
    if total == 0:
        return f"[{'?' * width}]  0%"
    pct = current / total
    filled = int(width * pct)
    bar = "=" * filled
    if filled < width:
        bar += ">"
        bar += " " * (width - filled - 1)
    else:
        bar = "=" * width
    return f"[{bar}] {pct*100:5.1f}%"


def format_time(seconds):
    """Format seconds as human-readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_file_header(filename, file_num, total_files, file_size_mb):
    """Print a flashy header for each file being processed."""
    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  FILE {file_num}/{total_files}: {filename:<42} │")
    print(f"  │  Size: {file_size_mb:.0f} MB{' ' * (47 - len(f'{file_size_mb:.0f} MB'))}│")
    print(f"  └──────────────────────────────────────────────────────────┘")




# ================================================================
# Quality Judge #5: Statistical Heuristics (no model needed, CPU)
# ================================================================

def statistical_quality_score(text):
    """
    Fast heuristic quality check. Returns score 0-5.
    Catches garbage that even LLMs might miss.
    """
    score = 5.0

    if len(text) < 200:
        return 0
    if len(text) < 500:
        score -= 1.0

    # Repetition detection
    sentences = text.split('.')
    if len(sentences) > 5:
        unique = set(s.strip().lower() for s in sentences if len(s.strip()) > 10)
        unique_ratio = len(unique) / len(sentences)
        if unique_ratio < 0.3:
            return 0
        if unique_ratio < 0.6:
            score -= 2.0

    # Character distribution
    sample = text[:3000]
    total_chars = len(sample)
    alpha = sum(1 for c in sample if c.isalpha())
    spaces = sum(1 for c in sample if c.isspace())
    digits = sum(1 for c in sample if c.isdigit())
    special = total_chars - alpha - digits - spaces

    alpha_ratio = alpha / total_chars
    special_ratio = special / total_chars

    if alpha_ratio < 0.40:
        if not any(kw in text[:500] for kw in ['def ', 'class ', 'import ', 'function ', '#include', 'public ', 'var ']):
            score -= 2.0

    if special_ratio > 0.30:
        score -= 1.5

    # Vocabulary richness
    words = text[:2000].lower().split()
    if len(words) > 20:
        richness = len(set(words)) / len(words)
        if richness < 0.15:
            score -= 2.0
        elif richness < 0.25:
            score -= 1.0

    # ALL CAPS spam
    if len(words) > 20:
        upper = sum(1 for w in words[:200] if w.isupper() and len(w) > 2)
        if upper / len(words[:200]) > 0.3:
            score -= 1.0

    # URL spam
    url_count = text[:3000].count('http') + text[:3000].count('www.')
    if url_count > 15:
        score -= 2.0

    return max(0, min(5, score))


# ================================================================
# LLM Quality Scorer
# ================================================================

QUALITY_PROMPT = """Rate this text's quality for training an AI. Consider coherence, informativeness, and writing quality.
Reply with ONLY a number 1-5 (1=garbage, 3=acceptable, 5=excellent).

Text:
{text}

Score:"""


def score_with_llm(model, tokenizer, texts, device, max_text_chars=800):
    """Score texts using a causal LM. Returns list of scores 1-5."""
    scores = []
    for text in texts:
        prompt = QUALITY_PROMPT.format(text=text[:max_text_chars])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        score = 3
        for char in response:
            if char.isdigit() and 1 <= int(char) <= 5:
                score = int(char)
                break
        scores.append(score)

    return scores


def score_with_classifier(model, tokenizer, texts, device, max_text_chars=1500):
    """Score texts using a classification model. Returns list of scores 0-5."""
    scores = []
    for text in texts:
        inputs = tokenizer(
            text[:max_text_chars],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits.squeeze()
            score = logits.item() if logits.dim() == 0 else logits[0].item()
            scores.append(max(0, min(5, score)))

    return scores


# ================================================================
# Load the 5 judges
# ================================================================

def load_models(num_gpus=4):
    """Load all 5 quality judges across available GPUs."""
    judges = []

    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║          LOADING 5 AI QUALITY JUDGES                    ║")
    print("  ╚══════════════════════════════════════════════════════════╝")

    # Judge 1: FineWeb-Edu Classifier (GPU 0)
    print("\n  [1/5] FineWeb-Edu Classifier (RoBERTa)")
    print("        The gold standard for educational quality scoring")
    device0 = "cuda:0" if num_gpus > 0 else "cpu"
    print(f"        Loading to {device0}...", end=" ", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
        mdl = AutoModelForSequenceClassification.from_pretrained(
            "HuggingFaceFW/fineweb-edu-classifier"
        ).to(device0).eval()
        judges.append(("FineWeb", lambda t, _m=mdl, _t=tok, _d=device0: score_with_classifier(_m, _t, t, _d), 2.5))
        print("LOADED")
    except Exception as e:
        print(f"SKIP ({e})")

    # Judge 2: Qwen2.5-0.5B-Instruct (GPU 1)
    print("\n  [2/5] Qwen2.5-0.5B-Instruct")
    print("        Alibaba's small model — fast, different training perspective")
    device1 = f"cuda:{min(1, num_gpus-1)}" if num_gpus > 0 else "cpu"
    print(f"        Loading to {device1}...", end=" ", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        mdl = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16
        ).to(device1).eval()
        judges.append(("Qwen", lambda t, _m=mdl, _t=tok, _d=device1: score_with_llm(_m, _t, t, _d), 3))
        print("LOADED")
    except Exception as e:
        print(f"SKIP ({e})")

    # Judge 3: SmolLM2-360M-Instruct (GPU 2)
    print("\n  [3/5] SmolLM2-360M-Instruct")
    print("        HuggingFace's tiny powerhouse — extremely fast scorer")
    device2 = f"cuda:{min(2, num_gpus-1)}" if num_gpus > 0 else "cpu"
    print(f"        Loading to {device2}...", end=" ", flush=True)
    try:
        smol_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        smol_mdl = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-360M-Instruct", torch_dtype=torch.bfloat16
        ).to(device2).eval()
        judges.append(("SmolLM", lambda t, _m=smol_mdl, _t=smol_tok, _d=device2: score_with_llm(_m, _t, t, _d), 3))
        print("LOADED")
    except Exception as e:
        print(f"SKIP ({e})")

    # Judge 4: Llama-3.2-1B-Instruct (GPU 3)
    print("\n  [4/5] Llama-3.2-1B-Instruct")
    print("        Meta's compact model — web-scale training perspective")
    device3 = f"cuda:{min(3, num_gpus-1)}" if num_gpus > 0 else "cpu"
    print(f"        Loading to {device3}...", end=" ", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        mdl = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16
        ).to(device3).eval()
        judges.append(("Llama", lambda t, _m=mdl, _t=tok, _d=device3: score_with_llm(_m, _t, t, _d), 3))
        print("LOADED")
    except Exception as e:
        print(f"SKIP ({e})")

    # Judge 5: Statistical (CPU)
    print("\n  [5/5] Statistical Quality Heuristics")
    print("        Catches repetition, spam, boilerplate, gibberish")
    judges.append(("Stats", lambda t: [statistical_quality_score(x) for x in t], 3.0))
    print("        LOADED")

    # Summary
    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  JUDGES READY: {len(judges)}/5 loaded                              │")
    for name, _, thresh in judges:
        status = f"  │    {name:<12} pass threshold >= {thresh:<4}                    │"
        print(status)
    print(f"  └──────────────────────────────────────────────────────────┘")

    return judges


# ================================================================
# Count lines in a file (for progress tracking)
# ================================================================

def count_lines(filepath):
    """Fast line count."""
    count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            count += 1
    return count


# ================================================================
# Filter a single file
# ================================================================

def filter_file(input_path, output_path, judges, vote_threshold, file_num, total_files):
    """Filter a JSONL file through the judge ensemble with live progress."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        return 0, 0, 0

    # Count total docs for progress bar
    file_size_mb = input_path.stat().st_size / (1024**2)
    print(f"\n  Counting docs in {input_path.name}...", end=" ", flush=True)
    total_lines = count_lines(input_path)
    print(f"{total_lines:,} documents")

    print_file_header(input_path.name, file_num, total_files, file_size_mb)

    total = 0
    kept = 0
    rejected = 0
    judge_pass_counts = Counter()
    judge_total_counts = Counter()
    start = time.time()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            data = json.loads(line)
            text = data.get("text", "")

            # Score with all judges
            votes_pass = 0
            for name, scorer, threshold in judges:
                scores = scorer([text])
                judge_total_counts[name] += 1
                if scores[0] >= threshold:
                    votes_pass += 1
                    judge_pass_counts[name] += 1

            # Majority vote
            if votes_pass >= vote_threshold:
                fout.write(line)
                kept += 1
            else:
                rejected += 1

            # Live progress update every 500 docs
            if total % 500 == 0 or total == total_lines:
                elapsed = time.time() - start
                rate = total / max(elapsed, 0.001)
                keep_pct = kept / max(total, 1) * 100
                remaining = (total_lines - total) / max(rate, 0.001)

                bar = progress_bar(total, total_lines)
                judge_str = "  ".join(
                    f"{n[:5]}:{judge_pass_counts[n]/max(judge_total_counts[n],1)*100:.0f}%"
                    for n, _, _ in judges
                )

                print(
                    f"\r  {bar}  "
                    f"{total:>8,}/{total_lines:,} | "
                    f"KEPT {keep_pct:4.1f}% | "
                    f"{rate:5,.0f} doc/s | "
                    f"ETA {format_time(remaining)} | "
                    f"{judge_str}",
                    end="", flush=True,
                )

    elapsed = time.time() - start
    keep_pct = kept / max(total, 1) * 100

    # Final summary for this file
    print()  # newline after progress bar
    print(f"  ┌── RESULT: {input_path.name}")
    print(f"  │  Scanned:  {total:>10,} documents")
    print(f"  │  Kept:     {kept:>10,} ({keep_pct:.1f}%)")
    print(f"  │  Rejected: {rejected:>10,} ({100-keep_pct:.1f}%)")
    print(f"  │  Speed:    {total/max(elapsed,1):>10,.0f} docs/sec")
    print(f"  │  Time:     {format_time(elapsed):>10}")
    print(f"  │")
    print(f"  │  Judge approval rates:")
    for name, _, _ in judges:
        jrate = judge_pass_counts[name] / max(judge_total_counts[name], 1) * 100
        bar_width = int(jrate / 100 * 20)
        print(f"  │    {name:<12} {'#' * bar_width}{'.' * (20-bar_width)} {jrate:.0f}%")
    print(f"  └──────────────────────────────────────────────────")

    return kept, rejected, total


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Filter training data with AI ensemble")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./data/filtered")
    parser.add_argument("--threshold", type=int, default=3,
                        help="Min votes to keep a document (default: 3 of 5)")
    parser.add_argument("--num-gpus", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Header
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║     TRYPLICITY DATA QUALITY FILTER                      ║")
    print("  ║     5-Model AI Ensemble — ZERO SLOP POLICY              ║")
    print("  ╚══════════════════════════════════════════════════════════╝")

    # Detect GPUs
    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus > 0:
        print(f"\n  GPUs: {num_gpus}x detected")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    GPU {i}: {name} ({mem:.0f} GB)")
    else:
        print("\n  No GPUs detected — running on CPU (will be slower)")

    # Find data files
    data_files = sorted(f for f in data_dir.glob("*.jsonl") if f.is_file())
    if not data_files:
        print(f"\n  ERROR: No .jsonl files in {data_dir}")
        return

    total_size_gb = sum(f.stat().st_size for f in data_files) / (1024**3)
    print(f"\n  Input data: {len(data_files)} files, {total_size_gb:.1f} GB")
    for f in data_files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"    {f.name}: {size_mb:.0f} MB")

    # Load judges
    judges = load_models(num_gpus)
    num_judges = len(judges)
    vote_threshold = min(args.threshold, num_judges)

    print(f"\n  Vote policy: document needs >= {vote_threshold}/{num_judges} judge approvals")

    # Filter all files
    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║              FILTERING — ZERO SLOP MODE                 ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    overall_start = time.time()
    grand_kept = 0
    grand_rejected = 0
    grand_total = 0

    for i, data_file in enumerate(data_files, 1):
        out_file = output_dir / data_file.name
        kept, rejected, docs = filter_file(
            data_file, out_file, judges, vote_threshold,
            file_num=i, total_files=len(data_files),
        )
        grand_kept += kept
        grand_rejected += rejected
        grand_total += docs

        # Running total
        elapsed = time.time() - overall_start
        overall_pct = grand_kept / max(grand_total, 1) * 100
        print(f"\n  >>> RUNNING TOTAL: {grand_kept:,}/{grand_total:,} kept ({overall_pct:.0f}%) | {format_time(elapsed)} elapsed <<<")

    # Grand summary
    elapsed = time.time() - overall_start
    keep_pct = grand_kept / max(grand_total, 1) * 100

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║              FILTERING COMPLETE                         ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Time:          {format_time(elapsed):>10}                            ║")
    print(f"  ║  Docs scanned:  {grand_total:>10,}                            ║")
    print(f"  ║  Docs KEPT:     {grand_kept:>10,}  ({keep_pct:.0f}%)                   ║")
    print(f"  ║  Docs REJECTED: {grand_rejected:>10,}  ({100-keep_pct:.0f}%)                   ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")

    # Show filtered output sizes
    filtered_size = 0
    for f in sorted(output_dir.glob("*.jsonl")):
        size_gb = f.stat().st_size / (1024**3)
        filtered_size += size_gb
        lines = count_lines(f)
        print(f"  ║  {f.name[:30]:<30} {lines:>8,} docs  {size_gb:.1f} GB  ║")

    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  FILTERED DATA: {filtered_size:.1f} GB  (from {total_size_gb:.1f} GB raw)          ║")
    print(f"  ║  Quality: top {keep_pct:.0f}% of all data                        ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Training will use: --data-dir {output_dir}")
    print()


if __name__ == "__main__":
    main()
