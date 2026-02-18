"""
Tryplicity Data Filter — Two-phase quality + article relevance filter.

Phase 1: Quality filter using pre-trained FineWeb-Edu classifier.
          Scores each document 0-5 for educational quality. Keeps >= 3.
          Distributes across all available GPUs via multiprocessing.

Phase 2: Article relevance filter using heuristic rules.
          Keeps only well-structured article-like prose.
          Runs on CPU, parallelized across processes for I/O speed.

Usage:
  python filter_data.py --phase 1                  # Quality filter
  python filter_data.py --phase 2                  # Article relevance filter
  python filter_data.py --phase 1 --threshold 2.5  # Lower quality bar
"""

import argparse, json, os, re, time, sys
from pathlib import Path
import multiprocessing as mp


# ── Phase 2 heuristics ───────────────────────────────────────────

def is_article_quality(text):
    """
    Check if text has the structure of a well-written article.
    Returns True if it passes all heuristic checks.
    Tuned for Opus 4.6 writing style: clear prose, good structure,
    no excessive code/lists, readable but not dumbed down.
    """
    if len(text) < 500:
        return False

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        return False

    # Check paragraph lengths (words)
    para_word_counts = [len(p.split()) for p in paragraphs]
    avg_para_words = sum(para_word_counts) / len(para_word_counts)
    if avg_para_words < 30 or avg_para_words > 600:
        return False

    # Sentence analysis
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) < 5:
        return False

    avg_sent_words = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    if avg_sent_words < 10 or avg_sent_words > 35:
        return False

    # Code/markup ratio — reject if too much code
    code_chars = sum(c in '{}[]()<>|\\/@#$%^&*=+~`' for c in text)
    if code_chars / len(text) > 0.05:
        return False

    # Bullet/list ratio — reject if mostly lists
    list_lines = sum(1 for line in text.split('\n')
                     if line.strip().startswith(('-', '*', '•', '1.', '2.', '3.')))
    total_lines = max(1, len(text.split('\n')))
    if list_lines / total_lines > 0.25:
        return False

    # Repetition check — reject if 3+ identical sentences
    seen = {}
    for s in sentences:
        key = s.lower().strip()[:100]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            return False

    # Must have some sentence-ending punctuation (proper prose)
    punct_count = text.count('.') + text.count('!') + text.count('?')
    if punct_count < 5:
        return False

    return True


# ── Phase 1 worker (GPU) ────────────────────────────────────────

def phase1_worker(gpu_id, num_gpus, input_files, output_dir, threshold, batch_size):
    """Classify documents using FineWeb-Edu classifier on assigned GPU."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    print(f"  [GPU {gpu_id}] Loading FineWeb-Edu classifier...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
    model = AutoModelForSequenceClassification.from_pretrained(
        "HuggingFaceFW/fineweb-edu-classifier"
    ).to(device).eval()
    print(f"  [GPU {gpu_id}] Classifier loaded. Processing...", flush=True)

    gpu_dir = Path(output_dir) / f"_gpu{gpu_id}"
    gpu_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_kept = 0

    for input_file in input_files:
        fname = Path(input_file).name
        out_file = gpu_dir / fname

        texts_batch = []
        lines_batch = []

        with open(input_file, "r", encoding="utf-8") as fin, \
             open(out_file, "w", encoding="utf-8") as fout:

            for line_num, raw_line in enumerate(fin):
                # Interleaved sharding: this GPU handles every Nth line
                if line_num % num_gpus != gpu_id:
                    continue

                try:
                    data = json.loads(raw_line)
                    text = data.get("text", "")
                except (json.JSONDecodeError, KeyError):
                    continue

                # Truncate for classifier (first 2000 chars is enough to judge quality)
                texts_batch.append(text[:2000])
                lines_batch.append(raw_line)

                if len(texts_batch) >= batch_size:
                    kept = _classify_and_write(
                        model, tokenizer, texts_batch, lines_batch,
                        fout, device, threshold
                    )
                    total_processed += len(texts_batch)
                    total_kept += kept
                    texts_batch, lines_batch = [], []

            # Flush remaining
            if texts_batch:
                kept = _classify_and_write(
                    model, tokenizer, texts_batch, lines_batch,
                    fout, device, threshold
                )
                total_processed += len(texts_batch)
                total_kept += kept

        print(f"  [GPU {gpu_id}] {fname}: {total_kept:,} kept / {total_processed:,} processed", flush=True)

    pct = (total_kept / max(1, total_processed)) * 100
    print(f"  [GPU {gpu_id}] DONE — {total_kept:,} / {total_processed:,} kept ({pct:.0f}%)", flush=True)


def _classify_and_write(model, tokenizer, texts, lines, fout, device, threshold):
    """Score a batch of texts with the classifier and write passing ones."""
    import torch

    kept = 0
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**inputs).logits
            scores = logits.squeeze(-1).float().cpu().tolist()

        # Handle single-item batch (scores becomes a float instead of list)
        if isinstance(scores, float):
            scores = [scores]

        for score, line in zip(scores, lines):
            if score >= threshold:
                fout.write(line)
                kept += 1
    except Exception as e:
        # On error, keep all docs in batch (fail-safe, don't lose data)
        for line in lines:
            fout.write(line)
        kept = len(lines)
        print(f"  WARNING: Batch error ({e}), kept all {len(lines)} docs", flush=True)

    return kept


# ── Phase 2 worker (CPU) ────────────────────────────────────────

def phase2_worker(worker_id, num_workers, input_files, output_dir):
    """Filter documents using article-quality heuristics."""
    worker_dir = Path(output_dir) / f"_worker{worker_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_kept = 0

    for input_file in input_files:
        fname = Path(input_file).name
        out_file = worker_dir / fname

        with open(input_file, "r", encoding="utf-8") as fin, \
             open(out_file, "w", encoding="utf-8") as fout:

            for line_num, raw_line in enumerate(fin):
                if line_num % num_workers != worker_id:
                    continue

                try:
                    data = json.loads(raw_line)
                    text = data.get("text", "")
                except (json.JSONDecodeError, KeyError):
                    continue

                total_processed += 1

                if is_article_quality(text):
                    fout.write(raw_line)
                    total_kept += 1

    pct = (total_kept / max(1, total_processed)) * 100
    print(f"  [Worker {worker_id}] DONE — {total_kept:,} / {total_processed:,} kept ({pct:.0f}%)", flush=True)


# ── Merge outputs ────────────────────────────────────────────────

def merge_outputs(output_dir, num_workers, input_files, prefix="_gpu"):
    """Merge per-worker output files into single files per source."""
    output_dir = Path(output_dir)

    for input_file in input_files:
        fname = Path(input_file).name
        merged = output_dir / fname
        total_lines = 0

        with open(merged, "w", encoding="utf-8") as fout:
            for wid in range(num_workers):
                part = output_dir / f"{prefix}{wid}" / fname
                if part.exists():
                    with open(part, "r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)
                            total_lines += 1

        print(f"    {fname}: {total_lines:,} docs", flush=True)

    # Cleanup temp directories
    for wid in range(num_workers):
        temp_dir = output_dir / f"{prefix}{wid}"
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tryplicity Data Filter")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True,
                        help="1 = quality filter (GPU), 2 = article relevance (CPU)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Input data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/filtered or data/final)")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Phase 1: minimum quality score 0-5 (default: 3.0)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Phase 1: batch size per GPU (default: 512)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Phase 2: number of CPU workers (default: auto)")
    args = parser.parse_args()

    if args.phase == 1:
        input_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir or "data/filtered")
    else:
        input_dir = Path(args.output_dir or "data/filtered")
        if args.output_dir:
            input_dir = Path(args.data_dir)
        else:
            input_dir = Path("data/filtered")
        output_dir = Path("data/final")

    input_files = sorted(input_dir.glob("*.jsonl"))
    if not input_files:
        print(f"  ERROR: No .jsonl files in {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Count input docs
    total_input = 0
    print()
    print(f"  PHASE {args.phase}: {'QUALITY FILTER' if args.phase == 1 else 'ARTICLE RELEVANCE FILTER'}")
    print(f"  Input:  {input_dir} ({len(input_files)} files)")
    print(f"  Output: {output_dir}")
    print()
    print("  Input files:")
    for f in input_files:
        n = sum(1 for _ in open(f, "r", encoding="utf-8"))
        total_input += n
        print(f"    {f.name}: {n:,} docs")
    print(f"    TOTAL: {total_input:,} docs")
    print()

    t0 = time.time()

    if args.phase == 1:
        # Phase 1: GPU-based quality classification
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except ImportError:
            print("  ERROR: PyTorch not installed. pip install torch")
            sys.exit(1)

        if num_gpus == 0:
            print("  ERROR: No CUDA GPUs found. Phase 1 requires GPU.")
            sys.exit(1)

        print(f"  Using {num_gpus} GPUs | Batch size: {args.batch_size} | Threshold: {args.threshold}")
        print()

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=phase1_worker,
                args=(gpu_id, num_gpus, [str(f) for f in input_files],
                      str(output_dir), args.threshold, args.batch_size)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print()
        print("  Merging GPU outputs...")
        merge_outputs(output_dir, num_gpus, [str(f) for f in input_files], prefix="_gpu")

    else:
        # Phase 2: CPU-based heuristic filtering
        num_workers = args.workers or min(mp.cpu_count(), 16)
        print(f"  Using {num_workers} CPU workers")
        print()

        processes = []
        for wid in range(num_workers):
            p = mp.Process(
                target=phase2_worker,
                args=(wid, num_workers, [str(f) for f in input_files],
                      str(output_dir))
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print()
        print("  Merging worker outputs...")
        merge_outputs(output_dir, num_workers, [str(f) for f in input_files], prefix="_worker")

    elapsed = time.time() - t0

    # Final summary
    total_output = 0
    print()
    print(f"  PHASE {args.phase} COMPLETE — {elapsed/60:.1f} minutes")
    print()
    print("  Output files:")
    for f in sorted(output_dir.glob("*.jsonl")):
        n = sum(1 for _ in open(f, "r", encoding="utf-8"))
        total_output += n
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name}: {n:,} docs ({size_mb:.0f} MB)")

    pct = (total_output / max(1, total_input)) * 100
    print(f"    TOTAL: {total_output:,} docs ({pct:.0f}% of input)")
    print()

    if args.phase == 1:
        print("  Next: python filter_data.py --phase 2")
    else:
        print("  Next: bash run.sh train")
    print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
