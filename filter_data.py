"""
Tryplicity Data Filter — 4-phase deep cleaning pipeline.

Phase 1: Quality classifier (FineWeb-Edu, threshold >= 3.0)    GPU
Phase 2: Quality classifier (stricter, threshold >= 3.5)       GPU
Phase 3: Article heuristics (standard rules)                   CPU
Phase 4: Article heuristics (strict rules — only the best)     CPU

Data flow:
  data/*.jsonl → data/phase1/ → data/phase2/ → data/phase3/ → data/final/

Usage:
  python filter_data.py --phase 1    # or 2, 3, 4
  bash run.sh filter                 # runs all 4 phases
"""

import argparse, json, os, re, time, sys
from pathlib import Path
import multiprocessing as mp


# ── Article heuristics ───────────────────────────────────────────

def is_article_quality(text, strict=False):
    """
    Check if text has the structure of a well-written article.
    strict=False: standard pass (Phase 3)
    strict=True:  tight pass (Phase 4) — only the purest prose survives
    """
    # --- Thresholds ---
    if strict:
        min_chars = 800
        min_paragraphs = 4
        min_para_words, max_para_words = 50, 400
        min_sent_words, max_sent_words = 12, 28
        max_code_ratio = 0.03
        max_list_ratio = 0.15
        min_sentences = 8
        min_punct = 8
        max_repeats = 2
    else:
        min_chars = 500
        min_paragraphs = 3
        min_para_words, max_para_words = 30, 600
        min_sent_words, max_sent_words = 10, 35
        max_code_ratio = 0.05
        max_list_ratio = 0.25
        min_sentences = 5
        min_punct = 5
        max_repeats = 3

    if len(text) < min_chars:
        return False

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < min_paragraphs:
        return False

    # Check paragraph lengths (words)
    para_word_counts = [len(p.split()) for p in paragraphs]
    avg_para_words = sum(para_word_counts) / len(para_word_counts)
    if avg_para_words < min_para_words or avg_para_words > max_para_words:
        return False

    # Sentence analysis
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) < min_sentences:
        return False

    avg_sent_words = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    if avg_sent_words < min_sent_words or avg_sent_words > max_sent_words:
        return False

    # Code/markup ratio
    code_chars = sum(c in '{}[]()<>|\\/@#$%^&*=+~`' for c in text)
    if code_chars / len(text) > max_code_ratio:
        return False

    # Bullet/list ratio
    list_lines = sum(1 for line in text.split('\n')
                     if line.strip().startswith(('-', '*', '•', '1.', '2.', '3.')))
    total_lines = max(1, len(text.split('\n')))
    if list_lines / total_lines > max_list_ratio:
        return False

    # Repetition check
    seen = {}
    for s in sentences:
        key = s.lower().strip()[:100]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= max_repeats:
            return False

    # Must have sentence-ending punctuation (proper prose)
    punct_count = text.count('.') + text.count('!') + text.count('?')
    if punct_count < min_punct:
        return False

    # --- Strict-only checks ---
    if strict:
        # Vocabulary diversity: unique words / total words > 0.3
        words = text.lower().split()
        if len(words) > 50:
            diversity = len(set(words)) / len(words)
            if diversity < 0.30:
                return False

        # No excessive caps (SHOUTING)
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        if caps_words / max(1, len(words)) > 0.05:
            return False

        # Paragraph length variance — good articles have varied paragraph sizes
        if len(para_word_counts) >= 3:
            mean_wc = sum(para_word_counts) / len(para_word_counts)
            variance = sum((wc - mean_wc) ** 2 for wc in para_word_counts) / len(para_word_counts)
            # If all paragraphs are exactly the same length, it's likely template/generated
            if variance < 10:
                return False

    return True


# ── GPU worker (Phases 1 & 2) ───────────────────────────────────

def classifier_worker(gpu_id, num_gpus, input_files, output_dir, threshold, batch_size):
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
    print(f"  [GPU {gpu_id}] Ready. Threshold: {threshold}", flush=True)

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
                if line_num % num_gpus != gpu_id:
                    continue

                try:
                    data = json.loads(raw_line)
                    text = data.get("text", "")
                except (json.JSONDecodeError, KeyError):
                    continue

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

            if texts_batch:
                kept = _classify_and_write(
                    model, tokenizer, texts_batch, lines_batch,
                    fout, device, threshold
                )
                total_processed += len(texts_batch)
                total_kept += kept

        if total_processed > 0:
            print(f"  [GPU {gpu_id}] {fname}: {total_kept:,} / {total_processed:,} kept", flush=True)

    pct = (total_kept / max(1, total_processed)) * 100
    print(f"  [GPU {gpu_id}] DONE — {total_kept:,} / {total_processed:,} ({pct:.0f}%)", flush=True)


def _classify_and_write(model, tokenizer, texts, lines, fout, device, threshold):
    """Score a batch of texts and write passing ones."""
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

        if isinstance(scores, float):
            scores = [scores]

        for score, line in zip(scores, lines):
            if score >= threshold:
                fout.write(line)
                kept += 1
    except Exception as e:
        for line in lines:
            fout.write(line)
        kept = len(lines)
        print(f"  WARNING: Batch error ({e}), kept all {len(lines)} docs", flush=True)

    return kept


# ── CPU worker (Phases 3 & 4) ───────────────────────────────────

def heuristic_worker(worker_id, num_workers, input_files, output_dir, strict):
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

                if is_article_quality(text, strict=strict):
                    fout.write(raw_line)
                    total_kept += 1

    pct = (total_kept / max(1, total_processed)) * 100
    print(f"  [Worker {worker_id}] DONE — {total_kept:,} / {total_processed:,} ({pct:.0f}%)", flush=True)


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


# ── Phase configs ────────────────────────────────────────────────

PHASES = {
    1: {
        "name": "QUALITY FILTER (broad)",
        "type": "classifier",
        "threshold": 3.0,
        "input_dir": "data",
        "output_dir": "data/phase1",
    },
    2: {
        "name": "QUALITY FILTER (strict)",
        "type": "classifier",
        "threshold": 3.5,
        "input_dir": "data/phase1",
        "output_dir": "data/phase2",
    },
    3: {
        "name": "ARTICLE HEURISTICS (standard)",
        "type": "heuristic",
        "strict": False,
        "input_dir": "data/phase2",
        "output_dir": "data/phase3",
    },
    4: {
        "name": "ARTICLE HEURISTICS (strict — only the best)",
        "type": "heuristic",
        "strict": True,
        "input_dir": "data/phase3",
        "output_dir": "data/final",
    },
}


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tryplicity 4-Phase Data Filter")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], required=True,
                        help="1-2 = quality classifier (GPU), 3-4 = article heuristics (CPU)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override input directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Phases 1-2: override quality score threshold")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Phases 1-2: batch size per GPU (default: 512)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Phases 3-4: number of CPU workers (default: auto)")
    args = parser.parse_args()

    cfg = PHASES[args.phase]
    input_dir = Path(args.data_dir or cfg["input_dir"])
    output_dir = Path(args.output_dir or cfg["output_dir"])

    input_files = sorted(input_dir.glob("*.jsonl"))
    if not input_files:
        print(f"  ERROR: No .jsonl files in {input_dir}")
        if args.phase > 1:
            print(f"  Run phase {args.phase - 1} first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Count input docs
    total_input = 0
    print()
    print(f"  PHASE {args.phase}/4: {cfg['name']}")
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

    if cfg["type"] == "classifier":
        threshold = args.threshold if args.threshold is not None else cfg["threshold"]

        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except ImportError:
            print("  ERROR: PyTorch not installed. pip install torch")
            sys.exit(1)

        if num_gpus == 0:
            print("  ERROR: No CUDA GPUs found. Phases 1-2 require GPU.")
            sys.exit(1)

        print(f"  {num_gpus} GPUs | Batch: {args.batch_size} | Threshold: {threshold}")
        print()

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=classifier_worker,
                args=(gpu_id, num_gpus, [str(f) for f in input_files],
                      str(output_dir), threshold, args.batch_size)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print()
        print("  Merging GPU outputs...")
        merge_outputs(output_dir, num_gpus, [str(f) for f in input_files], prefix="_gpu")

    else:
        strict = cfg["strict"]
        num_workers = args.workers or min(mp.cpu_count(), 16)
        print(f"  {num_workers} CPU workers | Mode: {'STRICT' if strict else 'standard'}")
        print()

        processes = []
        for wid in range(num_workers):
            p = mp.Process(
                target=heuristic_worker,
                args=(wid, num_workers, [str(f) for f in input_files],
                      str(output_dir), strict)
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
    print(f"  PHASE {args.phase}/4 COMPLETE — {elapsed/60:.1f} minutes")
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

    if args.phase < 4:
        print(f"  Next: python filter_data.py --phase {args.phase + 1}")
    else:
        print("  ALL 4 PHASES COMPLETE — data/final/ is ready for training")
        print("  Next: bash run.sh train")
    print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
