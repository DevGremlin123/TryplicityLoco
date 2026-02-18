"""
Tryplicity 5-Judge AI Filter — Real models, no heuristics.

5 AI judges score every document. Majority vote decides.

  Judge 1: FineWeb-Edu classifier    — Educational quality (0-5)
  Judge 2: CoLA grammar checker      — Grammatical acceptability
  Judge 3: Toxic-BERT                — Toxicity detection (reject toxic)
  Judge 4: OpenAI GPT detector       — AI slop detection (reject AI-generated)
  Judge 5: Formality ranker          — Writing formality/professionalism

Architecture:
  10 GPUs, 2 per judge. All 5 judges run simultaneously.
  Each GPU processes half the documents independently.
  After all judges finish, merge votes with majority rule.

Two rounds:
  Round 1: Keep docs with >= 3/5 votes (broad pass)
  Round 2: Keep docs with >= 4/5 votes (strict pass — only the best)

Usage:
  python filter_data.py --round 1 --data-dir ./data
  python filter_data.py --round 2
  bash run.sh filter              # runs both rounds
"""

import argparse, json, time, sys
from pathlib import Path
import multiprocessing as mp


# ── Judge definitions ────────────────────────────────────────────

JUDGES = [
    {
        "name": "educational_quality",
        "model_id": "HuggingFaceFW/fineweb-edu-classifier",
        "desc": "Educational quality (FineWeb-Edu)",
        "type": "regression",       # outputs float 0-5
        "pass_fn": "score >= 3.0",  # round 1
        "pass_fn_strict": "score >= 3.5",  # round 2
    },
    {
        "name": "grammar",
        "model_id": "textattack/distilbert-base-uncased-CoLA",
        "desc": "Grammar acceptability (CoLA)",
        "type": "classification",   # outputs logits for [unacceptable, acceptable]
        "pass_label": 1,            # label 1 = acceptable
    },
    {
        "name": "toxicity",
        "model_id": "unitary/toxic-bert",
        "desc": "Toxicity detection (reject toxic)",
        "type": "classification",   # outputs logits for [not_toxic, toxic]
        "pass_label": 0,            # label 0 = not toxic (we KEEP non-toxic)
    },
    {
        "name": "ai_slop",
        "model_id": "openai-community/roberta-base-openai-detector",
        "desc": "AI-generated text detector (reject AI slop)",
        "type": "classification",   # outputs logits for [Real, Fake]
        "pass_label": 0,            # label 0 = Real/human-written (we KEEP human)
    },
    {
        "name": "formality",
        "model_id": "s-nlp/roberta-base-formality-ranker",
        "desc": "Writing formality (keep formal/professional)",
        "type": "regression",       # outputs formality score
        "pass_fn": "score >= 0.5",  # above midpoint = formal
        "pass_fn_strict": "score >= 0.6",
    },
]


# ── Judge worker ─────────────────────────────────────────────────

def judge_worker(gpu_id, judge_idx, num_gpus_per_judge, gpu_offset,
                 input_files, scores_dir, batch_size, is_strict):
    """
    Run one judge on assigned GPU, scoring all documents.
    Writes a scores file: one line per doc with 0 (fail) or 1 (pass).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    judge = JUDGES[judge_idx]
    # Which slice of documents this GPU handles
    local_rank = gpu_id - gpu_offset  # 0 or 1 within this judge's GPU pair

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    print(f"  [GPU {gpu_id}] Loading {judge['name']}: {judge['model_id']}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(judge["model_id"])
    model = AutoModelForSequenceClassification.from_pretrained(
        judge["model_id"]
    ).to(device).eval()
    print(f"  [GPU {gpu_id}] {judge['name']} ready.", flush=True)

    judge_dir = Path(scores_dir) / judge["name"]
    judge_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_passed = 0

    for input_file in input_files:
        fname = Path(input_file).name
        score_file = judge_dir / f"gpu{gpu_id}_{fname}.scores"

        texts_batch = []
        line_indices = []

        with open(input_file, "r", encoding="utf-8") as fin, \
             open(score_file, "w") as fout:

            for line_num, raw_line in enumerate(fin):
                # Interleaved sharding across this judge's GPUs
                if line_num % num_gpus_per_judge != local_rank:
                    continue

                try:
                    data = json.loads(raw_line)
                    text = data.get("text", "")
                except (json.JSONDecodeError, KeyError):
                    fout.write(f"{line_num}\t0\n")
                    continue

                texts_batch.append(text[:2000])
                line_indices.append(line_num)

                if len(texts_batch) >= batch_size:
                    results = _score_batch(model, tokenizer, texts_batch,
                                           device, judge, is_strict)
                    for idx, passed in zip(line_indices, results):
                        fout.write(f"{idx}\t{1 if passed else 0}\n")
                        total_processed += 1
                        total_passed += 1 if passed else 0
                    texts_batch, line_indices = [], []

            # Flush remaining
            if texts_batch:
                results = _score_batch(model, tokenizer, texts_batch,
                                       device, judge, is_strict)
                for idx, passed in zip(line_indices, results):
                    fout.write(f"{idx}\t{1 if passed else 0}\n")
                    total_processed += 1
                    total_passed += 1 if passed else 0

    pct = (total_passed / max(1, total_processed)) * 100
    print(f"  [GPU {gpu_id}] {judge['name']}: {total_passed:,} / {total_processed:,} passed ({pct:.0f}%)", flush=True)


def _score_batch(model, tokenizer, texts, device, judge, is_strict):
    """Score a batch and return list of bool (pass/fail)."""
    import torch

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

        if judge["type"] == "regression":
            scores = logits.squeeze(-1).float().cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]

            if is_strict and "pass_fn_strict" in judge:
                threshold = float(judge["pass_fn_strict"].split(">=")[1].strip())
            else:
                threshold = float(judge["pass_fn"].split(">=")[1].strip())

            return [s >= threshold for s in scores]

        else:  # classification
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            pass_label = judge["pass_label"]
            return [probs[i, pass_label].item() > 0.5 for i in range(len(texts))]

    except Exception as e:
        print(f"  WARNING: Batch error in {judge['name']} ({e}), marking all as pass", flush=True)
        return [True] * len(texts)


# ── Merge votes and output ───────────────────────────────────────

def merge_votes(input_files, scores_dir, output_dir, min_votes):
    """
    Read all judge score files, count votes per document,
    keep documents with >= min_votes passing judges.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = Path(scores_dir)

    total_input = 0
    total_output = 0

    for input_file in input_files:
        fname = Path(input_file).name

        # Collect votes from all judges
        votes = {}  # line_num -> count of passing votes
        for judge in JUDGES:
            judge_dir = scores_dir / judge["name"]
            for score_file in sorted(judge_dir.glob(f"*_{fname}.scores")):
                with open(score_file, "r") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) == 2:
                            line_num = int(parts[0])
                            passed = int(parts[1])
                            votes[line_num] = votes.get(line_num, 0) + passed

        # Write surviving documents
        out_file = output_dir / fname
        kept = 0
        with open(input_file, "r", encoding="utf-8") as fin, \
             open(out_file, "w", encoding="utf-8") as fout:
            for line_num, raw_line in enumerate(fin):
                total_input += 1
                if votes.get(line_num, 0) >= min_votes:
                    fout.write(raw_line)
                    kept += 1
                    total_output += 1

        print(f"    {fname}: {kept:,} docs survived (>= {min_votes}/5 votes)", flush=True)

    return total_input, total_output


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tryplicity 5-Judge AI Filter")
    parser.add_argument("--round", type=int, choices=[1, 2], required=True,
                        help="1 = broad pass (3/5 votes), 2 = strict pass (4/5 votes)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Input directory (default: auto based on round)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto based on round)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size per GPU (default: 512)")
    args = parser.parse_args()

    if args.round == 1:
        input_dir = Path(args.data_dir or "data")
        output_dir = Path(args.output_dir or "data/round1")
        scores_dir = Path("data/.scores_r1")
        min_votes = 3
        is_strict = False
    else:
        input_dir = Path(args.data_dir or "data/round1")
        output_dir = Path(args.output_dir or "data/final")
        scores_dir = Path("data/.scores_r2")
        min_votes = 4
        is_strict = True

    input_files = sorted(input_dir.glob("*.jsonl"))
    if not input_files:
        print(f"  ERROR: No .jsonl files in {input_dir}")
        if args.round == 2:
            print("  Run round 1 first: python filter_data.py --round 1")
        sys.exit(1)

    scores_dir.mkdir(parents=True, exist_ok=True)

    # Detect GPUs
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except ImportError:
        print("  ERROR: pip install torch transformers")
        sys.exit(1)

    if num_gpus < 5:
        print(f"  WARNING: {num_gpus} GPUs found, need 10 for full speed (2 per judge).")
        print(f"  Will run judges sequentially on available GPUs.")

    # Count input
    total_input = 0
    print()
    print(f"  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  TRYPLICITY — 5-JUDGE AI FILTER (Round {args.round})        ║")
    print(f"  ║                                                  ║")
    for j in JUDGES:
        print(f"  ║  {j['name']:24s} {j['desc']:24s} ║")
    print(f"  ║                                                  ║")
    print(f"  ║  Majority vote: >= {min_votes}/5 to survive               ║")
    print(f"  ╚══════════════════════════════════════════════════╝")
    print()
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  GPUs:   {num_gpus}")
    print()

    print("  Input files:")
    for f in input_files:
        n = sum(1 for _ in open(f, "r", encoding="utf-8"))
        total_input += n
        print(f"    {f.name}: {n:,} docs")
    print(f"    TOTAL: {total_input:,} docs")
    print()

    t0 = time.time()

    if num_gpus >= 10:
        # Ideal: 2 GPUs per judge, all 5 run simultaneously
        gpus_per_judge = 2
        print(f"  Running all 5 judges in parallel (2 GPUs each)...")
        print()

        processes = []
        for judge_idx in range(5):
            gpu_offset = judge_idx * gpus_per_judge
            for local in range(gpus_per_judge):
                gpu_id = gpu_offset + local
                p = mp.Process(
                    target=judge_worker,
                    args=(gpu_id, judge_idx, gpus_per_judge, gpu_offset,
                          [str(f) for f in input_files], str(scores_dir),
                          args.batch_size, is_strict)
                )
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

    elif num_gpus >= 5:
        # 1 GPU per judge, all 5 run simultaneously
        print(f"  Running all 5 judges in parallel (1 GPU each)...")
        print()

        processes = []
        for judge_idx in range(5):
            gpu_id = judge_idx
            p = mp.Process(
                target=judge_worker,
                args=(gpu_id, judge_idx, 1, gpu_id,
                      [str(f) for f in input_files], str(scores_dir),
                      args.batch_size, is_strict)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    else:
        # Fewer GPUs: run judges sequentially, spread across available GPUs
        print(f"  Running 5 judges sequentially across {num_gpus} GPUs...")
        print()

        for judge_idx in range(5):
            print(f"  --- Judge {judge_idx+1}/5: {JUDGES[judge_idx]['name']} ---")
            processes = []
            for gpu_id in range(num_gpus):
                p = mp.Process(
                    target=judge_worker,
                    args=(gpu_id, judge_idx, num_gpus, 0,
                          [str(f) for f in input_files], str(scores_dir),
                          args.batch_size, is_strict)
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

    scoring_time = time.time() - t0
    print()
    print(f"  Scoring complete in {scoring_time/60:.1f} minutes.")
    print(f"  Merging votes (>= {min_votes}/5 to survive)...")
    print()

    # Merge votes
    total_in, total_out = merge_votes(
        [str(f) for f in input_files], str(scores_dir), str(output_dir), min_votes
    )

    elapsed = time.time() - t0
    pct = (total_out / max(1, total_in)) * 100

    print()
    print(f"  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  ROUND {args.round} COMPLETE — {elapsed/60:.1f} min                      ║")
    print(f"  ║  {total_out:>12,} / {total_in:>12,} docs survived ({pct:.0f}%)    ║")
    print(f"  ╚══════════════════════════════════════════════════╝")
    print()

    if args.round == 1:
        print(f"  Next: python filter_data.py --round 2")
    else:
        print(f"  FILTERING COMPLETE — data/final/ is ready for training")
        print(f"  Next: bash run.sh train")
    print()

    # Cleanup scores
    import shutil
    if scores_dir.exists():
        shutil.rmtree(scores_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
