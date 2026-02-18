"""
Tryplicity Training Data — 40 GB Extremely High Quality
========================================================

ZERO SLOP policy. Every source is either:
  - Pre-scored by quality classifiers (FineWeb-Edu score >= 3)
  - Naturally premium (Wikipedia, academic papers, classic books)
  - Synthetically generated textbooks (Cosmopedia — no web noise at all)
  - Expert-curated domain data (OpenWebMath, clean code)

Quality filters on ALL data:
  - Min 300 chars (kills stubs and fragments)
  - Alphanumeric ratio >= 50% (kills HTML/boilerplate/garbage)
  - FineWeb-Edu: only int_score >= 3 (top quality tier)
  - Truncate at 80K chars (prevents single mega-docs from dominating)

Downloads run in PARALLEL (4 workers) for speed.
Resume support: safe to re-run if interrupted.

Sources (all public, no auth required):
  1. FineWeb-Edu (score >= 3) — best educational web text
  2. Cosmopedia              — synthetic textbooks (zero slop)
  3. Wikipedia               — factual gold standard
  4. OpenWebMath             — curated math content
  5. CodeParrot-Clean        — deduplicated Python
  6. PG19 Books              — classic literature
  7. peS2o                   — academic papers (Semantic Scholar)

Target: ~40 GB on disk, ~10B tokens
Estimated download: ~45-90 min on a cloud pod
"""

import json
import time
import threading
from datasets import load_dataset
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Thread-safe print
_print_lock = threading.Lock()
def tprint(msg):
    with _print_lock:
        print(msg, flush=True)


def is_quality_text(text, min_chars=300):
    """
    Quality gate — rejects slop, boilerplate, and garbage.
    Returns True only for clean, substantive text.
    """
    if len(text) < min_chars:
        return False

    # Alphanumeric ratio check — kills HTML, encoded data, symbol spam
    alnum_count = sum(1 for c in text[:2000] if c.isalnum() or c.isspace())
    if alnum_count / min(len(text), 2000) < 0.50:
        return False

    # Must have some sentence structure (at least 3 periods/question marks)
    sentence_endings = text.count('.') + text.count('?') + text.count('!')
    if sentence_endings < 3 and not any(kw in text[:200] for kw in ['def ', 'class ', 'import ', 'function ', '#include']):
        # Exception: code files don't need sentence endings
        return False

    return True


def download_dataset(name, dataset_id, num_samples, text_key="text",
                     split="train", streaming=True, subset=None,
                     min_score=None, score_key="int_score",
                     min_chars=300, **kwargs):
    """
    Download a dataset with quality filtering and resume support.

    Args:
        min_score: If set, only keep docs where score_key >= min_score
                   (used for FineWeb-Edu quality scoring)
    """
    out_file = DATA_DIR / f"{name}.jsonl"

    existing = 0
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing >= num_samples:
            tprint(f"  [{name}] Already have {existing:,} samples (target: {num_samples:,})")
            return str(out_file)
        tprint(f"  [{name}] Resuming from {existing:,}...")
    else:
        pass

    tprint(f"  [{name}] Downloading {num_samples - existing:,} samples from {dataset_id}...")
    if min_score is not None:
        tprint(f"  [{name}] Quality filter: {score_key} >= {min_score}")

    load_kwargs = {"split": split, "streaming": streaming}
    if subset:
        load_kwargs["name"] = subset
    load_kwargs.update(kwargs)

    ds = load_dataset(dataset_id, **load_kwargs)

    saved = 0
    scanned = 0
    skipped_short = 0
    skipped_quality = 0
    skipped_score = 0
    start = time.time()

    mode = "a" if out_file.exists() else "w"
    with open(out_file, mode, encoding="utf-8") as f:
        for example in ds:
            scanned += 1

            # Skip already-downloaded samples (for resume)
            if existing > 0 and scanned <= existing:
                continue

            # Score filter (FineWeb-Edu quality tiers)
            if min_score is not None:
                score = example.get(score_key, 0)
                if score is None or score < min_score:
                    skipped_score += 1
                    continue

            text = example.get(text_key, "")

            # Quality gate
            if not is_quality_text(text, min_chars=min_chars):
                if len(text) < min_chars:
                    skipped_short += 1
                else:
                    skipped_quality += 1
                continue

            # Truncate very long docs
            if len(text) > 80000:
                text = text[:80000]

            f.write(json.dumps({"text": text}) + "\n")
            saved += 1

            if saved % 50_000 == 0:
                elapsed = time.time() - start
                rate = saved / max(elapsed, 1)
                eta = (num_samples - existing - saved) / max(rate, 1) / 60
                tprint(
                    f"  [{name}] {existing + saved:,}/{num_samples:,} saved "
                    f"| scanned {scanned:,} "
                    f"| rejected: {skipped_score + skipped_short + skipped_quality:,} "
                    f"| {rate:.0f} docs/s "
                    f"| ETA: {eta:.0f} min"
                )

            if existing + saved >= num_samples:
                break

    total_rejected = skipped_short + skipped_quality + skipped_score
    accept_rate = saved / max(scanned - existing, 1) * 100
    tprint(
        f"  [{name}] DONE: {saved:,} saved from {scanned:,} scanned "
        f"({accept_rate:.0f}% accept rate) "
        f"| rejected {total_rejected:,} "
        f"(short={skipped_short:,}, quality={skipped_quality:,}, score={skipped_score:,})"
    )
    return str(out_file)


# ================================================================
# Dataset definitions — ordered by quality importance
# ================================================================

DATASETS = [
    {
        "name": "fineweb_edu",
        "dataset_id": "HuggingFaceFW/fineweb-edu",
        "num_samples": 2_000_000,
        "subset": "sample-10BT",
        "min_score": 3,           # Only top quality tier (score 3-5 out of 0-5)
        "score_key": "int_score",
        "description": "FineWeb-Edu score>=3 (best educational web text)",
        "est_gb": 16,
        "share": "40%",
    },
    {
        "name": "cosmopedia",
        "dataset_id": "HuggingFaceTB/cosmopedia",
        "num_samples": 500_000,
        "subset": "web_samples_v2",
        "description": "Cosmopedia (synthetic textbooks — zero web slop)",
        "est_gb": 5,
        "share": "12.5%",
    },
    {
        "name": "wikipedia",
        "dataset_id": "wikimedia/wikipedia",
        "num_samples": 1_000_000,
        "subset": "20231101.en",
        "description": "Wikipedia (factual gold standard)",
        "est_gb": 6,
        "share": "15%",
    },
    {
        "name": "openwebmath",
        "dataset_id": "open-web-math/open-web-math",
        "num_samples": 350_000,
        "min_chars": 200,         # Math docs can be shorter but still valuable
        "description": "OpenWebMath (curated math & reasoning)",
        "est_gb": 3,
        "share": "7.5%",
    },
    {
        "name": "code_python",
        "dataset_id": "codeparrot/codeparrot-clean",
        "num_samples": 350_000,
        "text_key": "content",
        "min_chars": 200,         # Code files can be shorter
        "description": "CodeParrot-Clean (deduplicated Python)",
        "est_gb": 4,
        "share": "10%",
    },
    {
        "name": "books_pg19",
        "dataset_id": "emozilla/pg19",
        "num_samples": 15_000,
        "description": "PG19 Books (classic literature, long-form coherence)",
        "est_gb": 3,
        "share": "7.5%",
    },
    {
        "name": "peso2_academic",
        "dataset_id": "allenai/peS2o",
        "num_samples": 400_000,
        "subset": "v2",
        "description": "peS2o (Semantic Scholar academic papers)",
        "est_gb": 3,
        "share": "7.5%",
    },
]


def download_one(ds_config):
    """Download a single dataset (called by thread pool)."""
    # Extract download_dataset args from config
    kwargs = {k: v for k, v in ds_config.items()
              if k not in ("description", "est_gb", "share")}
    return download_dataset(**kwargs)


if __name__ == "__main__":
    print("=" * 70)
    print("TRYPLICITY DATA DOWNLOAD — 40 GB EXTREMELY HIGH QUALITY")
    print("=" * 70)
    print()
    print("Quality policy: ZERO SLOP")
    print("  - FineWeb-Edu: only int_score >= 3 (top quality tier)")
    print("  - All data: min 300 chars, >= 50% alphanumeric, sentence structure")
    print("  - Cosmopedia: synthetic textbooks (no web noise at all)")
    print("  - Wikipedia + Academic papers: naturally premium sources")
    print()

    # Show plan
    total_est_gb = 0
    print("  Download plan:")
    print(f"  {'Source':<50} {'Samples':>10}  {'~Size':>6}  {'Share':>6}")
    print(f"  {'─'*50} {'─'*10}  {'─'*6}  {'─'*6}")
    for ds in DATASETS:
        print(f"  {ds['description']:<50} {ds['num_samples']:>10,}  {ds['est_gb']:>4} GB  {ds['share']:>6}")
        total_est_gb += ds["est_gb"]
    print(f"  {'─'*50} {'─'*10}  {'─'*6}")
    print(f"  {'TOTAL':<50} {'':>10}  {total_est_gb:>4} GB")
    print()

    # Download in parallel — 4 workers for speed
    # (not 7 because HuggingFace rate limits + disk I/O contention)
    NUM_WORKERS = 4
    print(f"  Downloading with {NUM_WORKERS} parallel workers...")
    print("=" * 70)

    start_time = time.time()
    results = {}

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(download_one, ds): ds["name"] for ds in DATASETS}

        for future in as_completed(futures):
            name = futures[future]
            try:
                path = future.result()
                results[name] = path
                tprint(f"\n  >>> [{name}] Complete! <<<\n")
            except Exception as e:
                tprint(f"\n  >>> [{name}] FAILED: {e} <<<")
                tprint(f"  >>> Re-run the script to retry (resume support will skip completed data) <<<\n")
                results[name] = None

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print(f"Total download time: {elapsed/60:.1f} minutes")
    print("=" * 70)

    total_size = 0
    total_samples = 0
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        lines = sum(1 for _ in open(f, "r", encoding="utf-8"))
        total_size += size_mb
        total_samples += lines
        print(f"  {f.name}: {lines:,} samples, {size_mb/1024:.1f} GB")

    est_tokens = total_samples * 2400  # Conservative avg tokens per doc
    print(f"\n  TOTAL: {total_samples:,} samples, {total_size/1024:.1f} GB")
    print(f"  Estimated tokens: ~{est_tokens / 1e9:.1f}B")
    print(f"\n  4x B200 throughput: ~200K-400K tokens/sec")
    print(f"  Single pass: ~{est_tokens / 300_000 / 60:.0f} min at 300K tok/s")
    if est_tokens > 0:
        print(f"  11-min run sees: ~{11 * 60 * 300_000 / 1e9:.1f}B tokens ({11 * 60 * 300_000 / est_tokens * 100:.0f}% of data)")
    print("=" * 70)
