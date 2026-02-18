"""
Download training data for Tryplicity MI300X run.
Target: ~2.5B tokens of diverse, high-quality data.

This is LESS than the 12-hour run (which targets 50B tokens via cycling)
because 8x MI300X processes ~750K-1.2M tokens/sec, and we only have
~36 minutes of pretraining time for $10.

The data will be cycled through ~1-2 times during training.

Sources (all public, freely available):
1. FineWeb-Edu (10BT sample) — highest quality educational web text
2. Wikipedia — encyclopedia knowledge (public domain)
3. CodeParrot-Clean — Python code
4. Open-Web-Math — mathematical content
5. SlimPajama — diverse web text for variety

Estimated download time: ~20-40 min depending on connection speed.
Estimated disk usage: ~12-15 GB.
"""

import os
import json
from datasets import load_dataset
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


def download_dataset(name, dataset_id, num_samples, text_key="text",
                     split="train", streaming=True, subset=None, **kwargs):
    """Generic dataset downloader with resume support."""
    out_file = DATA_DIR / f"{name}.jsonl"

    existing = 0
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing >= num_samples:
            print(f"  [{name}] Already have {existing:,} samples (target: {num_samples:,})")
            return str(out_file)
        print(f"  [{name}] Resuming from {existing:,}...")
        remaining = num_samples - existing
    else:
        remaining = num_samples

    print(f"  [{name}] Downloading {remaining:,} samples from {dataset_id}...")

    load_kwargs = {"split": split, "streaming": streaming}
    if subset:
        load_kwargs["name"] = subset
    load_kwargs.update(kwargs)

    ds = load_dataset(dataset_id, **load_kwargs)

    count = 0
    skipped = 0
    mode = "a" if out_file.exists() else "w"
    with open(out_file, mode, encoding="utf-8") as f:
        for example in ds:
            # Skip already-downloaded samples
            if existing > 0 and count < existing:
                count += 1
                continue

            text = example.get(text_key, "")
            if len(text) < 100:  # Slightly stricter quality filter
                skipped += 1
                continue
            # Truncate very long docs to 80K chars (keep more context than before)
            if len(text) > 80000:
                text = text[:80000]
            f.write(json.dumps({"text": text}) + "\n")
            count += 1
            if count % 100_000 == 0:
                print(f"  [{name}] {count:,}/{num_samples:,}... (skipped {skipped:,} short docs)")
            if count >= num_samples:
                break

    print(f"  [{name}] Saved {count:,} samples (skipped {skipped:,} short docs)")
    return str(out_file)


if __name__ == "__main__":
    print("=" * 70)
    print("TRYPLICITY MI300X DATA DOWNLOAD")
    print("Target: ~2.5B tokens for 8x MI300X training ($10 budget)")
    print("=" * 70)

    # Token budget breakdown:
    # - Educational web text: ~1.2B tokens (48%)  — broad knowledge
    # - Wikipedia:            ~400M tokens (16%)   — factual knowledge
    # - Code:                 ~400M tokens (16%)   — programming ability
    # - Math:                 ~300M tokens (12%)   — reasoning ability
    # - Books/stories:        ~200M tokens (8%)    — long-form coherence
    # Total:                  ~2.5B tokens

    # 1. FineWeb-Edu: Primary data source. ~1.2B tokens.
    # Average doc ~2400 tokens → 500K samples ≈ 1.2B tokens
    print("\n[1/5] FineWeb-Edu (educational web text — 48% of data)...")
    download_dataset(
        name="fineweb_edu_mi300x",
        dataset_id="HuggingFaceFW/fineweb-edu",
        num_samples=500_000,
        subset="sample-10BT",
    )

    # 2. Wikipedia: ~400M tokens
    # Average article ~1300 tokens → 300K articles ≈ 390M tokens
    print("\n[2/5] Wikipedia (encyclopedia — 16% of data)...")
    download_dataset(
        name="wikipedia_mi300x",
        dataset_id="wikimedia/wikipedia",
        num_samples=300_000,
        subset="20231101.en",
    )

    # 3. Code: ~400M tokens
    # Average file ~4000 tokens → 100K files ≈ 400M tokens
    print("\n[3/5] Code (Python — 16% of data)...")
    download_dataset(
        name="code_mi300x",
        dataset_id="codeparrot/codeparrot-clean",
        num_samples=100_000,
        text_key="content",
    )

    # 4. Math: ~300M tokens
    # Average doc ~3000 tokens → 100K docs ≈ 300M tokens
    print("\n[4/5] Math (OpenWebMath — 12% of data)...")
    download_dataset(
        name="math_mi300x",
        dataset_id="open-web-math/open-web-math",
        num_samples=100_000,
    )

    # 5. Books: ~200M tokens for long-form coherence
    # Average book ~50K tokens → 4K books ≈ 200M tokens
    print("\n[5/5] Books (Project Gutenberg — 8% of data)...")
    download_dataset(
        name="books_mi300x",
        dataset_id="emozilla/pg19",
        num_samples=4_000,
    )

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    total_size = 0
    total_samples = 0
    for f in sorted(DATA_DIR.glob("*_mi300x.jsonl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        lines = sum(1 for _ in open(f, "r", encoding="utf-8"))
        total_size += size_mb
        total_samples += lines
        print(f"  {f.name}: {lines:,} samples, {size_mb:.1f} MB")

    est_tokens = total_samples * 2400  # Average tokens per doc
    print(f"\n  TOTAL: {total_samples:,} samples, {total_size/1024:.1f} GB")
    print(f"  Estimated tokens: ~{est_tokens / 1e9:.1f}B")
    print(f"\n  MI300X throughput estimate: ~750K-1.2M tokens/sec")
    print(f"  Training time at this data volume: ~{est_tokens / 900_000 / 60:.0f} min (single pass)")
    print(f"  Budget allows: ~36 min of pretraining")
    print("=" * 70)
