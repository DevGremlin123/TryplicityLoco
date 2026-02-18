"""
Download FULL training data for Tryplicity 12-hour run.
Target: enough data for ~50B tokens of training.

Sources (all public, freely available):
1. FineWeb-Edu (10BT sample) -- highest quality educational web text
2. Wikipedia -- encyclopedia knowledge (public domain)
3. CodeParrot-Clean -- Python code
4. Open-Web-Math -- mathematical content
5. SlimPajama (sample) -- diverse web text

All datasets are streamed to avoid downloading terabytes upfront.
"""

import os
import json
from datasets import load_dataset
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


def download_dataset(name, dataset_id, num_samples, text_key="text",
                     split="train", streaming=True, subset=None, **kwargs):
    """Generic dataset downloader."""
    out_file = DATA_DIR / f"{name}.jsonl"

    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing >= num_samples:
            print(f"  [{name}] Already have {existing:,} samples")
            return str(out_file)
        print(f"  [{name}] Resuming from {existing:,}...")

    print(f"  [{name}] Downloading {num_samples:,} samples from {dataset_id}...")

    load_kwargs = {"split": split, "streaming": streaming}
    if subset:
        load_kwargs["name"] = subset
    load_kwargs.update(kwargs)

    ds = load_dataset(dataset_id, **load_kwargs)

    count = 0
    mode = "a" if out_file.exists() else "w"
    with open(out_file, mode, encoding="utf-8") as f:
        for example in ds:
            text = example.get(text_key, "")
            if len(text) < 50:
                continue
            # Truncate very long docs to 50K chars (save disk space)
            if len(text) > 50000:
                text = text[:50000]
            f.write(json.dumps({"text": text}) + "\n")
            count += 1
            if count % 50_000 == 0:
                print(f"  [{name}] {count:,}/{num_samples:,}...")
            if count >= num_samples:
                break

    print(f"  [{name}] Saved {count:,} samples")
    return str(out_file)


if __name__ == "__main__":
    print("=" * 60)
    print("TRYPLICITY FULL DATA DOWNLOAD")
    print("Target: Enough data for 12-hour training run")
    print("=" * 60)

    # 1. FineWeb-Edu: Our primary data source. Getting 500K samples
    # (~2.5 GB, roughly 1B tokens). Highest quality web text.
    print("\n[1/5] FineWeb-Edu (educational web text)...")
    download_dataset(
        name="fineweb_edu_full",
        dataset_id="HuggingFaceFW/fineweb-edu",
        num_samples=500_000,
        subset="sample-10BT",
    )

    # 2. Wikipedia: Encyclopedia knowledge, public domain
    # 300K articles, ~1B tokens
    print("\n[2/5] Wikipedia (encyclopedia)...")
    download_dataset(
        name="wikipedia",
        dataset_id="wikimedia/wikipedia",
        num_samples=300_000,
        subset="20231101.en",
    )

    # 3. More code: Python from CodeParrot
    # 100K samples, ~400M tokens
    print("\n[3/5] Code (Python)...")
    download_dataset(
        name="code_full",
        dataset_id="codeparrot/codeparrot-clean",
        num_samples=100_000,
        text_key="content",
    )

    # 4. Math: OpenWebMath
    # 100K samples, ~400M tokens
    print("\n[4/5] Math (OpenWebMath)...")
    download_dataset(
        name="math_full",
        dataset_id="open-web-math/open-web-math",
        num_samples=100_000,
    )

    # 5. Books / stories: pg19 (Project Gutenberg books, public domain)
    # 10K books, ~500M tokens
    print("\n[5/5] Books (Project Gutenberg)...")
    download_dataset(
        name="books",
        dataset_id="emozilla/pg19",
        num_samples=10_000,
    )

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    total_size = 0
    total_samples = 0
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        lines = sum(1 for _ in open(f, "r", encoding="utf-8"))
        total_size += size_mb
        total_samples += lines
        print(f"  {f.name}: {lines:,} samples, {size_mb:.1f} MB")
    print(f"\n  TOTAL: {total_samples:,} samples, {total_size:.1f} MB")
    print(f"  Estimated tokens: ~{total_samples * 500 / 1e9:.1f}B")
    print("=" * 60)
