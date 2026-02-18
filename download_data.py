"""
Download training data for Tryplicity.
FineWeb-Edu (educational web text) - primary corpus.
We download a small streaming sample first for the 1-min test,
then continue downloading for the full 12-hour run.
"""

import os
import json
from datasets import load_dataset
from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

def download_fineweb_edu_sample(num_samples=100_000, split_name="sample"):
    """Download a sample of FineWeb-Edu for quick testing and tokenizer training."""
    print(f"Downloading FineWeb-Edu sample ({num_samples:,} examples)...")
    out_file = DATA_DIR / f"fineweb_edu_{split_name}.jsonl"

    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing >= num_samples:
            print(f"  Already have {existing:,} samples in {out_file}")
            return str(out_file)
        print(f"  Resuming from {existing:,} samples...")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    count = 0
    with open(out_file, "a" if out_file.exists() else "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if len(text) < 50:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            count += 1
            if count % 10_000 == 0:
                print(f"  Downloaded {count:,}/{num_samples:,}...")
            if count >= num_samples:
                break

    print(f"  Saved {count:,} samples to {out_file}")
    return str(out_file)


def download_code_sample(num_samples=20_000):
    """Download code samples from codeparrot (public, no auth needed)."""
    print(f"Downloading code samples ({num_samples:,} examples)...")
    out_file = DATA_DIR / "code_sample.jsonl"

    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing >= num_samples:
            print(f"  Already have {existing:,} samples in {out_file}")
            return str(out_file)

    # Use codeparrot-clean (public, no gating)
    ds = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True,
    )

    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("content", "")
            if len(text) < 100 or len(text) > 10000:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            count += 1
            if count % 5_000 == 0:
                print(f"  Downloaded {count:,}/{num_samples:,}...")
            if count >= num_samples:
                break

    print(f"  Saved {count:,} samples to {out_file}")
    return str(out_file)


def download_math_sample(num_samples=10_000):
    """Download math samples from open-web-math."""
    print(f"Downloading math samples ({num_samples:,} examples)...")
    out_file = DATA_DIR / "math_sample.jsonl"

    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing >= num_samples:
            print(f"  Already have {existing:,} samples in {out_file}")
            return str(out_file)

    ds = load_dataset(
        "open-web-math/open-web-math",
        split="train",
        streaming=True,
    )

    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if len(text) < 100:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            count += 1
            if count % 2_000 == 0:
                print(f"  Downloaded {count:,}/{num_samples:,}...")
            if count >= num_samples:
                break

    print(f"  Saved {count:,} samples to {out_file}")
    return str(out_file)


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("TRYPLICITY DATA DOWNLOAD")
    print("=" * 60)

    # Check what we need to download
    skip_fineweb = (DATA_DIR / "fineweb_edu_sample.jsonl").exists()

    if not skip_fineweb:
        print("\n--- Phase 1: Quick sample for tokenizer + testing ---")
        download_fineweb_edu_sample(num_samples=100_000)
    else:
        print("\n--- Phase 1: FineWeb-Edu already downloaded ---")

    print("\n--- Phase 2: Code samples ---")
    code_file = download_code_sample(num_samples=20_000)

    print("\n--- Phase 3: Math samples ---")
    math_file = download_math_sample(num_samples=10_000)

    print("\n" + "=" * 60)
    print("DATA DOWNLOAD COMPLETE")
    # Show file sizes
    for f in DATA_DIR.glob("*.jsonl"):
        size_mb = f.stat().st_size / (1024 * 1024)
        lines = sum(1 for _ in open(f, "r", encoding="utf-8"))
        print(f"  {f.name}: {lines:,} samples, {size_mb:.1f} MB")
    print("=" * 60)
