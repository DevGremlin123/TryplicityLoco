"""
Tryplicity Data Collection — Download ~200GB of article-focused training data.

All sources chosen for article-quality prose and factual knowledge.
No code, no math, no academic papers. Just well-written articles.

Sources:
  1. FineWeb-Edu (sample-100BT)  — Educational web articles, pre-scored
  2. Wikipedia (20231101.en)      — Factual backbone, clean article structure
  3. Cosmopedia (web_samples_v2)  — Synthetic textbook-quality articles
  4. PG19 Books                   — Long-form prose, writing quality
  5. C4 RealNews                  — News-style articles
  6. C4 English                   — Diverse well-written web content

Run:  python collect_data.py
"""

import json, os, time, threading, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("data")
MANIFEST_FILE = DATA_DIR / ".manifest.json"
_print_lock = threading.Lock()
_manifest_lock = threading.Lock()


# ── Dataset configs ──────────────────────────────────────────────

DATASETS = [
    {
        "name": "fineweb_edu",
        "dataset_id": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-100BT",
        "num_samples": 8_000_000,
        "text_key": "text",
        "min_score": 3,
        "score_key": "int_score",
        "min_chars": 300,
        "est_gb": 100,
        "desc": "Educational web articles (score >= 3)",
    },
    {
        "name": "wikipedia",
        "dataset_id": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "num_samples": 6_800_000,
        "text_key": "text",
        "min_chars": 200,
        "est_gb": 22,
        "desc": "English Wikipedia — factual articles",
    },
    {
        "name": "cosmopedia",
        "dataset_id": "HuggingFaceTB/cosmopedia",
        "subset": "web_samples_v2",
        "num_samples": 2_000_000,
        "text_key": "text",
        "min_chars": 300,
        "est_gb": 20,
        "desc": "Synthetic textbook-quality articles",
    },
    {
        "name": "books_pg19",
        "dataset_id": "emozilla/pg19",
        "subset": None,
        "num_samples": 28_000,
        "text_key": "text",
        "min_chars": 500,
        "est_gb": 11,
        "desc": "Classic books — long-form prose",
    },
    {
        "name": "c4_realnews",
        "dataset_id": "allenai/c4",
        "subset": "realnewslike",
        "num_samples": 13_000_000,
        "text_key": "text",
        "min_chars": 300,
        "est_gb": 15,
        "desc": "News-style articles from Common Crawl",
    },
    {
        "name": "c4_english",
        "dataset_id": "allenai/c4",
        "subset": "en",
        "num_samples": 5_000_000,
        "text_key": "text",
        "min_chars": 300,
        "est_gb": 35,
        "desc": "Diverse well-written English web content",
    },
]


# ── Helpers ──────────────────────────────────────────────────────

def tprint(msg):
    with _print_lock:
        print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_manifest():
    with _manifest_lock:
        if MANIFEST_FILE.exists():
            return json.loads(MANIFEST_FILE.read_text())
        return {}


def save_manifest(manifest):
    with _manifest_lock:
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))


def is_quality_text(text, min_chars=300):
    """Basic pre-filter. Real filtering happens in filter_data.py."""
    if len(text) < min_chars:
        return False
    # Alphanumeric ratio check
    alnum = sum(c.isalnum() or c.isspace() for c in text[:2000])
    if alnum / min(len(text), 2000) < 0.50:
        return False
    # Skip if too many special chars (likely code or markup)
    special = sum(c in '{}[]()<>|\\/@#$%^&*=+~`' for c in text[:2000])
    if special / min(len(text), 2000) > 0.08:
        return False
    return True


def count_lines(filepath):
    """Fast line count."""
    count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            count += 1
    return count


# ── Download function ────────────────────────────────────────────

def download_dataset(cfg):
    name = cfg["name"]
    out_file = DATA_DIR / f"{name}.jsonl"

    # Check manifest for completion
    manifest = load_manifest()
    if manifest.get(name, {}).get("complete"):
        saved = manifest[name].get("lines_written", 0)
        tprint(f"[{name}] Already complete ({saved:,} docs). Skipping.")
        return str(out_file)

    try:
        from datasets import load_dataset
    except ImportError:
        tprint(f"[{name}] ERROR: pip install datasets")
        return None

    # Resume support
    existing = 0
    if out_file.exists():
        existing = count_lines(out_file)
        if existing > 0:
            tprint(f"[{name}] Resuming from {existing:,} docs...")

    tprint(f"[{name}] Loading {cfg['dataset_id']}"
           + (f" ({cfg['subset']})" if cfg.get("subset") else "")
           + f" — target {cfg['num_samples']:,} docs, ~{cfg['est_gb']} GB")

    try:
        kwargs = {
            "split": "train",
            "streaming": True,
            "trust_remote_code": True,
        }
        if cfg.get("subset"):
            ds = load_dataset(cfg["dataset_id"], cfg["subset"], **kwargs)
        else:
            ds = load_dataset(cfg["dataset_id"], **kwargs)
    except Exception as e:
        tprint(f"[{name}] FAILED to load: {e}")
        return None

    text_key = cfg["text_key"]
    min_chars = cfg.get("min_chars", 300)
    min_score = cfg.get("min_score")
    score_key = cfg.get("score_key")
    target = cfg["num_samples"]
    saved = existing
    skipped_resume = 0
    skipped_quality = 0
    skipped_score = 0

    mode = "a" if existing > 0 else "w"
    with open(out_file, mode, encoding="utf-8") as f:
        for i, row in enumerate(ds):
            # Skip already-saved rows when resuming
            if skipped_resume < existing:
                skipped_resume += 1
                continue

            if saved >= target:
                break

            text = row.get(text_key, "")
            if not text:
                continue

            # Score-based filter (FineWeb-Edu)
            if min_score and score_key:
                score = row.get(score_key, 0)
                if score is not None and score < min_score:
                    skipped_score += 1
                    continue

            # Truncate very long docs to 100K chars
            if len(text) > 100_000:
                text = text[:100_000]

            # Basic quality check
            if not is_quality_text(text, min_chars):
                skipped_quality += 1
                continue

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            saved += 1

            if saved % 100_000 == 0:
                tprint(f"[{name}] {saved:,} / {target:,} docs saved"
                       + f" (skipped: {skipped_quality:,} quality, {skipped_score:,} score)")

    tprint(f"[{name}] DONE — {saved:,} docs saved"
           + f" (skipped: {skipped_quality:,} quality, {skipped_score:,} score)")

    # Update manifest
    manifest = load_manifest()
    manifest[name] = {"complete": True, "lines_written": saved}
    save_manifest(manifest)

    return str(out_file)


# ── Main ─────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  TRYPLICITY — DATA COLLECTION")
    print("  Target: ~200 GB of article-focused training data")
    print("=" * 60)
    print()
    print("  Sources:")
    total_gb = 0
    total_docs = 0
    for cfg in DATASETS:
        print(f"    {cfg['name']:20s} ~{cfg['est_gb']:>3d} GB  {cfg['num_samples']:>10,} docs  {cfg['desc']}")
        total_gb += cfg["est_gb"]
        total_docs += cfg["num_samples"]
    print(f"    {'TOTAL':20s} ~{total_gb:>3d} GB  {total_docs:>10,} docs")
    print()

    # Check manifest for already-completed datasets
    manifest = load_manifest()
    remaining = [cfg for cfg in DATASETS if not manifest.get(cfg["name"], {}).get("complete")]
    done = len(DATASETS) - len(remaining)
    if done > 0:
        print(f"  {done} datasets already complete (from manifest). {len(remaining)} remaining.")
        print()

    if not remaining:
        print("  All datasets already downloaded!")
        print()
        # Print summary
        for cfg in DATASETS:
            f = DATA_DIR / f"{cfg['name']}.jsonl"
            if f.exists():
                n = manifest.get(cfg["name"], {}).get("lines_written", "?")
                size_mb = f.stat().st_size / 1e6
                print(f"    {cfg['name']:20s} {n:>10} docs  {size_mb:>8.0f} MB")
        return

    t0 = time.time()

    # Download with 3 parallel workers
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download_dataset, cfg): cfg["name"] for cfg in remaining}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result:
                    tprint(f"[{name}] Complete: {result}")
                else:
                    tprint(f"[{name}] FAILED")
            except Exception as e:
                tprint(f"[{name}] ERROR: {e}")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  COLLECTION COMPLETE — {elapsed/60:.0f} minutes")
    print("=" * 60)
    print()

    # Final summary
    total_size = 0
    total_lines = 0
    for cfg in DATASETS:
        f = DATA_DIR / f"{cfg['name']}.jsonl"
        if f.exists():
            size_mb = f.stat().st_size / 1e6
            n = load_manifest().get(cfg["name"], {}).get("lines_written", 0)
            print(f"    {cfg['name']:20s} {n:>10,} docs  {size_mb/1000:>6.1f} GB")
            total_size += size_mb
            total_lines += n

    print(f"    {'TOTAL':20s} {total_lines:>10,} docs  {total_size/1000:>6.1f} GB")
    print()
    print("  Next step: bash run.sh filter")
    print()


if __name__ == "__main__":
    main()
