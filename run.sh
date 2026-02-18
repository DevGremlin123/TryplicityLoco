#!/bin/bash
# ============================================================
# TRYPLICITY — UNIVERSAL LAUNCHER
#
# Auto-detects your GPUs and picks the best strategy.
#
#   bash run.sh prep      <- Download + AI filter + tokenizer
#   bash run.sh train     <- Train (auto-detects DDP/FSDP/single)
#   bash run.sh           <- Both back-to-back
#
# Works on ANY hardware: 1x RTX 5090, 5x B200, 8x H200, etc.
# ============================================================

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

MODE="${1:-all}"

run_prep() {
    echo ""
    echo "  TRYPLICITY — DATA PREPARATION"
    echo "  [1] Check GPUs  [2] Install deps  [3] Download data"
    echo "  [4] AI quality filter  [5] Build tokenizer"
    echo ""

    echo "  [1/5] GPUs"
    python -c "
import torch
n = torch.cuda.device_count()
print(f'  {n} GPUs found')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'    GPU {i}: {name} ({mem:.0f} GB)')
total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1e9
print(f'  Total VRAM: {total:.0f} GB')
"
    echo ""

    echo "  [2/5] Dependencies"
    pip install -q torch datasets tokenizers transformers tqdm numpy sentencepiece 2>/dev/null || true
    echo "  Done"
    echo ""

    echo "  [3/5] Downloading data"
    if [ ! -d "data" ] || [ -z "$(ls data/*.jsonl 2>/dev/null)" ]; then
        python download_data_mi300x.py
    else
        echo "  Data already present"
        for f in data/*.jsonl; do [ -f "$f" ] && echo "    $(basename $f): $(wc -l < "$f") samples"; done
    fi
    echo ""

    echo "  [4/5] AI quality filter"
    if [ ! -d "data/filtered" ] || [ -z "$(ls data/filtered/*.jsonl 2>/dev/null)" ]; then
        python filter_data_quality.py --data-dir ./data --output-dir ./data/filtered --threshold 3
    else
        echo "  Filtered data already present"
        for f in data/filtered/*.jsonl; do [ -f "$f" ] && echo "    $(basename $f): $(wc -l < "$f") docs"; done
    fi
    echo ""

    echo "  [5/5] Tokenizer"
    if [ ! -f "tokenizer/tryplicity.json" ]; then
        python training/build_tokenizer.py
    else
        echo "  Already built"
    fi

    echo ""
    echo "  PREP COMPLETE — data saved to ./data/filtered/"
    echo "  Run: bash run.sh train"
    echo ""
}

run_train() {
    if [ ! -d "data/filtered" ] || [ -z "$(ls data/filtered/*.jsonl 2>/dev/null)" ]; then
        echo "  ERROR: No filtered data. Run: bash run.sh prep"
        exit 1
    fi
    if [ ! -f "tokenizer/tryplicity.json" ]; then
        echo "  ERROR: No tokenizer. Run: bash run.sh prep"
        exit 1
    fi

    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "  Launching on ${NUM_GPUS} GPUs with torchrun..."
        torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 \
            training/train.py \
            --minutes "${TRAIN_MINUTES:-11}" \
            --rate "${TRAIN_RATE:-0}" \
            --data-dir ./data/filtered \
            --tokenizer ./tokenizer/tryplicity.json \
            --checkpoint-dir ./checkpoints
    else
        echo "  Launching on 1 GPU..."
        python training/train.py \
            --minutes "${TRAIN_MINUTES:-30}" \
            --rate "${TRAIN_RATE:-0}" \
            --data-dir ./data/filtered \
            --tokenizer ./tokenizer/tryplicity.json \
            --checkpoint-dir ./checkpoints
    fi

    echo ""
    echo "  TRAINING COMPLETE"
    echo "  Download: scp <pod>:~/Tryplicity/checkpoints/final.pt ./"
    echo ""
}

case "$MODE" in
    prep) run_prep ;;
    train) run_train ;;
    all) run_prep; sleep 3; run_train ;;
    *) echo "Usage: bash run.sh [prep|train|all]"; exit 1 ;;
esac
