#!/bin/bash
# ============================================================
# TRYPLICITY - 5x NVIDIA B200 — FULL PIPELINE
#
# TWO MODES (so you can save filtered data before training):
#
#   bash run_b200_5x.sh prep     <- Download + AI filter + tokenizer (safe to stop after)
#   bash run_b200_5x.sh train    <- Train the 9.4B model on filtered data
#   bash run_b200_5x.sh          <- Run everything back-to-back
#
# ONLY 2 COMMANDS TOTAL:
#   1. git clone https://github.com/DevGremlin123/TryplicityLoco.git ~/Tryplicity
#   2. cd ~/Tryplicity && bash run_b200_5x.sh prep
#      (check your filtered data, then...)
#      bash run_b200_5x.sh train
#
# HARDWARE: 5x B200 (192 GB HBM3e each, 960 GB total)
# STRATEGY: DDP (Distributed Data Parallel) — each GPU holds full model
#           Model base = ~150 GB, B200 = 192 GB = 42 GB headroom per GPU
# COST:     ~$4.58 for 11 min at $25.00/hr (5x $5.00/GPU)
#
# ============================================================

set -e

# Memory optimization — MUST be set before any CUDA allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

MODE="${1:-all}"  # "prep", "train", or "all" (default)

# ================================================================
# PREP MODE: Download + Filter + Tokenizer
# ================================================================
run_prep() {
    echo ""
    echo "  +=========================================================+"
    echo "  |         TRYPLICITY -- DATA PREPARATION                  |"
    echo "  |         Target: 5x B200                                 |"
    echo "  |                                                         |"
    echo "  |   [1] Check GPUs                                        |"
    echo "  |   [2] Install dependencies                              |"
    echo "  |   [3] Download 40 GB training data                      |"
    echo "  |   [4] AI quality filter (5-model ensemble)              |"
    echo "  |   [5] Build tokenizer                                   |"
    echo "  |                                                         |"
    echo "  |   Filtered data saves to data/filtered/ on disk.        |"
    echo "  |   Safe to stop after this -- nothing is lost.           |"
    echo "  +=========================================================+"
    echo ""

    # ---- [1/5] Check GPUs ----
    echo "=========================================================="
    echo "  [1/5] CHECKING GPUs"
    echo "=========================================================="
    python -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs found: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'    GPU {i}: {name} ({mem:.0f} GB)')
if n < 5:
    print(f'  WARNING: Expected 5 GPUs, found {n}')
total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1e9
print(f'  Total VRAM: {total:.0f} GB')
if n >= 1:
    per_gpu = torch.cuda.get_device_properties(0).total_memory / 1e9
    if per_gpu >= 150:
        print(f'  NOTE: {per_gpu:.0f} GB/GPU -- DDP (each GPU holds full model)')
    else:
        print(f'  NOTE: {per_gpu:.0f} GB/GPU -- may need FSDP if < 150 GB')
"
    echo ""

    # ---- [2/5] Install dependencies ----
    echo "=========================================================="
    echo "  [2/5] INSTALLING DEPENDENCIES"
    echo "=========================================================="
    pip install -q torch datasets tokenizers transformers wandb tqdm numpy sentencepiece 2>/dev/null || true
    echo "  Done"
    echo ""

    # ---- [3/5] Download data ----
    echo "=========================================================="
    echo "  [3/5] DOWNLOADING 40 GB TRAINING DATA"
    echo "=========================================================="
    if [ ! -d "data" ] || [ -z "$(ls data/*.jsonl 2>/dev/null)" ]; then
        python download_data_mi300x.py
    else
        echo "  Data already present:"
        for f in data/*.jsonl; do
            if [ -f "$f" ]; then
                lines=$(wc -l < "$f")
                size=$(du -h "$f" | cut -f1)
                echo "    $(basename $f): $lines samples, $size"
            fi
        done
        echo ""
    fi

    # ---- [4/5] AI quality filter ----
    echo "=========================================================="
    echo "  [4/5] AI QUALITY FILTER -- 5-MODEL ENSEMBLE"
    echo "=========================================================="
    if [ ! -d "data/filtered" ] || [ -z "$(ls data/filtered/*.jsonl 2>/dev/null)" ]; then
        python filter_data_quality.py --data-dir ./data --output-dir ./data/filtered --threshold 3
    else
        echo "  Filtered data already present:"
        for f in data/filtered/*.jsonl; do
            if [ -f "$f" ]; then
                lines=$(wc -l < "$f")
                size=$(du -h "$f" | cut -f1)
                echo "    $(basename $f): $lines docs, $size"
            fi
        done
        echo ""
    fi

    # ---- [5/5] Build tokenizer ----
    echo "=========================================================="
    echo "  [5/5] BUILDING TOKENIZER"
    echo "=========================================================="
    if [ ! -f "tokenizer/tryplicity.json" ]; then
        python training/build_tokenizer.py
    else
        echo "  Tokenizer already built"
    fi
    echo ""

    # ---- PREP DONE ----
    echo ""
    echo "  +=========================================================+"
    echo "  |              PREP COMPLETE -- DATA IS SAVED              |"
    echo "  +---------------------------------------------------------+"
    echo "  |                                                         |"
    echo "  |  Filtered data:  ./data/filtered/*.jsonl                |"
    echo "  |  Tokenizer:      ./tokenizer/tryplicity.json            |"
    echo "  |                                                         |"
    echo "  |  Everything is on disk. Safe to stop here.              |"
    echo "  |  GPU memory is FREE -- judge models already unloaded.   |"
    echo "  |                                                         |"
    echo "  |  When ready to train, run:                              |"
    echo "  |    bash run_b200_5x.sh train                            |"
    echo "  |                                                         |"
    echo "  +=========================================================+"
    echo ""

    # Show what was saved
    echo "  Saved files:"
    if [ -d "data/filtered" ]; then
        for f in data/filtered/*.jsonl; do
            if [ -f "$f" ]; then
                size=$(du -h "$f" | cut -f1)
                lines=$(wc -l < "$f")
                echo "    $size  $(basename $f)  ($lines docs)"
            fi
        done
        echo ""
        du -sh data/filtered/ 2>/dev/null | while read size dir; do
            echo "  Total filtered data: $size"
        done
    fi
    echo ""
}

# ================================================================
# TRAIN MODE: DDP training on 5x B200
# ================================================================
run_train() {
    echo ""
    echo "  +=========================================================+"
    echo "  |       TRYPLICITY -- TRAINING 9.4B MODEL (DDP)           |"
    echo "  |       5x NVIDIA B200 -- 192 GB HBM3e each              |"
    echo "  +=========================================================+"
    echo ""

    # Verify filtered data exists
    if [ ! -d "data/filtered" ] || [ -z "$(ls data/filtered/*.jsonl 2>/dev/null)" ]; then
        echo "  ERROR: No filtered data found!"
        echo "  Run 'bash run_b200_5x.sh prep' first."
        exit 1
    fi

    # Verify tokenizer exists
    if [ ! -f "tokenizer/tryplicity.json" ]; then
        echo "  ERROR: No tokenizer found!"
        echo "  Run 'bash run_b200_5x.sh prep' first."
        exit 1
    fi

    # Show what we're training on
    echo "  Training data (AI quality-filtered):"
    for f in data/filtered/*.jsonl; do
        if [ -f "$f" ]; then
            lines=$(wc -l < "$f")
            size=$(du -h "$f" | cut -f1)
            echo "    $(basename $f): $lines docs, $size"
        fi
    done
    echo ""

    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "=========================================================="
    echo "  LAUNCHING: 9.4B model on ${NUM_GPUS}x B200"
    echo "  Strategy: DDP (each GPU holds full model)"
    echo "  B200: 192 GB/GPU, model base ~150 GB = 42 GB headroom"
    echo "  Batch: 4/GPU | Grad checkpointing: ON | bf16 mixed precision"
    echo "=========================================================="
    echo ""

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29500 \
        training/train_b200.py \
        --minutes 11 \
        --rate 25.00 \
        --data-dir ./data/filtered \
        --tokenizer ./tokenizer/tryplicity.json \
        --checkpoint-dir ./checkpoints

    echo ""
    echo "  +=========================================================+"
    echo "  |              TRAINING COMPLETE                           |"
    echo "  |                                                         |"
    echo "  |  DOWNLOAD YOUR MODEL BEFORE SHUTTING DOWN:              |"
    echo "  |  scp <pod>:~/Tryplicity/checkpoints/final_b200.pt ./   |"
    echo "  +=========================================================+"
}

# ================================================================
# Route based on mode
# ================================================================
case "$MODE" in
    prep)
        run_prep
        ;;
    train)
        run_train
        ;;
    all)
        run_prep
        echo ""
        echo "  --- Prep done. Starting training in 5 seconds... ---"
        echo "  --- (Ctrl+C now to stop -- your filtered data is saved) ---"
        echo ""
        sleep 5
        run_train
        ;;
    *)
        echo "Usage: bash run_b200_5x.sh [prep|train|all]"
        echo ""
        echo "  prep   -- Download data, AI quality filter, build tokenizer"
        echo "  train  -- Train the 9.4B model with DDP (requires prep first)"
        echo "  all    -- Run prep then train back-to-back (default)"
        echo ""
        exit 1
        ;;
esac
