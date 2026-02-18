#!/bin/bash
# ============================================================
# Tryplicity — 8x MI300X Training Launch Script
# ============================================================
#
# Budget: $10 at $12.08/hr = ~50 min total runtime
# Hardware: 8x AMD Instinct MI300X (1536 GB HBM3)
# Throughput: ~750K-1.2M tokens/sec
# Token target: ~2B tokens
#
# STEP 1: Download data (~20-40 min, do this BEFORE renting)
#   python download_data_mi300x.py
#
# STEP 2: Build tokenizer (if not already built)
#   python training/build_tokenizer.py
#
# STEP 3: Launch training (this script)
#   bash launch_mi300x.sh
#
# Total cost: ~$10
# ============================================================

set -euo pipefail

echo "============================================================"
echo "TRYPLICITY — 8x MI300X TRAINING"
echo "Budget: \$10 | Rate: \$12.08/hr | ~50 min runtime"
echo "============================================================"

# ---- Environment setup for AMD MI300X ----
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

# ROCm-specific optimizations
export MIOPEN_FIND_MODE=NORMAL
export MIOPEN_DEBUG_DISABLE_FIND_DB=0
export HSA_FORCE_FINE_GRAIN_PCIE=1

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=0
export OMP_NUM_THREADS=24  # 192 vCPU / 8 GPUs = 24 per GPU

# Memory: disable gradient checkpointing overhead (we have 1.5TB VRAM)
export TRYPLICITY_NO_GRAD_CKPT=1

# ---- Check prerequisites ----
if [ ! -d "./data" ] || [ -z "$(ls -A ./data/*mi300x*.jsonl 2>/dev/null)" ]; then
    echo ""
    echo "ERROR: No MI300X training data found!"
    echo "Run this first:  python download_data_mi300x.py"
    echo "(Do this BEFORE you start renting the GPU to save money!)"
    exit 1
fi

if [ ! -f "./tokenizer/tryplicity.json" ]; then
    echo ""
    echo "Building tokenizer..."
    python training/build_tokenizer.py
fi

# ---- Count data ----
echo ""
echo "Data files:"
for f in ./data/*mi300x*.jsonl; do
    lines=$(wc -l < "$f")
    size=$(du -h "$f" | cut -f1)
    echo "  $(basename $f): $lines samples, $size"
done

# ---- Launch distributed training ----
echo ""
echo "Launching training on 8x MI300X..."
echo "  Per-GPU batch: 32 seqs × 4096 tokens"
echo "  Effective batch: 1024 seqs = 4.2M tokens/step"
echo "  Token target: 2B"
echo ""

# torchrun handles distributed setup across 8 GPUs
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    training/train_mi300x.py \
    --budget 10.0 \
    --rate 12.08 \
    --data-dir ./data \
    --tokenizer ./tokenizer/tryplicity.json \
    --checkpoint-dir ./checkpoints

echo ""
echo "============================================================"
echo "Training complete! Check ./checkpoints/ for the model."
echo "============================================================"
