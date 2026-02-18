#!/bin/bash
# ============================================================
# TRYPLICITY - 8x MI300X Training Launch Script
# Run this on the rented supercomputer
# ============================================================
#
# STEP 1: Upload this entire project to the pod:
#   scp -r ./Tryplicity <pod>:~/Tryplicity
#
# STEP 2: SSH into the pod and run:
#   cd ~/Tryplicity && bash run_mi300x.sh
#
# STEP 3: After training, download your model:
#   scp <pod>:~/Tryplicity/checkpoints/final_mi300x.pt ./
#
# ============================================================

set -e

echo "============================================================"
echo "TRYPLICITY - 8x MI300X Setup & Training"
echo "============================================================"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Check GPUs
echo ""
echo "[1/4] Checking GPUs..."
python -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs found: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'  GPU {i}: {name} ({mem:.0f} GB)')
if n < 8:
    print(f'  WARNING: Expected 8 GPUs, found {n}')
print(f'  Total VRAM: {sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1e9:.0f} GB')
"

# Install dependencies if needed
echo ""
echo "[2/4] Checking dependencies..."
pip install -q torch datasets tokenizers wandb tqdm numpy 2>/dev/null || true

# Download data if not present
echo ""
echo "[3/4] Checking data..."
if [ ! -d "data" ] || [ -z "$(ls data/*.jsonl 2>/dev/null)" ]; then
    echo "  Downloading training data..."
    python download_data_mi300x.py
else
    echo "  Data already present:"
    for f in data/*.jsonl; do
        lines=$(wc -l < "$f")
        size=$(du -h "$f" | cut -f1)
        echo "    $(basename $f): $lines samples, $size"
    done
fi

# Build tokenizer if not present
if [ ! -f "tokenizer/tryplicity.json" ]; then
    echo "  Building tokenizer..."
    python training/build_tokenizer.py
else
    echo "  Tokenizer already built"
fi

# Launch training
echo ""
echo "[4/4] Launching training on $(python -c 'import torch; print(torch.cuda.device_count())') GPUs..."
echo "============================================================"
echo ""

# 11 minutes default. Change --minutes for longer runs.
torchrun \
    --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
    --master_port=29500 \
    training/train_mi300x.py \
    --minutes 11 \
    --rate 15.92 \
    --data-dir ./data \
    --tokenizer ./tokenizer/tryplicity.json \
    --checkpoint-dir ./checkpoints

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo ""
echo "DOWNLOAD YOUR MODEL BEFORE SHUTTING DOWN THE POD:"
echo "  scp <pod>:~/Tryplicity/checkpoints/final_mi300x.pt ./"
echo "============================================================"
