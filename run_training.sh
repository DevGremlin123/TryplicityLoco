#!/bin/bash
# ============================================================
# TRYPLICITY — Complete Training Pipeline (One Command)
# For RunPod single-machine GPU pods (8x MI300X, 8x H100, etc.)
#
# Usage:
#   chmod +x run_training.sh
#   ./run_training.sh [BUDGET] [RATE_PER_HOUR] [NUM_GPUS]
#
# Examples:
#   ./run_training.sh 10 15.92 8    # 8x MI300X at $15.92/hr
#   ./run_training.sh 10 12.08 8    # 8x MI300X at $12.08/hr (spot)
#   ./run_training.sh 15 25.84 8    # 8x H100 SXM at $25.84/hr
#   ./run_training.sh 5 3.89 4      # 4x A100 at $3.89/hr
# ============================================================
set -e

BUDGET=${1:-10}
RATE=${2:-15.92}
NUM_GPUS=${3:-8}

RUNTIME_MIN=$(python3 -c "print(f'{$BUDGET / $RATE * 60:.0f}')")

echo "============================================================"
echo "TRYPLICITY TRAINING PIPELINE"
echo "  Budget: \$$BUDGET at \$$RATE/hr = ~${RUNTIME_MIN} min"
echo "  GPUs: $NUM_GPUS"
echo "  Host: $(hostname)"
echo "============================================================"

# ── Step 1: Check hardware ──
echo ""
echo "[1/5] Hardware check..."
nvidia-smi -L
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "  Found $GPU_COUNT GPU(s)"
if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo "  WARNING: Requested $NUM_GPUS GPUs but found $GPU_COUNT. Adjusting."
    NUM_GPUS=$GPU_COUNT
fi

# ── Step 2: Install dependencies ──
echo ""
echo "[2/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision torchaudio 2>/dev/null || \
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --quiet tokenizers datasets sentencepiece fastapi uvicorn
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# ── Step 3: Download data ──
echo ""
echo "[3/5] Downloading training data (~20-40 min)..."
python3 download_data_mi300x.py

# ── Step 4: Build tokenizer ──
echo ""
echo "[4/5] Building tokenizer..."
python3 training/build_tokenizer.py

# ── Step 5: TRAIN ──
echo ""
echo "[5/5] STARTING TRAINING..."
echo "  Budget: \$$BUDGET | Rate: \$$RATE/hr | GPUs: $NUM_GPUS"
echo ""

START_TIME=$(date +%s)

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 \
    training/train_mi300x.py \
    --budget $BUDGET \
    --rate $RATE \
    --data-dir ./data \
    --tokenizer ./tokenizer/tryplicity.json \
    --checkpoint-dir ./checkpoints

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "============================================================"
echo "TRAINING COMPLETE IN ${ELAPSED} MINUTES"
echo "============================================================"
echo ""
echo "Checkpoint saved to: ./checkpoints/"
ls -lh ./checkpoints/*.pt 2>/dev/null || echo "  (no checkpoints found)"
echo ""
echo "To start the inference server:"
echo "  python3 inference/serve.py --checkpoint ./checkpoints/final_mi300x_pretrain.pt --port 8080"
echo ""
echo "============================================================"
