#!/bin/bash
# ============================================================
# TRYPLICITY — RTX 5090 Training Pipeline (Single GPU)
#
# Usage:
#   chmod +x run_5090.sh
#   ./run_5090.sh [BUDGET] [RATE_PER_HOUR]
#
# Examples:
#   ./run_5090.sh 10 1.50    # RTX 5090 at $1.50/hr
#   ./run_5090.sh 5 0.69     # RTX 5090 spot pricing
# ============================================================
set -e

BUDGET=${1:-10}
RATE=${2:-1.50}

RUNTIME_MIN=$(python3 -c "print(f'{$BUDGET / $RATE * 60:.0f}')")

echo "============================================================"
echo "TRYPLICITY TRAINING — RTX 5090"
echo "  Budget: \$$BUDGET at \$$RATE/hr = ~${RUNTIME_MIN} min"
echo "  GPU: Single RTX 5090 (32 GB VRAM)"
echo "  Host: $(hostname)"
echo "============================================================"

# ── Step 1: Check hardware ──
echo ""
echo "[1/5] Hardware check..."
nvidia-smi -L
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  GPU VRAM: ${GPU_MEM} MiB"

# ── Step 2: Install dependencies ──
echo ""
echo "[2/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision torchaudio 2>/dev/null || \
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --quiet tokenizers datasets sentencepiece fastapi uvicorn
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# ── Step 3: Download data ──
echo ""
echo "[3/5] Downloading training data..."
python3 download_data_mi300x.py

# ── Step 4: Build tokenizer ──
echo ""
echo "[4/5] Building tokenizer..."
python3 training/build_tokenizer.py

# ── Step 5: TRAIN (single GPU, no torchrun) ──
echo ""
echo "[5/5] STARTING TRAINING..."
echo "  Budget: \$$BUDGET | Rate: \$$RATE/hr"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START_TIME=$(date +%s)

python3 training/train_5090.py \
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
echo "  python3 inference/serve.py --checkpoint ./checkpoints/final_5090_pretrain.pt --port 8080"
echo ""
echo "============================================================"
