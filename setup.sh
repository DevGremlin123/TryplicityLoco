#!/bin/bash
# ============================================================
# TRYPLICITY — RunPod One-Shot Setup
# Run this on BOTH pods. Takes ~5-8 minutes.
# ============================================================
set -e

echo "============================================================"
echo "TRYPLICITY SETUP — $(hostname)"
echo "============================================================"

# ── 1. Hardware check ──
echo ""
echo "[1/6] Checking hardware..."
nvidia-smi -L
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "  Found $GPU_COUNT GPU(s)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  VRAM per GPU: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"

# ── 2. Install Python deps ──
echo ""
echo "[2/6] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --quiet tokenizers datasets sentencepiece wandb fastapi uvicorn
echo "  Done."

# Verify torch sees CUDA
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# ── 3. Get the code ──
echo ""
echo "[3/6] Code check..."
if [ ! -f "model/architecture.py" ]; then
    echo "  ERROR: Code not found. Upload the Tryplicity repo first!"
    echo "  From your local machine run:"
    echo "    scp -r -P <PORT> /path/to/Tryplicity/* root@<POD_IP>:~/Tryplicity/"
    echo "  Or: git clone <your-repo-url>"
    exit 1
fi
echo "  Code found."

# ── 4. Download training data ──
echo ""
echo "[4/6] Downloading training data..."
python3 download_data.py
echo "  Data ready."

# ── 5. Build tokenizer ──
echo ""
echo "[5/6] Building tokenizer..."
python3 training/build_tokenizer.py
echo "  Tokenizer ready."

# ── 6. Smoke test ──
echo ""
echo "[6/6] Smoke test..."
python3 -c "
from model.architecture import create_tiny_model
import torch
model = create_tiny_model().cuda()
x = torch.randint(0, 32000, (2, 64)).cuda()
result = model(x, labels=x)
print(f'  Model OK — Loss: {result[\"loss\"].item():.4f}')
result['loss'].backward()
print('  Backward OK')
"

echo ""
echo "============================================================"
echo "SETUP COMPLETE — Ready to train!"
echo "  GPUs: $GPU_COUNT"
echo "  Host: $(hostname)"
echo "============================================================"

# Print internal IP (needed for multi-pod training)
INTERNAL_IP=$(hostname -I | awk '{print $1}')
echo ""
echo ">>> INTERNAL IP: $INTERNAL_IP <<<"
echo ">>> Save this — Pod 0's IP is needed as MASTER_ADDR <<<"
