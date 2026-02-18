#!/bin/bash
# ============================================================
# Tryplicity — 16x H100 SXM Cluster Training
# ============================================================
#
# Hardware: 2 pods × 8x H100 SXM = 16 GPUs (1280 GB VRAM)
# Rate: $51.65/hr | Budget: ~$13-17 | Runtime: ~15 min
# Throughput: ~3M tokens/sec | Target: 2B tokens
#
# BEFORE RENTING: Download data locally, push to git/HuggingFace,
# then clone on the pod.
#
# ON EACH POD:
#   1. git clone <your-repo> /workspace/Tryplicity
#   2. cd /workspace/Tryplicity
#   3. pip install tokenizers datasets
#   4. python download_data_mi300x.py  (same data, works for any GPU)
#   5. python training/build_tokenizer.py
#   6. bash launch_h100.sh
# ============================================================

set -euo pipefail

echo "============================================================"
echo "TRYPLICITY — 16x H100 SXM CLUSTER"
echo "Rate: \$51.65/hr | ~15 min | ~\$13"
echo "============================================================"

# ---- NCCL settings for multi-node ----
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# ---- PyTorch optimizations ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=20  # 320 vCPU / 16 GPUs = 20 per GPU
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Better NCCL overlap

# ---- Detect cluster topology ----
# RunPod sets these env vars for multi-node
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=8

echo ""
echo "  Nodes: $NNODES | This node: $NODE_RANK"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  GPUs/node: $GPUS_PER_NODE | Total: $((NNODES * GPUS_PER_NODE))"

# ---- Check data ----
if [ ! -d "./data" ] || [ -z "$(ls -A ./data/*.jsonl 2>/dev/null)" ]; then
    echo ""
    echo "ERROR: No training data found!"
    echo "Run: python download_data_mi300x.py"
    exit 1
fi

if [ ! -f "./tokenizer/tryplicity.json" ]; then
    echo "Building tokenizer..."
    python training/build_tokenizer.py
fi

echo ""
echo "Launching on node $NODE_RANK..."

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    training/train_h100.py \
    --budget 17.0 \
    --rate 51.65 \
    --data-dir ./data \
    --tokenizer ./tokenizer/tryplicity.json \
    --checkpoint-dir ./checkpoints

echo ""
echo "============================================================"
echo "Done! Model in ./checkpoints/"
echo "============================================================"
