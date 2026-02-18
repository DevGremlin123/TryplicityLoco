#!/bin/bash
# ============================================================
# TRYPLICITY — UNIVERSAL LAUNCHER
#
# Data pipeline (can run anywhere):
#   bash run.sh collect     <- Download ~200GB article data
#   bash run.sh filter      <- Phase 1 (quality) + Phase 2 (article relevance)
#   bash run.sh tokenizer   <- Build tokenizer from filtered data
#
# Training (5x B200):
#   bash run.sh train       <- Train model (~23 hrs on 5x B200)
#   bash run.sh all         <- filter + tokenizer + train
#
# ============================================================

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false

MODE="${1:-help}"

run_collect() {
    echo ""
    echo "  TRYPLICITY — COLLECTING ~200GB ARTICLE DATA"
    echo ""
    pip install -q datasets tokenizers tqdm numpy 2>/dev/null || true
    python collect_data.py
}

run_filter() {
    echo ""
    echo "  TRYPLICITY — FILTERING DATA (Phase 1 + Phase 2)"
    echo ""

    pip install -q torch transformers 2>/dev/null || true

    # Check for raw data
    if [ ! -d "data" ] || [ -z "$(ls data/*.jsonl 2>/dev/null)" ]; then
        echo "  ERROR: No raw data. Run: bash run.sh collect"
        exit 1
    fi

    # Phase 1: Quality filter (GPU)
    if [ ! -d "data/filtered" ] || [ -z "$(ls data/filtered/*.jsonl 2>/dev/null)" ]; then
        echo "  ── Phase 1: Quality Filter (FineWeb-Edu classifier) ──"
        python filter_data.py --phase 1 --data-dir ./data --threshold 3.0 --batch-size 512
    else
        echo "  Phase 1 already done:"
        for f in data/filtered/*.jsonl; do
            [ -f "$f" ] && echo "    $(basename $f): $(wc -l < "$f") docs"
        done
    fi

    echo ""

    # Phase 2: Article relevance (CPU heuristics)
    if [ ! -d "data/final" ] || [ -z "$(ls data/final/*.jsonl 2>/dev/null)" ]; then
        echo "  ── Phase 2: Article Relevance Filter (heuristics) ──"
        python filter_data.py --phase 2
    else
        echo "  Phase 2 already done:"
        for f in data/final/*.jsonl; do
            [ -f "$f" ] && echo "    $(basename $f): $(wc -l < "$f") docs"
        done
    fi

    echo ""
    echo "  FILTERING COMPLETE"
    echo ""
}

run_tokenizer() {
    if [ ! -d "data/final" ] || [ -z "$(ls data/final/*.jsonl 2>/dev/null)" ]; then
        echo "  ERROR: No filtered data. Run: bash run.sh filter"
        exit 1
    fi

    echo "  Building tokenizer from filtered data..."
    python training/build_tokenizer.py
    echo "  Tokenizer built: tokenizer/tryplicity.json"
}

run_train() {
    if [ ! -d "data/final" ] || [ -z "$(ls data/final/*.jsonl 2>/dev/null)" ]; then
        echo "  ERROR: No filtered data. Run: bash run.sh filter"
        exit 1
    fi

    if [ ! -f "tokenizer/tryplicity.json" ]; then
        echo "  Building tokenizer first..."
        python training/build_tokenizer.py
    fi

    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

    echo ""
    echo "  TRYPLICITY — TRAINING"
    echo "  GPUs: ${NUM_GPUS} | Data: ./data/final/"
    echo ""

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 \
            training/train.py \
            --minutes "${TRAIN_MINUTES:-1380}" \
            --rate "${TRAIN_RATE:-21.20}" \
            --data-dir ./data/final \
            --tokenizer ./tokenizer/tryplicity.json \
            --checkpoint-dir ./checkpoints
    else
        python training/train.py \
            --minutes "${TRAIN_MINUTES:-1380}" \
            --rate "${TRAIN_RATE:-0}" \
            --data-dir ./data/final \
            --tokenizer ./tokenizer/tryplicity.json \
            --checkpoint-dir ./checkpoints
    fi

    echo ""
    echo "  TRAINING COMPLETE"
    echo "  Download: scp <pod>:~/Tryplicity/checkpoints/final.pt ./"
    echo ""
}

run_all() {
    run_filter
    sleep 2
    run_tokenizer
    sleep 2
    run_train
}

case "$MODE" in
    collect)    run_collect ;;
    filter)     run_filter ;;
    tokenizer)  run_tokenizer ;;
    train)      run_train ;;
    all)        run_all ;;
    help|*)
        echo ""
        echo "  TRYPLICITY LAUNCHER"
        echo ""
        echo "  Data pipeline:"
        echo "    bash run.sh collect     Download ~200GB article data"
        echo "    bash run.sh filter      Phase 1 (quality) + Phase 2 (relevance)"
        echo "    bash run.sh tokenizer   Build tokenizer from filtered data"
        echo ""
        echo "  Training (5x B200, \$21.20/hr):"
        echo "    bash run.sh train       Train model (~23 hrs)"
        echo "    bash run.sh all         filter + tokenizer + train"
        echo ""
        echo "  Env vars:"
        echo "    TRAIN_MINUTES=1380  TRAIN_RATE=21.20  bash run.sh train"
        echo ""
        ;;
esac
