#!/bin/bash
# ============================================================
# TRYPLICITY — UNIVERSAL LAUNCHER
#
# Data pipeline:
#   bash run.sh collect     <- Download ~200GB article data
#   bash run.sh filter      <- 5-judge AI filter (2 rounds)
#   bash run.sh tokenizer   <- Build tokenizer from clean data
#   bash run.sh upload      <- Upload filtered data to HuggingFace
#   bash run.sh download    <- Download filtered data from HuggingFace
#
# Training (5x B200):
#   bash run.sh train       <- Train model (~23 hrs)
#   bash run.sh all         <- filter + tokenizer + train
#
# 5 AI Judges (all run simultaneously on 10 GPUs):
#   1. FineWeb-Edu classifier  — Educational quality
#   2. CoLA grammar checker    — Grammatical acceptability
#   3. Toxic-BERT              — Toxicity detection
#   4. OpenAI GPT detector     — AI slop detection
#   5. Formality ranker        — Writing professionalism
#
# Round 1: Keep docs with >= 3/5 judge votes
# Round 2: Keep docs with >= 4/5 judge votes (only the best)
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
    echo "  ╔══════════════════════════════════════════════════╗"
    echo "  ║  TRYPLICITY — 5-JUDGE AI FILTER                  ║"
    echo "  ║                                                  ║"
    echo "  ║  Judge 1: Educational quality (FineWeb-Edu)      ║"
    echo "  ║  Judge 2: Grammar (CoLA)                         ║"
    echo "  ║  Judge 3: Toxicity (Toxic-BERT)                  ║"
    echo "  ║  Judge 4: AI slop (GPT detector)                 ║"
    echo "  ║  Judge 5: Formality (writing professionalism)    ║"
    echo "  ║                                                  ║"
    echo "  ║  Round 1: >= 3/5 votes to survive                ║"
    echo "  ║  Round 2: >= 4/5 votes to survive                ║"
    echo "  ║  Only the purest data remains.                   ║"
    echo "  ╚══════════════════════════════════════════════════╝"
    echo ""

    pip install -q torch transformers 2>/dev/null || true

    # Check for raw data
    if [ ! -d "data" ] || [ -z "$(ls data/*.jsonl 2>/dev/null)" ]; then
        echo "  ERROR: No raw data. Run: bash run.sh collect"
        exit 1
    fi

    # Round 1: Broad pass (3/5 votes)
    if [ ! -d "data/round1" ] || [ -z "$(ls data/round1/*.jsonl 2>/dev/null)" ]; then
        echo "  ── ROUND 1: 5 judges, keep >= 3/5 votes ──"
        echo ""
        python filter_data.py --round 1
    else
        echo "  Round 1 already done:"
        for f in data/round1/*.jsonl; do
            [ -f "$f" ] && echo "    $(basename $f): $(wc -l < "$f") docs"
        done
        echo ""
    fi

    # Round 2: Strict pass (4/5 votes)
    if [ ! -d "data/final" ] || [ -z "$(ls data/final/*.jsonl 2>/dev/null)" ]; then
        echo "  ── ROUND 2: 5 judges again, keep >= 4/5 votes ──"
        echo ""
        python filter_data.py --round 2
    else
        echo "  Round 2 already done:"
        for f in data/final/*.jsonl; do
            [ -f "$f" ] && echo "    $(basename $f): $(wc -l < "$f") docs"
        done
        echo ""
    fi

    echo ""
    echo "  ╔══════════════════════════════════════════════════╗"
    echo "  ║  FILTERING COMPLETE                              ║"
    echo "  ║  Only the purest data remains in data/final/     ║"
    echo "  ╚══════════════════════════════════════════════════╝"
    echo ""
    echo "  Final data:"
    for f in data/final/*.jsonl; do
        [ -f "$f" ] && echo "    $(basename $f): $(wc -l < "$f") docs"
    done
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

run_upload() {
    HF_REPO="${HF_REPO:-DevGremlin123/tryplicity-data}"
    pip install -q huggingface-hub 2>/dev/null || true

    echo ""
    echo "  TRYPLICITY — UPLOADING DATA TO HUGGINGFACE"
    echo "  Repo: ${HF_REPO}"
    echo ""

    if [ ! -d "data/final" ] || [ -z "$(ls data/final/*.jsonl 2>/dev/null)" ]; then
        echo "  ERROR: No filtered data in data/final/. Run: bash run.sh filter"
        exit 1
    fi

    # Upload filtered data
    echo "  Uploading data/final/ ..."
    huggingface-cli upload "${HF_REPO}" ./data/final --repo-type dataset --path-in-repo final

    # Upload tokenizer if it exists
    if [ -f "tokenizer/tryplicity.json" ]; then
        echo "  Uploading tokenizer..."
        huggingface-cli upload "${HF_REPO}" ./tokenizer --repo-type dataset --path-in-repo tokenizer
    fi

    echo ""
    echo "  UPLOAD COMPLETE"
    echo "  On any new pod: bash run.sh download"
    echo ""
}

run_download() {
    HF_REPO="${HF_REPO:-DevGremlin123/tryplicity-data}"
    pip install -q huggingface-hub 2>/dev/null || true

    echo ""
    echo "  TRYPLICITY — DOWNLOADING DATA FROM HUGGINGFACE"
    echo "  Repo: ${HF_REPO}"
    echo ""

    mkdir -p data/final tokenizer

    echo "  Downloading filtered data..."
    huggingface-cli download "${HF_REPO}" --repo-type dataset --include "final/*" --local-dir ./data

    if huggingface-cli download "${HF_REPO}" --repo-type dataset --include "tokenizer/*" --local-dir ./tokenizer 2>/dev/null; then
        echo "  Tokenizer downloaded."
    fi

    echo ""
    echo "  DOWNLOAD COMPLETE — data/final/ is ready"
    echo "  Next: bash run.sh train"
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
    upload)     run_upload ;;
    download)   run_download ;;
    train)      run_train ;;
    all)        run_all ;;
    help|*)
        echo ""
        echo "  TRYPLICITY LAUNCHER"
        echo ""
        echo "  Data pipeline:"
        echo "    bash run.sh collect     Download ~200GB article data"
        echo "    bash run.sh filter      5-judge AI filter (2 rounds)"
        echo "    bash run.sh tokenizer   Build tokenizer"
        echo "    bash run.sh upload      Upload filtered data to HuggingFace"
        echo "    bash run.sh download    Download filtered data from HuggingFace"
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
