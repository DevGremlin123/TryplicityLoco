# Tryplicity — Development Guidelines

## Project Purpose
Tryplicity is an AI model that writes high-quality HTML articles from web search results.
The model writes indistinguishably from Claude Opus 4.6's style — clear, direct, well-structured prose.
Target: GPT-3 level intelligence + Opus 4.6 writing quality.

## AI Model Architecture
- **Hybrid**: Mamba-2 (SSM) + GQA Attention + MoE (Mixture of Experts) + BitNet 1.58-bit quantization
- **Brain-inspired modules**: Spike activations, dynamic sparsity, lateral inhibition, Hebbian learning, predictive coding, sleep consolidation, neuro-curriculum
- **Full model**: 9.4B total params, ~600M active per token (64 experts, 8 active), hidden=2048, 32 layers
- **5090 model**: ~1B total params, hidden=1024, 24 layers, 32 experts — fits in 32 GB VRAM
- **BitNet**: Ternary weights {-1, 0, +1}, 10x memory compression
- **Multi-token prediction**: Predicts 2-4 tokens ahead simultaneously

## Key Files
- `model/architecture.py` — TryplicityModel, create_5090_model(), create_full_model()
- `model/bitnet.py` — BitLinear (ternary weight quantization)
- `model/mamba_layer.py` — Mamba-2 SSM blocks
- `model/attention_layer.py` — GQA attention
- `model/moe_layer.py` — Mixture of Experts routing
- `model/hyper_connections.py` — Multi-stream residual connections
- `brain/spike_activation.py` — Biological spike activations with learnable thresholds
- `brain/dynamic_sparsity.py` — DynamicSparsityRouter + LateralInhibitionRouter
- `brain/hebbian_loss.py` — Hebbian auxiliary loss
- `brain/predictive_coding.py` — Predictive coding wrapper
- `brain/sleep_consolidation.py` — Between-phase memory consolidation
- `brain/curriculum.py` — NeuroCurriculum (infancy → childhood → adolescence → mastery)
- `training/train.py` — Universal training script (auto-detects single/DDP/FSDP)
- `training/data_pipeline.py` — StreamingTextDataset
- `training/build_tokenizer.py` — BPE tokenizer builder (32K vocab)
- `collect_data.py` — Downloads ~200GB article-focused training data
- `filter_data.py` — 5-judge AI filter (2 rounds, majority vote)
- `run.sh` — Universal launcher

## Data Pipeline
Training data is 100% article-focused. No code, no math, no academic papers.

Sources (~200GB total):
- FineWeb-Edu (sample-100BT) — Educational web articles, pre-scored >= 3
- Wikipedia (20231101.en) — Factual backbone
- Cosmopedia (web_samples_v2) — Synthetic textbook-quality articles
- PG19 Books — Long-form prose
- C4 RealNews — News-style articles
- C4 English — Diverse well-written web content

5-Judge AI Filter (all run simultaneously on GPUs):
1. FineWeb-Edu classifier — Educational quality (score >= 3.0 / 3.5)
2. CoLA DistilBERT — Grammatical acceptability
3. Toxic-BERT — Toxicity detection (reject toxic)
4. OpenAI GPT detector — AI slop detection (reject AI-generated)
5. Formality ranker — Writing professionalism (score >= 0.5 / 0.6)

Round 1: Keep docs with >= 3/5 judge votes (broad pass)
Round 2: Keep docs with >= 4/5 judge votes (only the best)

## Pipeline Commands
```
bash run.sh collect     # Download ~200GB article data
bash run.sh filter      # 5-judge AI filter (2 rounds)
bash run.sh tokenizer   # Build tokenizer
bash run.sh train       # Train model (5x B200, ~23 hrs)
bash run.sh all         # filter + tokenizer + train
```

Env vars: `TRAIN_MINUTES=1380 TRAIN_RATE=21.20 bash run.sh train`

## Training Hardware: 5x B200
- $21.20/hr, $500 budget ≈ 23.6 hours
- Strategy: DDP (192 GB/GPU, each holds full model)
- Checkpoint every 30-60 min

## Critical Dtype Rules (LEARNED FROM CRASHES)
- When model is `.to(dtype=torch.bfloat16)`, ALL buffers become bf16
- `lerp_()` requires matching dtypes — always use `.to(self.buffer.dtype)` not `.float()`
- BitNet `activation_quant()` promotes bf16→float32 via rounding ops — cast back with `.to(w_quant.dtype)`
- `F.linear()` requires matching dtypes between activations and weights

## Code Standards
- Always read files before editing
- No over-engineering — only changes directly needed
- Test on target GPU before deploying to expensive hardware
- Match existing patterns (spacing, naming, structure)

## GitHub
https://github.com/DevGremlin123/TryplicityLoco
