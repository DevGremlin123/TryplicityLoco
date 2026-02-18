# Tryplicity — Development Guidelines

## Project Overview
Tryplicity is a custom AI model AND search/chat interface. Two parts:
1. **AI Model** — Custom 9.4B parameter (3B active) hybrid architecture trained from scratch
2. **Frontend** — Dark theme + purple accent (#8b5cf6) chat interface. Node.js + Express backend.

## AI Model Architecture
- **Hybrid**: Mamba-2 (SSM) + GQA Attention + MoE (Mixture of Experts) + BitNet 1.58-bit quantization
- **Brain-inspired modules**: Spike activations, dynamic sparsity, lateral inhibition, Hebbian learning, predictive coding, sleep consolidation, neuro-curriculum
- **Full model**: 9.4B total params, ~600M active per token (64 experts, 8 active), hidden=2048, 32 layers
- **5090 model**: ~1B total params, hidden=1024, 24 layers, 32 experts — fits in 32 GB VRAM
- **BitNet**: Ternary weights {-1, 0, +1}, 10x memory compression
- **Multi-token prediction**: Predicts 2-4 tokens ahead simultaneously

## Key Files — Model & Training
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
- `training/train.py` — **Universal training script** (auto-detects single/DDP/FSDP)
- `training/data_pipeline.py` — StreamingTextDataset
- `training/build_tokenizer.py` — BPE tokenizer builder (32K vocab)
- `training/config.py` — FullConfig, ModelConfig, BrainConfig, TrainingConfig
- `run.sh` — **Universal launcher** (`bash run.sh prep`, `bash run.sh train`, `bash run.sh`)
- `download_data_mi300x.py` — Downloads training data
- `filter_data_quality.py` — 5-model AI quality filter ensemble

## Critical Dtype Rules (LEARNED FROM CRASHES)
- When model is `.to(dtype=torch.bfloat16)`, ALL buffers become bf16
- `lerp_()` requires matching dtypes — always use `.to(self.buffer.dtype)` not `.float()`
- BitNet `activation_quant()` promotes bf16→float32 via rounding ops — cast back with `.to(w_quant.dtype)`
- `F.linear()` requires matching dtypes between activations and weights
- These fixes are universal — work for both bf16 models AND fp32+autocast

## VRAM Budgets (auto-detected by train.py)
- **Single GPU <= 48 GB** (e.g. RTX 5090): `create_5090_model()`, batch=1-2, accum=32-64
- **Multi-GPU >= 150 GB/GPU** (e.g. B200): DDP, `create_full_model()`, batch=4/GPU
- **Multi-GPU < 150 GB/GPU** (e.g. H200 SXM): FSDP FULL_SHARD, `create_full_model()`, batch=4/GPU
- Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation

## Strategy Auto-Detection (train.py)
- 1 GPU → single mode (no distributed)
- Multi-GPU + >= 150 GB/GPU → DDP (each GPU holds full model)
- Multi-GPU + < 150 GB/GPU → FSDP (shards across GPUs)
- FSDP: `transformer_auto_wrap_policy` wrapping `TryplicityBlock`, gradient clipping via `model.clip_grad_norm_()`
- DDP: `find_unused_parameters=False`, gradient clipping via `torch.nn.utils.clip_grad_norm_()`

## Training Pipeline
1. `bash run.sh prep` — Install deps, download data, AI quality filter, build tokenizer
2. `bash run.sh train` — Auto-detects GPUs, picks single/DDP/FSDP, trains
3. `bash run.sh` — Both back-to-back
4. Download checkpoint before shutting down cloud pod!

Env vars: `TRAIN_MINUTES=30 TRAIN_RATE=25.00 bash run.sh train`

## Current Status (Feb 2026)
- **Unified pipeline**: One `run.sh` + one `training/train.py` for ANY hardware
- All dtype bugs fixed (4 separate dtype crashes resolved)
- 40 GB data pipeline + 5-model AI quality filter ready (`filter_data_quality.py`)
- GitHub: https://github.com/DevGremlin123/TryplicityLoco

## Frontend Design
- Dark theme: --bg: #10101f, --white: #1a1a2e, --text: #eeeef5, --purple: #7c3aed, --purple-light: #8b5cf6
- Logo: Circle-in-circle (hollow outer, solid inner) in #8b5cf6
- Font: Inter (Google Fonts)
- All frontend in `public/index.html` (single file)
- Server in `server.js` (Express, port 5000)
- No frameworks. Vanilla JS only.

## Code Standards
- Always read files before editing
- No over-engineering — only changes directly needed
- Test on target GPU before deploying to expensive hardware
- Match existing patterns (spacing, naming, structure)
