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
- `training/train_5090.py` — Single GPU training script for RTX 5090 (32 GB VRAM)
- `training/train_mi300x.py` — Multi-GPU training script for 8x MI300X (DDP)
- `training/train_b200.py` — Multi-GPU training script for 4x NVIDIA B200 (DDP)
- `training/train_h200.py` — Multi-GPU training script for 8x H200 SXM (FSDP)
- `training/data_pipeline.py` — StreamingTextDataset
- `training/build_tokenizer.py` — BPE tokenizer builder (32K vocab)
- `training/config.py` — FullConfig, ModelConfig, BrainConfig, TrainingConfig
- `run_5090.sh` — Launch script for 5090 training
- `run_training.sh` — Launch script for multi-GPU training
- `run_b200.sh` — Launch script for 4x B200 training
- `run_h200.sh` — Launch script for 8x H200 SXM training (FSDP)
- `download_data_mi300x.py` — Downloads training data
- `filter_data_quality.py` — 5-model AI quality filter ensemble

## Critical Dtype Rules (LEARNED FROM CRASHES)
- When model is `.to(dtype=torch.bfloat16)`, ALL buffers become bf16
- `lerp_()` requires matching dtypes — always use `.to(self.buffer.dtype)` not `.float()`
- BitNet `activation_quant()` promotes bf16→float32 via rounding ops — cast back with `.to(w_quant.dtype)`
- `F.linear()` requires matching dtypes between activations and weights
- These fixes are universal — work for both bf16 models AND fp32+autocast

## VRAM Budgets
- **RTX 5090 (32 GB)**: Use `create_5090_model()`, batch=1, accumulation=64, gradient_checkpointing=ON
- **4x B200 (192 GB each)**: DDP, `create_full_model()`, batch=4/GPU, accum=8, grad_ckpt=ON, ~175 GB/GPU
- **8x MI300X (192 GB each)**: DDP, `create_full_model()`, batch=2/GPU, accum=8, grad_ckpt=ON, ~170 GB/GPU
- **8x H200 SXM (141 GB each)**: FSDP FULL_SHARD (DDP won't fit! 150 GB > 141 GB), batch=4/GPU, accum=4, ~41 GB/GPU
- Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
- DDP: weights=4B/param, optimizer=8B/param, grads=4B/param → 150 GB base for 9.4B (needs 150+ GB/GPU)
- FSDP: shards across GPUs → 18.8 GB base per GPU for 8 GPUs (fits in 141 GB H200)

## FSDP Notes (H200)
- Uses `transformer_auto_wrap_policy` wrapping each `TryplicityBlock` as a separate FSDP unit
- Weight tying (embed <-> lm_head) preserved with `use_orig_params=True`
- Predictive coding uses logit confidence instead of `raw_model.embed()` (can't access model.module with FSDP)
- Checkpoint saving uses `FSDP.state_dict_type()` with `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)`
- Generation test uses `FSDP.summon_full_params()` to temporarily gather all params
- Gradient clipping: use `model.clip_grad_norm_()` (FSDP method), NOT `torch.nn.utils.clip_grad_norm_()`

## Training Pipeline
1. Install deps (torch, tokenizers, datasets, sentencepiece)
2. Download data (`download_data_mi300x.py`)
3. Build tokenizer (`training/build_tokenizer.py`)
4. Train (`training/train_5090.py` or `training/train_b200.py` or `training/train_h200.py` or `training/train_mi300x.py`)
5. Download checkpoint before shutting down cloud pod!

## Current Status (Feb 2026)
- 5090 script with medium model (~1B params) ready but UNTESTED
- All dtype bugs fixed (4 separate dtype crashes resolved)
- 4x B200 script ready (`train_b200.py` + `run_b200.sh`): DDP, batch=4/GPU, ~175 GB/GPU, ~$3.67 for 11 min
- **8x H200 SXM script ready** (`train_h200.py` + `run_h200.sh`): FSDP, batch=4/GPU, ~41 GB/GPU, ~$5.26 for 11 min
- 8x MI300X script ready (`train_mi300x.py` + `run_mi300x.sh`): DDP
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
