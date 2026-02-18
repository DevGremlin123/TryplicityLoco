"""
Tryplicity Training Configuration
All hyperparameters in one place. Brain optimization modules have bypass flags.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Core model architecture parameters."""
    hidden_size: int = 2048
    num_layers: int = 32
    num_attention_layers: int = 4  # Placed at layers 8, 16, 24, 32
    num_mamba_layers: int = 28

    # Attention
    num_attention_heads: int = 16
    num_kv_heads: int = 4  # GQA
    head_dim: int = 128  # hidden_size // num_attention_heads

    # Mamba-2
    mamba_state_dim: int = 128
    mamba_conv_dim: int = 4
    mamba_expand_factor: int = 2

    # MoE
    num_experts: int = 64
    num_shared_experts: int = 2
    base_active_experts: int = 8
    expert_hidden_size: int = 512

    # BitNet
    use_bitnet: bool = True

    # Hyper-connections
    use_hyper_connections: bool = True
    hyper_num_streams: int = 4

    # Vocabulary and context
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # Norms
    rms_norm_eps: float = 1e-6

    # Dropout (zero during pretraining, small during fine-tuning)
    dropout: float = 0.0

    @property
    def attention_layer_indices(self):
        """Layers that use attention instead of Mamba (0-indexed)."""
        # Layers 8, 16, 24, 32 in 1-indexed = 7, 15, 23, 31 in 0-indexed
        return [7, 15, 23, 31]


@dataclass
class BrainConfig:
    """Brain-inspired optimization flags and parameters."""

    # Master toggle
    enable_brain_optimizations: bool = True

    # 1. Predictive Coding
    enable_predictive_coding: bool = True
    prediction_threshold: float = 0.7
    predictor_layers: int = 2
    predictor_ema_alpha: float = 0.01

    # 2. Dynamic Sparsity
    enable_dynamic_sparsity: bool = True
    min_active_experts: int = 4
    max_active_experts: int = 12
    difficulty_aux_loss_weight: float = 0.01
    difficulty_entropy_weight: float = 0.001

    # 3. Lateral Inhibition
    enable_lateral_inhibition: bool = True
    inhibition_strength: float = 0.3
    similarity_update_interval: int = 1000

    # 4. Spike Activations
    enable_spike_activations: bool = True
    initial_threshold: float = 0.1
    target_sparsity: float = 0.7  # 70% zeros target

    # 5. Hebbian Auxiliary Losses
    enable_hebbian_loss: bool = True
    hebbian_decorr_weight: float = 0.01
    hebbian_sparsity_weight: float = 0.01
    hebbian_predictive_weight: float = 0.005
    hebbian_depth_decay: float = 0.8  # Multiply weight by this per layer

    # 6. Curriculum
    enable_curriculum: bool = True

    # 7. Sleep Consolidation
    enable_sleep_consolidation: bool = True
    sleep_replay_samples: int = 10000
    sleep_replay_lr: float = 1e-5
    sleep_prune_threshold: float = 0.01  # Prune experts used < 1%
    sleep_checkpoint_average_n: int = 3


@dataclass
class CurriculumStageConfig:
    """Configuration for a single curriculum stage."""
    name: str
    hours_start: float
    hours_end: float
    max_seq_len: int
    max_reading_level: Optional[float]
    text_ratio: float
    code_ratio: float
    math_ratio: float
    batch_size_tokens: int
    lr_multiplier: float
    hebbian_weight_multiplier: float


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Phases (default: 12-hour RTX 5070 schedule)
    pretrain_hours: float = 9.5
    instruct_hours: float = 50 / 60  # 50 minutes
    grpo_hours: float = 55 / 60  # 55 minutes
    sleep1_minutes: float = 10.0
    sleep2_minutes: float = 5.0

    # Hardware profile (overridden by get_mi300x_config)
    num_gpus: int = 1
    hardware_profile: str = "rtx5070"  # "rtx5070" or "mi300x"

    # Optimizer: Muon
    optimizer: str = "muon"
    peak_lr: float = 3e-3
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)

    # WSD Schedule
    warmup_ratio: float = 0.02
    stable_ratio: float = 0.88
    decay_ratio: float = 0.10

    # Instruct-tune
    instruct_lr: float = 3e-4
    instruct_warmup_ratio: float = 0.05

    # GRPO
    grpo_lr: float = 1e-5
    grpo_num_completions: int = 8
    grpo_steps: int = 200
    grpo_math_ratio: float = 0.4
    grpo_code_ratio: float = 0.3
    grpo_reasoning_ratio: float = 0.3

    # Multi-token prediction
    num_predict_tokens: int = 4

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    # GaLore
    use_galore: bool = True
    galore_rank: int = 256

    # Logging
    log_interval: int = 100
    save_interval_hours: float = 1.5  # Save checkpoint every 1.5 hours
    use_wandb: bool = True
    wandb_project: str = "tryplicity"

    # Data
    total_tokens_target: int = 50_000_000_000  # 50B tokens
    tokenizer_vocab_size: int = 32000

    # Curriculum stages
    curriculum_stages: list = field(default_factory=lambda: [
        CurriculumStageConfig(
            name="infancy",
            hours_start=0.5, hours_end=2.0,
            max_seq_len=512,
            max_reading_level=6.0,
            text_ratio=0.9, code_ratio=0.1, math_ratio=0.0,
            batch_size_tokens=256_000,
            lr_multiplier=0.5,
            hebbian_weight_multiplier=1.0,
        ),
        CurriculumStageConfig(
            name="childhood",
            hours_start=2.0, hours_end=5.0,
            max_seq_len=2048,
            max_reading_level=10.0,
            text_ratio=0.75, code_ratio=0.15, math_ratio=0.10,
            batch_size_tokens=1_000_000,
            lr_multiplier=1.0,
            hebbian_weight_multiplier=0.5,
        ),
        CurriculumStageConfig(
            name="adolescence",
            hours_start=5.0, hours_end=8.0,
            max_seq_len=4096,
            max_reading_level=None,
            text_ratio=0.65, code_ratio=0.20, math_ratio=0.15,
            batch_size_tokens=2_000_000,
            lr_multiplier=1.0,
            hebbian_weight_multiplier=0.2,
        ),
        CurriculumStageConfig(
            name="mastery",
            hours_start=8.0, hours_end=10.0,
            max_seq_len=4096,
            max_reading_level=None,
            text_ratio=0.50, code_ratio=0.25, math_ratio=0.25,
            batch_size_tokens=2_000_000,
            lr_multiplier=0.7,
            hebbian_weight_multiplier=0.1,
        ),
    ])


@dataclass
class FullConfig:
    """Complete configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"  # For activations; weights are BitNet ternary
    seed: int = 42

    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    tokenizer_path: str = "./tokenizer"
    log_dir: str = "./logs"


# Default config instance (RTX 5070 / 12-hour schedule)
def get_default_config() -> FullConfig:
    return FullConfig()


def get_mi300x_config(budget_dollars: float = 10.0, rate_per_hour: float = 12.08) -> FullConfig:
    """
    Optimized config for 8x MI300X cluster.

    Hardware: 8x AMD Instinct MI300X
      - 1536 GB HBM3 total VRAM
      - ~10,460 TFLOPS BF16 aggregate
      - ~42.4 TB/s memory bandwidth
      - 2264 GB system RAM, 192 vCPU

    Speed vs RTX 5070: ~50-80x faster (NOT 100Kx)
    Budget: $10 at $12.08/hr = ~50 min runtime

    Estimated throughput: ~750K-1.2M tokens/sec for our 3B BitNet MoE model
    Target: ~2B tokens in ~40 min of pretraining
    """
    runtime_hours = budget_dollars / rate_per_hour  # ~0.83 hours = ~50 min
    runtime_minutes = runtime_hours * 60  # ~50 min total

    # Time allocation (within budget)
    pretrain_minutes = runtime_minutes * 0.72   # ~36 min pretraining
    instruct_minutes = runtime_minutes * 0.12   # ~6 min instruct-tune
    grpo_minutes = runtime_minutes * 0.10       # ~5 min GRPO alignment
    sleep1_minutes = runtime_minutes * 0.04     # ~2 min sleep consolidation
    sleep2_minutes = runtime_minutes * 0.02     # ~1 min final consolidation

    config = FullConfig()

    # -- Training schedule for MI300X --
    config.training.pretrain_hours = pretrain_minutes / 60.0
    config.training.instruct_hours = instruct_minutes / 60.0
    config.training.grpo_hours = grpo_minutes / 60.0
    config.training.sleep1_minutes = sleep1_minutes
    config.training.sleep2_minutes = sleep2_minutes
    config.training.num_gpus = 8
    config.training.hardware_profile = "mi300x"

    # -- Massive batch sizes (1536 GB VRAM allows this) --
    # Per-GPU batch: 32 sequences × 4096 tokens = 131K tokens/GPU
    # 8 GPUs × 32 = 256 sequences per step
    # Gradient accumulation 4 = effective 1024 sequences = 4.2M tokens/step
    # This is near-optimal for 3B model training (large batch = stable gradients)

    # -- Token target based on throughput --
    # Conservative: 750K tok/s × 36 min × 60 = 1.62B tokens
    # Optimistic: 1.2M tok/s × 36 min × 60 = 2.59B tokens
    config.training.total_tokens_target = 2_000_000_000  # 2B tokens (realistic middle)

    # -- Optimizer: Higher LR for shorter training --
    config.training.peak_lr = 6e-3     # Aggressive: short schedule needs fast learning
    config.training.min_lr = 6e-5
    config.training.optimizer = "muon"

    # -- WSD Schedule tuned for short run --
    config.training.warmup_ratio = 0.03    # 3% warmup (~65 sec)
    config.training.stable_ratio = 0.82    # 82% stable phase
    config.training.decay_ratio = 0.15     # 15% cooldown (important for quality)

    # -- Instruct-tune (more aggressive for short window) --
    config.training.instruct_lr = 5e-4
    config.training.instruct_warmup_ratio = 0.08

    # -- GRPO alignment (aggressive) --
    config.training.grpo_lr = 3e-5
    config.training.grpo_num_completions = 16  # More completions for better signal
    config.training.grpo_steps = 400           # Pack more steps in

    # -- Multi-token prediction (keep at 4) --
    config.training.num_predict_tokens = 4

    # -- Gradient checkpointing OFF (we have 1.5 TB VRAM, use it) --
    config.training.gradient_checkpointing = False

    # -- GaLore OFF (enough VRAM for full optimizer states) --
    config.training.use_galore = False
    config.training.galore_rank = 0

    # -- Logging (more frequent for short run) --
    config.training.log_interval = 25
    config.training.save_interval_hours = 0.15  # Save every ~9 minutes

    # -- Compressed curriculum (4 stages in ~36 minutes instead of 10 hours) --
    config.training.curriculum_stages = [
        CurriculumStageConfig(
            name="infancy",
            hours_start=0.0, hours_end=pretrain_minutes * 0.12 / 60,  # ~4 min
            max_seq_len=1024,       # Start longer than before (we have VRAM)
            max_reading_level=6.0,
            text_ratio=0.85, code_ratio=0.10, math_ratio=0.05,
            batch_size_tokens=2_000_000,   # 2M tokens/batch (8 GPUs can handle it)
            lr_multiplier=0.6,
            hebbian_weight_multiplier=1.0,
        ),
        CurriculumStageConfig(
            name="childhood",
            hours_start=pretrain_minutes * 0.12 / 60,
            hours_end=pretrain_minutes * 0.40 / 60,  # ~10 min
            max_seq_len=2048,
            max_reading_level=10.0,
            text_ratio=0.70, code_ratio=0.18, math_ratio=0.12,
            batch_size_tokens=4_000_000,   # 4M tokens/batch
            lr_multiplier=1.0,
            hebbian_weight_multiplier=0.5,
        ),
        CurriculumStageConfig(
            name="adolescence",
            hours_start=pretrain_minutes * 0.40 / 60,
            hours_end=pretrain_minutes * 0.75 / 60,  # ~13 min
            max_seq_len=4096,
            max_reading_level=None,
            text_ratio=0.60, code_ratio=0.22, math_ratio=0.18,
            batch_size_tokens=4_000_000,
            lr_multiplier=1.0,
            hebbian_weight_multiplier=0.2,
        ),
        CurriculumStageConfig(
            name="mastery",
            hours_start=pretrain_minutes * 0.75 / 60,
            hours_end=pretrain_minutes / 60,  # final ~9 min
            max_seq_len=4096,
            max_reading_level=None,
            text_ratio=0.45, code_ratio=0.28, math_ratio=0.27,
            batch_size_tokens=4_000_000,
            lr_multiplier=0.7,
            hebbian_weight_multiplier=0.1,
        ),
    ]

    # -- Brain optimizations (all ON, tuned for speed) --
    config.brain.enable_brain_optimizations = True
    config.brain.enable_predictive_coding = True
    config.brain.enable_dynamic_sparsity = True
    config.brain.enable_lateral_inhibition = True
    config.brain.enable_spike_activations = True
    config.brain.enable_hebbian_loss = True
    config.brain.enable_curriculum = True
    config.brain.enable_sleep_consolidation = True
    config.brain.sleep_replay_samples = 50000   # More replay (we have the VRAM)
    config.brain.sleep_replay_lr = 2e-5
    config.brain.sleep_checkpoint_average_n = 4  # Average more checkpoints

    # -- Dynamic sparsity tuned for 8 GPUs --
    config.brain.min_active_experts = 6     # More experts active (we have compute)
    config.brain.max_active_experts = 16    # Let it use up to 16
    config.brain.difficulty_aux_loss_weight = 0.008

    return config


def get_h100_config(budget_dollars: float = 17.0, rate_per_hour: float = 51.65) -> FullConfig:
    """
    Optimized config for 16x H100 SXM cluster (2 pods × 8 GPUs).

    Hardware: 16x NVIDIA H100 SXM (RunPod Instant Cluster, 2 pods)
      - 1280 GB HBM3 total VRAM (80 GB per GPU)
      - ~15,824 TFLOPS BF16 aggregate
      - 3200 Gbps inter-pod, NVLink intra-pod
      - 3008 GB system RAM, 320 vCPU

    Cost: $51.65/hr total ($3.23/GPU)
    Budget: ~$13 for full training run (~15 min)

    Estimated throughput: ~2.5-3.5M tokens/sec for our 3B BitNet MoE model
    Target: ~2B tokens in ~11 min of pretraining
    """
    runtime_hours = budget_dollars / rate_per_hour
    runtime_minutes = runtime_hours * 60  # ~20 min at $17

    # Time allocation
    pretrain_minutes = runtime_minutes * 0.72
    instruct_minutes = runtime_minutes * 0.12
    grpo_minutes = runtime_minutes * 0.10
    sleep1_minutes = runtime_minutes * 0.04
    sleep2_minutes = runtime_minutes * 0.02

    config = FullConfig()

    # -- Training schedule --
    config.training.pretrain_hours = pretrain_minutes / 60.0
    config.training.instruct_hours = instruct_minutes / 60.0
    config.training.grpo_hours = grpo_minutes / 60.0
    config.training.sleep1_minutes = sleep1_minutes
    config.training.sleep2_minutes = sleep2_minutes
    config.training.num_gpus = 16
    config.training.hardware_profile = "h100"

    # -- Token target --
    # ~3M tok/s × 11 min × 60 = ~2B tokens
    config.training.total_tokens_target = 2_000_000_000  # 2B tokens

    # -- Optimizer: High LR for fast convergence --
    config.training.peak_lr = 6e-3
    config.training.min_lr = 6e-5
    config.training.optimizer = "muon"

    # -- WSD Schedule --
    config.training.warmup_ratio = 0.03
    config.training.stable_ratio = 0.82
    config.training.decay_ratio = 0.15

    # -- Instruct-tune --
    config.training.instruct_lr = 5e-4
    config.training.instruct_warmup_ratio = 0.08

    # -- GRPO alignment --
    config.training.grpo_lr = 3e-5
    config.training.grpo_num_completions = 16
    config.training.grpo_steps = 400

    # -- Multi-token prediction --
    config.training.num_predict_tokens = 4

    # -- Gradient checkpointing OFF (1280 GB VRAM is plenty) --
    config.training.gradient_checkpointing = False

    # -- GaLore OFF (full optimizer states fit in 80 GB/GPU) --
    config.training.use_galore = False
    config.training.galore_rank = 0

    # -- Logging --
    config.training.log_interval = 20
    config.training.save_interval_hours = 0.08  # Save every ~5 min

    # -- Compressed curriculum (4 stages in ~11 min) --
    config.training.curriculum_stages = [
        CurriculumStageConfig(
            name="infancy",
            hours_start=0.0, hours_end=pretrain_minutes * 0.12 / 60,
            max_seq_len=1024,
            max_reading_level=6.0,
            text_ratio=0.85, code_ratio=0.10, math_ratio=0.05,
            batch_size_tokens=4_000_000,   # 16 GPUs can push big batches
            lr_multiplier=0.6,
            hebbian_weight_multiplier=1.0,
        ),
        CurriculumStageConfig(
            name="childhood",
            hours_start=pretrain_minutes * 0.12 / 60,
            hours_end=pretrain_minutes * 0.40 / 60,
            max_seq_len=2048,
            max_reading_level=10.0,
            text_ratio=0.70, code_ratio=0.18, math_ratio=0.12,
            batch_size_tokens=8_000_000,
            lr_multiplier=1.0,
            hebbian_weight_multiplier=0.5,
        ),
        CurriculumStageConfig(
            name="adolescence",
            hours_start=pretrain_minutes * 0.40 / 60,
            hours_end=pretrain_minutes * 0.75 / 60,
            max_seq_len=4096,
            max_reading_level=None,
            text_ratio=0.60, code_ratio=0.22, math_ratio=0.18,
            batch_size_tokens=8_000_000,
            lr_multiplier=1.0,
            hebbian_weight_multiplier=0.2,
        ),
        CurriculumStageConfig(
            name="mastery",
            hours_start=pretrain_minutes * 0.75 / 60,
            hours_end=pretrain_minutes / 60,
            max_seq_len=4096,
            max_reading_level=None,
            text_ratio=0.45, code_ratio=0.28, math_ratio=0.27,
            batch_size_tokens=8_000_000,
            lr_multiplier=0.7,
            hebbian_weight_multiplier=0.1,
        ),
    ]

    # -- Brain optimizations (all ON) --
    config.brain.enable_brain_optimizations = True
    config.brain.enable_predictive_coding = True
    config.brain.enable_dynamic_sparsity = True
    config.brain.enable_lateral_inhibition = True
    config.brain.enable_spike_activations = True
    config.brain.enable_hebbian_loss = True
    config.brain.enable_curriculum = True
    config.brain.enable_sleep_consolidation = True
    config.brain.sleep_replay_samples = 50000
    config.brain.sleep_replay_lr = 2e-5
    config.brain.sleep_checkpoint_average_n = 4

    # -- Dynamic sparsity tuned for 16 GPUs --
    config.brain.min_active_experts = 6
    config.brain.max_active_experts = 16
    config.brain.difficulty_aux_loss_weight = 0.008

    return config


if __name__ == "__main__":
    config = get_default_config()
    print(f"=== DEFAULT CONFIG (RTX 5070 / 12h) ===")
    print(f"Model: {config.model.hidden_size}d, {config.model.num_layers} layers")
    print(f"MoE: {config.model.num_experts} experts, {config.model.base_active_experts} active")
    print(f"BitNet: {config.model.use_bitnet}")
    print(f"Brain optimizations: {config.brain.enable_brain_optimizations}")
    print(f"Curriculum stages: {[s.name for s in config.training.curriculum_stages]}")
    print(f"Total tokens target: {config.training.total_tokens_target / 1e9:.0f}B")

    print(f"\n=== MI300X CONFIG ($10 budget) ===")
    mi_config = get_mi300x_config(budget_dollars=10.0)
    print(f"Runtime: {mi_config.training.pretrain_hours * 60:.0f} min pretrain")
    print(f"Token target: {mi_config.training.total_tokens_target / 1e9:.1f}B")
    print(f"GPUs: {mi_config.training.num_gpus}")
    print(f"Peak LR: {mi_config.training.peak_lr}")
    print(f"Gradient checkpointing: {mi_config.training.gradient_checkpointing}")
    print(f"GaLore: {mi_config.training.use_galore}")
    print(f"Curriculum stages: {[s.name for s in mi_config.training.curriculum_stages]}")
    for stage in mi_config.training.curriculum_stages:
        duration = (stage.hours_end - stage.hours_start) * 60
        print(f"  {stage.name}: {duration:.1f} min, seq_len={stage.max_seq_len}, "
              f"batch_tokens={stage.batch_size_tokens:,}")
    print("Config OK.")
