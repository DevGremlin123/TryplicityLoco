"""
Neuro-Curriculum: Developmental stages for training data.
Mirrors how the brain bootstraps understanding through critical periods.
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CurriculumStage:
    """A single developmental stage."""
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


# Default stages
DEFAULT_STAGES = [
    CurriculumStage(
        name="infancy",
        hours_start=0.0, hours_end=2.0,
        max_seq_len=512,
        max_reading_level=6.0,
        text_ratio=0.9, code_ratio=0.1, math_ratio=0.0,
        batch_size_tokens=256_000,
        lr_multiplier=0.5,
        hebbian_weight_multiplier=1.0,
    ),
    CurriculumStage(
        name="childhood",
        hours_start=2.0, hours_end=5.0,
        max_seq_len=2048,
        max_reading_level=10.0,
        text_ratio=0.75, code_ratio=0.15, math_ratio=0.10,
        batch_size_tokens=1_000_000,
        lr_multiplier=1.0,
        hebbian_weight_multiplier=0.5,
    ),
    CurriculumStage(
        name="adolescence",
        hours_start=5.0, hours_end=8.0,
        max_seq_len=4096,
        max_reading_level=None,
        text_ratio=0.65, code_ratio=0.20, math_ratio=0.15,
        batch_size_tokens=2_000_000,
        lr_multiplier=1.0,
        hebbian_weight_multiplier=0.2,
    ),
    CurriculumStage(
        name="mastery",
        hours_start=8.0, hours_end=10.0,
        max_seq_len=4096,
        max_reading_level=None,
        text_ratio=0.50, code_ratio=0.25, math_ratio=0.25,
        batch_size_tokens=2_000_000,
        lr_multiplier=0.7,
        hebbian_weight_multiplier=0.1,
    ),
]


class NeuroCurriculum:
    """
    Manages curriculum progression during training.
    Determines which stage we're in based on elapsed time
    and adjusts data pipeline parameters accordingly.
    """

    def __init__(self, stages=None, total_hours: float = 10.0, enabled: bool = True):
        self.stages = stages or DEFAULT_STAGES
        self.total_hours = total_hours
        self.enabled = enabled
        self.start_time = None
        self._current_stage_idx = 0

    def start(self):
        """Mark the beginning of training."""
        self.start_time = time.time()
        self._current_stage_idx = 0

    def elapsed_hours(self) -> float:
        """Hours since training started."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 3600.0

    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage based on elapsed time."""
        if not self.enabled or not self.stages:
            return self.stages[-1] if self.stages else None

        elapsed = self.elapsed_hours()

        # Scale stages to actual training time
        time_scale = self.total_hours / self.stages[-1].hours_end

        for i, stage in enumerate(self.stages):
            scaled_end = stage.hours_end * time_scale
            if elapsed < scaled_end:
                self._current_stage_idx = i
                return stage

        # Past all stages, stay on last one
        self._current_stage_idx = len(self.stages) - 1
        return self.stages[-1]

    def get_stage_name(self) -> str:
        return self.get_current_stage().name

    def get_max_seq_len(self) -> int:
        return self.get_current_stage().max_seq_len

    def get_batch_size_tokens(self) -> int:
        return self.get_current_stage().batch_size_tokens

    def get_lr_multiplier(self) -> float:
        return self.get_current_stage().lr_multiplier

    def get_data_mix(self) -> dict:
        stage = self.get_current_stage()
        return {
            "text": stage.text_ratio,
            "code": stage.code_ratio,
            "math": stage.math_ratio,
        }

    def get_hebbian_multiplier(self) -> float:
        return self.get_current_stage().hebbian_weight_multiplier

    def should_transition(self) -> bool:
        """Check if we just moved to a new stage."""
        old_idx = self._current_stage_idx
        self.get_current_stage()
        return self._current_stage_idx != old_idx

    def progress_string(self) -> str:
        """Human-readable progress string."""
        stage = self.get_current_stage()
        elapsed = self.elapsed_hours()
        return (f"Stage: {stage.name} | "
                f"Time: {elapsed:.2f}h / {self.total_hours:.1f}h | "
                f"Seq len: {stage.max_seq_len} | "
                f"Mix: text={stage.text_ratio:.0%} code={stage.code_ratio:.0%} math={stage.math_ratio:.0%}")


if __name__ == "__main__":
    print("Neuro-curriculum smoke test...")
    curriculum = NeuroCurriculum(total_hours=0.17)  # ~10 minutes for testing
    curriculum.start()

    stage = curriculum.get_current_stage()
    print(f"  Current: {curriculum.progress_string()}")
    print(f"  LR multiplier: {curriculum.get_lr_multiplier()}")
    print(f"  Hebbian multiplier: {curriculum.get_hebbian_multiplier()}")
    print(f"  Data mix: {curriculum.get_data_mix()}")
    print("  Curriculum OK.")
