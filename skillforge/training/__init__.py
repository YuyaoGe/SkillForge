"""Framework-neutral training hooks and a CPU reference loop."""

from .loop import EpisodeResult, SkillForgeLoop, TrainingSummary

__all__ = ["EpisodeResult", "SkillForgeLoop", "TrainingSummary"]
