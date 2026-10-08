"""Provider-neutral environment adapter contracts."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

from ..training.loop import EpisodeResult


class EnvironmentAdapter(Protocol):
    """Minimal contract required by the SkillForge training loop."""

    def run_episode(
        self,
        task: str,
        prompt: str,
        policy: Callable[[str], str],
    ) -> EpisodeResult:
        ...


def normalize_result(result: Mapping[str, Any]) -> EpisodeResult:
    """Convert common environment result mappings into ``EpisodeResult``."""
    return EpisodeResult(
        success=bool(result.get("success", result.get("done", False))),
        reward=float(result.get("reward", 1.0 if result.get("success") else 0.0)),
        trajectory=list(result.get("trajectory", [])),
        info=dict(result.get("info", {})),
    )
