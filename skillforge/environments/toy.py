"""Deterministic CPU environment used by tests and documentation."""

from __future__ import annotations

from typing import Callable

from ..training.loop import EpisodeResult


class ToyEnvironment:
    """A tiny task environment with no model, network, or GPU dependency."""

    def run_episode(
        self,
        task: str,
        prompt: str,
        policy: Callable[[str], str],
    ) -> EpisodeResult:
        action = policy(f"Task: {task}\n\nSkills:\n{prompt}")
        success = task.lower() in action.lower()
        return EpisodeResult(
            success=success,
            reward=1.0 if success else 0.0,
            trajectory=[{"task": task, "action": action}],
        )
