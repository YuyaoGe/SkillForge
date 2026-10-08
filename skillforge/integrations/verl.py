"""Trainer hooks shared by the bundled ``verl`` integration and other runners.

This module deliberately does not import Ray, torch, vLLM, or a provider SDK at
module import time. That keeps the lifecycle testable on CPU-only machines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional

from ..bank import SkillBank
from ..evolution import ForgeEngine


@dataclass
class EpisodeContext:
    task: str
    prompt: str
    references: List[dict[str, Any]]


class SkillForgeHooks:
    """Retrieve skills before rollout and evolve them after validation."""

    def __init__(
        self,
        bank: SkillBank,
        engine: ForgeEngine,
        *,
        mutate_fn: Optional[Callable[[dict[str, Any], SkillBank], Optional[dict[str, Any]]]] = None,
        top_k: int = 6,
    ) -> None:
        self.bank = bank
        self.engine = engine
        self.mutate_fn = mutate_fn
        self.top_k = top_k

    def before_episode(self, task: str) -> EpisodeContext:
        retrieved = self.bank.retrieve(task, top_k=self.top_k)
        return EpisodeContext(
            task=task,
            prompt=self.bank.format_prompt(retrieved),
            references=self.bank.references(retrieved),
        )

    def after_episode(
        self,
        context: EpisodeContext,
        *,
        success: bool,
        trajectory: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> None:
        serialized = json.dumps(list(trajectory or []), ensure_ascii=False)
        self.engine.record_episode(
            self.bank,
            context.references,
            success,
            task=context.task,
            trajectory=serialized,
        )

    def after_validation(self, global_step: int) -> dict[str, Any]:
        return self.engine.maybe_evolve(
            self.bank,
            global_step=global_step,
            mutate_fn=self.mutate_fn,
        )
