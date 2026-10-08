"""A small end-to-end loop for testing the skill lifecycle without a GPU.

The loop mirrors the hook order used by a distributed trainer:
retrieve skills, run an episode, assign its outcome, then evolve at validation
boundaries. A real trainer can use the same hooks through
``skillforge.integrations.verl``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Mapping, Optional, Protocol

from ..bank import SkillBank
from ..evolution import ForgeEngine


@dataclass
class EpisodeResult:
    """Outcome returned by an environment adapter after one episode."""

    success: bool
    reward: float
    trajectory: List[Mapping[str, Any]] = field(default_factory=list)
    info: Mapping[str, Any] = field(default_factory=dict)


class EpisodeEnvironment(Protocol):
    def run_episode(
        self,
        task: str,
        prompt: str,
        policy: Callable[[str], str],
    ) -> EpisodeResult:
        """Run one task using the supplied policy."""


@dataclass
class TrainingSummary:
    """Serializable record of one CPU or distributed training invocation."""

    episodes: int = 0
    successes: int = 0
    validation_summaries: List[dict[str, Any]] = field(default_factory=list)


class SkillForgeLoop:
    """Connect an environment adapter, policy, skill bank, and evolution."""

    def __init__(
        self,
        bank: SkillBank,
        engine: ForgeEngine,
        environment: EpisodeEnvironment,
        policy: Callable[[str], str],
        *,
        mutate_fn: Optional[Callable[[dict[str, Any], SkillBank], Optional[dict[str, Any]]]] = None,
        top_k: int = 6,
    ) -> None:
        self.bank = bank
        self.engine = engine
        self.environment = environment
        self.policy = policy
        self.mutate_fn = mutate_fn
        self.top_k = top_k

    def run(self, tasks: Iterable[str], *, validations: int = 1) -> TrainingSummary:
        """Run the complete retrieve → rollout → reward → evolve cycle."""
        task_list = list(tasks)
        summary = TrainingSummary()
        for validation_index in range(validations):
            for task in task_list:
                retrieved = self.bank.retrieve(task, top_k=self.top_k)
                references = self.bank.references(retrieved)
                prompt = self.bank.format_prompt(retrieved)
                result = self.environment.run_episode(task, prompt, self.policy)
                trajectory = json.dumps(result.trajectory, ensure_ascii=False)
                self.engine.record_episode(
                    self.bank,
                    references,
                    result.success,
                    task=task,
                    trajectory=trajectory,
                )
                summary.episodes += 1
                summary.successes += int(result.success)

            summary.validation_summaries.append(
                self.engine.maybe_evolve(
                    self.bank,
                    global_step=validation_index + 1,
                    mutate_fn=self.mutate_fn,
                )
            )
        return summary
