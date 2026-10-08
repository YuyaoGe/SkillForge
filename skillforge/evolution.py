"""Fitness-driven skill lifecycle management."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .bank import SkillBank


@dataclass
class EvolutionConfig:
    """Thresholds and budgets for the four-state skill lifecycle."""

    fitness_warmup: int = 5
    default_fitness: float = 0.5
    retire_fitness_threshold: float = 0.4
    demote_fitness_threshold: float = 0.6
    stabilize_fitness_threshold: float = 0.7
    retire_min_usage: int = 20
    promote_min_usage: int = 10
    stabilize_min_usage: int = 30
    mutate_fitness_low: float = 0.3
    mutate_fitness_high: float = 0.6
    mutate_min_usage: int = 10
    max_mutations_per_cycle: int = 2
    max_retirements_per_cycle: int = 3
    max_total_skills: int = 80
    evolve_every_n_validations: int = 1
    original_skill_protection_usage: int = 50
    retire_warmup_steps: int = 0


class ForgeEngine:
    """Apply promotion, demotion, retirement, stabilization, and mutation."""

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        *,
        tracker_dir: Optional[str | Path] = None,
    ) -> None:
        self.config = config or EvolutionConfig()
        self.validation_count = 0
        self.tracker_dir = Path(tracker_dir) if tracker_dir else None
        self.history: List[Dict[str, Any]] = []

    def fitness(self, skill: Dict[str, Any]) -> float:
        usage = int(skill.get("usage_count", 0))
        if usage < self.config.fitness_warmup:
            return self.config.default_fitness
        return float(skill.get("success_count", 0)) / max(usage, 1)

    def record_episode(
        self,
        bank: SkillBank,
        skills: List[Dict[str, Any] | str],
        success: bool,
        *,
        task: Optional[str] = None,
        trajectory: Optional[str] = None,
    ) -> None:
        bank.record_episode(skills, success, task=task, trajectory=trajectory)

    def pre_retire(self, bank: SkillBank) -> List[Dict[str, Any]]:
        """Filter the seed bank before SFT or RL using proto-fitness."""
        retired: List[Dict[str, Any]] = []
        for skill in list(bank.all_skills()):
            if (
                skill.get("lifecycle") == "active"
                and self.fitness(skill) < self.config.retire_fitness_threshold
                and skill.get("usage_count", 0) >= self.config.retire_min_usage
            ):
                skill_id = skill.get("skill_id", "unknown")
                score = self.fitness(skill)
                bank.remove(skill_id)
                retired.append({
                    "skill_id": skill_id,
                    "fitness": score,
                    "reason": "pre_retirement",
                })
        return retired

    def maybe_evolve(
        self,
        bank: SkillBank,
        *,
        global_step: int,
        mutate_fn: Optional[
            Callable[[Dict[str, Any], SkillBank], Optional[Dict[str, Any]]]
        ] = None,
    ) -> Dict[str, Any]:
        """Run one forging cycle when the validation cadence is reached."""
        self.validation_count += 1
        summary: Dict[str, Any] = {
            "validation_count": self.validation_count,
            "global_step": global_step,
            "promoted": [],
            "demoted": [],
            "retired": [],
            "stabilized": [],
            "mutated": [],
            "skipped": (
                self.validation_count % self.config.evolve_every_n_validations != 0
            ),
        }
        if summary["skipped"]:
            return summary

        for skill in bank.all_skills():
            SkillBank.ensure_fields(skill)

        # The order is part of the method: promotion, demotion, retirement,
        # stabilization, mutation, then capacity enforcement.
        for skill in bank.all_skills():
            if (
                skill.get("lifecycle") == "trial"
                and skill.get("usage_count", 0) >= self.config.promote_min_usage
            ):
                skill["lifecycle"] = "active"
                summary["promoted"].append(skill.get("skill_id"))

        for skill in bank.all_skills():
            if (
                skill.get("lifecycle") == "stable"
                and self.fitness(skill) < self.config.demote_fitness_threshold
            ):
                skill["lifecycle"] = "active"
                summary["demoted"].append(skill.get("skill_id"))

        if global_step >= self.config.retire_warmup_steps:
            candidates = []
            for skill in bank.all_skills():
                generation = int(skill.get("generation", 0))
                min_usage = self.config.retire_min_usage
                if generation == 0:
                    min_usage = max(min_usage, self.config.original_skill_protection_usage)
                if (
                    skill.get("lifecycle") == "active"
                    and self.fitness(skill) < self.config.retire_fitness_threshold
                    and skill.get("usage_count", 0) >= min_usage
                ):
                    candidates.append((self.fitness(skill), skill))
            candidates.sort(key=lambda item: item[0])
            for score, skill in candidates[: self.config.max_retirements_per_cycle]:
                skill_id = skill.get("skill_id", "unknown")
                if bank.remove(skill_id):
                    summary["retired"].append({
                        "skill_id": skill_id,
                        "fitness": round(score, 4),
                    })

        for skill in bank.all_skills():
            if (
                skill.get("lifecycle") == "active"
                and self.fitness(skill) >= self.config.stabilize_fitness_threshold
                and skill.get("usage_count", 0) >= self.config.stabilize_min_usage
            ):
                skill["lifecycle"] = "stable"
                summary["stabilized"].append(skill.get("skill_id"))

        if mutate_fn is not None:
            candidates = [
                skill for skill in bank.all_skills()
                if (
                    skill.get("lifecycle") == "active"
                    and self.config.mutate_fitness_low
                    <= self.fitness(skill)
                    <= self.config.mutate_fitness_high
                    and skill.get("usage_count", 0) >= self.config.mutate_min_usage
                )
            ]
            available = max(0, self.config.max_total_skills - len(bank))
            n_mutations = min(
                self.config.max_mutations_per_cycle,
                len(candidates),
                available,
            )
            for _ in range(n_mutations):
                if not candidates:
                    break
                weights = [max(1e-6, 1.0 - self.fitness(skill)) for skill in candidates]
                parent = random.choices(candidates, weights=weights, k=1)[0]
                candidates.remove(parent)
                child = mutate_fn(parent, bank)
                if child is None:
                    continue
                child = dict(child)
                child["parent_id"] = parent.get("skill_id")
                child["generation"] = int(parent.get("generation", 0)) + 1
                child["lifecycle"] = "trial"
                child["usage_count"] = 0
                child["success_count"] = 0
                child["children"] = []
                if bank.add(child):
                    parent.setdefault("children", []).append(child["skill_id"])
                    parent["last_mutated_step"] = global_step
                    summary["mutated"].append({
                        "parent_id": parent.get("skill_id"),
                        "child_id": child["skill_id"],
                        "parent_fitness": round(self.fitness(parent), 4),
                    })

        self._enforce_cap(bank, summary)
        self.history.append(summary)
        self._save_tracker()
        return summary

    def _enforce_cap(self, bank: SkillBank, summary: Dict[str, Any]) -> None:
        while len(bank) > self.config.max_total_skills:
            candidates = [
                skill for skill in bank.all_skills()
                if skill.get("lifecycle") in {"active", "trial"}
            ]
            if not candidates:
                break
            victim = min(candidates, key=self.fitness)
            skill_id = victim.get("skill_id", "unknown")
            score = self.fitness(victim)
            if bank.remove(skill_id):
                summary["retired"].append({
                    "skill_id": skill_id,
                    "fitness": round(score, 4),
                    "reason": "capacity",
                })

    def _save_tracker(self) -> None:
        if self.tracker_dir is None:
            return
        self.tracker_dir.mkdir(parents=True, exist_ok=True)
        with (self.tracker_dir / "evolution.json").open("w") as handle:
            json.dump(self.history, handle, indent=2)
