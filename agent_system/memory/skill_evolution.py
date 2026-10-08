"""
Skill Evolution Tree — fitness-based lifecycle management for skills.

Each skill goes through a lifecycle: trial → active → stable/retired.
Low-fitness skills are retired (removed), medium-fitness skills are
mutated (LLM generates improved variants), and high-fitness skills
are stabilized.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any


@dataclass
class EvolutionConfig:
    """Configuration for the skill evolution system."""
    enable: bool = False
    retire_fitness_threshold: float = 0.2
    retire_min_usage: int = 20
    stabilize_fitness_threshold: float = 0.7
    stabilize_min_usage: int = 30
    mutate_fitness_low: float = 0.3
    mutate_fitness_high: float = 0.6
    mutate_min_usage: int = 15
    max_mutations_per_step: int = 2
    max_retirements_per_step: int = 3
    max_total_skills: int = 80
    evolve_every_n_validations: int = 2
    original_skill_protection_usage: int = 50
    retire_warmup_steps: int = 50  # disable retirement before this step


class SkillEvolutionManager:
    """
    Manages the evolutionary lifecycle of skills during RL training.

    Tracks skill usage/success, computes fitness, and periodically
    retires, stabilizes, or mutates skills.
    """

    def __init__(self, config: EvolutionConfig, tracker_save_dir: Optional[str] = None):
        self.config = config
        self.validation_count = 0
        self.tracker_save_dir = tracker_save_dir
        self.evolution_log: List[Dict] = []

    @staticmethod
    def compute_fitness(skill: Dict) -> float:
        """Compute fitness as success_count / usage_count.

        Returns 0.5 for skills with fewer than 5 uses (insufficient data).
        """
        usage = skill.get('usage_count', 0)
        if usage < 5:
            return 0.5
        return skill.get('success_count', 0) / usage

    @staticmethod
    def ensure_evolution_fields(skill: Dict) -> Dict:
        """Add evolution tracking fields to a skill dict if missing.

        This provides backward compatibility — existing skills without
        evolution fields get sensible defaults.  Original skills
        (no parent) start as 'stable' generation 0.
        """
        defaults = {
            'usage_count': 0,
            'success_count': 0,
            'lifecycle': 'stable',  # original skills start stable
            'parent_id': None,
            'generation': 0,
            'children': [],
        }
        for key, val in defaults.items():
            if key not in skill:
                skill[key] = val
        return skill

    def record_episode_outcome(
        self,
        skills_used: List[Dict],
        success: bool,
    ) -> None:
        """Record the outcome of an episode for each skill used.

        Updates usage_count and success_count in-place on the skill dicts.
        """
        for skill in skills_used:
            self.ensure_evolution_fields(skill)
            skill['usage_count'] += 1
            if success:
                skill['success_count'] += 1

    def _should_evolve(self) -> bool:
        """Check whether it's time to run evolution."""
        return (self.validation_count > 0 and
                self.validation_count % self.config.evolve_every_n_validations == 0)

    def maybe_evolve(
        self,
        skills_memory,  # SkillsOnlyMemory instance
        global_step: int,
        mutate_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run one evolution cycle if the timing is right.

        Args:
            skills_memory: The SkillsOnlyMemory that holds the skill bank.
            global_step:   Current training step (for logging).
            mutate_fn:     Callable(skill, current_skills) -> Optional[new_skill_dict].

        Returns:
            Summary dict with counts of retired/stabilized/mutated skills.
        """
        self.validation_count += 1

        summary = {
            'validation_count': self.validation_count,
            'global_step': global_step,
            'retired': [],
            'stabilized': [],
            'mutated': [],
            'skipped': not self._should_evolve(),
        }

        if not self._should_evolve():
            return summary

        all_skills = self.get_all_skills_flat(skills_memory)

        # Ensure every skill has evolution fields
        for skill in all_skills:
            self.ensure_evolution_fields(skill)

        # --- Retirement ---
        retired_count = 0
        if global_step < self.config.retire_warmup_steps:
            print(f"[SkillEvolution] Step {global_step}: retirement skipped (warmup, step < {self.config.retire_warmup_steps})")
        for skill in list(all_skills):  # copy list since we may remove
            if global_step < self.config.retire_warmup_steps:
                break
            if retired_count >= self.config.max_retirements_per_step:
                break

            fitness = self.compute_fitness(skill)
            usage = skill.get('usage_count', 0)
            generation = skill.get('generation', 0)

            # Protection for original skills
            min_usage = self.config.retire_min_usage
            if generation == 0:
                min_usage = max(min_usage, self.config.original_skill_protection_usage)

            if (fitness < self.config.retire_fitness_threshold and
                    usage >= min_usage):
                skill_id = skill.get('skill_id', 'unknown')
                skills_memory.remove_skill(skill_id)
                summary['retired'].append({
                    'skill_id': skill_id,
                    'fitness': round(fitness, 3),
                    'usage': usage,
                })
                retired_count += 1

        # --- Stabilization ---
        for skill in all_skills:
            fitness = self.compute_fitness(skill)
            usage = skill.get('usage_count', 0)
            if (fitness >= self.config.stabilize_fitness_threshold and
                    usage >= self.config.stabilize_min_usage and
                    skill.get('lifecycle') != 'stable'):
                skill['lifecycle'] = 'stable'
                summary['stabilized'].append({
                    'skill_id': skill.get('skill_id', 'unknown'),
                    'fitness': round(fitness, 3),
                    'usage': usage,
                })

        # --- Lifecycle promotion: trial → active ---
        for skill in all_skills:
            if skill.get('lifecycle') == 'trial' and skill.get('usage_count', 0) >= 10:
                skill['lifecycle'] = 'active'

        # --- Mutation (randomized selection with cooldown) ---
        if mutate_fn is not None:
            import random
            total_skills = self._count_total_skills(skills_memory)

            # Collect all candidates that meet mutation criteria
            candidates = []
            for skill in all_skills:
                fitness = self.compute_fitness(skill)
                usage = skill.get('usage_count', 0)
                if (self.config.mutate_fitness_low <= fitness <= self.config.mutate_fitness_high and
                        usage >= self.config.mutate_min_usage and
                        skill.get('lifecycle') in ('active', 'stable')):
                    # Weight: closer to fitness midpoint = higher weight
                    mid = (self.config.mutate_fitness_low + self.config.mutate_fitness_high) / 2
                    weight = 1.0 - abs(fitness - mid) / (self.config.mutate_fitness_high - self.config.mutate_fitness_low)
                    weight = max(weight, 0.1)
                    # Cooldown: reduce weight if recently mutated
                    last_mut = skill.get('last_mutated_step', -999)
                    if global_step - last_mut < 20:
                        weight *= 0.5
                    candidates.append((skill, fitness, weight))

            # Weighted random sampling without replacement
            n_to_mutate = min(self.config.max_mutations_per_step, len(candidates))
            if total_skills >= self.config.max_total_skills:
                n_to_mutate = 0
            selected = []
            remaining = list(candidates)
            for _ in range(n_to_mutate):
                if not remaining or total_skills >= self.config.max_total_skills:
                    break
                weights = [w for _, _, w in remaining]
                chosen_idx = random.choices(range(len(remaining)), weights=weights, k=1)[0]
                selected.append(remaining.pop(chosen_idx))

            for skill, fitness, _ in selected:
                try:
                    new_skill = mutate_fn(skill, skills_memory.skills,
                                          skill.get('recent_failures'))
                except Exception as e:
                    print(f"[SkillEvolution] Mutation failed for {skill.get('skill_id')}: {e}")
                    continue

                if new_skill is not None:
                    # Set evolution fields on the child
                    new_skill['lifecycle'] = 'trial'
                    new_skill['parent_id'] = skill.get('skill_id')
                    new_skill['generation'] = skill.get('generation', 0) + 1
                    new_skill['usage_count'] = 0
                    new_skill['success_count'] = 0
                    new_skill['children'] = []

                    # Record parent-child relationship and cooldown
                    skill.setdefault('children', []).append(new_skill.get('skill_id'))
                    skill['last_mutated_step'] = global_step

                    # Add to skill bank
                    added = skills_memory.add_skills([new_skill], category='general')
                    if added > 0:
                        total_skills += 1
                        summary['mutated'].append({
                            'parent_id': skill.get('skill_id'),
                            'child_id': new_skill.get('skill_id'),
                            'parent_fitness': round(fitness, 3),
                        })

        # Save tracker
        self.evolution_log.append(summary)
        self._save_tracker(global_step)

        # One-line summary: actions + fitness distribution
        fb = [0, 0, 0, 0]  # <0.2 | 0.2-0.6 | 0.6-0.7 | >=0.7
        for skill in all_skills:
            f = self.compute_fitness(skill)
            if f < 0.2: fb[0] += 1
            elif f <= 0.6: fb[1] += 1
            elif f < 0.7: fb[2] += 1
            else: fb[3] += 1
        print(f"[SkillEvolution] Step {global_step}: "
              f"retired={len(summary['retired'])}, stabilized={len(summary['stabilized'])}, "
              f"mutated={len(summary['mutated'])}, total={self._count_total_skills(skills_memory)}, "
              f"fitness[<.2|.2-.6|.6-.7|>=.7]={fb}")

        return summary

    @staticmethod
    def get_all_skills_flat(skills_memory) -> List[Dict]:
        """Return a flat list of all skill dicts (references, not copies)."""
        all_skills = []
        all_skills.extend(skills_memory.skills.get('general_skills', []))
        for task_skills in skills_memory.skills.get('task_specific_skills', {}).values():
            all_skills.extend(task_skills)
        return all_skills

    @staticmethod
    def _count_total_skills(skills_memory) -> int:
        """Count total skills excluding common_mistakes."""
        count = len(skills_memory.skills.get('general_skills', []))
        for task_skills in skills_memory.skills.get('task_specific_skills', {}).values():
            count += len(task_skills)
        return count

    def _save_tracker(self, global_step: int) -> None:
        """Persist evolution log to JSON file."""
        if self.tracker_save_dir is None:
            return
        os.makedirs(self.tracker_save_dir, exist_ok=True)
        path = os.path.join(self.tracker_save_dir, 'evolution_tracker.json')
        with open(path, 'w') as f:
            json.dump(self.evolution_log, f, indent=2)
