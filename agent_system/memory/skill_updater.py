"""Compatibility adapter for the public SkillForge teacher interface.

The original trainer imports ``SkillUpdater``. Keeping this small adapter lets
that trainer use the cleaned provider-neutral implementation without carrying
private endpoints, machine paths, or provider SDK details into the repository.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from skillforge.mutation import SkillMutator, TeacherClient


LLMClientWrapper = TeacherClient


class SkillUpdater:
    """Expose the legacy trainer methods through ``SkillMutator``."""

    def __init__(self, max_new_skills_per_update: int = 3, max_tokens: int = 2048):
        del max_tokens
        self.max_new_skills_per_update = max_new_skills_per_update
        self.mutator = SkillMutator()
        self.model = getattr(self.mutator.client, "model", "configured-teacher")

    @staticmethod
    def _bank_view(skills: Dict[str, Any]):
        class BankView:
            def __init__(self, payload: Dict[str, Any]):
                self.skills = payload

            def all_skills(self, *, include_retired: bool = False):
                groups = list(self.skills.get("general_skills", []))
                groups.extend(
                    item
                    for values in self.skills.get("task_specific_skills", {}).values()
                    for item in values
                )
                if include_retired:
                    return groups
                return [item for item in groups if item.get("lifecycle") != "retired"]

        return BankView(skills)

    def mutate_skill(
        self,
        skill: Dict[str, Any],
        current_skills: Dict[str, Any],
        recent_failures: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.mutator.mutate(
            skill,
            self._bank_view(current_skills),
            recent_failures=recent_failures,
        )

    def analyze_failures(
        self,
        failed_trajectories: List[Dict[str, Any]],
        current_skills: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return self.mutator.analyze_failures(
            failed_trajectories,
            self._bank_view(current_skills),
            max_new_skills=self.max_new_skills_per_update,
        )
