"""Skill storage, retrieval, prompt formatting, and episode-level credit assignment.

The bank is independent of a particular RL trainer or environment. An integration
only needs to retrieve skills before an episode and report the retrieved skill ids
with the episode outcome afterwards.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class SkillBank:
    """Mutable skill library used by SkillForge."""

    def __init__(
        self,
        skills: Mapping[str, Any],
        *,
        task_keywords: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.skills: Dict[str, Any] = {
            "general_skills": list(skills.get("general_skills", [])),
            "task_specific_skills": {
                key: list(value)
                for key, value in skills.get("task_specific_skills", {}).items()
            },
            "common_mistakes": list(skills.get("common_mistakes", [])),
        }
        self.task_keywords = {
            key: list(value)
            for key, value in (task_keywords or {}).items()
        }

        for skill in self.all_skills():
            self.ensure_fields(skill)

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> "SkillBank":
        path = Path(path)
        with path.open() as handle:
            return cls(json.load(handle), **kwargs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.skills, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def ensure_fields(skill: Dict[str, Any], *, lifecycle: str = "active") -> Dict[str, Any]:
        """Add runtime fields without changing the skill's paper-facing content."""
        defaults = {
            "usage_count": 0,
            "success_count": 0,
            "lifecycle": lifecycle,
            "parent_id": None,
            "generation": 0,
            "children": [],
        }
        for key, value in defaults.items():
            skill.setdefault(key, value)
        return skill

    def all_skills(self, *, include_retired: bool = False) -> List[Dict[str, Any]]:
        skills = list(self.skills["general_skills"])
        for group in self.skills["task_specific_skills"].values():
            skills.extend(group)
        if include_retired:
            return skills
        return [skill for skill in skills if skill.get("lifecycle") != "retired"]

    def __len__(self) -> int:
        return len(self.all_skills())

    def get_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        for skill in self.all_skills(include_retired=True):
            if skill.get("skill_id") == skill_id:
                return skill
        return None

    def skill_ids(self) -> set[str]:
        return {
            skill["skill_id"]
            for skill in self.all_skills(include_retired=True)
            if skill.get("skill_id")
        }

    def detect_task_type(self, task_description: str) -> str:
        """Choose a task group using configured keywords or token overlap."""
        groups = self.skills["task_specific_skills"]
        if not groups:
            return "unknown"

        text = task_description.lower()
        scores: Dict[str, int] = {}
        for group in groups:
            keywords = self.task_keywords.get(group, [])
            if not keywords:
                keywords = re.findall(r"[a-z0-9]+", group.lower().replace("_", " "))
            scores[group] = sum(text.count(keyword.lower()) for keyword in keywords)

        best_group, best_score = max(scores.items(), key=lambda item: item[1])
        return best_group if best_score else next(iter(groups))

    def retrieve(
        self,
        task_description: str,
        *,
        top_k: int = 6,
        task_specific_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieve general and task-specific skills for one episode."""
        task_type = self.detect_task_type(task_description)
        general = [
            skill for skill in self.skills["general_skills"]
            if skill.get("lifecycle") != "retired"
        ][:top_k]
        specific = [
            skill for skill in self.skills["task_specific_skills"].get(task_type, [])
            if skill.get("lifecycle") != "retired"
        ]
        if task_specific_top_k is not None:
            specific = specific[:task_specific_top_k]

        return {
            "general_skills": general,
            "task_specific_skills": specific,
            "mistakes_to_avoid": self.skills["common_mistakes"][:5],
            "task_type": task_type,
        }

    @staticmethod
    def format_prompt(retrieved: Mapping[str, Any]) -> str:
        """Render retrieved skills as a prompt section."""
        sections: list[str] = []

        general = retrieved.get("general_skills", [])
        if general:
            lines = ["General principles"]
            for skill in general:
                lines.append(
                    f"- {skill.get('title', '')}: {skill.get('principle', '')}"
                )
            sections.append("\n".join(lines))

        specific = retrieved.get("task_specific_skills", [])
        if specific:
            title = retrieved.get("task_type", "Task").replace("_", " ").title()
            lines = [f"{title} skills"]
            for skill in specific:
                lines.append(
                    f"- {skill.get('title', '')}: {skill.get('principle', '')}"
                )
                if skill.get("when_to_apply"):
                    lines.append(f"  Apply when: {skill['when_to_apply']}")
            sections.append("\n".join(lines))

        mistakes = retrieved.get("mistakes_to_avoid", [])
        if mistakes:
            lines = ["Mistakes to avoid"]
            for mistake in mistakes:
                description = mistake.get("description", "")
                correction = mistake.get("how_to_avoid", "")
                if description:
                    lines.append(f"- Avoid: {description}")
                if correction:
                    lines.append(f"  Instead: {correction}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) or "No relevant skills found."

    def references(self, retrieved: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Resolve retrieved records back to mutable bank entries."""
        ids = {
            skill.get("skill_id")
            for key in ("general_skills", "task_specific_skills")
            for skill in retrieved.get(key, [])
            if skill.get("skill_id")
        }
        return [
            skill
            for skill in self.all_skills(include_retired=True)
            if skill.get("skill_id") in ids
        ]

    def record_episode(
        self,
        skills: Iterable[Dict[str, Any] | str],
        success: bool,
        *,
        task: Optional[str] = None,
        trajectory: Optional[str] = None,
    ) -> None:
        """Assign one binary episode outcome to every retrieved skill."""
        for item in skills:
            skill = self.get_skill(item) if isinstance(item, str) else item
            if skill is None:
                continue
            self.ensure_fields(skill)
            skill["usage_count"] += 1
            if success:
                skill["success_count"] += 1
            if not success and task:
                failures = skill.setdefault("recent_failures", [])
                failures.append({
                    "task": task[:500],
                    "trajectory": (trajectory or "")[:1000],
                })
                del failures[:-15]

    def add(self, skill: Mapping[str, Any], *, category: str = "general") -> bool:
        """Add a new child skill if its id is not already present."""
        skill_id = skill.get("skill_id")
        if not skill_id or skill_id in self.skill_ids():
            return False
        child = dict(skill)
        self.ensure_fields(child, lifecycle="trial")
        child["lifecycle"] = "trial"
        if category == "general":
            self.skills["general_skills"].append(child)
        else:
            self.skills["task_specific_skills"].setdefault(category, []).append(child)
        return True

    def remove(self, skill_id: str) -> bool:
        removed = False
        general = self.skills["general_skills"]
        kept = [skill for skill in general if skill.get("skill_id") != skill_id]
        removed |= len(kept) != len(general)
        self.skills["general_skills"] = kept
        for group, skills in list(self.skills["task_specific_skills"].items()):
            kept = [skill for skill in skills if skill.get("skill_id") != skill_id]
            removed |= len(kept) != len(skills)
            self.skills["task_specific_skills"][group] = kept
        return removed

    def counts(self) -> Dict[str, int]:
        task_specific = sum(
            len(skills) for skills in self.skills["task_specific_skills"].values()
        )
        return {
            "general": len(self.skills["general_skills"]),
            "task_specific": task_specific,
            "total": len(self),
        }
