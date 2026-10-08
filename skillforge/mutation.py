"""Optional external teacher for failure analysis and skill mutation.

No provider, endpoint, or credential is bundled. Configure an LLM only when
using LLM-guided mutation:
  SKILLFORGE_LLM_BACKEND=openai|anthropic|kimi
  SKILLFORGE_LLM_API_KEY=...
  SKILLFORGE_LLM_BASE_URL=...  (optional)
  SKILLFORGE_LLM_MODEL=...
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


class TeacherClient:
    def __init__(self) -> None:
        self.backend = os.getenv("SKILLFORGE_LLM_BACKEND", "openai").lower()
        if self.backend == "kimi":
            # Kimi's public API is OpenAI-compatible. The endpoint remains an
            # environment setting so no provider-specific URL is published.
            self.backend = "openai"
        self.api_key = os.getenv("SKILLFORGE_LLM_API_KEY")
        self.base_url = os.getenv("SKILLFORGE_LLM_BASE_URL")
        self.model = os.getenv("SKILLFORGE_LLM_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("SKILLFORGE_LLM_MAX_TOKENS", "2048"))
        if not self.api_key:
            raise EnvironmentError(
                "Set SKILLFORGE_LLM_API_KEY before using SkillMutator."
            )

        if self.backend == "anthropic":
            from anthropic import Anthropic

            self.client = Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=2,
            )
        elif self.backend == "openai":
            from openai import OpenAI

            kwargs = {"api_key": self.api_key, "max_retries": 3}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)
        else:
            raise ValueError("SKILLFORGE_LLM_BACKEND must be openai, anthropic, or kimi")

    def complete(self, prompt: str) -> Optional[str]:
        try:
            if self.backend == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as exc:
            # Do not echo provider responses or configured endpoint details.
            del exc
            print("[SkillForge] teacher request failed")
            return None


class SkillMutator:
    """Generate new skills or improved child skills with an external teacher."""

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
    ) -> None:
        self.client = client or TeacherClient()

    @staticmethod
    def _next_id(skills: Mapping[str, Any], prefix: str) -> str:
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        index = 0
        groups = [skills.get("general_skills", [])]
        groups += list(skills.get("task_specific_skills", {}).values())
        for group in groups:
            for skill in group:
                match = pattern.match(skill.get("skill_id", ""))
                if match:
                    index = max(index, int(match.group(1)))
        return f"{prefix}_{index + 1:03d}"

    @staticmethod
    def _extract_json(text: str, *, array: bool) -> Any:
        start_char, end_char = ("[", "]") if array else ("{", "}")
        start, end = text.find(start_char), text.rfind(end_char)
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    def mutate(
        self,
        parent: Mapping[str, Any],
        bank: Any,
        *,
        recent_failures: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        child_id = self._next_id(bank.skills, "evo")
        fitness = int(parent.get("success_count", 0)) / max(
            int(parent.get("usage_count", 0)), 1
        )
        failures = list(recent_failures or [])[:8]
        failure_text = "\n".join(
            f"- Task: {item.get('task', '')[:300]}\n"
            f"  Trajectory: {item.get('trajectory', '')[:500]}"
            for item in failures
        ) or "No failure trace was recorded."

        prompt = f"""Improve the following reusable agent skill.
Return only one JSON object with title, principle, and when_to_apply.

Parent skill:
- title: {parent.get('title', '')}
- principle: {parent.get('principle', '')}
- when to apply: {parent.get('when_to_apply', '')}
- fitness: {fitness:.3f}
- usage: {parent.get('usage_count', 0)}
- successes: {parent.get('success_count', 0)}

Recent failures:
{failure_text}
"""
        response = self.client.complete(prompt)
        parsed = self._extract_json(response or "", array=False)
        if not isinstance(parsed, dict) or not parsed.get("title") or not parsed.get("principle"):
            return None
        parsed["skill_id"] = child_id
        return parsed

    def analyze_failures(
        self,
        failed_trajectories: Iterable[Mapping[str, Any]],
        bank: Any,
        *,
        max_new_skills: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate new skills from failed trajectories."""
        failures = list(failed_trajectories)[:8]
        if not failures:
            return []
        existing = [
            skill.get("title", "")
            for skill in bank.all_skills(include_retired=True)
        ]
        examples = "\n".join(
            f"- Task: {item.get('task', '')[:300]}\n"
            f"  Trajectory: {item.get('trajectory', '')[:600]}"
            for item in failures
        )
        prompt = f"""Suggest up to {max_new_skills} reusable skills for an agent.
Return only a JSON array. Every item must contain title, principle, and
when_to_apply. Do not repeat an existing title.

Existing titles:
{existing[:100]}

Failed episodes:
{examples}
"""
        response = self.client.complete(prompt)
        parsed = self._extract_json(response or "", array=True)
        if not isinstance(parsed, list):
            return []
        start = self._next_id(bank.skills, "dyn")
        try:
            start_index = int(start.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            start_index = 1
        result = []
        for offset, item in enumerate(parsed[:max_new_skills]):
            if not isinstance(item, dict) or not item.get("title") or not item.get("principle"):
                continue
            child = dict(item)
            child["skill_id"] = f"dyn_{start_index + offset:03d}"
            result.append(child)
        return result
