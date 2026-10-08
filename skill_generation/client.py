"""Provider-neutral teacher adapter used by skill-bank generation scripts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from skillforge.mutation import TeacherClient


class ConfiguredTeacher:
    """Adapt the shared teacher client to the generator scripts' message API."""

    def __init__(self, *, model: Optional[str] = None, max_new_tokens: Optional[int] = None) -> None:
        self.client = TeacherClient()
        if model:
            self.client.model = model
        if max_new_tokens is not None:
            self.client.max_tokens = max_new_tokens

    def generate_response(self, messages: Iterable[Mapping[str, Any]]) -> str:
        prompt = "\n\n".join(str(item.get("content", "")) for item in messages)
        return self.client.complete(prompt) or ""
