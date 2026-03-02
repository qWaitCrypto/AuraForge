from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AgentContext:
    """
    Assembled layered prompt context for one agent session.
    """

    system_prompt: str
    layers: dict[str, str] = field(default_factory=dict)
    agent_id: str | None = None
    role: str | None = None
    issue_key: str | None = None
    sandbox_id: str | None = None
    trigger: str | None = None
