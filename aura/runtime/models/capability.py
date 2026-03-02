from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.types import ToolSpec as LlmToolSpec

ROLE_WORKER = "worker"
ROLE_INTEGRATOR = "integrator"


@dataclass(frozen=True, slots=True)
class AgentCapabilitySurface:
    """
    Runtime capability surface for one agent session.
    """

    agent_id: str
    role: str
    tool_allowlist: list[str]
    tool_specs: list[LlmToolSpec]
    external_executors: dict[str, object] = field(default_factory=dict)
    max_turns: int = 20
    max_tool_calls: int = 60
    resolved_from: str = "agent_spec"
    warnings: list[str] = field(default_factory=list)
