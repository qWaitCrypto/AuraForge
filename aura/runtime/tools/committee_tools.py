from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models.signal import SignalType
from ..signal.bus import SignalBus
from .runtime import ToolExecutionContext


def _required_non_empty(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid '{field_name}' (expected non-empty string).")
    return value.strip()


def _optional_str_list(value: Any, *, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Invalid '{field_name}' (expected list of strings).")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Invalid '{field_name}' (expected list of strings).")
        cleaned = item.strip()
        if cleaned:
            out.append(cleaned)
    return out or None


@dataclass(frozen=True, slots=True)
class CommitteeSubmitTool:
    signal_bus: SignalBus
    name: str = "committee__submit"
    description: str = (
        "Submit a project-level request to Committee. "
        "Use this when the user asks for multi-step work that should be decomposed and coordinated."
    )
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "High-level goal in one sentence."},
                "context": {"type": "string", "description": "Detailed context and background."},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                "references": {"type": "array", "items": {"type": "string"}},
                "candidate_agents": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["goal", "context"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        goal = _required_non_empty(args.get("goal"), field_name="goal")
        req_context = _required_non_empty(args.get("context"), field_name="context")

        priority = str(args.get("priority") or "medium").strip().lower() or "medium"
        if priority not in {"high", "medium", "low"}:
            raise ValueError("Invalid 'priority' (expected high|medium|low).")

        payload = {
            "type": "project_request",
            "goal": goal,
            "context": req_context,
            "constraints": _optional_str_list(args.get("constraints"), field_name="constraints"),
            "priority": priority,
            "references": _optional_str_list(args.get("references"), field_name="references"),
            "candidate_agents": _optional_str_list(args.get("candidate_agents"), field_name="candidate_agents"),
        }

        from_agent = "super_agent"
        if context is not None and isinstance(context.agent_id, str) and context.agent_id.strip():
            from_agent = context.agent_id.strip()

        signal = self.signal_bus.send(
            from_agent=from_agent,
            to_agent="committee",
            signal_type=SignalType.WAKE,
            brief=f"New project request: {goal[:120]}",
            payload=payload,
        )
        return {
            "status": "submitted",
            "signal_id": signal.signal_id,
            "to_agent": "committee",
            "message": f"Project request submitted to Committee as signal {signal.signal_id}.",
        }
