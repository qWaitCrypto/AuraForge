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


@dataclass(frozen=True, slots=True)
class SignalSendTool:
    signal_bus: SignalBus
    name: str = "signal__send"
    description: str = (
        "Send a lightweight signal to another agent. "
        "Use WAKE to ask an agent to check a Linear issue, "
        "TASK_ASSIGNED to notify assignment, or NOTIFY for FYI."
    )
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "to_agent": {"type": "string", "description": "Target agent ID."},
                "signal_type": {"type": "string", "enum": ["wake", "task_assigned", "notify"]},
                "brief": {"type": "string", "maxLength": 200},
                "issue_key": {"type": "string"},
                "sandbox_id": {"type": "string"},
                "payload": {"type": "object"},
                "from_agent": {"type": "string", "description": "Sender agent ID (optional)."},
            },
            "required": ["to_agent", "signal_type", "brief"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root

        to_agent = _required_non_empty(args.get("to_agent"), field_name="to_agent")
        brief = _required_non_empty(args.get("brief"), field_name="brief")

        signal_type_raw = _required_non_empty(args.get("signal_type"), field_name="signal_type")
        signal_type = SignalType(signal_type_raw)

        from_agent = (
            str(args.get("from_agent")).strip()
            if isinstance(args.get("from_agent"), str) and str(args.get("from_agent")).strip()
            else None
        )
        if from_agent is None and context is not None and isinstance(context.agent_id, str) and context.agent_id.strip():
            from_agent = context.agent_id.strip()
        if from_agent is None and context is not None and isinstance(context.session_id, str) and context.session_id.strip():
            from_agent = context.session_id.strip()
        if from_agent is None:
            from_agent = "system"

        signal = self.signal_bus.send(
            from_agent=from_agent,
            to_agent=to_agent,
            signal_type=signal_type,
            brief=brief,
            issue_key=(str(args.get("issue_key")).strip() if isinstance(args.get("issue_key"), str) else None),
            sandbox_id=(str(args.get("sandbox_id")).strip() if isinstance(args.get("sandbox_id"), str) else None),
            payload=(dict(args.get("payload")) if isinstance(args.get("payload"), dict) else None),
        )
        return {"ok": True, "signal": signal.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class SignalPollTool:
    signal_bus: SignalBus
    name: str = "signal__poll"
    description: str = "Poll for signals addressed to this agent."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "to_agent": {"type": "string", "description": "Target agent ID (optional)."},
                "unconsumed_only": {"type": "boolean", "default": True},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "consume": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root

        to_agent = (
            str(args.get("to_agent")).strip()
            if isinstance(args.get("to_agent"), str) and str(args.get("to_agent")).strip()
            else None
        )
        if to_agent is None and context is not None and isinstance(context.agent_id, str) and context.agent_id.strip():
            to_agent = context.agent_id.strip()
        if to_agent is None and context is not None and isinstance(context.session_id, str) and context.session_id.strip():
            to_agent = context.session_id.strip()
        if to_agent is None:
            raise ValueError("Missing target agent. Provide 'to_agent' or use a bound session context.")

        unconsumed_only = bool(args.get("unconsumed_only", True))
        limit_raw = args.get("limit", 20)
        try:
            limit = max(1, min(50, int(limit_raw)))
        except Exception:
            raise ValueError(f"Invalid 'limit': {limit_raw!r}")

        signals = self.signal_bus.poll(to_agent=to_agent, unconsumed_only=unconsumed_only, limit=limit)

        consume = bool(args.get("consume", True))
        if consume:
            consumed_ids: set[str] = set()
            for signal in signals:
                if signal.consumed:
                    continue
                self.signal_bus.consume(signal.signal_id)
                consumed_ids.add(signal.signal_id)
            if consumed_ids:
                signals = [
                    signal.model_copy(update={"consumed": True}) if signal.signal_id in consumed_ids else signal
                    for signal in signals
                ]

        return {
            "ok": True,
            "count": len(signals),
            "signals": [signal.model_dump(mode="json") for signal in signals],
        }
