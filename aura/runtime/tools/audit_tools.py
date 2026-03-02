from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..event_log.logger import EventLog
from ..models.event_log import LogEventKind
from .runtime import ToolExecutionContext


def _parse_since(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return max(0, value)
    raw = str(value).strip()
    if not raw:
        return None
    if raw == "last_hour":
        return int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp() * 1000)
    if raw == "last_day":
        return int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp() * 1000)
    if raw.isdigit():
        return int(raw)
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception as exc:
        raise ValueError(f"Invalid 'since' value: {value!r}") from exc
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class AuditQueryTool:
    event_log: EventLog
    name: str = "audit__query"
    description: str = (
        "Query the audit log for agent activity records. "
        "Filter by agent_id, issue_key, tool_name, time range."
    )
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "session_id": {"type": "string"},
                "sandbox_id": {"type": "string"},
                "issue_key": {"type": "string"},
                "tool_name": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": [
                        "tool_call",
                        "llm_request",
                        "session_start",
                        "session_end",
                        "signal_sent",
                        "signal_received",
                    ],
                },
                "since": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root, context

        kind: LogEventKind | None = None
        kind_raw = args.get("kind")
        if isinstance(kind_raw, str) and kind_raw.strip():
            kind = LogEventKind(kind_raw.strip())

        limit_raw = args.get("limit", 500)
        try:
            limit = max(1, min(1000, int(limit_raw)))
        except Exception:
            raise ValueError(f"Invalid 'limit': {limit_raw!r}")

        events = self.event_log.query(
            agent_id=(str(args.get("agent_id")).strip() if isinstance(args.get("agent_id"), str) else None),
            session_id=(str(args.get("session_id")).strip() if isinstance(args.get("session_id"), str) else None),
            sandbox_id=(str(args.get("sandbox_id")).strip() if isinstance(args.get("sandbox_id"), str) else None),
            issue_key=(str(args.get("issue_key")).strip() if isinstance(args.get("issue_key"), str) else None),
            tool_name=(str(args.get("tool_name")).strip() if isinstance(args.get("tool_name"), str) else None),
            kind=kind,
            since_ms=_parse_since(args.get("since")),
            limit=limit,
        )
        return {
            "ok": True,
            "count": len(events),
            "events": [event.model_dump(mode="json") for event in events],
        }


@dataclass(frozen=True, slots=True)
class AuditRefsTool:
    event_log: EventLog
    name: str = "audit__refs"
    description: str = "Query external references (commits, PRs, Linear comments) produced by agents."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "issue_key": {"type": "string"},
                "ref_type": {"type": "string", "enum": ["commit", "pr", "linear", "push"]},
                "since": {"type": "string"},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root, context

        ref_type = args.get("ref_type")
        ref_prefix = None
        if isinstance(ref_type, str) and ref_type.strip():
            ref_prefix = f"{ref_type.strip()}:"

        refs = self.event_log.query_external_refs(
            agent_id=(str(args.get("agent_id")).strip() if isinstance(args.get("agent_id"), str) else None),
            issue_key=(str(args.get("issue_key")).strip() if isinstance(args.get("issue_key"), str) else None),
            ref_prefix=ref_prefix,
            since_ms=_parse_since(args.get("since")),
        )
        return {"ok": True, "count": len(refs), "refs": refs}
