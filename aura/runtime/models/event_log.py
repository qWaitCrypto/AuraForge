from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from ..ids import now_ts_ms


class LogEventKind(StrEnum):
    TOOL_CALL = "tool_call"
    LLM_REQUEST = "llm_request"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SIGNAL_SENT = "signal_sent"
    SIGNAL_RECEIVED = "signal_received"


class LogEvent(BaseModel):
    event_id: str
    timestamp: int = Field(default_factory=now_ts_ms, ge=0)
    session_id: str
    agent_id: str
    sandbox_id: str | None = None
    issue_key: str | None = None

    kind: LogEventKind
    tool_name: str | None = None
    tool_args_summary: str | None = None
    tool_result_summary: str | None = None
    tool_ok: bool | None = None
    duration_ms: int | None = Field(default=None, ge=0)

    external_refs: list[str] = Field(default_factory=list)

    @field_validator("event_id", "session_id", "agent_id")
    @classmethod
    def _validate_required_non_empty(cls, value: str, info) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return cleaned

    @field_validator("sandbox_id", "issue_key", "tool_name", "tool_args_summary", "tool_result_summary")
    @classmethod
    def _validate_optional_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("external_refs")
    @classmethod
    def _validate_refs(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            cleaned = str(raw or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            out.append(cleaned)
        return out
