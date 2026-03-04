from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from ..ids import now_ts_ms


class AgentSessionState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    STOPPING = "stopping"
    STOPPED = "stopped"


class AgentSession(BaseModel):
    session_id: str
    agent_id: str
    state: AgentSessionState = AgentSessionState.STARTING
    current_signal_id: str | None = None
    current_issue_key: str | None = None
    sandbox_id: str | None = None
    started_at: int = Field(default_factory=now_ts_ms)
    last_active_at: int = Field(default_factory=now_ts_ms)
    signals_processed: int = 0
    pid: int | None = None

    @field_validator("session_id", "agent_id")
    @classmethod
    def _validate_non_empty(cls, value: str, info) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return cleaned

    @field_validator("current_signal_id", "current_issue_key", "sandbox_id")
    @classmethod
    def _validate_optional_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("signals_processed")
    @classmethod
    def _validate_signals_processed(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError("signals_processed must be >= 0")
        return int(value)
