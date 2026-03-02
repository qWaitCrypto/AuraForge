from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from ..ids import now_ts_ms


class SignalType(StrEnum):
    WAKE = "wake"
    TASK_ASSIGNED = "task_assigned"
    NOTIFY = "notify"


class Signal(BaseModel):
    signal_id: str
    from_agent: str
    to_agent: str
    signal_type: SignalType
    brief: str
    issue_key: str | None = None
    sandbox_id: str | None = None
    payload: dict | None = None
    created_at: int = Field(default_factory=now_ts_ms, ge=0)
    consumed: bool = False

    @field_validator("signal_id", "from_agent", "to_agent")
    @classmethod
    def _validate_required_non_empty(cls, value: str, info) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return cleaned

    @field_validator("brief")
    @classmethod
    def _validate_brief(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("brief must be a non-empty string.")
        if len(cleaned) > 200:
            raise ValueError("brief must be <= 200 characters.")
        return cleaned

    @field_validator("issue_key", "sandbox_id")
    @classmethod
    def _validate_optional_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None
