from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from ..ids import now_ts_ms

_SANDBOX_ID_RE = re.compile(r"^sb_[A-Za-z0-9._:-]+$")


class Sandbox(BaseModel):
    """Physical isolated worktree metadata for one agent task."""

    sandbox_id: str
    agent_id: str
    issue_key: str
    worktree_path: str
    branch: str
    base_branch: str = "main"
    created_at: int = Field(default_factory=now_ts_ms, ge=0)

    @field_validator("sandbox_id")
    @classmethod
    def _validate_sandbox_id(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not _SANDBOX_ID_RE.fullmatch(cleaned):
            raise ValueError("sandbox_id must match pattern sb_<token>.")
        return cleaned

    @field_validator("agent_id", "issue_key", "worktree_path", "branch", "base_branch")
    @classmethod
    def _validate_non_empty(cls, value: str, info) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return cleaned
