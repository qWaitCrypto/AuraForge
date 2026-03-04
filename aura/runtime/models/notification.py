from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from ..ids import now_ts_ms


class NotificationType(StrEnum):
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    PR_CREATED = "pr_created"
    PR_MERGED = "pr_merged"
    REVIEW_NEEDED = "review_needed"


class UserNotification(BaseModel):
    notification_id: str
    notification_type: NotificationType
    title: str
    summary: str
    issue_key: str | None = None
    pr_url: str | None = None
    details: dict | None = None
    read: bool = False
    created_at: int = Field(default_factory=now_ts_ms)
