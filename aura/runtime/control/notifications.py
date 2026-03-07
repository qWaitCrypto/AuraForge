from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ids import new_id
from ..models.notification import NotificationType, UserNotification


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


class NotificationStore:
    def __init__(self, *, project_root: Path) -> None:
        root = project_root.expanduser().resolve()
        self._root = root / ".aura" / "state" / "notifications"
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / "notifications.jsonl"

    def _read_all(self) -> list[UserNotification]:
        if not self._path.exists():
            return []
        out: list[UserNotification] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = UserNotification.model_validate_json(line)
            except Exception:
                continue
            out.append(item)
        out.sort(key=lambda item: item.created_at, reverse=True)
        return out

    def _write_all(self, rows: list[UserNotification]) -> None:
        ordered = sorted(rows, key=lambda item: item.created_at)
        tmp = self._path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for item in ordered:
                handle.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(self._path)

    def create(
        self,
        *,
        notification_type: NotificationType,
        title: str,
        summary: str,
        issue_key: str | None = None,
        pr_url: str | None = None,
        details: dict | None = None,
    ) -> UserNotification:
        issue_key_value = _clean_text(issue_key) if isinstance(issue_key, str) else ""
        pr_url_value = _clean_text(pr_url) if isinstance(pr_url, str) else ""
        item = UserNotification(
            notification_id=new_id("notif"),
            notification_type=notification_type,
            title=_clean_text(title) or notification_type.value,
            summary=_clean_text(summary) or "-",
            issue_key=issue_key_value or None,
            pr_url=pr_url_value or None,
            details=dict(details or {}) if isinstance(details, dict) else None,
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
        return item

    def list(self, *, unread_only: bool = False, limit: int = 100) -> list[UserNotification]:
        rows = self._read_all()
        if unread_only:
            rows = [item for item in rows if not item.read]
        if limit > 0 and len(rows) > limit:
            return rows[:limit]
        return rows

    def mark_read(self, notification_id: str) -> UserNotification | None:
        target = _clean_text(notification_id)
        if not target:
            return None
        rows = self._read_all()
        changed = False
        found: UserNotification | None = None
        updated: list[UserNotification] = []
        for item in rows:
            if item.notification_id == target:
                next_item = item.model_copy(update={"read": True})
                updated.append(next_item)
                found = next_item
                changed = True
                continue
            updated.append(item)
        if changed:
            self._write_all(updated)
        return found
