from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ..models.event_log import LogEvent


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._:-]+", "_", str(value or "").strip()) or "na"


def _day_from_ts(timestamp_ms: int) -> str:
    dt = datetime.fromtimestamp(max(0, int(timestamp_ms)) / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


class EventLogStoreError(RuntimeError):
    pass


class EventLogFileStore:
    """Append-only JSONL storage for audit log events."""

    def __init__(self, *, project_root: Path) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._audit_root = self._project_root / ".aura" / "events" / "audit"
        self._audit_root.mkdir(parents=True, exist_ok=True)

    def _session_file(self, *, day: str, session_id: str) -> Path:
        return self._audit_root / day / f"session_{_safe_token(session_id)}.jsonl"

    def append(self, event: LogEvent) -> None:
        day = _day_from_ts(event.timestamp)
        path = self._session_file(day=day, session_id=event.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read(
        self,
        *,
        session_id: str | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> list[LogEvent]:
        files: list[Path] = []
        if isinstance(session_id, str) and session_id.strip():
            suffix = f"session_{_safe_token(session_id)}.jsonl"
            files = sorted(self._audit_root.glob(f"*/{suffix}"))
        else:
            files = sorted(self._audit_root.glob("*/*.jsonl"))

        events: list[LogEvent] = []
        for path in files:
            try:
                raw_lines = path.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                raise EventLogStoreError(f"Failed to read event log file: {path}") from exc
            for line in raw_lines:
                if not line.strip():
                    continue
                try:
                    item = LogEvent.model_validate_json(line)
                except Exception:
                    continue
                if since_ms is not None and item.timestamp < since_ms:
                    continue
                if until_ms is not None and item.timestamp > until_ms:
                    continue
                events.append(item)

        events.sort(key=lambda item: item.timestamp)
        return events
