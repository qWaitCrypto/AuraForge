from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ..models.signal import Signal, SignalType


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._:-]+", "_", str(value or "").strip()) or "na"


def _day_from_ts(timestamp_ms: int) -> str:
    dt = datetime.fromtimestamp(max(0, int(timestamp_ms)) / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


class SignalStoreError(RuntimeError):
    pass


class SignalStore:
    """JSONL inbox storage for lightweight agent signals."""

    def __init__(self, *, project_root: Path) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._root = self._project_root / ".aura" / "state" / "signals"
        self._inbox = self._root / "inbox"
        self._archive = self._root / "archive"
        self._inbox.mkdir(parents=True, exist_ok=True)
        self._archive.mkdir(parents=True, exist_ok=True)

    def _inbox_file(self, agent_id: str) -> Path:
        return self._inbox / f"{_safe_token(agent_id)}.jsonl"

    def append(self, signal: Signal) -> None:
        path = self._inbox_file(signal.to_agent)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(signal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read_inbox(self, agent_id: str, *, unconsumed_only: bool = True, limit: int = 20) -> list[Signal]:
        path = self._inbox_file(agent_id)
        if not path.exists():
            return []

        out: list[Signal] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                signal = Signal.model_validate_json(line)
            except Exception:
                continue
            if unconsumed_only and signal.consumed:
                continue
            out.append(signal)

        out.sort(key=lambda item: item.created_at)
        if limit > 0:
            out = out[:limit]
        return out

    def mark_consumed(self, signal_id: str) -> None:
        target = str(signal_id or "").strip()
        if not target:
            raise ValueError("signal_id must be a non-empty string.")

        found = False
        for path in sorted(self._inbox.glob("*.jsonl")):
            changed = False
            records: list[Signal] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    signal = Signal.model_validate_json(line)
                except Exception:
                    continue
                if signal.signal_id == target and not signal.consumed:
                    signal = signal.model_copy(update={"consumed": True})
                    changed = True
                    found = True
                records.append(signal)
            if not changed:
                continue

            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                for signal in records:
                    handle.write(json.dumps(signal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
            tmp.replace(path)

            for signal in records:
                if signal.signal_id == target and signal.consumed:
                    day = _day_from_ts(signal.created_at)
                    archive_path = self._archive / f"{day}.jsonl"
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    with archive_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(signal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
                    return

        if not found:
            raise SignalStoreError(f"Signal not found: {target}")

    def query_all(
        self,
        *,
        from_agent: str | None = None,
        to_agent: str | None = None,
        signal_type: SignalType | None = None,
        issue_key: str | None = None,
        since_ms: int | None = None,
        limit: int = 100,
    ) -> list[Signal]:
        items: list[Signal] = []
        for path in sorted(self._inbox.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    signal = Signal.model_validate_json(line)
                except Exception:
                    continue
                if from_agent is not None and signal.from_agent != from_agent:
                    continue
                if to_agent is not None and signal.to_agent != to_agent:
                    continue
                if signal_type is not None and signal.signal_type is not signal_type:
                    continue
                if issue_key is not None and signal.issue_key != issue_key:
                    continue
                if since_ms is not None and signal.created_at < since_ms:
                    continue
                items.append(signal)

        items.sort(key=lambda item: item.created_at)
        if limit > 0 and len(items) > limit:
            items = items[-limit:]
        return items
