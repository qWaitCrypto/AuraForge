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
        self._index_path = self._root / "_signal_index.json"
        self._inbox.mkdir(parents=True, exist_ok=True)
        self._archive.mkdir(parents=True, exist_ok=True)

    def _inbox_file(self, agent_id: str) -> Path:
        return self._inbox / f"{_safe_token(agent_id)}.jsonl"

    def _load_index(self) -> dict[str, str]:
        if not self._index_path.exists():
            return {}
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            token = value.strip()
            if not token:
                continue
            out[key] = token
        return out

    def _write_index(self, index_obj: dict[str, str]) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(index_obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._index_path)

    def _index_signal(self, *, signal_id: str, inbox_file_name: str) -> None:
        if not signal_id.strip():
            return
        index_obj = self._load_index()
        index_obj[signal_id] = inbox_file_name
        self._write_index(index_obj)

    def _lookup_indexed_inbox(self, signal_id: str) -> Path | None:
        index_obj = self._load_index()
        file_name = index_obj.get(signal_id)
        if not file_name:
            return None
        path = self._inbox / file_name
        if path.exists():
            return path
        return None

    def _scan_inbox_path_for_signal(self, signal_id: str) -> Path | None:
        target = str(signal_id or "").strip()
        if not target:
            return None
        for path in sorted(self._inbox.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    signal = Signal.model_validate_json(line)
                except Exception:
                    continue
                if signal.signal_id != target:
                    continue
                self._index_signal(signal_id=target, inbox_file_name=path.name)
                return path
        return None

    def append(self, signal: Signal) -> None:
        path = self._inbox_file(signal.to_agent)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(signal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self._index_signal(signal_id=signal.signal_id, inbox_file_name=path.name)

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

        path = self._lookup_indexed_inbox(target)
        if path is None:
            path = self._scan_inbox_path_for_signal(target)
        if path is None:
            raise SignalStoreError(f"Signal not found: {target}")

        changed = False
        matched: Signal | None = None
        records: list[Signal] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                signal = Signal.model_validate_json(line)
            except Exception:
                continue
            if signal.signal_id == target:
                if not signal.consumed:
                    signal = signal.model_copy(update={"consumed": True})
                    changed = True
                matched = signal
            records.append(signal)

        if matched is None:
            raise SignalStoreError(f"Signal not found: {target}")

        if not changed:
            return

        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for signal in records:
                handle.write(json.dumps(signal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(path)

        day = _day_from_ts(matched.created_at)
        archive_path = self._archive / f"{day}.jsonl"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(matched.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")

    def find_by_id(self, signal_id: str) -> Signal | None:
        target = str(signal_id or "").strip()
        if not target:
            return None
        paths = [*sorted(self._inbox.glob("*.jsonl")), *sorted(self._archive.glob("*.jsonl"))]
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    signal = Signal.model_validate_json(line)
                except Exception:
                    continue
                if signal.signal_id == target:
                    return signal
        return None

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
        items_by_id: dict[str, Signal] = {}

        paths = [*sorted(self._inbox.glob("*.jsonl")), *sorted(self._archive.glob("*.jsonl"))]
        for path in paths:
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
                existing = items_by_id.get(signal.signal_id)
                if existing is None:
                    items_by_id[signal.signal_id] = signal
                    continue
                # Prefer consumed record when duplicate appears in inbox + archive.
                if signal.consumed and not existing.consumed:
                    items_by_id[signal.signal_id] = signal

        items = sorted(items_by_id.values(), key=lambda item: item.created_at)
        if limit > 0 and len(items) > limit:
            items = items[-limit:]
        return items
