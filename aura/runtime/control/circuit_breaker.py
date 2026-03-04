from __future__ import annotations

import json
import threading
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from ..ids import now_ts_ms


class CircuitOpenError(RuntimeError):
    def __init__(self, name: str) -> None:
        cleaned = str(name or "").strip() or "unknown"
        self.name = cleaned
        super().__init__(f"circuit_open:{cleaned}")


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BreakerConfig(BaseModel):
    failure_threshold: int = Field(default=5, ge=1)
    success_threshold: int = Field(default=2, ge=1)
    open_duration_sec: int = Field(default=60, ge=1)


class BreakerRecord(BaseModel):
    name: str
    state: BreakerState = BreakerState.CLOSED
    failure_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    opened_at_ms: int | None = None
    last_attempt_ms: int | None = None


class CircuitBreaker:
    def __init__(
        self,
        *,
        project_root: Path,
        config: BreakerConfig | None = None,
    ) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._config = config or BreakerConfig()
        self._state_path = self._project_root / ".aura" / "state" / "control" / "circuit_breakers.json"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _load_state(self) -> dict[str, BreakerRecord]:
        if not self._state_path.exists():
            return {}
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}

        out: dict[str, BreakerRecord] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                continue
            try:
                record = BreakerRecord.model_validate(value)
            except Exception:
                continue
            out[key] = record
        return out

    def _write_state(self, rows: dict[str, BreakerRecord]) -> None:
        payload = {
            key: value.model_dump(mode="json")
            for key, value in sorted(rows.items(), key=lambda item: item[0])
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._state_path)

    @staticmethod
    def _normalize_name(name: str) -> str:
        cleaned = str(name or "").strip()
        if not cleaned:
            raise ValueError("breaker name must be a non-empty string.")
        return cleaned

    def _get_or_default(self, rows: dict[str, BreakerRecord], name: str) -> BreakerRecord:
        current = rows.get(name)
        if current is not None:
            return current
        return BreakerRecord(name=name)

    def can_call(self, name: str) -> bool:
        target = self._normalize_name(name)
        now = now_ts_ms()
        open_duration_ms = int(self._config.open_duration_sec) * 1000

        with self._lock:
            rows = self._load_state()
            record = self._get_or_default(rows, target)

            if record.state is BreakerState.OPEN:
                opened = int(record.opened_at_ms or 0)
                if opened > 0 and (now - opened) >= open_duration_ms:
                    # Timeout elapsed: allow a trial call in HALF_OPEN.
                    record = record.model_copy(
                        update={
                            "state": BreakerState.HALF_OPEN,
                            "failure_count": 0,
                            "success_count": 0,
                            "last_attempt_ms": now,
                        }
                    )
                    rows[target] = record
                    self._write_state(rows)
                    return True
                return False
            return True

    def record_success(self, name: str) -> None:
        target = self._normalize_name(name)
        now = now_ts_ms()

        with self._lock:
            rows = self._load_state()
            current = rows.get(target)
            if current is None:
                return
            record = current

            if record.state is BreakerState.HALF_OPEN:
                new_success_count = record.success_count + 1
                if new_success_count >= self._config.success_threshold:
                    record = record.model_copy(
                        update={
                            "state": BreakerState.CLOSED,
                            "failure_count": 0,
                            "success_count": 0,
                            "opened_at_ms": None,
                            "last_attempt_ms": now,
                        }
                    )
                else:
                    record = record.model_copy(
                        update={
                            "success_count": new_success_count,
                            "last_attempt_ms": now,
                        }
                    )
            elif record.state is BreakerState.CLOSED:
                # In CLOSED, success_count is not used for state transitions.
                if record.failure_count == 0 and record.success_count == 0 and record.opened_at_ms is None:
                    return
                record = record.model_copy(update={"failure_count": 0, "success_count": 0, "opened_at_ms": None})
            else:
                # OPEN state success should not normally happen; keep state unchanged.
                record = record.model_copy(update={"last_attempt_ms": now})

            rows[target] = record
            self._write_state(rows)

    def record_failure(self, name: str) -> None:
        target = self._normalize_name(name)
        now = now_ts_ms()

        with self._lock:
            rows = self._load_state()
            record = self._get_or_default(rows, target)

            if record.state is BreakerState.HALF_OPEN:
                record = record.model_copy(
                    update={
                        "state": BreakerState.OPEN,
                        "failure_count": max(1, self._config.failure_threshold),
                        "success_count": 0,
                        "opened_at_ms": now,
                        "last_attempt_ms": now,
                    }
                )
            else:
                failure_count = record.failure_count + 1
                state = record.state
                opened_at_ms = record.opened_at_ms
                if failure_count >= self._config.failure_threshold:
                    state = BreakerState.OPEN
                    opened_at_ms = now
                record = record.model_copy(
                    update={
                        "state": state,
                        "failure_count": failure_count,
                        "success_count": 0,
                        "opened_at_ms": opened_at_ms,
                        "last_attempt_ms": now,
                    }
                )

            rows[target] = record
            self._write_state(rows)

    def get_state(self, name: str) -> BreakerRecord:
        target = self._normalize_name(name)
        with self._lock:
            rows = self._load_state()
            return self._get_or_default(rows, target)

    def force_close(self, name: str) -> BreakerRecord:
        target = self._normalize_name(name)
        now = now_ts_ms()
        with self._lock:
            rows = self._load_state()
            record = self._get_or_default(rows, target).model_copy(
                update={
                    "state": BreakerState.CLOSED,
                    "failure_count": 0,
                    "success_count": 0,
                    "opened_at_ms": None,
                    "last_attempt_ms": now,
                }
            )
            rows[target] = record
            self._write_state(rows)
            return record

    def list_all(self) -> list[BreakerRecord]:
        with self._lock:
            rows = self._load_state()
            return [rows[key] for key in sorted(rows.keys())]


_SHARED_BREAKERS_LOCK = threading.Lock()
_SHARED_BREAKERS: dict[str, CircuitBreaker] = {}


def get_shared_circuit_breaker(
    *,
    project_root: Path,
    config: BreakerConfig | None = None,
) -> CircuitBreaker:
    root = project_root.expanduser().resolve()
    key = str(root)
    with _SHARED_BREAKERS_LOCK:
        existing = _SHARED_BREAKERS.get(key)
        if existing is not None:
            return existing
        created = CircuitBreaker(project_root=root, config=config)
        _SHARED_BREAKERS[key] = created
        return created
