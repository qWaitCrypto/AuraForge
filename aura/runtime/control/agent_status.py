from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from ..event_log import EventLog
from ..ids import now_ts_ms
from ..sandbox import SandboxManager
from ..signal import SignalBus

_DAY_MS = 86_400_000
_DEFAULT_SANDBOX_IDLE_TIMEOUT_MS = 3_600_000


class AgentState(StrEnum):
    IDLE = "idle"
    WAITING = "waiting"
    ACTIVE = "active"
    STUCK = "stuck"
    FAILED = "failed"
    COOLING = "cooling"


class AgentStatusRecord(BaseModel):
    agent_id: str
    state: AgentState
    active_sandbox_ids: list[str] = Field(default_factory=list)
    active_issue_keys: list[str] = Field(default_factory=list)
    pending_signal_count: int = 0
    last_event_ts: int | None = None
    last_signal_ts: int | None = None
    failure_count_24h: int = 0
    cooling_until_ts: int | None = None
    updated_at: int = Field(default_factory=now_ts_ms)

    @field_validator("agent_id")
    @classmethod
    def _validate_agent_id(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("agent_id must be a non-empty string.")
        return cleaned

    @field_validator("active_sandbox_ids", "active_issue_keys")
    @classmethod
    def _dedupe_list(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            cleaned = str(raw or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            out.append(cleaned)
        return out


class AgentStatusTracker:
    def __init__(
        self,
        *,
        project_root: Path,
        event_log: EventLog,
        signal_bus: SignalBus,
        sandbox_manager: SandboxManager,
    ) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._event_log = event_log
        self._signal_bus = signal_bus
        self._sandbox_manager = sandbox_manager
        self._state_path = self._project_root / ".aura" / "state" / "control" / "agent_status.json"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_snapshot(self) -> dict[str, AgentStatusRecord]:
        if not self._state_path.exists():
            return {}
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, AgentStatusRecord] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                continue
            try:
                record = AgentStatusRecord.model_validate(value)
            except Exception:
                continue
            out[key] = record
        return out

    def _write_snapshot(self, records: dict[str, AgentStatusRecord]) -> None:
        payload = {
            key: value.model_dump(mode="json")
            for key, value in sorted(records.items(), key=lambda item: item[0])
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._state_path)

    def _known_agents(self) -> set[str]:
        known: set[str] = set(self._load_snapshot().keys())
        for sandbox in self._sandbox_manager.list_active():
            known.add(sandbox.agent_id)
        for signal in self._signal_bus.query(limit=0):
            known.add(signal.to_agent)
        return {item for item in known if item.strip()}

    def refresh(
        self,
        agent_id: str,
        *,
        sandbox_idle_timeout_ms: int | None = None,
    ) -> AgentStatusRecord:
        agent = str(agent_id or "").strip()
        if not agent:
            raise ValueError("agent_id must be a non-empty string.")

        snapshot = self._load_snapshot()
        previous = snapshot.get(agent)
        now = now_ts_ms()

        active_sandboxes = self._sandbox_manager.find_by_agent(agent)
        active_sandbox_ids = [item.sandbox_id for item in active_sandboxes]
        active_issue_keys = sorted({item.issue_key for item in active_sandboxes})

        signals = self._signal_bus.query(to_agent=agent, limit=0)
        pending_signal_count = sum(1 for item in signals if not item.consumed)
        last_signal_ts = max((item.created_at for item in signals), default=None)

        events = self._event_log.query(agent_id=agent, limit=10_000)
        last_event_ts = max((item.timestamp for item in events), default=None)
        since_ts = now - _DAY_MS
        failure_count_24h = sum(1 for item in events if item.timestamp >= since_ts and item.tool_ok is False)

        cooling_until_ts = previous.cooling_until_ts if previous is not None else None
        if isinstance(cooling_until_ts, int) and cooling_until_ts <= now:
            cooling_until_ts = None

        idle_timeout_ms = _DEFAULT_SANDBOX_IDLE_TIMEOUT_MS
        if isinstance(sandbox_idle_timeout_ms, int) and sandbox_idle_timeout_ms > 0:
            idle_timeout_ms = sandbox_idle_timeout_ms
        last_event_failed = bool(events and events[-1].tool_ok is False)
        if isinstance(cooling_until_ts, int) and cooling_until_ts > now:
            state = AgentState.COOLING
        elif active_sandbox_ids:
            ref_ts = last_event_ts
            if ref_ts is None:
                ref_ts = max((item.created_at for item in active_sandboxes), default=None)
            if ref_ts is None or (now - ref_ts) > idle_timeout_ms:
                state = AgentState.STUCK
            else:
                state = AgentState.ACTIVE
        elif pending_signal_count > 0:
            state = AgentState.WAITING
        elif last_event_failed:
            state = AgentState.FAILED
        else:
            state = AgentState.IDLE

        record = AgentStatusRecord(
            agent_id=agent,
            state=state,
            active_sandbox_ids=active_sandbox_ids,
            active_issue_keys=active_issue_keys,
            pending_signal_count=pending_signal_count,
            last_event_ts=last_event_ts,
            last_signal_ts=last_signal_ts,
            failure_count_24h=failure_count_24h,
            cooling_until_ts=cooling_until_ts,
            updated_at=now,
        )
        snapshot[agent] = record
        self._write_snapshot(snapshot)
        return record

    def refresh_all(self, *, sandbox_idle_timeout_ms: int | None = None) -> list[AgentStatusRecord]:
        records: list[AgentStatusRecord] = []
        for agent_id in sorted(self._known_agents()):
            records.append(self.refresh(agent_id, sandbox_idle_timeout_ms=sandbox_idle_timeout_ms))
        return records

    def get(self, agent_id: str) -> AgentStatusRecord | None:
        agent = str(agent_id or "").strip()
        if not agent:
            return None
        snapshot = self._load_snapshot()
        return snapshot.get(agent)

    def list_all(self) -> list[AgentStatusRecord]:
        snapshot = self._load_snapshot()
        return [snapshot[key] for key in sorted(snapshot.keys())]

    def mark_cooling(self, agent_id: str, *, duration_ms: int) -> None:
        agent = str(agent_id or "").strip()
        if not agent:
            raise ValueError("agent_id must be a non-empty string.")
        duration = int(duration_ms)
        if duration <= 0:
            raise ValueError("duration_ms must be > 0.")

        snapshot = self._load_snapshot()
        now = now_ts_ms()
        cooling_until = now + duration
        current = snapshot.get(agent)
        if current is None:
            current = self.refresh(agent)
            snapshot = self._load_snapshot()

        updated = current.model_copy(
            update={
                "state": AgentState.COOLING,
                "cooling_until_ts": cooling_until,
                "updated_at": now,
            }
        )
        snapshot[agent] = updated
        self._write_snapshot(snapshot)
