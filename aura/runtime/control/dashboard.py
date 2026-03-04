from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..event_log import EventLog
from ..ids import now_ts_ms
from ..sandbox import SandboxManager
from ..signal import SignalBus
from .agent_status import AgentState, AgentStatusRecord, AgentStatusTracker


class AgentRow(BaseModel):
    agent_id: str
    state: AgentState
    active_issue_keys: list[str] = Field(default_factory=list)
    active_sandboxes: int = 0
    pending_signal_count: int = 0
    failure_count_24h: int = 0


class IssueRow(BaseModel):
    issue_key: str
    agents: list[str] = Field(default_factory=list)
    sandbox_count: int = 0
    signal_count: int = 0
    last_activity_ms: int | None = None


class SystemSummary(BaseModel):
    total_agents: int = 0
    active_agents: int = 0
    total_sandboxes: int = 0
    total_signals_today: int = 0
    stuck_count: int = 0
    dead_letter_count: int = 0


class DashboardSnapshot(BaseModel):
    ts_ms: int = Field(default_factory=now_ts_ms)
    summary: SystemSummary
    agents: list[AgentRow] = Field(default_factory=list)
    issues: list[IssueRow] = Field(default_factory=list)


class DashboardAggregator:
    def __init__(
        self,
        *,
        project_root: Path,
        status_tracker: AgentStatusTracker,
        signal_bus: SignalBus,
        sandbox_manager: SandboxManager,
        event_log: EventLog,
    ) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._status_tracker = status_tracker
        self._signal_bus = signal_bus
        self._sandbox_manager = sandbox_manager
        self._event_log = event_log
        self._control_state_dir = self._project_root / ".aura" / "state" / "control"
        self._dead_letters_path = self._control_state_dir / "dead_letters.jsonl"
        self._recovery_log_path = self._control_state_dir / "recovery_log.jsonl"

    @staticmethod
    def _start_of_day_ms(now_ms: int) -> int:
        dt = datetime.fromtimestamp(max(0, int(now_ms)) / 1000.0, tz=timezone.utc)
        midnight = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        return int(midnight.timestamp() * 1000)

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                count += 1
        return count

    def _read_recovery_records(
        self,
        *,
        issue_key: str | None = None,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not self._recovery_log_path.exists():
            return []

        issue_filter = str(issue_key or "").strip() if issue_key is not None else None
        agent_filter = str(agent_id or "").strip() if agent_id is not None else None
        rows: list[dict[str, Any]] = []
        for line in self._recovery_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if issue_filter is not None and str(payload.get("issue_key") or "").strip() != issue_filter:
                continue
            if agent_filter is not None and str(payload.get("agent_id") or "").strip() != agent_filter:
                continue
            rows.append(payload)

        rows.sort(key=lambda item: int(item.get("ts_ms") or 0))
        if limit > 0 and len(rows) > limit:
            rows = rows[-limit:]
        return rows

    def snapshot(self) -> DashboardSnapshot:
        now = now_ts_ms()
        since_ms = self._start_of_day_ms(now)

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_records = pool.submit(self._status_tracker.refresh_all)
            f_sandboxes = pool.submit(self._sandbox_manager.list_active)
            f_signals = pool.submit(
                self._signal_bus.query,
                since_ms=since_ms,
                limit=0,
                include_archive=True,
            )

            records = f_records.result()
            sandboxes = f_sandboxes.result()
            signals_today = f_signals.result()

        agent_rows: list[AgentRow] = []
        for record in records:
            agent_rows.append(
                AgentRow(
                    agent_id=record.agent_id,
                    state=record.state,
                    active_issue_keys=list(record.active_issue_keys),
                    active_sandboxes=len(record.active_sandbox_ids),
                    pending_signal_count=record.pending_signal_count,
                    failure_count_24h=record.failure_count_24h,
                )
            )

        by_issue: dict[str, dict[str, Any]] = {}
        for sandbox in sandboxes:
            issue = str(sandbox.issue_key or "").strip()
            if not issue:
                continue
            row = by_issue.setdefault(
                issue,
                {"agents": set(), "sandbox_count": 0, "signal_count": 0, "last_activity_ms": None},
            )
            row["agents"].add(sandbox.agent_id)
            row["sandbox_count"] += 1
            created_at = int(sandbox.created_at)
            last = row["last_activity_ms"]
            row["last_activity_ms"] = created_at if last is None else max(int(last), created_at)

        for signal in signals_today:
            issue = str(signal.issue_key or "").strip()
            if not issue:
                continue
            row = by_issue.setdefault(
                issue,
                {"agents": set(), "sandbox_count": 0, "signal_count": 0, "last_activity_ms": None},
            )
            row["signal_count"] += 1
            row["agents"].add(signal.to_agent)
            row["agents"].add(signal.from_agent)
            created_at = int(signal.created_at)
            last = row["last_activity_ms"]
            row["last_activity_ms"] = created_at if last is None else max(int(last), created_at)

        issue_rows: list[IssueRow] = []
        for issue_key in sorted(by_issue.keys()):
            row = by_issue[issue_key]
            issue_rows.append(
                IssueRow(
                    issue_key=issue_key,
                    agents=sorted({str(item).strip() for item in row["agents"] if str(item).strip()}),
                    sandbox_count=int(row["sandbox_count"]),
                    signal_count=int(row["signal_count"]),
                    last_activity_ms=row["last_activity_ms"],
                )
            )

        summary = SystemSummary(
            total_agents=len(agent_rows),
            active_agents=sum(1 for row in agent_rows if row.state is AgentState.ACTIVE),
            total_sandboxes=len(sandboxes),
            total_signals_today=len(signals_today),
            stuck_count=sum(1 for row in agent_rows if row.state is AgentState.STUCK),
            dead_letter_count=self._count_lines(self._dead_letters_path),
        )
        return DashboardSnapshot(
            ts_ms=now,
            summary=summary,
            agents=agent_rows,
            issues=issue_rows,
        )

    def agent_detail(self, agent_id: str) -> dict[str, Any]:
        agent = str(agent_id or "").strip()
        if not agent:
            raise ValueError("agent_id is required.")

        record: AgentStatusRecord = self._status_tracker.refresh(agent)
        active_sandboxes = self._sandbox_manager.find_by_agent(agent)
        signals_to = self._signal_bus.query(
            to_agent=agent,
            limit=20,
            include_archive=True,
        )
        signals_from = self._signal_bus.query(
            from_agent=agent,
            limit=20,
            include_archive=True,
        )
        merged_signals = {item.signal_id: item for item in [*signals_to, *signals_from]}
        recent_signals = sorted(
            merged_signals.values(),
            key=lambda item: int(item.created_at),
            reverse=True,
        )[:20]
        recent_events = self._event_log.query(agent_id=agent, limit=10)
        recent_recovery = self._read_recovery_records(agent_id=agent, limit=10)

        return {
            "agent_id": agent,
            "status": record.model_dump(mode="json"),
            "active_sandboxes": [item.model_dump(mode="json") for item in active_sandboxes],
            "recent_signals": [item.model_dump(mode="json") for item in recent_signals],
            "recent_events": [item.model_dump(mode="json") for item in recent_events],
            "recent_recovery": recent_recovery,
        }

    def issue_detail(self, issue_key: str) -> dict[str, Any]:
        issue = str(issue_key or "").strip()
        if not issue:
            raise ValueError("issue_key is required.")

        sandboxes = self._sandbox_manager.find_by_issue(issue)
        signals = self._signal_bus.query(issue_key=issue, limit=0, include_archive=True)
        events = self._event_log.query(issue_key=issue, limit=100)
        recovery = self._read_recovery_records(issue_key=issue, limit=500)
        agents = {
            *(item.agent_id for item in sandboxes),
            *(item.to_agent for item in signals),
            *(item.from_agent for item in signals),
        }
        agent_status: dict[str, dict[str, Any]] = {}
        for agent in sorted({str(item).strip() for item in agents if str(item).strip()}):
            record = self._status_tracker.get(agent)
            if record is not None:
                agent_status[agent] = record.model_dump(mode="json")

        return {
            "issue_key": issue,
            "agents": sorted(agent_status.keys()),
            "agent_status": agent_status,
            "sandboxes": [item.model_dump(mode="json") for item in sandboxes],
            "signals": [item.model_dump(mode="json") for item in signals],
            "events": [item.model_dump(mode="json") for item in events],
            "recovery": recovery,
        }
