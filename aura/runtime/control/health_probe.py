from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from ..ids import new_id, now_ts_ms
from ..mcp.config import load_mcp_config, mcp_stdio_errlog_context
from ..signal import SignalBus
from .agent_status import AgentState, AgentStatusTracker
from .policy import PolicyGate


class ProbeIssueKind(StrEnum):
    STUCK_SANDBOX = "stuck_sandbox"
    UNRESPONDED_SIGNAL = "unresponded_signal"
    COOLING_EXPIRED = "cooling_expired"
    EXCESSIVE_FAILURES = "excessive_failures"
    MCP_UNREACHABLE = "mcp_unreachable"


class ProbeIssue(BaseModel):
    kind: ProbeIssueKind
    agent_id: str | None = None
    sandbox_id: str | None = None
    signal_id: str | None = None
    issue_key: str | None = None
    detail: str = ""
    detected_at: int = Field(default_factory=now_ts_ms)


class ProbeReport(BaseModel):
    probe_id: str = Field(default_factory=lambda: new_id("probe"))
    probed_at_ms: int = Field(default_factory=now_ts_ms)
    issues: list[ProbeIssue] = Field(default_factory=list)
    agents_checked: int = 0
    healthy: bool = True


class HealthProbe:
    def __init__(
        self,
        *,
        project_root: Path,
        status_tracker: AgentStatusTracker,
        signal_bus: SignalBus,
        policy_gate: PolicyGate,
    ) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._status_tracker = status_tracker
        self._signal_bus = signal_bus
        self._policy_gate = policy_gate
        self._mcp_health_path = self._project_root / ".aura" / "state" / "control" / "mcp_health.json"
        self._mcp_health_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_mcp_health(self) -> dict[str, dict]:
        if not self._mcp_health_path.exists():
            return {}
        try:
            raw = json.loads(self._mcp_health_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            out[key] = dict(value)
        return out

    def _write_mcp_health(self, data: dict[str, dict]) -> None:
        self._mcp_health_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._mcp_health_path.with_suffix(self._mcp_health_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._mcp_health_path)

    def _record_mcp_health(self, *, server_name: str, ok: bool, detail: str) -> None:
        entries = self._load_mcp_health()
        entries[server_name] = {
            "ok": bool(ok),
            "detail": str(detail or ""),
            "checked_at_ms": now_ts_ms(),
        }
        self._write_mcp_health(entries)

    def probe(self) -> ProbeReport:
        policy = self._policy_gate.load_policy()
        now = now_ts_ms()
        records = self._status_tracker.refresh_all(sandbox_idle_timeout_ms=policy.sandbox_idle_timeout_ms)
        issues: list[ProbeIssue] = []

        for record in records:
            if record.state is AgentState.STUCK:
                if record.active_sandbox_ids:
                    for sandbox_id in record.active_sandbox_ids:
                        issues.append(
                            ProbeIssue(
                                kind=ProbeIssueKind.STUCK_SANDBOX,
                                agent_id=record.agent_id,
                                sandbox_id=sandbox_id,
                                issue_key=record.active_issue_keys[0] if record.active_issue_keys else None,
                                detail=(
                                    f"idle beyond {policy.sandbox_idle_timeout_ms}ms; "
                                    f"last_event_ts={record.last_event_ts}"
                                ),
                            )
                        )
                else:
                    issues.append(
                        ProbeIssue(
                            kind=ProbeIssueKind.STUCK_SANDBOX,
                            agent_id=record.agent_id,
                            detail=(
                                f"state=stuck without sandbox; "
                                f"idle_timeout_ms={policy.sandbox_idle_timeout_ms}"
                            ),
                        )
                    )

            if isinstance(record.cooling_until_ts, int) and record.cooling_until_ts <= now:
                issues.append(
                    ProbeIssue(
                        kind=ProbeIssueKind.COOLING_EXPIRED,
                        agent_id=record.agent_id,
                        detail=f"cooling_until_ts={record.cooling_until_ts} expired",
                    )
                )

            if (
                policy.max_failures_per_agent_24h > 0
                and record.failure_count_24h >= policy.max_failures_per_agent_24h
            ):
                issues.append(
                    ProbeIssue(
                        kind=ProbeIssueKind.EXCESSIVE_FAILURES,
                        agent_id=record.agent_id,
                        detail=(
                            f"failure_count_24h={record.failure_count_24h} "
                            f">= {policy.max_failures_per_agent_24h}"
                        ),
                    )
                )

        signals = self._signal_bus.query(limit=0, include_consumed=False, include_archive=False)
        for signal in signals:
            age_ms = now - signal.created_at
            if age_ms <= policy.signal_unresponded_timeout_ms:
                continue
            issues.append(
                ProbeIssue(
                    kind=ProbeIssueKind.UNRESPONDED_SIGNAL,
                    agent_id=signal.to_agent,
                    signal_id=signal.signal_id,
                    issue_key=signal.issue_key,
                    detail=f"age_ms={age_ms} > timeout_ms={policy.signal_unresponded_timeout_ms}",
                )
            )

        return ProbeReport(
            issues=issues,
            agents_checked=len(records),
            healthy=(len(issues) == 0),
        )

    def probe_agent(self, agent_id: str) -> ProbeReport:
        policy = self._policy_gate.load_policy()
        agent = str(agent_id or "").strip()
        if not agent:
            return ProbeReport(issues=[], agents_checked=0, healthy=True)

        now = now_ts_ms()
        record = self._status_tracker.refresh(agent, sandbox_idle_timeout_ms=policy.sandbox_idle_timeout_ms)
        issues: list[ProbeIssue] = []

        if record.state is AgentState.STUCK:
            for sandbox_id in record.active_sandbox_ids:
                issues.append(
                    ProbeIssue(
                        kind=ProbeIssueKind.STUCK_SANDBOX,
                        agent_id=record.agent_id,
                        sandbox_id=sandbox_id,
                        issue_key=record.active_issue_keys[0] if record.active_issue_keys else None,
                        detail=f"idle beyond {policy.sandbox_idle_timeout_ms}ms",
                    )
                )

        if isinstance(record.cooling_until_ts, int) and record.cooling_until_ts <= now:
            issues.append(
                ProbeIssue(
                    kind=ProbeIssueKind.COOLING_EXPIRED,
                    agent_id=record.agent_id,
                    detail=f"cooling_until_ts={record.cooling_until_ts} expired",
                )
            )

        if (
            policy.max_failures_per_agent_24h > 0
            and record.failure_count_24h >= policy.max_failures_per_agent_24h
        ):
            issues.append(
                ProbeIssue(
                    kind=ProbeIssueKind.EXCESSIVE_FAILURES,
                    agent_id=record.agent_id,
                    detail=(
                        f"failure_count_24h={record.failure_count_24h} "
                        f">= {policy.max_failures_per_agent_24h}"
                    ),
                )
            )

        signals = self._signal_bus.query(
            to_agent=agent,
            limit=0,
            include_consumed=False,
            include_archive=False,
        )
        for signal in signals:
            age_ms = now - signal.created_at
            if age_ms <= policy.signal_unresponded_timeout_ms:
                continue
            issues.append(
                ProbeIssue(
                    kind=ProbeIssueKind.UNRESPONDED_SIGNAL,
                    agent_id=signal.to_agent,
                    signal_id=signal.signal_id,
                    issue_key=signal.issue_key,
                    detail=f"age_ms={age_ms} > timeout_ms={policy.signal_unresponded_timeout_ms}",
                )
            )

        return ProbeReport(issues=issues, agents_checked=1, healthy=(len(issues) == 0))

    def probe_mcp(self, server_name: str) -> ProbeIssue | None:
        server = str(server_name or "").strip()
        if not server:
            issue = ProbeIssue(
                kind=ProbeIssueKind.MCP_UNREACHABLE,
                detail="server_name is required",
            )
            self._record_mcp_health(server_name="unknown", ok=False, detail=issue.detail)
            return issue

        try:
            cfg = load_mcp_config(project_root=self._project_root)
        except Exception as exc:
            issue = ProbeIssue(
                kind=ProbeIssueKind.MCP_UNREACHABLE,
                detail=f"mcp config load failed: {exc}",
            )
            self._record_mcp_health(server_name=server, ok=False, detail=issue.detail)
            return issue

        target = cfg.servers.get(server)
        if target is None:
            issue = ProbeIssue(
                kind=ProbeIssueKind.MCP_UNREACHABLE,
                detail=f"server not configured: {server}",
            )
            self._record_mcp_health(server_name=server, ok=False, detail=issue.detail)
            return issue
        if not target.enabled:
            issue = ProbeIssue(
                kind=ProbeIssueKind.MCP_UNREACHABLE,
                detail=f"server disabled: {server}",
            )
            self._record_mcp_health(server_name=server, ok=False, detail=issue.detail)
            return issue
        if not str(target.command or "").strip():
            issue = ProbeIssue(
                kind=ProbeIssueKind.MCP_UNREACHABLE,
                detail=f"command missing for server: {server}",
            )
            self._record_mcp_health(server_name=server, ok=False, detail=issue.detail)
            return issue

        try:
            from agno.tools.mcp.mcp import MCPTools
            from mcp import StdioServerParameters
            from mcp.client.stdio import get_default_environment
        except Exception as exc:
            issue = ProbeIssue(
                kind=ProbeIssueKind.MCP_UNREACHABLE,
                detail=f"MCP tooling unavailable: {exc}",
            )
            self._record_mcp_health(server_name=server, ok=False, detail=issue.detail)
            return issue

        async def _probe_once() -> None:
            params = StdioServerParameters(
                command=target.command,
                args=list(target.args or []),
                env={**get_default_environment(), **dict(target.env or {})},
                cwd=target.cwd,
            )
            toolkit = MCPTools(
                server_params=params,
                transport="stdio",
                timeout_seconds=5,
                tool_name_prefix=f"mcp__{server}__",
            )
            with mcp_stdio_errlog_context(project_root=self._project_root, server_name=server):
                async with toolkit as entered:
                    _ = entered.get_async_functions()

        try:
            asyncio.run(asyncio.wait_for(_probe_once(), timeout=5.0))
        except Exception as exc:
            issue = ProbeIssue(
                kind=ProbeIssueKind.MCP_UNREACHABLE,
                detail=f"probe failed: {exc}",
            )
            self._record_mcp_health(server_name=server, ok=False, detail=issue.detail)
            return issue

        self._record_mcp_health(server_name=server, ok=True, detail="ok")
        return None
