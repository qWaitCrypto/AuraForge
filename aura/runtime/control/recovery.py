from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..event_log import EventLog
from ..ids import new_id, now_ts_ms
from ..models.event_log import LogEvent, LogEventKind
from ..models.signal import SignalType
from ..sandbox import SandboxManager
from ..signal import SignalBus
from .agent_status import AgentStatusTracker
from .health_probe import ProbeIssue, ProbeIssueKind, ProbeReport
from .policy import PolicyGate


class RecoveryAction(StrEnum):
    KILL_SANDBOX = "kill_sandbox"
    RESEND_SIGNAL = "resend_signal"
    DEAD_LETTER = "dead_letter"
    CLEAR_COOLING = "clear_cooling"
    MANUAL_TAKEOVER = "manual_takeover"
    NOOP = "noop"


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    record_id: str
    ts_ms: int
    action: RecoveryAction
    outcome: str
    ok: bool
    operator: str
    issue_key: str | None = None
    agent_id: str | None = None
    sandbox_id: str | None = None
    signal_id: str | None = None
    old_signal_id: str | None = None
    new_signal_id: str | None = None
    reason: str | None = None
    error: str | None = None


class RecoveryManager:
    def __init__(
        self,
        *,
        project_root: Path,
        sandbox_manager: SandboxManager,
        signal_bus: SignalBus,
        status_tracker: AgentStatusTracker,
        policy_gate: PolicyGate,
        event_log: EventLog,
    ) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._sandbox_manager = sandbox_manager
        self._signal_bus = signal_bus
        self._status_tracker = status_tracker
        self._policy_gate = policy_gate
        self._event_log = event_log
        self._state_dir = self._project_root / ".aura" / "state" / "control"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._recovery_log_path = self._state_dir / "recovery_log.jsonl"
        self._dead_letters_path = self._state_dir / "dead_letters.jsonl"

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _record_event(
        self,
        *,
        action: RecoveryAction,
        ok: bool,
        summary: str,
        agent_id: str,
        issue_key: str | None = None,
        sandbox_id: str | None = None,
        refs: list[str] | None = None,
    ) -> None:
        self._event_log.record(
            LogEvent(
                event_id=new_id("evt"),
                session_id=f"recovery:{action.value}:{new_id('sess')}",
                agent_id=agent_id,
                sandbox_id=sandbox_id,
                issue_key=issue_key,
                kind=LogEventKind.TOOL_CALL,
                tool_name=f"recovery__{action.value}",
                tool_result_summary=summary,
                tool_ok=ok,
                external_refs=list(refs or []),
            )
        )

    def _emit_record(self, record: RecoveryRecord) -> RecoveryRecord:
        payload = asdict(record)
        payload["action"] = record.action.value
        self._append_jsonl(self._recovery_log_path, payload)
        return record

    def list_dead_letters(self) -> list[dict[str, Any]]:
        if not self._dead_letters_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self._dead_letters_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                out.append(item)
        return out

    def _load_resend_counts(self) -> dict[tuple[str, str], int]:
        if not self._recovery_log_path.exists():
            return {}
        counts: dict[tuple[str, str], int] = {}
        for line in self._recovery_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            if item.get("action") != RecoveryAction.RESEND_SIGNAL.value:
                continue
            issue_key = str(item.get("issue_key") or "").strip()
            agent_id = str(item.get("agent_id") or "").strip()
            if not issue_key or not agent_id:
                continue
            key = (issue_key, agent_id)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def auto_recover(self, report: ProbeReport) -> list[RecoveryRecord]:
        out: list[RecoveryRecord] = []
        resend_counts = self._load_resend_counts()
        for issue in report.issues:
            if issue.kind is ProbeIssueKind.STUCK_SANDBOX and issue.sandbox_id:
                out.append(
                    self.kill_sandbox(
                        issue.sandbox_id,
                        reason=issue.detail or "stuck_sandbox",
                        operator="auto",
                    )
                )
                continue

            if issue.kind is ProbeIssueKind.UNRESPONDED_SIGNAL and issue.signal_id:
                signal = self._signal_bus.find_signal(issue.signal_id)
                if signal is None:
                    out.append(
                        self._emit_record(
                            RecoveryRecord(
                                record_id=new_id("rcv"),
                                ts_ms=now_ts_ms(),
                                action=RecoveryAction.NOOP,
                                outcome="signal_not_found",
                                ok=False,
                                operator="auto",
                                signal_id=issue.signal_id,
                                issue_key=issue.issue_key,
                                agent_id=issue.agent_id,
                                reason=issue.detail or "unresponded_signal",
                            )
                        )
                    )
                    continue
                resend_key = (
                    str(signal.issue_key or issue.issue_key or "").strip(),
                    signal.to_agent,
                )
                resend_count = resend_counts.get(resend_key, 0)
                if resend_count >= 2:
                    out.append(
                        self.dead_letter(
                            signal.signal_id,
                            reason=issue.detail or "resend_limit_reached",
                            operator="auto",
                        )
                    )
                else:
                    record = self.resend_signal(
                        issue_key=resend_key[0],
                        agent_id=resend_key[1],
                        brief=signal.brief,
                        reason=issue.detail or "unresponded_signal",
                        old_signal_id=signal.signal_id,
                        operator="auto",
                    )
                    if record.ok:
                        resend_counts[resend_key] = resend_count + 1
                    out.append(record)
                continue

            if issue.kind is ProbeIssueKind.COOLING_EXPIRED and issue.agent_id:
                out.append(self.clear_cooling(issue.agent_id, operator="auto"))
                continue

            out.append(
                self._emit_record(
                    RecoveryRecord(
                        record_id=new_id("rcv"),
                        ts_ms=now_ts_ms(),
                        action=RecoveryAction.NOOP,
                        outcome=f"no_auto_action:{issue.kind.value}",
                        ok=True,
                        operator="auto",
                        issue_key=issue.issue_key,
                        agent_id=issue.agent_id,
                        sandbox_id=issue.sandbox_id,
                        signal_id=issue.signal_id,
                        reason=issue.detail or issue.kind.value,
                    )
                )
            )
        return out

    def kill_sandbox(self, sandbox_id: str, *, reason: str, operator: str = "manual") -> RecoveryRecord:
        sid = str(sandbox_id or "").strip()
        item = self._sandbox_manager.get(sid)
        agent_id = item.agent_id if item is not None else "operator"
        issue_key = item.issue_key if item is not None else None
        action = RecoveryAction.KILL_SANDBOX

        try:
            self._sandbox_manager.destroy(sid)
            policy = self._policy_gate.load_policy()
            if item is not None:
                self._status_tracker.mark_cooling(
                    item.agent_id,
                    duration_ms=policy.agent_cooldown_after_failure_ms,
                )
            record = RecoveryRecord(
                record_id=new_id("rcv"),
                ts_ms=now_ts_ms(),
                action=action,
                outcome="killed",
                ok=True,
                operator=operator,
                issue_key=issue_key,
                agent_id=agent_id,
                sandbox_id=sid,
                reason=reason,
            )
            self._record_event(
                action=action,
                ok=True,
                summary="sandbox killed",
                agent_id=agent_id,
                issue_key=issue_key,
                sandbox_id=sid,
                refs=[f"sandbox:{sid}"],
            )
            return self._emit_record(record)
        except Exception as exc:
            record = RecoveryRecord(
                record_id=new_id("rcv"),
                ts_ms=now_ts_ms(),
                action=action,
                outcome="kill_failed",
                ok=False,
                operator=operator,
                issue_key=issue_key,
                agent_id=agent_id,
                sandbox_id=sid,
                reason=reason,
                error=str(exc),
            )
            self._record_event(
                action=action,
                ok=False,
                summary=str(exc),
                agent_id=agent_id,
                issue_key=issue_key,
                sandbox_id=sid,
                refs=[f"sandbox:{sid}"],
            )
            return self._emit_record(record)

    def resend_signal(
        self,
        *,
        issue_key: str,
        agent_id: str,
        brief: str,
        reason: str,
        old_signal_id: str | None = None,
        operator: str = "manual",
    ) -> RecoveryRecord:
        issue = str(issue_key or "").strip()
        agent = str(agent_id or "").strip()
        text = str(brief or "").strip() or "wake"
        text = text[:200]
        old_id = str(old_signal_id or "").strip() or None
        action = RecoveryAction.RESEND_SIGNAL

        try:
            signal = self._signal_bus.send(
                from_agent="control.recovery",
                to_agent=agent,
                signal_type=SignalType.WAKE,
                brief=text,
                issue_key=issue or None,
                payload={"reason": reason, "old_signal_id": old_id},
            )
            if old_id is not None:
                try:
                    self._signal_bus.consume(old_id)
                except Exception:
                    pass
            record = RecoveryRecord(
                record_id=new_id("rcv"),
                ts_ms=now_ts_ms(),
                action=action,
                outcome="resent",
                ok=True,
                operator=operator,
                issue_key=issue or None,
                agent_id=agent,
                signal_id=signal.signal_id,
                old_signal_id=old_id,
                new_signal_id=signal.signal_id,
                reason=reason,
            )
            self._record_event(
                action=action,
                ok=True,
                summary=f"resent signal {signal.signal_id}",
                agent_id=agent,
                issue_key=issue or None,
                refs=[f"signal:{signal.signal_id}"] + ([f"signal:{old_id}"] if old_id else []),
            )
            return self._emit_record(record)
        except Exception as exc:
            record = RecoveryRecord(
                record_id=new_id("rcv"),
                ts_ms=now_ts_ms(),
                action=action,
                outcome="resend_failed",
                ok=False,
                operator=operator,
                issue_key=issue or None,
                agent_id=agent,
                old_signal_id=old_id,
                reason=reason,
                error=str(exc),
            )
            self._record_event(
                action=action,
                ok=False,
                summary=str(exc),
                agent_id=agent or "operator",
                issue_key=issue or None,
                refs=([f"signal:{old_id}"] if old_id else []),
            )
            return self._emit_record(record)

    def dead_letter(self, signal_id: str, *, reason: str, operator: str = "manual") -> RecoveryRecord:
        sid = str(signal_id or "").strip()
        action = RecoveryAction.DEAD_LETTER
        signal = self._signal_bus.find_signal(sid)
        issue_key = signal.issue_key if signal is not None else None
        agent_id = signal.to_agent if signal is not None else "operator"

        try:
            if signal is not None:
                self._append_jsonl(
                    self._dead_letters_path,
                    {
                        "signal": signal.model_dump(mode="json"),
                        "reason": reason,
                        "recorded_at_ms": now_ts_ms(),
                    },
                )
            self._signal_bus.consume(sid)
            record = RecoveryRecord(
                record_id=new_id("rcv"),
                ts_ms=now_ts_ms(),
                action=action,
                outcome="dead_lettered",
                ok=True,
                operator=operator,
                issue_key=issue_key,
                agent_id=agent_id,
                signal_id=sid,
                reason=reason,
            )
            self._record_event(
                action=action,
                ok=True,
                summary="signal moved to dead letters",
                agent_id=agent_id,
                issue_key=issue_key,
                refs=[f"signal:{sid}"],
            )
            return self._emit_record(record)
        except Exception as exc:
            record = RecoveryRecord(
                record_id=new_id("rcv"),
                ts_ms=now_ts_ms(),
                action=action,
                outcome="dead_letter_failed",
                ok=False,
                operator=operator,
                issue_key=issue_key,
                agent_id=agent_id,
                signal_id=sid,
                reason=reason,
                error=str(exc),
            )
            self._record_event(
                action=action,
                ok=False,
                summary=str(exc),
                agent_id=agent_id,
                issue_key=issue_key,
                refs=[f"signal:{sid}"],
            )
            return self._emit_record(record)

    def clear_cooling(self, agent_id: str, *, operator: str = "manual") -> RecoveryRecord:
        agent = str(agent_id or "").strip()
        action = RecoveryAction.CLEAR_COOLING
        try:
            self._status_tracker.clear_cooling(agent)
            record = RecoveryRecord(
                record_id=new_id("rcv"),
                ts_ms=now_ts_ms(),
                action=action,
                outcome="cooling_cleared",
                ok=True,
                operator=operator,
                agent_id=agent,
            )
            self._record_event(
                action=action,
                ok=True,
                summary="cooling cleared",
                agent_id=agent,
            )
            return self._emit_record(record)
        except Exception as exc:
            record = RecoveryRecord(
                record_id=new_id("rcv"),
                ts_ms=now_ts_ms(),
                action=action,
                outcome="clear_cooling_failed",
                ok=False,
                operator=operator,
                agent_id=agent,
                error=str(exc),
            )
            self._record_event(
                action=action,
                ok=False,
                summary=str(exc),
                agent_id=agent or "operator",
            )
            return self._emit_record(record)

    def manual_takeover(
        self,
        issue_key: str,
        *,
        new_agent_id: str,
        base_branch: str | None = None,
        operator: str = "manual",
        reason: str = "manual",
    ) -> RecoveryRecord:
        issue = str(issue_key or "").strip()
        new_agent = str(new_agent_id or "").strip()
        action = RecoveryAction.MANUAL_TAKEOVER
        if not issue or not new_agent:
            return self._emit_record(
                RecoveryRecord(
                    record_id=new_id("rcv"),
                    ts_ms=now_ts_ms(),
                    action=action,
                    outcome="invalid_input",
                    ok=False,
                    operator=operator,
                    issue_key=issue or None,
                    agent_id=new_agent or None,
                    reason=reason,
                    error="issue_key and new_agent_id are required",
                )
            )

        old_sandboxes = list(self._sandbox_manager.find_by_issue(issue))
        selected_base_branch = self._resolve_takeover_base_branch(
            requested=base_branch,
            old_sandboxes=old_sandboxes,
        )
        for sandbox in old_sandboxes:
            try:
                self._sandbox_manager.destroy(sandbox.sandbox_id)
            except Exception:
                pass

        sandbox = None
        try:
            sandbox = self._sandbox_manager.create(
                agent_id=new_agent,
                issue_key=issue,
                base_branch=selected_base_branch,
            )
            brief = f"Manual takeover for {issue}: {reason}".strip()[:200] or "manual takeover"
            signal = self._signal_bus.send(
                from_agent="control.recovery",
                to_agent=new_agent,
                signal_type=SignalType.TASK_ASSIGNED,
                brief=brief,
                issue_key=issue,
                sandbox_id=sandbox.sandbox_id,
                payload={"reason": reason, "operator": operator},
            )
            record = RecoveryRecord(
                record_id=new_id("rcv"),
                ts_ms=now_ts_ms(),
                action=action,
                outcome="takeover_assigned",
                ok=True,
                operator=operator,
                issue_key=issue,
                agent_id=new_agent,
                sandbox_id=sandbox.sandbox_id,
                signal_id=signal.signal_id,
                new_signal_id=signal.signal_id,
                reason=reason,
            )
            self._record_event(
                action=action,
                ok=True,
                summary="manual takeover assigned",
                agent_id=new_agent,
                issue_key=issue,
                sandbox_id=sandbox.sandbox_id,
                refs=[f"sandbox:{sandbox.sandbox_id}", f"signal:{signal.signal_id}"],
            )
            return self._emit_record(record)
        except Exception as exc:
            if sandbox is not None:
                try:
                    self._sandbox_manager.destroy(sandbox.sandbox_id)
                except Exception:
                    pass
            record = RecoveryRecord(
                record_id=new_id("rcv"),
                ts_ms=now_ts_ms(),
                action=action,
                outcome="takeover_failed",
                ok=False,
                operator=operator,
                issue_key=issue,
                agent_id=new_agent,
                sandbox_id=sandbox.sandbox_id if sandbox is not None else None,
                reason=reason,
                error=str(exc),
            )
            self._record_event(
                action=action,
                ok=False,
                summary=str(exc),
                agent_id=new_agent,
                issue_key=issue,
                sandbox_id=sandbox.sandbox_id if sandbox is not None else None,
            )
            return self._emit_record(record)

    def _resolve_takeover_base_branch(self, *, requested: str | None, old_sandboxes: list[Any]) -> str:
        explicit = str(requested or "").strip()
        if explicit:
            return explicit

        if old_sandboxes:
            base = str(getattr(old_sandboxes[0], "base_branch", "") or "").strip()
            if base:
                return base

        try:
            proc = subprocess.run(
                ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
                cwd=self._project_root,
                text=True,
                capture_output=True,
            )
            if proc.returncode == 0:
                ref = str(proc.stdout or "").strip()
                if ref.startswith("origin/") and len(ref) > len("origin/"):
                    return ref[len("origin/") :]
        except Exception:
            pass

        return "main"
