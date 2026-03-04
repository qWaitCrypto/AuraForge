from __future__ import annotations

import asyncio
import fnmatch
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .committee import COMMITTEE_AGENT_ID, CommitteeCoordinator
from .engine import Engine, ToolDecision, build_engine_for_session
from .event_bus import EventBus
from .event_log import EventLog, EventLogFileStore
from .ids import new_id, now_ts_ms
from .llm.config_io import load_model_config_layers_for_dir
from .models.agent_session import AgentSession, AgentSessionState
from .models.event_log import LogEvent, LogEventKind
from .models.signal import Signal, SignalType
from .project import RuntimePaths
from .signal import SignalBus
from .stores import FileApprovalStore, FileArtifactStore, FileEventLogStore, FileSessionStore
from .tools.runtime import ToolApprovalMode
from .protocol import Op, OpKind


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    poll_interval_s: float = 2.0
    idle_timeout_s: float = 300.0
    bid_check_interval_s: float = 120.0
    max_concurrent_agents: int = 5
    max_signals_per_agent: int = 0
    max_auto_approval_loops: int = 8
    op_timeout_s: float | None = 300.0


@dataclass(frozen=True, slots=True)
class RunnerApprovalPolicy:
    auto_approve: tuple[str, ...]
    auto_deny: tuple[str, ...]

    def decide(self, tool_name: str) -> str | None:
        name = str(tool_name or "").strip().lower()
        if not name:
            return "deny"
        for pattern in self.auto_deny:
            if fnmatch.fnmatch(name, pattern.lower()):
                return "deny"
        for pattern in self.auto_approve:
            if fnmatch.fnmatch(name, pattern.lower()):
                return "approve"
        return None

    @staticmethod
    def for_wake() -> "RunnerApprovalPolicy":
        return RunnerApprovalPolicy(
            auto_approve=(
                "mcp__*linear*",
                "signal__send",
                "signal__poll",
                "spec__query",
                "spec__get",
                "spec__list_assets",
                "spec__get_asset",
                "audit__query",
                "audit__refs",
            ),
            auto_deny=(
                "shell__run",
                "project__apply_*",
                "project__patch",
                "snapshot__rollback",
                "subagent__run",
            ),
        )

    @staticmethod
    def for_task_assigned() -> "RunnerApprovalPolicy":
        return RunnerApprovalPolicy(auto_approve=("*",), auto_deny=())

    @staticmethod
    def for_notify() -> "RunnerApprovalPolicy":
        return RunnerApprovalPolicy(
            auto_approve=(
                "mcp__*linear*",
                "signal__send",
                "signal__poll",
                "audit__query",
                "audit__refs",
                "project__read_text",
                "project__read_text_many",
                "project__list_dir",
                "project__glob",
                "project__search_text",
                "project__text_stats",
                "snapshot__list",
                "snapshot__diff",
                "snapshot__read_text",
            ),
            auto_deny=("shell__run", "project__apply_*", "project__patch", "snapshot__rollback"),
        )


EngineFactory = Callable[[str], Engine]


class AgentRunner:
    """
    Poll pending signals and drive one long-lived Engine session per agent.

    The runner guarantees in-process serial handling per `agent_id` while allowing
    different agents to run concurrently up to `max_concurrent_agents`.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        signal_bus: SignalBus,
        config: RunnerConfig | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._signal_bus = signal_bus
        self._config = config or RunnerConfig()

        self._paths = RuntimePaths.for_project(self._project_root)
        self._artifact_store = FileArtifactStore(self._paths.artifacts_dir)
        self._session_store = FileSessionStore(self._paths.sessions_dir)
        self._approval_store = FileApprovalStore(self._paths.state_dir / "approvals")
        self._event_log_store = FileEventLogStore(
            self._paths.events_dir,
            artifact_store=self._artifact_store,
            session_store=self._session_store,
        )
        self._event_bus = EventBus(event_log_store=self._event_log_store)
        self._audit_log = EventLog(store=EventLogFileStore(project_root=self._project_root))
        self._committee = CommitteeCoordinator(project_root=self._project_root, signal_bus=self._signal_bus)

        self._state_dir = self._paths.state_dir / "runner"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_path = self._state_dir / "sessions.json"
        self._metrics_path = self._state_dir / "metrics.jsonl"

        self._sessions: dict[str, AgentSession] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._queues: dict[str, asyncio.Queue[Signal]] = {}
        self._engines: dict[str, Engine] = {}
        self._inflight_signal_ids: set[str] = set()

        self._engine_factory = engine_factory or self._build_engine_for_session
        self._running = False
        self._next_bid_check_ts_ms = 0

    async def start(self) -> None:
        self._running = True
        self._next_bid_check_ts_ms = 0
        self._append_metric("runner_started", {"pid": os.getpid()})
        self._write_sessions_snapshot()
        try:
            while self._running:
                await self._poll_and_dispatch()
                self._emit_periodic_bid_check()
                await self._reap_finished_sessions()
                await asyncio.sleep(max(0.1, float(self._config.poll_interval_s)))
        finally:
            await self._shutdown_all()
            self._append_metric("runner_stopped", {"pid": os.getpid()})

    async def stop(self) -> None:
        self._running = False

    def list_sessions(self) -> list[AgentSession]:
        items = list(self._sessions.values())
        items.sort(key=lambda item: item.started_at)
        return items

    def get_session(self, agent_id: str) -> AgentSession | None:
        return self._sessions.get(str(agent_id or "").strip())

    def sessions_snapshot(self) -> dict[str, Any]:
        return {
            "updated_at": now_ts_ms(),
            "count": len(self._sessions),
            "sessions": [item.model_dump(mode="json") for item in self.list_sessions()],
        }

    def send_wake(self, *, agent_id: str, issue_key: str, brief: str, from_agent: str = "runner") -> Signal:
        return self._signal_bus.send(
            from_agent=str(from_agent or "runner").strip() or "runner",
            to_agent=str(agent_id or "").strip(),
            signal_type=SignalType.WAKE,
            brief=str(brief or "wake").strip()[:200] or "wake",
            issue_key=str(issue_key or "").strip() or None,
        )

    def _emit_periodic_bid_check(self) -> None:
        interval = float(self._config.bid_check_interval_s)
        if interval <= 0:
            return
        now = now_ts_ms()
        if self._next_bid_check_ts_ms <= 0:
            self._next_bid_check_ts_ms = now + int(interval * 1000)
            return
        if now < self._next_bid_check_ts_ms:
            return
        self._next_bid_check_ts_ms = now + int(interval * 1000)
        signal = self._signal_bus.send(
            from_agent="runner.timer",
            to_agent=COMMITTEE_AGENT_ID,
            signal_type=SignalType.NOTIFY,
            brief="Periodic bid check",
            payload={
                "type": "bid_check",
                "auto_evaluate": True,
                "fetch_linear_comments": True,
            },
        )
        self._append_metric(
            "bid_check_emitted",
            {
                "signal_id": signal.signal_id,
                "interval_s": interval,
            },
        )

    async def _poll_and_dispatch(self) -> None:
        signals = self._signal_bus.query(
            include_consumed=False,
            include_archive=False,
            limit=0,
        )
        if not signals:
            return

        for signal in signals:
            if not self._running:
                return
            if signal.signal_id in self._inflight_signal_ids:
                continue
            enqueued = await self._enqueue_signal(signal)
            if enqueued:
                self._inflight_signal_ids.add(signal.signal_id)

    async def _enqueue_signal(self, signal: Signal) -> bool:
        agent_id = str(signal.to_agent or "").strip()
        if not agent_id:
            return False

        if agent_id not in self._tasks or self._tasks[agent_id].done():
            ok = await self._start_agent_session(agent_id=agent_id)
            if not ok:
                return False

        queue = self._queues.get(agent_id)
        if queue is None:
            return False
        await queue.put(signal)
        self._append_metric(
            "signal_enqueued",
            {
                "agent_id": agent_id,
                "signal_id": signal.signal_id,
                "signal_type": signal.signal_type.value,
                "issue_key": signal.issue_key,
            },
        )
        self._write_sessions_snapshot()
        return True

    async def _start_agent_session(self, *, agent_id: str) -> bool:
        active = sum(1 for task in self._tasks.values() if not task.done())
        if agent_id not in self._tasks and active >= int(self._config.max_concurrent_agents):
            self._append_metric("agent_start_skipped", {"agent_id": agent_id, "reason": "max_concurrent_agents"})
            return False

        role = "integrator" if agent_id == "committee" else "worker"
        session_id = self._session_store.create_session(
            {
                "project_ref": str(self._project_root),
                "mode": "agent_runner",
                "agent_id": agent_id,
                "role": role,
                "tool_approval_mode": ToolApprovalMode.STANDARD.value,
                "llm_streaming": True,
            }
        )

        try:
            engine = self._engine_factory(session_id)
        except Exception as exc:
            self._append_metric("agent_start_failed", {"agent_id": agent_id, "error": str(exc)})
            return False

        queue: asyncio.Queue[Signal] = self._queues.get(agent_id) or asyncio.Queue()
        self._queues[agent_id] = queue

        session = AgentSession(
            session_id=session_id,
            agent_id=agent_id,
            state=AgentSessionState.STARTING,
            pid=os.getpid(),
        )
        self._sessions[agent_id] = session
        self._engines[agent_id] = engine

        task = asyncio.create_task(self._agent_loop(agent_id=agent_id), name=f"agent_runner:{agent_id}")
        self._tasks[agent_id] = task

        session.state = AgentSessionState.RUNNING
        session.last_active_at = now_ts_ms()
        self._record_session_event(session=session, kind=LogEventKind.SESSION_START, summary="agent session started")
        self._append_metric("agent_started", {"agent_id": agent_id, "session_id": session_id})
        self._write_sessions_snapshot()
        return True

    async def _agent_loop(self, *, agent_id: str) -> None:
        session = self._sessions.get(agent_id)
        engine = self._engines.get(agent_id)
        queue = self._queues.get(agent_id)
        if session is None or engine is None or queue is None:
            return

        idle_timeout = -1.0 if agent_id == COMMITTEE_AGENT_ID else float(self._config.idle_timeout_s)

        try:
            while self._running:
                try:
                    if idle_timeout > 0:
                        signal = await asyncio.wait_for(queue.get(), timeout=idle_timeout)
                    else:
                        signal = await queue.get()
                except asyncio.TimeoutError:
                    session.state = AgentSessionState.IDLE
                    self._append_metric(
                        "agent_idle_timeout",
                        {"agent_id": agent_id, "session_id": session.session_id, "idle_timeout_s": idle_timeout},
                    )
                    break

                session.state = AgentSessionState.RUNNING
                session.current_signal_id = signal.signal_id
                session.current_issue_key = signal.issue_key
                session.sandbox_id = signal.sandbox_id
                session.last_active_at = now_ts_ms()
                self._write_sessions_snapshot()
                handled_ok = False
                should_consume = False

                try:
                    should_consume = await self._handle_signal(
                        agent_id=agent_id,
                        session=session,
                        engine=engine,
                        signal=signal,
                    )
                    handled_ok = True
                except Exception as exc:
                    self._append_metric(
                        "signal_handle_failed",
                        {
                            "agent_id": agent_id,
                            "session_id": session.session_id,
                            "signal_id": signal.signal_id,
                            "error": str(exc),
                        },
                    )
                finally:
                    if handled_ok and should_consume:
                        try:
                            self._signal_bus.consume(signal.signal_id)
                        except Exception as exc:
                            self._append_metric(
                                "signal_consume_failed",
                                {
                                    "agent_id": agent_id,
                                    "session_id": session.session_id,
                                    "signal_id": signal.signal_id,
                                    "error": str(exc),
                                },
                            )
                    self._inflight_signal_ids.discard(signal.signal_id)
                    session.signals_processed += 1
                    session.last_active_at = now_ts_ms()
                    session.current_signal_id = None
                    session.current_issue_key = None
                    session.sandbox_id = None
                    session.state = AgentSessionState.IDLE
                    queue.task_done()
                    self._write_sessions_snapshot()

                max_signals = int(self._config.max_signals_per_agent)
                if max_signals > 0 and session.signals_processed >= max_signals:
                    self._append_metric(
                        "agent_rotation",
                        {
                            "agent_id": agent_id,
                            "session_id": session.session_id,
                            "signals_processed": session.signals_processed,
                        },
                    )
                    break
        finally:
            session.state = AgentSessionState.STOPPED
            session.last_active_at = now_ts_ms()
            self._record_session_event(session=session, kind=LogEventKind.SESSION_END, summary="agent session stopped")

            self._engines.pop(agent_id, None)
            self._tasks.pop(agent_id, None)
            self._sessions.pop(agent_id, None)
            if queue.empty():
                self._queues.pop(agent_id, None)
            self._write_sessions_snapshot()

    async def _handle_signal(self, *, agent_id: str, session: AgentSession, engine: Engine, signal: Signal) -> bool:
        if agent_id == COMMITTEE_AGENT_ID:
            try:
                decision = self._committee.handle_signal(signal)
            except Exception as exc:
                self._append_metric(
                    "committee_signal_failed",
                    {
                        "agent_id": agent_id,
                        "session_id": session.session_id,
                        "signal_id": signal.signal_id,
                        "signal_type": signal.signal_type.value,
                        "error": str(exc),
                    },
                )
            else:
                if bool(decision.get("handled")):
                    self._append_metric(
                        "committee_signal_processed",
                        {
                            "agent_id": agent_id,
                            "session_id": session.session_id,
                            "signal_id": signal.signal_id,
                            "signal_type": signal.signal_type.value,
                            "decision": decision,
                        },
                    )
                    return True

        # Update metadata before engine.arun so ContextBuilder can resolve `signal_id`
        # from the session record for this turn's prompt assembly.
        self._session_store.update_session(
            session.session_id,
            {
                "signal_id": signal.signal_id,
                "signal_type": signal.signal_type.value,
                "signal_from_agent": signal.from_agent,
                "signal_to_agent": signal.to_agent,
                "signal_brief": signal.brief,
                "signal_payload": dict(signal.payload or {}),
                "issue_key": signal.issue_key,
                "sandbox_id": signal.sandbox_id,
            },
        )

        if engine.tool_runtime is not None:
            if signal.signal_type is SignalType.TASK_ASSIGNED:
                engine.tool_runtime.set_approval_mode(ToolApprovalMode.TRUSTED)
            else:
                engine.tool_runtime.set_approval_mode(ToolApprovalMode.STANDARD)

        text = self._signal_to_text(signal)
        op = Op(
            kind=OpKind.CHAT.value,
            payload={"text": text},
            session_id=session.session_id,
            request_id=new_id("req"),
            timestamp=now_ts_ms(),
            turn_id=new_id("turn"),
        )

        run = await engine.arun(op, timeout_s=self._config.op_timeout_s)
        policy = self._approval_policy_for_signal(signal)
        loops = 0
        while run.status == "needs_approval":
            if not run.pending_tools:
                break
            decisions: list[ToolDecision] = []
            unresolved_pending: list[str] = []
            for pending in run.pending_tools:
                action = policy.decide(pending.tool_name)
                if action == "approve":
                    decisions.append(ToolDecision(tool_call_id=pending.tool_call_id, decision="approve"))
                    continue
                if action == "deny":
                    note = (
                        f"Runner automatic policy denied tool '{pending.tool_name}' during "
                        f"{signal.signal_type.value} phase."
                    )
                    decisions.append(
                        ToolDecision(
                            tool_call_id=pending.tool_call_id,
                            decision="deny",
                            note=note,
                        )
                    )
                    continue
                unresolved_pending.append(pending.tool_name)

            if decisions:
                run = await engine.continue_run(
                    run_id=run.run_id,
                    decisions=decisions,
                    timeout_s=self._config.op_timeout_s,
                )
            if unresolved_pending:
                self._append_metric(
                    "approval_manual_required",
                    {
                        "agent_id": agent_id,
                        "session_id": session.session_id,
                        "signal_id": signal.signal_id,
                        "run_id": run.run_id,
                        "pending_tools": unresolved_pending,
                    },
                )
                break
            loops += 1
            if loops >= int(self._config.max_auto_approval_loops):
                self._append_metric(
                    "approval_loop_guard",
                    {
                        "agent_id": agent_id,
                        "session_id": session.session_id,
                        "signal_id": signal.signal_id,
                        "run_id": run.run_id,
                    },
                )
                break

        self._append_metric(
            "signal_handled",
            {
                "agent_id": agent_id,
                "session_id": session.session_id,
                "signal_id": signal.signal_id,
                "signal_type": signal.signal_type.value,
                "status": run.status,
                "run_id": run.run_id,
                "error": run.error,
            },
        )
        return True

    async def _reap_finished_sessions(self) -> None:
        for agent_id, task in list(self._tasks.items()):
            if task.done():
                try:
                    _ = task.result()
                except Exception as exc:
                    self._append_metric("agent_task_failed", {"agent_id": agent_id, "error": str(exc)})
                # Normal cleanup happens in `_agent_loop` finally.
                self._tasks.pop(agent_id, None)
        self._write_sessions_snapshot()

    async def _shutdown_all(self) -> None:
        self._running = False
        for session in self._sessions.values():
            session.state = AgentSessionState.STOPPING
            session.last_active_at = now_ts_ms()
        self._write_sessions_snapshot()

        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        self._tasks.clear()
        self._engines.clear()
        self._sessions.clear()
        self._queues.clear()
        self._inflight_signal_ids.clear()
        self._write_sessions_snapshot()

    def _build_engine_for_session(self, session_id: str) -> Engine:
        layers = load_model_config_layers_for_dir(self._project_root, require_project=True)
        model_config = layers.merged()
        engine = build_engine_for_session(
            project_root=self._project_root,
            session_id=session_id,
            event_bus=self._event_bus,
            session_store=self._session_store,
            event_log_store=self._event_log_store,
            artifact_store=self._artifact_store,
            approval_store=self._approval_store,
            model_config=model_config,
            system_prompt=None,
            tools_enabled=True,
            max_tool_turns=30,
        )
        engine.load_history_from_events()
        engine.apply_memory_summary_retention()
        return engine

    @staticmethod
    def _signal_to_text(signal: Signal) -> str:
        issue_key = signal.issue_key or "UNKNOWN"
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        if signal.signal_type is SignalType.WAKE:
            if (
                str(signal.to_agent or "").strip() == COMMITTEE_AGENT_ID
                and str(payload.get("type") or "").strip().lower() == "project_request"
            ):
                goal = str(payload.get("goal") or signal.brief or "project request").strip()
                return (
                    "Project request handed off to Committee.\n"
                    f"Goal: {goal}\n"
                    "Decompose into executable tasks, prepare issues, and wake matching candidates."
                )
            return (
                f"Wake signal received for issue {issue_key}.\n"
                "Read the issue in Linear, evaluate fit against your capabilities, "
                "and place a bid comment if you are a strong fit."
            )
        if signal.signal_type is SignalType.TASK_ASSIGNED:
            sandbox = signal.sandbox_id or ""
            return (
                f"Task assigned: {issue_key}.\n"
                f"Sandbox: {sandbox}.\n"
                "Work in the assigned sandbox, produce concrete changes, and notify committee when done."
            )
        return f"Notification received: {signal.brief}"

    @staticmethod
    def _approval_policy_for_signal(signal: Signal) -> RunnerApprovalPolicy:
        if signal.signal_type is SignalType.WAKE:
            return RunnerApprovalPolicy.for_wake()
        if signal.signal_type is SignalType.TASK_ASSIGNED:
            return RunnerApprovalPolicy.for_task_assigned()
        return RunnerApprovalPolicy.for_notify()

    def _write_sessions_snapshot(self) -> None:
        payload = self.sessions_snapshot()
        tmp = self._sessions_path.with_suffix(".json.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._sessions_path)

    def _append_metric(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "event": str(event or "runner"),
            "ts_ms": now_ts_ms(),
            "pid": os.getpid(),
            "payload": dict(payload or {}),
        }
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _record_session_event(self, *, session: AgentSession, kind: LogEventKind, summary: str) -> None:
        try:
            self._audit_log.record(
                LogEvent(
                    event_id=new_id("evt"),
                    session_id=session.session_id,
                    agent_id=session.agent_id,
                    sandbox_id=session.sandbox_id,
                    issue_key=session.current_issue_key,
                    kind=kind,
                    tool_name="agent_runner",
                    tool_result_summary=summary,
                    tool_ok=True,
                    external_refs=[f"runner_session:{session.session_id}"],
                )
            )
        except Exception:
            return


def load_runner_sessions_snapshot(*, project_root: Path) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    path = root / ".aura" / "state" / "runner" / "sessions.json"
    if not path.exists():
        return {"updated_at": None, "count": 0, "sessions": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"updated_at": None, "count": 0, "sessions": []}
    if not isinstance(raw, dict):
        return {"updated_at": None, "count": 0, "sessions": []}
    sessions = raw.get("sessions") if isinstance(raw.get("sessions"), list) else []
    return {
        "updated_at": raw.get("updated_at"),
        "count": int(raw.get("count") or len(sessions)),
        "sessions": sessions,
    }
