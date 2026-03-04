from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from aura.cli import EXIT_OK, main
from aura.runtime.agent_runner import AgentRunner, RunnerApprovalPolicy, RunnerConfig
from aura.runtime.control_hub import ControlHub
from aura.runtime.engine import PendingToolCall, RunResult
from aura.runtime.engine_agno_async import AgnoAsyncEngine
from aura.runtime.event_bus import EventBus
from aura.runtime.llm.config import ModelConfig
from aura.runtime.models.signal import SignalType
from aura.runtime.project import RuntimePaths
from aura.runtime.signal import SignalBus, SignalStore
from aura.runtime.stores import FileApprovalStore, FileArtifactStore, FileEventLogStore, FileSessionStore
from aura.runtime.tools.runtime import ToolApprovalMode


class _FakeToolRuntime:
    def __init__(self) -> None:
        self._mode = ToolApprovalMode.STANDARD

    def set_approval_mode(self, mode: ToolApprovalMode) -> None:
        self._mode = mode

    def get_approval_mode(self) -> ToolApprovalMode:
        return self._mode


class _FakeEngine:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.tool_runtime = _FakeToolRuntime()
        self.handled_payloads: list[dict] = []

    async def arun(self, op, *, timeout_s=None, cancel=None):  # noqa: ANN001
        del timeout_s, cancel
        self.handled_payloads.append(dict(op.payload))
        return RunResult(status="completed", run_id=op.request_id, session_id=op.session_id)

    async def continue_run(self, *, run_id, decisions, timeout_s=None, cancel=None):  # noqa: ANN001
        del run_id, decisions, timeout_s, cancel
        raise AssertionError("continue_run should not be called for completed fake runs")


class _NeedsApprovalEngine(_FakeEngine):
    def __init__(self, session_id: str, *, pending_tool_name: str) -> None:
        super().__init__(session_id)
        self.pending_tool_name = pending_tool_name
        self.continue_calls: int = 0

    async def arun(self, op, *, timeout_s=None, cancel=None):  # noqa: ANN001
        del op, timeout_s, cancel
        return RunResult(
            status="needs_approval",
            run_id="run_approval",
            session_id=self.session_id,
            approval_id="apr_1",
            pending_tools=[
                PendingToolCall(
                    tool_call_id="tc_1",
                    tool_name=self.pending_tool_name,
                    args={"path": "README.md"},
                )
            ],
        )

    async def continue_run(self, *, run_id, decisions, timeout_s=None, cancel=None):  # noqa: ANN001
        del run_id, timeout_s, cancel
        self.continue_calls += 1
        if decisions and decisions[0].decision == "deny":
            return RunResult(status="completed", run_id="run_approval", session_id=self.session_id)
        return RunResult(
            status="needs_approval",
            run_id="run_approval",
            session_id=self.session_id,
            approval_id="apr_1",
            pending_tools=[
                PendingToolCall(
                    tool_call_id="tc_1",
                    tool_name=self.pending_tool_name,
                    args={"path": "README.md"},
                )
            ],
        )


class _FailingEngine(_FakeEngine):
    async def arun(self, op, *, timeout_s=None, cancel=None):  # noqa: ANN001
        del op, timeout_s, cancel
        raise RuntimeError("simulated engine failure")


class _SlowEngine(_FakeEngine):
    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.calls = 0

    async def arun(self, op, *, timeout_s=None, cancel=None):  # noqa: ANN001
        del timeout_s, cancel
        self.calls += 1
        await asyncio.sleep(0.25)
        return await super().arun(op, timeout_s=None, cancel=None)


def _build_engine(project_root: Path, *, session_meta: dict | None = None) -> tuple[AgnoAsyncEngine, str]:
    session_store = FileSessionStore(project_root / ".aura" / "sessions")
    artifact_store = FileArtifactStore(project_root / ".aura" / "artifacts")
    event_store = FileEventLogStore(project_root / ".aura" / "events", artifact_store=artifact_store, session_store=session_store)
    approval_store = FileApprovalStore(project_root / ".aura" / "approvals")

    base_meta = {"mode": "chat", "project_ref": str(project_root), "agent_id": "python-pro"}
    if isinstance(session_meta, dict):
        base_meta.update(session_meta)
    session_id = session_store.create_session(base_meta)

    engine = AgnoAsyncEngine(
        project_root=project_root,
        session_id=session_id,
        event_bus=EventBus(),
        session_store=session_store,
        event_log_store=event_store,
        artifact_store=artifact_store,
        approval_store=approval_store,
        model_config=ModelConfig(),
        tools_enabled=True,
    )
    return engine, session_id


def test_runner_approval_policy_wake_defaults() -> None:
    policy = RunnerApprovalPolicy.for_wake()
    assert policy.decide("mcp__linear_abcd__list_issues") == "approve"
    assert policy.decide("shell__run") == "deny"
    assert policy.decide("project__read_text") is None


def test_engine_injects_signal_into_context_builder(tmp_path: Path) -> None:
    engine, session_id = _build_engine(tmp_path)
    sig = engine.signal_bus.send(
        from_agent="committee",
        to_agent="python-pro",
        signal_type=SignalType.WAKE,
        brief="check issue",
        issue_key="PROJ-P1",
    )
    engine.session_store.update_session(session_id, {"signal_id": sig.signal_id})

    req = engine._build_request()
    assert "Wake signal received." in req.system
    assert "Issue: PROJ-P1" in req.system


def test_agent_runner_consumes_signal_and_handles_once(tmp_path: Path) -> None:
    paths = RuntimePaths.for_project(tmp_path)
    paths.system_dir.mkdir(parents=True, exist_ok=True)

    from aura.runtime.signal import SignalBus, SignalStore

    bus = SignalBus(store=SignalStore(project_root=tmp_path))
    created: dict[str, _FakeEngine] = {}

    def _factory(session_id: str):
        eng = _FakeEngine(session_id)
        created[session_id] = eng
        return eng

    runner = AgentRunner(
        project_root=tmp_path,
        signal_bus=bus,
        engine_factory=_factory,
    )

    sent = bus.send(
        from_agent="tester",
        to_agent="python-pro",
        signal_type=SignalType.WAKE,
        brief="please bid",
        issue_key="PROJ-P2",
    )

    async def _run_once() -> None:
        task = asyncio.create_task(runner.start())
        await asyncio.sleep(0.4)
        await runner.stop()
        await task

    asyncio.run(_run_once())

    all_signals = bus.query(include_consumed=True, include_archive=True, limit=0)
    hit = [item for item in all_signals if item.signal_id == sent.signal_id]
    assert hit and hit[0].consumed is True

    metrics_path = tmp_path / ".aura" / "state" / "runner" / "metrics.jsonl"
    assert metrics_path.exists()
    metric_lines = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(item.get("event") == "signal_handled" for item in metric_lines)


def test_runner_unknown_pending_tool_keeps_needs_approval(tmp_path: Path) -> None:
    bus = SignalBus(store=SignalStore(project_root=tmp_path))
    created: dict[str, _NeedsApprovalEngine] = {}

    def _factory(session_id: str):
        engine = _NeedsApprovalEngine(session_id, pending_tool_name="project__read_text")
        created[session_id] = engine
        return engine

    runner = AgentRunner(project_root=tmp_path, signal_bus=bus, engine_factory=_factory)
    bus.send(
        from_agent="tester",
        to_agent="python-pro",
        signal_type=SignalType.WAKE,
        brief="approval needed",
        issue_key="PROJ-P4",
    )

    async def _run_once() -> None:
        task = asyncio.create_task(runner.start())
        await asyncio.sleep(0.4)
        await runner.stop()
        await task

    asyncio.run(_run_once())

    assert created
    assert all(engine.continue_calls == 0 for engine in created.values())
    all_signals = bus.query(include_consumed=True, include_archive=True, limit=0)
    hit = [item for item in all_signals if item.issue_key == "PROJ-P4"]
    assert hit and hit[0].consumed is True

    metrics_path = tmp_path / ".aura" / "state" / "runner" / "metrics.jsonl"
    metric_lines = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(item.get("event") == "approval_manual_required" for item in metric_lines)


def test_runner_failed_handle_does_not_consume_signal(tmp_path: Path) -> None:
    bus = SignalBus(store=SignalStore(project_root=tmp_path))

    def _factory(session_id: str):
        return _FailingEngine(session_id)

    runner = AgentRunner(project_root=tmp_path, signal_bus=bus, engine_factory=_factory)
    sent = bus.send(
        from_agent="tester",
        to_agent="python-pro",
        signal_type=SignalType.WAKE,
        brief="will fail",
        issue_key="PROJ-P5",
    )

    async def _run_once() -> None:
        task = asyncio.create_task(runner.start())
        await asyncio.sleep(0.35)
        await runner.stop()
        await task

    asyncio.run(_run_once())

    all_signals = bus.query(include_consumed=True, include_archive=True, limit=0)
    hit = [item for item in all_signals if item.signal_id == sent.signal_id]
    assert hit and hit[0].consumed is False


def test_runner_config_default_timeout_guard() -> None:
    cfg = RunnerConfig()
    assert cfg.op_timeout_s == 300.0


def test_runner_inflight_dedup_avoids_duplicate_enqueue(tmp_path: Path) -> None:
    bus = SignalBus(store=SignalStore(project_root=tmp_path))
    created: dict[str, _SlowEngine] = {}

    def _factory(session_id: str):
        engine = _SlowEngine(session_id)
        created[session_id] = engine
        return engine

    runner = AgentRunner(
        project_root=tmp_path,
        signal_bus=bus,
        engine_factory=_factory,
        config=RunnerConfig(poll_interval_s=0.05, idle_timeout_s=0.6),
    )
    bus.send(
        from_agent="tester",
        to_agent="python-pro",
        signal_type=SignalType.WAKE,
        brief="dedup check",
        issue_key="PROJ-P6",
    )

    async def _run_once() -> None:
        task = asyncio.create_task(runner.start())
        await asyncio.sleep(0.55)
        await runner.stop()
        await task

    asyncio.run(_run_once())

    assert created
    assert sum(engine.calls for engine in created.values()) == 1


def test_control_hub_stop_running_writes_stop_file(tmp_path: Path, monkeypatch) -> None:
    pid_path = tmp_path / ".aura" / "control_hub.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text('{"pid": 4242, "started_at": 1, "project_root": "x"}\n', encoding="utf-8")

    called: dict[str, int] = {}

    def _fake_kill(pid: int, sig: int) -> None:
        called["pid"] = pid
        called["sig"] = int(sig)

    monkeypatch.setattr(os, "kill", _fake_kill)

    ok = ControlHub.stop_running(tmp_path)
    assert ok is True
    assert (tmp_path / ".aura" / "control_hub.stop").exists()
    if os.name != "nt":
        assert called.get("pid") == 4242


def test_phase1_cli_runner_and_daemon_status(tmp_path: Path, monkeypatch, capsys) -> None:
    assert main(["init", str(tmp_path)]) == EXIT_OK
    monkeypatch.chdir(tmp_path)

    rc_wake = main(
        [
            "runner",
            "wake",
            "--agent-id",
            "python-pro",
            "--issue-key",
            "PROJ-P3",
            "--brief",
            "wake now",
        ]
    )
    out_wake = capsys.readouterr().out
    assert rc_wake == EXIT_OK
    assert '"ok": true' in out_wake
    assert '"signal"' in out_wake

    rc_status = main(["daemon", "status"])
    out_status = capsys.readouterr().out
    assert rc_status == EXIT_OK
    assert '"running"' in out_status
    assert '"runner"' in out_status
