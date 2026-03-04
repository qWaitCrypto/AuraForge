from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aura.cli import EXIT_OK, main
from aura.runtime.agent_runner import AgentRunner, RunnerApprovalPolicy
from aura.runtime.engine import RunResult
from aura.runtime.engine_agno_async import AgnoAsyncEngine
from aura.runtime.event_bus import EventBus
from aura.runtime.llm.config import ModelConfig
from aura.runtime.models.signal import SignalType
from aura.runtime.project import RuntimePaths
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
