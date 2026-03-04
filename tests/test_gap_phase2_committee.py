from __future__ import annotations

import asyncio
from pathlib import Path

from aura.runtime.agent_runner import AgentRunner
from aura.runtime.committee import CommitteeCoordinator
from aura.runtime.engine import RunResult
from aura.runtime.engine_agno_async import AgnoAsyncEngine
from aura.runtime.event_bus import EventBus
from aura.runtime.llm.config import ModelConfig
from aura.runtime.models.signal import SignalType
from aura.runtime.signal import SignalBus, SignalStore
from aura.runtime.stores import FileApprovalStore, FileArtifactStore, FileEventLogStore, FileSessionStore
from aura.runtime.tools.committee_tools import CommitteeSubmitTool
from aura.runtime.tools.runtime import ToolApprovalMode


class _FakeToolRuntime:
    def __init__(self) -> None:
        self._mode = ToolApprovalMode.STANDARD

    def set_approval_mode(self, mode: ToolApprovalMode) -> None:
        self._mode = mode


class _FakeEngine:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.tool_runtime = _FakeToolRuntime()
        self.payloads: list[dict] = []

    async def arun(self, op, *, timeout_s=None, cancel=None):  # noqa: ANN001
        del timeout_s, cancel
        self.payloads.append(dict(op.payload))
        return RunResult(status="completed", run_id=op.request_id, session_id=op.session_id)

    async def continue_run(self, *, run_id, decisions, timeout_s=None, cancel=None):  # noqa: ANN001
        del run_id, decisions, timeout_s, cancel
        raise AssertionError("continue_run should not be called for completed fake runs")


def _build_engine(project_root: Path, *, session_meta: dict | None = None) -> tuple[AgnoAsyncEngine, str]:
    session_store = FileSessionStore(project_root / ".aura" / "sessions")
    artifact_store = FileArtifactStore(project_root / ".aura" / "artifacts")
    event_store = FileEventLogStore(project_root / ".aura" / "events", artifact_store=artifact_store, session_store=session_store)
    approval_store = FileApprovalStore(project_root / ".aura" / "approvals")
    base_meta = {"mode": "chat", "project_ref": str(project_root)}
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


def test_committee_submit_tool_emits_project_request_signal(tmp_path: Path) -> None:
    bus = SignalBus(store=SignalStore(project_root=tmp_path))
    tool = CommitteeSubmitTool(signal_bus=bus)

    result = tool.execute(
        args={
            "goal": "Refactor auth module",
            "context": "Current auth implementation is spread across legacy files.",
            "constraints": ["Do not break login API"],
            "priority": "high",
            "references": ["auth/legacy.py"],
            "candidate_agents": ["market_fused__backend-developer"],
        },
        project_root=tmp_path,
    )

    assert result["status"] == "submitted"
    signals = bus.poll(to_agent="committee", unconsumed_only=True, limit=10)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type is SignalType.WAKE
    assert signal.payload is not None
    assert signal.payload.get("type") == "project_request"
    assert signal.payload.get("goal") == "Refactor auth module"
    assert signal.payload.get("priority") == "high"


def test_engine_registers_committee_submit_tool(tmp_path: Path) -> None:
    engine, _ = _build_engine(tmp_path)
    names = {spec.name for spec in engine.tool_registry.list_specs()}
    assert "committee__submit" in names

    request = engine._build_request()
    exposed = {spec.name for spec in request.tools}
    assert "committee__submit" in exposed


def test_committee_coordinator_persists_request_and_dispatches_wakes(tmp_path: Path) -> None:
    bus = SignalBus(store=SignalStore(project_root=tmp_path))
    coordinator = CommitteeCoordinator(project_root=tmp_path, signal_bus=bus)

    incoming = bus.send(
        from_agent="super_agent",
        to_agent="committee",
        signal_type=SignalType.WAKE,
        brief="New project request",
        payload={
            "type": "project_request",
            "goal": "Harden authentication",
            "context": "Move password hashing to bcrypt and add migration.",
            "candidate_agents": ["agent.auth", "agent.test"],
            "tasks": [
                {
                    "issue_key": "AUTO-AUTH-1",
                    "title": "Migrate password hashing",
                    "description": "Replace legacy hashes and add migration script.",
                    "candidate_agents": ["agent.auth"],
                },
                {
                    "issue_key": "AUTO-AUTH-2",
                    "title": "Update tests",
                    "description": "Add migration and compatibility coverage.",
                    "candidate_agents": ["agent.test"],
                },
            ],
        },
    )

    decision = coordinator.handle_signal(incoming)
    assert decision["handled"] is True
    assert decision["status"] == "dispatched"
    assert decision["wake_count"] == 2

    requests = coordinator.store.list_requests(limit=10)
    assert len(requests) == 1
    request = requests[0]
    assert request.status == "dispatched"
    assert len(request.tasks) == 2
    assert request.tasks[0].issue_key == "AUTO-AUTH-1"

    wakes = bus.query(from_agent="committee", signal_type=SignalType.WAKE, include_archive=True, limit=0)
    assert len(wakes) == 2
    assert {item.to_agent for item in wakes} == {"agent.auth", "agent.test"}


def test_runner_handles_committee_project_request_signal(tmp_path: Path) -> None:
    bus = SignalBus(store=SignalStore(project_root=tmp_path))
    created: dict[str, _FakeEngine] = {}

    def _factory(session_id: str):
        engine = _FakeEngine(session_id)
        created[session_id] = engine
        return engine

    runner = AgentRunner(
        project_root=tmp_path,
        signal_bus=bus,
        engine_factory=_factory,
    )

    bus.send(
        from_agent="super_agent",
        to_agent="committee",
        signal_type=SignalType.WAKE,
        brief="Please execute this project request",
        payload={
            "type": "project_request",
            "goal": "Build dashboard filters",
            "context": "Need backend and UI filter wiring.",
            "candidate_agents": ["agent.ui"],
        },
    )

    async def _run_once() -> None:
        task = asyncio.create_task(runner.start())
        await asyncio.sleep(0.5)
        await runner.stop()
        await task

    asyncio.run(_run_once())

    requests_path = tmp_path / ".aura" / "state" / "committee" / "pending_requests.jsonl"
    assert requests_path.exists()
    rows = [line for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows

    wakes = bus.query(from_agent="committee", signal_type=SignalType.WAKE, include_archive=True, limit=0)
    assert len(wakes) == 1
    assert wakes[0].to_agent == "agent.ui"
