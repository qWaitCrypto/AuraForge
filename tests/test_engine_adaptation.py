from __future__ import annotations

import asyncio
from pathlib import Path

from aura.runtime.engine_agno_async import AgnoAsyncEngine
from aura.runtime.event_bus import EventBus
from aura.runtime.llm.config import ModelConfig
from aura.runtime.stores import FileApprovalStore, FileArtifactStore, FileEventLogStore, FileSessionStore
from aura.runtime.tools.runtime import InspectionDecision


def _build_engine(project_root: Path) -> tuple[AgnoAsyncEngine, str]:
    session_store = FileSessionStore(project_root / ".aura" / "sessions")
    artifact_store = FileArtifactStore(project_root / ".aura" / "artifacts")
    event_store = FileEventLogStore(project_root / ".aura" / "events", artifact_store=artifact_store, session_store=session_store)
    approval_store = FileApprovalStore(project_root / ".aura" / "approvals")
    session_id = session_store.create_session({"mode": "chat", "project_ref": str(project_root)})
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


def test_engine_registers_audit_and_signal_tools(tmp_path: Path) -> None:
    engine, _ = _build_engine(tmp_path)
    names = {spec.name for spec in engine.tool_registry.list_specs()}

    assert "audit__query" in names
    assert "audit__refs" in names
    assert "signal__send" in names
    assert "signal__poll" in names


def test_engine_records_event_log_on_tool_execution(tmp_path: Path, monkeypatch) -> None:
    engine, session_id = _build_engine(tmp_path)
    target = tmp_path / "README.md"
    target.write_text("hello\n", encoding="utf-8")

    async def _inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)

    planned = engine.tool_runtime.plan(
        tool_execution_id="tool_evt_1",
        tool_name="project__read_text",
        tool_call_id="call_evt_1",
        arguments={"path": "README.md"},
    )
    message = asyncio.run(engine._execute_tool(planned=planned, request_id="req_evt_1", turn_id="turn_evt_1"))

    assert '"ok": true' in message.lower()
    events = engine.event_log.query(session_id=session_id, tool_name="project__read_text")
    assert events
    assert events[-1].tool_ok is True
    assert events[-1].tool_name == "project__read_text"


def test_workspace_approval_block_removed_from_runtime_inspection(tmp_path: Path) -> None:
    engine, _ = _build_engine(tmp_path)
    planned = engine.tool_runtime.plan(
        tool_execution_id="tool_ws_1",
        tool_name="workspace__close_workspace",
        tool_call_id="call_ws_1",
        arguments={"workspace_id": "ws_TEST_001"},
    )
    inspection = engine.tool_runtime.inspect(planned)

    assert inspection.decision in {InspectionDecision.ALLOW, InspectionDecision.REQUIRE_APPROVAL}
    assert "Workspace lifecycle tools mutate local state/worktrees" not in str(inspection.reason or "")
