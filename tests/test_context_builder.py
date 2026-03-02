from __future__ import annotations

from pathlib import Path

from aura.runtime.context import ContextBuilder
from aura.runtime.llm.types import ToolSpec as LlmToolSpec
from aura.runtime.models.capability import AgentCapabilitySurface, ROLE_WORKER
from aura.runtime.models.sandbox import Sandbox
from aura.runtime.models.signal import Signal, SignalType


def _surface() -> AgentCapabilitySurface:
    return AgentCapabilitySurface(
        agent_id="agent.test.context.v1",
        role=ROLE_WORKER,
        tool_allowlist=["project__read_text", "signal__send"],
        tool_specs=[
            LlmToolSpec(
                name="project__read_text",
                description="Read text file",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
            LlmToolSpec(
                name="signal__send",
                description="Send signal",
                input_schema={"type": "object", "properties": {"to_agent": {"type": "string"}}},
            ),
        ],
    )


def test_context_builder_wake_trigger(tmp_path: Path) -> None:
    builder = ContextBuilder(project_root=tmp_path)
    signal = Signal(
        signal_id="sig_1",
        from_agent="committee",
        to_agent="agent.test.context.v1",
        signal_type=SignalType.WAKE,
        brief="check issue",
        issue_key="PROJ-101",
    )
    context = builder.build(
        surface=_surface(),
        signal=signal,
        task_description="Propose a concise approach in issue comments.",
    )

    assert context.trigger == "wake"
    assert context.issue_key == "PROJ-101"
    assert "Issue: PROJ-101" in context.system_prompt
    assert "Task Description:" in context.system_prompt
    assert "Available Tools" in context.system_prompt


def test_context_builder_assigned_includes_sandbox_and_project_knowledge(tmp_path: Path) -> None:
    knowledge_path = tmp_path / ".aura" / "context" / "project_knowledge.md"
    knowledge_path.parent.mkdir(parents=True, exist_ok=True)
    knowledge_path.write_text("Coding standard: keep tests deterministic.", encoding="utf-8")

    builder = ContextBuilder(project_root=tmp_path)
    sandbox = Sandbox(
        sandbox_id="sb_proj101_agent_01",
        agent_id="agent.test.context.v1",
        issue_key="PROJ-101",
        worktree_path=".aura/sandboxes/sb_proj101_agent_01",
        branch="agent/PROJ-101/test/01",
    )
    signal = Signal(
        signal_id="sig_2",
        from_agent="committee",
        to_agent="agent.test.context.v1",
        signal_type=SignalType.TASK_ASSIGNED,
        brief="you won",
        issue_key="PROJ-101",
        sandbox_id=sandbox.sandbox_id,
    )
    context = builder.build(surface=_surface(), sandbox=sandbox, signal=signal)

    assert context.trigger == "task_assigned"
    assert context.sandbox_id == "sb_proj101_agent_01"
    assert "Worktree: .aura/sandboxes/sb_proj101_agent_01" in context.system_prompt
    assert "Coding standard: keep tests deterministic." in context.system_prompt
