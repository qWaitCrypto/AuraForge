from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aura.runtime.capability.builder import CapabilityBuilder
from aura.runtime.models.agent_spec import AgentExecutionSpec, AgentSpec
from aura.runtime.models.capability import ROLE_WORKER
from aura.runtime.models.tool_spec import (
    SideEffectLevel,
    ToolAccessPolicy,
    ToolEffectProfile,
    ToolEntrypoint,
    ToolEntrypointType,
    ToolKind,
    ToolRuntimePolicy,
    ToolSpec,
)
from aura.runtime.registry import SpecRegistry, SpecResolver
from aura.runtime.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class _DummyTool:
    name: str
    description: str = "dummy tool"
    input_schema: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.input_schema is None:
            object.__setattr__(self, "input_schema", {"type": "object", "properties": {}})

    def execute(self, *, args: dict, project_root):  # pragma: no cover - never executed in this test
        del args, project_root
        return {"ok": True}


def _make_tool_spec(name: str, *, spec_id: str) -> ToolSpec:
    return ToolSpec(
        id=spec_id,
        name=name,
        runtime_name=name,
        description=f"Tool {name}",
        kind=ToolKind.LOCAL,
        entrypoint=ToolEntrypoint(type=ToolEntrypointType.PYTHON_CALLABLE, ref=f"aura.tools.{name}"),
        effects=ToolEffectProfile(side_effect_level=SideEffectLevel.LOW, idempotent=True),
        runtime=ToolRuntimePolicy(timeout_sec=30),
        policy=ToolAccessPolicy(approval_required=False),
    )


def _build_builder(tmp_path: Path) -> tuple[CapabilityBuilder, str]:
    registry = SpecRegistry(project_root=tmp_path)
    resolver = SpecResolver(registry=registry)
    tool_registry = ToolRegistry()

    runtime_tool_names = [
        "signal__send",
        "signal__poll",
        "audit__query",
        "audit__refs",
        "session__search",
        "project__read_text",
        "shell__run",
    ]
    for name in runtime_tool_names:
        tool_registry.register(_DummyTool(name=name))

    tool_spec = _make_tool_spec("project__read_text", spec_id="tool.test.project_read_text.v1")
    registry.register_tool(tool_spec)

    agent = AgentSpec(
        id="agent.test.capability_builder.v1",
        name="capability-builder-test-agent",
        tool_ids=[tool_spec.id],
        execution=AgentExecutionSpec(
            default_max_turns=7,
            default_max_tool_calls=11,
        ),
    )
    registry.register_agent(agent)

    builder = CapabilityBuilder(
        spec_resolver=resolver,
        tool_registry=tool_registry,
        project_root=tmp_path,
    )
    return builder, agent.id


def test_capability_builder_build_from_agent_spec(tmp_path: Path) -> None:
    builder, agent_id = _build_builder(tmp_path)
    surface = builder.build(agent_id=agent_id, role=ROLE_WORKER)

    assert surface.agent_id == agent_id
    assert surface.role == ROLE_WORKER
    assert surface.resolved_from == "agent_spec"
    assert surface.max_turns == 7
    assert surface.max_tool_calls == 11
    assert "project__read_text" in surface.tool_allowlist
    assert any(spec.name == "project__read_text" for spec in surface.tool_specs)


def test_capability_builder_fallback_for_unknown_agent(tmp_path: Path) -> None:
    builder, _ = _build_builder(tmp_path)
    surface = builder.build(agent_id="agent.unknown.v1", role=ROLE_WORKER)

    assert surface.agent_id == "agent.unknown.v1"
    assert surface.role == ROLE_WORKER
    assert surface.resolved_from == "preset_fallback"
    assert "signal__send" in surface.tool_allowlist
    assert any("resolve_agent_failed" in warning for warning in surface.warnings)
