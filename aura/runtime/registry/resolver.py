from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..models.agent_spec import AgentExecutionMode, AgentSpec
from ..models.mcp_spec import McpServerSpec, McpTransport
from ..models.skill_spec import SkillSpec
from ..models.tool_spec import ToolEntrypointType, ToolSpec
from .spec_registry import SpecRegistry


class ResolutionSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class SpecResolutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResolutionIssue:
    severity: ResolutionSeverity
    reference: str
    message: str


@dataclass(frozen=True, slots=True)
class ResolvedAgentBundle:
    agent: AgentSpec
    skills: list[SkillSpec] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    mcp_servers: list[McpServerSpec] = field(default_factory=list)
    issues: list[ResolutionIssue] = field(default_factory=list)

    def has_errors(self) -> bool:
        return any(issue.severity is ResolutionSeverity.ERROR for issue in self.issues)

    def tool_runtime_names(self) -> list[str]:
        out: list[str] = []
        for spec in self.tools:
            runtime_name = spec.runtime_name if isinstance(spec.runtime_name, str) else spec.name
            runtime_name = runtime_name.strip()
            if not runtime_name:
                continue
            if runtime_name in out:
                continue
            out.append(runtime_name)
        return out

    def runnable_tool_runtime_names(self) -> list[str]:
        out: list[str] = []
        for spec in self.tools:
            if spec.entrypoint.type is ToolEntrypointType.UNKNOWN:
                continue
            runtime_name = spec.runtime_name if isinstance(spec.runtime_name, str) else spec.name
            runtime_name = runtime_name.strip()
            if not runtime_name:
                continue
            if runtime_name in out:
                continue
            out.append(runtime_name)
        return out


class SpecResolver:
    """
    Resolve declarative spec relations into an executable bundle.
    """

    def __init__(self, *, registry: SpecRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> SpecRegistry:
        return self._registry

    @staticmethod
    def _tool_is_runnable(tool: ToolSpec) -> bool:
        if tool.entrypoint.type is ToolEntrypointType.UNKNOWN:
            return False
        return True

    @staticmethod
    def _mcp_server_is_runnable(server: McpServerSpec) -> bool:
        if not server.enabled:
            return False
        if server.transport is McpTransport.STDIO:
            return bool(server.command)
        if server.transport in {McpTransport.HTTP, McpTransport.WS}:
            return bool(server.endpoint)
        return False

    def resolve_agent(self, identifier: str, *, strict: bool = True) -> ResolvedAgentBundle:
        agent_id = self._registry.resolve_agent_id(identifier)
        if agent_id is None:
            raise SpecResolutionError(f"Unknown agent: {identifier!r}")

        agent = self._registry.get_agent(agent_id)
        if agent is None:
            raise SpecResolutionError(f"Agent disappeared from registry: {agent_id}")

        issues: list[ResolutionIssue] = []
        skills: list[SkillSpec] = []
        tools: list[ToolSpec] = []
        mcp_servers: list[McpServerSpec] = []

        tool_ids: list[str] = list(agent.tool_ids)

        for skill_id in agent.skill_ids:
            skill = self._registry.get_skill(skill_id)
            if skill is None:
                issues.append(
                    ResolutionIssue(
                        severity=ResolutionSeverity.ERROR,
                        reference=skill_id,
                        message=f"Agent references missing skill: {skill_id}",
                    )
                )
                continue
            skills.append(skill)
            for req_tool_id in skill.requires_tool_ids:
                if req_tool_id not in tool_ids:
                    tool_ids.append(req_tool_id)

        for mcp_id in agent.mcp_server_ids:
            server = self._registry.get_mcp_server(mcp_id)
            if server is None:
                issues.append(
                    ResolutionIssue(
                        severity=ResolutionSeverity.ERROR,
                        reference=mcp_id,
                        message=f"Agent references missing MCP server: {mcp_id}",
                    )
                )
                continue
            mcp_servers.append(server)
            if not self._mcp_server_is_runnable(server):
                issues.append(
                    ResolutionIssue(
                        severity=ResolutionSeverity.WARNING,
                        reference=server.id,
                        message=(
                            f"MCP server is not runtime-ready (enabled={server.enabled}, "
                            f"transport={server.transport.value})."
                        ),
                    )
                )
            for tool_id in server.provided_tool_ids():
                if tool_id not in tool_ids:
                    tool_ids.append(tool_id)

        for tool_id in tool_ids:
            tool = self._registry.get_tool(tool_id)
            if tool is None:
                issues.append(
                    ResolutionIssue(
                        severity=ResolutionSeverity.ERROR,
                        reference=tool_id,
                        message=f"Agent references missing tool: {tool_id}",
                    )
                )
                continue
            tools.append(tool)
            if not self._tool_is_runnable(tool):
                issues.append(
                    ResolutionIssue(
                        severity=ResolutionSeverity.WARNING,
                        reference=tool.id,
                        message=f"Tool is template-only or not runtime-ready (entrypoint={tool.entrypoint.type.value}).",
                    )
                )

        bundle = ResolvedAgentBundle(
            agent=agent,
            skills=skills,
            tools=tools,
            mcp_servers=mcp_servers,
            issues=issues,
        )
        if strict and bundle.has_errors():
            details = "; ".join(f"{i.reference}: {i.message}" for i in bundle.issues if i.severity is ResolutionSeverity.ERROR)
            raise SpecResolutionError(f"Failed to resolve agent {agent.id}: {details}")
        return bundle

    def resolve_subagent(self, identifier: str, *, strict: bool = True) -> ResolvedAgentBundle:
        bundle = self.resolve_agent(identifier, strict=strict)
        mode = bundle.agent.execution.mode
        if mode is not AgentExecutionMode.SUBAGENT_PRESET:
            raise SpecResolutionError(
                f"Agent {bundle.agent.id} cannot run in subagent tool path (execution.mode={mode.value})."
            )
        if not bundle.agent.execution.preset_name:
            raise SpecResolutionError(f"Agent {bundle.agent.id} is missing execution.preset_name.")
        return bundle
