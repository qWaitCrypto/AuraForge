from __future__ import annotations

from pathlib import Path
from typing import Any

from ..llm.types import ToolSpec as LlmToolSpec
from ..models.capability import AgentCapabilitySurface, ROLE_INTEGRATOR, ROLE_WORKER
from ..registry import ResolvedAgentBundle, SpecResolutionError, SpecResolver
from ..tools.registry import ToolRegistry

PLATFORM_BASE_TOOLS: list[str] = [
    "signal__send",
    "signal__poll",
    "audit__query",
    "audit__refs",
    "spec__query",
    "spec__get",
    "spec__list_assets",
    "spec__get_asset",
    "skill__list",
    "skill__load",
    "skill__read_file",
    "session__search",
]

WORKER_TOOLS: list[str] = [
    "project__list_dir",
    "project__glob",
    "project__read_text",
    "project__read_text_many",
    "project__search_text",
    "project__text_stats",
    "project__apply_edits",
    "project__apply_patch",
    "project__patch",
    "shell__run",
    "snapshot__create",
    "snapshot__diff",
    "snapshot__list",
    "snapshot__read_text",
]


class CapabilityBuilder:
    """
    Build runtime capability surface from registry specs.
    """

    def __init__(
        self,
        *,
        spec_resolver: SpecResolver,
        tool_registry: ToolRegistry,
        project_root: Path,
    ) -> None:
        self._resolver = spec_resolver
        self._registry = tool_registry
        self._project_root = project_root.expanduser().resolve()

    def build(
        self,
        *,
        agent_id: str,
        role: str = ROLE_WORKER,
        extra_tools: list[str] | None = None,
    ) -> AgentCapabilitySurface:
        cleaned_agent_id = str(agent_id or "").strip() or "anonymous"
        warnings: list[str] = []
        bundle: ResolvedAgentBundle | None = None
        try:
            bundle = self._resolver.resolve_agent(cleaned_agent_id, strict=False)
        except SpecResolutionError as exc:
            warnings.append(f"resolve_agent_failed: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"resolve_agent_exception: {exc}")

        if bundle is None:
            allowlist = self._merge_allowlist(
                role=role,
                agent_tools=[],
                extra_tools=extra_tools,
            )
            specs = self._collect_runtime_specs(allowlist)
            return AgentCapabilitySurface(
                agent_id=cleaned_agent_id,
                role=self._normalize_role(role),
                tool_allowlist=allowlist,
                tool_specs=specs,
                external_executors={},
                max_turns=20,
                max_tool_calls=60,
                resolved_from="preset_fallback",
                warnings=warnings,
            )

        surface = self.build_from_bundle(bundle=bundle, role=role, extra_tools=extra_tools)
        if warnings:
            merged = list(surface.warnings)
            merged.extend(warnings)
            return AgentCapabilitySurface(
                agent_id=surface.agent_id,
                role=surface.role,
                tool_allowlist=list(surface.tool_allowlist),
                tool_specs=list(surface.tool_specs),
                external_executors=dict(surface.external_executors),
                max_turns=surface.max_turns,
                max_tool_calls=surface.max_tool_calls,
                resolved_from=surface.resolved_from,
                warnings=merged,
            )
        return surface

    def build_from_bundle(
        self,
        *,
        bundle: ResolvedAgentBundle,
        role: str = ROLE_WORKER,
        extra_tools: list[str] | None = None,
    ) -> AgentCapabilitySurface:
        agent_tools = bundle.runnable_tool_runtime_names()
        if not agent_tools:
            agent_tools = list(bundle.agent.execution.default_allowlist)
        allowlist = self._merge_allowlist(
            role=role,
            agent_tools=agent_tools,
            extra_tools=extra_tools,
        )
        runtime_specs = self._collect_runtime_specs(allowlist)
        mcp_specs, mcp_executors, mcp_warnings = self._build_mcp_executors(bundle=bundle)

        merged_specs: list[LlmToolSpec] = list(runtime_specs)
        seen_names = {item.name for item in merged_specs}
        for spec in mcp_specs:
            if spec.name in seen_names:
                continue
            merged_specs.append(spec)
            seen_names.add(spec.name)

        warnings: list[str] = []
        for issue in bundle.issues:
            warnings.append(f"{issue.severity.value}:{issue.reference}:{issue.message}")
        warnings.extend(mcp_warnings)

        max_turns, max_tool_calls = self._resolve_limits(bundle=bundle)
        return AgentCapabilitySurface(
            agent_id=bundle.agent.id,
            role=self._normalize_role(role),
            tool_allowlist=allowlist,
            tool_specs=merged_specs,
            external_executors=mcp_executors,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            resolved_from="agent_spec",
            warnings=warnings,
        )

    def _merge_allowlist(
        self,
        *,
        role: str,
        agent_tools: list[str],
        extra_tools: list[str] | None,
    ) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()

        def _add(name: str) -> None:
            cleaned = str(name or "").strip()
            if not cleaned or cleaned in seen:
                return
            seen.add(cleaned)
            merged.append(cleaned)

        normalized_role = self._normalize_role(role)
        for item in PLATFORM_BASE_TOOLS:
            _add(item)
        for item in WORKER_TOOLS:
            _add(item)
        if normalized_role == ROLE_INTEGRATOR:
            # Integrator currently shares worker surface; runtime enforces privileged actions.
            pass
        for item in agent_tools:
            _add(item)
        for item in extra_tools or []:
            _add(item)
        return merged

    def _collect_runtime_specs(self, allowlist: list[str]) -> list[LlmToolSpec]:
        available: dict[str, LlmToolSpec] = {
            spec.name: spec for spec in self._registry.list_specs() if isinstance(spec.name, str) and spec.name
        }
        out: list[LlmToolSpec] = []
        for name in allowlist:
            spec = available.get(name)
            if spec is None:
                continue
            out.append(spec)
        return out

    def _build_mcp_executors(
        self,
        *,
        bundle: ResolvedAgentBundle,
    ) -> tuple[list[LlmToolSpec], dict[str, object], list[str]]:
        try:
            from ..tools.subagent_runner import SubagentRunTool

            extra_specs, executors, warning_rows = SubagentRunTool._build_external_mcp_tools(
                bundle=bundle,
                project_root=self._project_root,
            )
            warnings: list[str] = []
            for item in warning_rows:
                if isinstance(item, dict):
                    code = str(item.get("code") or "").strip()
                    message = str(item.get("message") or "").strip()
                    tool_name = str(item.get("tool") or "").strip()
                    details = ":".join(part for part in [code, tool_name, message] if part)
                    warnings.append(details or str(item))
                else:
                    warnings.append(str(item))
            return list(extra_specs), dict(executors), warnings
        except Exception as exc:
            return [], {}, [f"mcp_executor_build_failed:{exc}"]

    def _resolve_limits(self, *, bundle: ResolvedAgentBundle | None) -> tuple[int, int]:
        max_turns = 20
        max_tool_calls = 60
        if bundle is None:
            return max_turns, max_tool_calls

        turns = bundle.agent.execution.default_max_turns
        calls = bundle.agent.execution.default_max_tool_calls
        if isinstance(turns, int) and turns > 0:
            max_turns = turns
        if isinstance(calls, int) and calls > 0:
            max_tool_calls = calls
        return max_turns, max_tool_calls

    @staticmethod
    def _normalize_role(role: str) -> str:
        cleaned = str(role or "").strip().lower()
        if cleaned == ROLE_INTEGRATOR:
            return ROLE_INTEGRATOR
        return ROLE_WORKER
