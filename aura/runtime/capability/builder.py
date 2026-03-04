from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..control.circuit_breaker import CircuitBreaker, CircuitOpenError, get_shared_circuit_breaker
from ..ids import new_id
from ..llm.types import ToolSpec as LlmToolSpec
from ..mcp.config import load_mcp_config
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

INTEGRATOR_TOOLS: list[str] = [
    "session__export",
    "spec__propose",
    "spec__apply",
    "spec__seal",
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
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._resolver = spec_resolver
        self._registry = tool_registry
        self._project_root = project_root.expanduser().resolve()
        self._circuit_breaker = circuit_breaker or get_shared_circuit_breaker(project_root=self._project_root)

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
            for item in INTEGRATOR_TOOLS:
                _add(item)
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
            mcp_cfg = load_mcp_config(project_root=self._project_root)
        except Exception as exc:
            return [], {}, [f"mcp_config_load_failed:{exc}"]
        servers_by_id: dict[str, Any] = {}
        for server in list(getattr(bundle, "mcp_servers", []) or []):
            sid = str(getattr(server, "id", "") or "").strip()
            if sid:
                servers_by_id[sid] = server

        extra_specs: list[LlmToolSpec] = []
        executors: dict[str, object] = {}
        warnings: list[str] = []
        seen_names: set[str] = set()

        for tool in list(getattr(bundle, "tools", []) or []):
            entrypoint = getattr(tool, "entrypoint", None)
            entry_type = str(getattr(getattr(entrypoint, "type", None), "value", getattr(entrypoint, "type", "")) or "")
            if entry_type.strip().lower() != "mcp":
                continue

            runtime_name = str(getattr(tool, "runtime_name", "") or "").strip()
            if not runtime_name or runtime_name in seen_names:
                continue

            binding = getattr(tool, "mcp_binding", None)
            server_id = str(getattr(binding, "server_id", "") or "").strip()
            remote_name = str(getattr(binding, "remote_tool", "") or "").strip()
            if not server_id or not remote_name:
                warnings.append(f"mcp_binding_missing:{runtime_name}")
                continue

            server_spec = servers_by_id.get(server_id)
            server_name = str(getattr(server_spec, "name", "") or "").strip()
            if not server_name:
                warnings.append(f"mcp_server_missing:{runtime_name}:{server_id}")
                continue

            cfg_server = mcp_cfg.servers.get(server_name)
            if cfg_server is None or not bool(cfg_server.enabled) or not str(cfg_server.command or "").strip():
                warnings.append(f"mcp_server_not_configured:{runtime_name}:{server_name}")
                continue

            schema = getattr(tool, "params_schema", None)
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            description = str(getattr(tool, "description", "") or "").strip() or f"MCP tool {runtime_name}"
            extra_specs.append(
                LlmToolSpec(
                    name=runtime_name,
                    description=description,
                    input_schema=schema,
                )
            )
            executors[runtime_name] = self._make_mcp_stdio_executor(
                server_name=server_name,
                command=str(cfg_server.command or "").strip(),
                args=list(cfg_server.args or []),
                env=dict(cfg_server.env or {}),
                cwd=cfg_server.cwd,
                timeout_s=float(cfg_server.timeout_s),
                runtime_name=runtime_name,
                remote_name=remote_name,
            )
            seen_names.add(runtime_name)

        return extra_specs, executors, warnings

    def _make_mcp_stdio_executor(
        self,
        *,
        server_name: str,
        command: str,
        args: list[str],
        env: dict[str, str],
        cwd: str | None,
        timeout_s: float,
        runtime_name: str,
        remote_name: str,
    ):
        def _execute(call_args: dict[str, Any]) -> Any:
            if not self._circuit_breaker.can_call(server_name):
                raise CircuitOpenError(server_name)
            try:
                from agno.tools.mcp.mcp import MCPTools
                from mcp import StdioServerParameters
                from mcp.client.stdio import get_default_environment
            except Exception as e:  # pragma: no cover - optional dependency
                raise RuntimeError(f"MCP tooling unavailable: {e}") from e

            async def _run_once() -> Any:
                prefix = f"mcp__{server_name}__"
                server_params = StdioServerParameters(
                    command=command,
                    args=list(args or []),
                    env={**get_default_environment(), **dict(env or {})},
                    cwd=cwd,
                )
                toolkit = MCPTools(
                    server_params=server_params,
                    transport="stdio",
                    timeout_seconds=int(max(1.0, float(timeout_s))),
                    tool_name_prefix=prefix,
                )
                async with toolkit as entered:
                    async_functions = entered.get_async_functions()
                    target = async_functions.get(runtime_name)
                    normalized_runtime = runtime_name.replace("___", "__")
                    if target is None:
                        for k, fn in async_functions.items():
                            fn_name = str(getattr(fn, "name", k) or k)
                            if (
                                fn_name == runtime_name
                                or fn_name.replace("___", "__") == normalized_runtime
                                or str(k) == runtime_name
                                or str(k).replace("___", "__") == normalized_runtime
                                or fn_name == remote_name
                                or str(k) == remote_name
                            ):
                                target = fn
                                break
                    if target is None:
                        available = sorted(str(k) for k in async_functions.keys())
                        raise RuntimeError(
                            f"MCP tool not found for server={server_name!r}, runtime_name={runtime_name!r}, "
                            f"remote_name={remote_name!r}, available={available[:20]!r}"
                        )
                    from agno.run import RunContext
                    from agno.tools.function import FunctionCall
                    from agno.tools.function import ToolResult as AgnoToolResult

                    try:
                        target._run_context = RunContext(
                            run_id=f"mcp_{new_id('run')}",
                            session_id=f"mcp_{new_id('session')}",
                            metadata={},
                        )
                    except Exception:
                        pass

                    fc = FunctionCall(
                        function=target,
                        arguments=dict(call_args or {}),
                        call_id=f"call_{new_id('mcp')}",
                    )
                    res = await fc.aexecute()
                    if str(getattr(res, "status", "") or "") != "success":
                        raise RuntimeError(str(getattr(res, "error", "") or "MCP tool execution failed"))
                    raw = getattr(res, "result", None)
                    if isinstance(raw, AgnoToolResult):
                        out: dict[str, Any] = {"content": raw.content}
                        if getattr(raw, "images", None):
                            out["images"] = [
                                img.to_dict() if hasattr(img, "to_dict") else img for img in list(raw.images or [])
                            ]
                        return out
                    return raw

            timeout = max(1.0, float(timeout_s))
            try:
                result = asyncio.run(asyncio.wait_for(_run_once(), timeout=timeout))
            except Exception:
                self._circuit_breaker.record_failure(server_name)
                raise
            self._circuit_breaker.record_success(server_name)
            return result

        return _execute

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
