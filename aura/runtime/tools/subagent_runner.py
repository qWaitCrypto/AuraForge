from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

from ..ids import new_id
from ..llm.router import ModelRouter
from ..llm.types import ToolSpec as LlmToolSpec
from ..mcp.config import load_mcp_config
from ..models import WorkSpec
from ..models.workspace import WorkbenchRole
from ..registry import SpecResolutionError, SpecResolver
from ..stores import ArtifactStore
from ..subagents.presets import get_preset, preset_input_schema
from ..subagents.runner import run_subagent
from ..workspace import WorkspaceManager, WorkspaceManagerError
from .registry import ToolRegistry
from .runtime import InspectionDecision, ToolExecutionContext, ToolRuntime


@dataclass(slots=True)
class SubagentRunTool:
    model_router: ModelRouter
    tool_registry: ToolRegistry
    tool_runtime: ToolRuntime
    artifact_store: ArtifactStore
    spec_resolver: SpecResolver | None = None
    workspace_manager: WorkspaceManager | None = None

    name: ClassVar[str] = "subagent__run"
    description: ClassVar[str] = (
        "Run a bounded delegated task in an isolated subagent context. "
        "Use for verification (preset=verifier), documents (preset=doc_worker), spreadsheets (preset=sheet_worker), "
        "browser automation (preset=browser_worker), or file operations (preset=file_ops_worker). "
        "Prefer `agent_id` for spec-driven routing; `preset` remains supported for compatibility. "
        "The runner enforces a per-run tool allowlist, prevents recursion, and never nests interactive approvals: "
        "if an approval-required tool is requested, it stops and returns an actionable report."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "Spec agent id or alias (preferred). Must resolve to a subagent_preset agent.",
            },
            "preset": preset_input_schema(),
            "task": {"type": "string", "description": "Delegated task instruction."},
            "context": {
                "type": "object",
                "description": "Optional extra context passed to the subagent.",
                "properties": {
                    "text": {"type": "string", "description": "Extra context text."},
                    "files": {
                        "type": "array",
                        "description": "Optional file hints; the subagent may read these via project tools if needed.",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                        "max_chars": {"type": "integer", "minimum": 1},
                                    },
                                    "required": ["path"],
                                    "additionalProperties": False,
                                },
                            ]
                        },
                    },
                },
                "additionalProperties": False,
            },
            "work_spec": {
                "type": "object",
                "description": (
                    "WorkSpec for this delegated run. "
                    "Used for tool gating (workspace roots, domain allowlist, allowed file types) and approval decisions."
                ),
                "properties": {
                    "goal": {"type": "string", "description": "What the subagent must accomplish."},
                    "expected_outputs": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Concrete files/artifacts that should be produced.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["document", "spreadsheet", "index", "report", "other"],
                                    "description": "Output category. Must use one of the allowed enum values (do not invent values like 'draft').",
                                },
                                "format": {
                                    "type": "string",
                                    "description": "Output format (e.g. docx, xlsx, pdf, md, json).",
                                },
                                "path": {
                                    "type": "string",
                                    "description": "Workspace-relative output path (recommended).",
                                },
                            },
                            "required": ["type", "format"],
                            "additionalProperties": True,
                        },
                    },
                    "resource_scope": {
                        "type": "object",
                        "description": "Hard boundary for tool access (workspace/file types/domains).",
                        "properties": {
                            "workspace_roots": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Allowed workspace roots (relative paths).",
                            },
                            "file_type_allowlist": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Optional allowlist of file extensions/types (e.g. docx, md, json). "
                                    "If omitted or empty, file types are not restricted. "
                                    "If set, include any intermediate artifact types you expect to write (e.g. json for plan files)."
                                ),
                            },
                            "domain_allowlist": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Allowed external domains (if any).",
                            },
                        },
                        "additionalProperties": True,
                    },
                },
                "required": ["goal", "expected_outputs", "resource_scope"],
                "additionalProperties": True,
            },
            "workspace": {
                "type": "object",
                "description": (
                    "Optional workspace routing hints. If provided, the runner binds this subagent "
                    "to a dedicated workbench and enforces WorkSpec workspace_roots accordingly."
                ),
                "properties": {
                    "workspace_id": {"type": "string"},
                    "workbench_id": {"type": "string"},
                    "role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
                    "bind_session": {"type": "boolean"},
                    "lease_seconds": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
            "tool_allowlist": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional override allowlist (tool names or glob patterns).",
            },
            "max_turns": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max internal turns."},
            "max_tool_calls": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": "Max tool calls executed inside the subagent.",
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    }

    def _execute_internal_workspace_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        tool = self.tool_runtime.get_tool(tool_name)
        if tool is None:
            raise RuntimeError(f"Internal workspace tool not found: {tool_name}")

        planned = self.tool_runtime.plan(
            tool_execution_id=f"tool_{new_id('call')}",
            tool_name=tool_name,
            tool_call_id=new_id("call"),
            arguments=dict(arguments),
            caller_kind="system",
        )
        inspection = self.tool_runtime.inspect(planned)
        if inspection.decision is not InspectionDecision.ALLOW:
            reason = inspection.reason or inspection.action_summary or f"Internal tool denied: {tool_name}"
            raise PermissionError(reason)

        sys_ctx = ToolExecutionContext(
            session_id=(context.session_id if context is not None else ""),
            request_id=(context.request_id if context is not None else None),
            turn_id=(context.turn_id if context is not None else None),
            tool_execution_id=planned.tool_execution_id,
            event_bus=(context.event_bus if context is not None else None),
            workspace_id=(context.workspace_id if context is not None else None),
            workbench_id=(context.workbench_id if context is not None else None),
            worktree_path=(context.worktree_path if context is not None else None),
            workspace_role=(context.workspace_role if context is not None else None),
            caller_kind="system",
        )

        from inspect import Parameter, signature

        params = signature(tool.execute).parameters
        accepts_context = "context" in params or any(p.kind is Parameter.VAR_KEYWORD for p in params.values())
        if accepts_context:
            raw = tool.execute(args=planned.arguments, project_root=self.tool_runtime.project_root, context=sys_ctx)
        else:
            raw = tool.execute(args=planned.arguments, project_root=self.tool_runtime.project_root)
        if not isinstance(raw, dict):
            raise RuntimeError(f"Internal tool returned invalid payload: {tool_name}")
        return raw

    @staticmethod
    def _load_agent_card_text(*, project_root: Path, prompt_ref: str | None, max_chars: int = 20000) -> str | None:
        ref = str(prompt_ref or "").strip()
        if not ref:
            return None
        root = project_root.expanduser().resolve()
        try:
            candidate = (root / ref).resolve()
        except Exception:
            return None
        if candidate != root and root not in candidate.parents:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return None
        if not text:
            return None
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars]
        return text

    @staticmethod
    def _build_minimal_work_spec(*, goal: str, workspace_root: str) -> WorkSpec:
        root = str(workspace_root or "").strip().rstrip("/")
        default_output_path = f"{root}/artifacts/subagent_report.md" if root else "artifacts/subagent_report.md"
        return WorkSpec.model_validate(
            {
                "goal": goal,
                "expected_outputs": [
                    {
                        "type": "other",
                        "format": "md",
                        "path": default_output_path,
                    }
                ],
                "resource_scope": {
                    "workspace_roots": ([root] if root else []),
                    "file_type_allowlist": [],
                    "domain_allowlist": [],
                },
            }
        )

    @staticmethod
    def _make_mcp_stdio_executor(
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
            return asyncio.run(asyncio.wait_for(_run_once(), timeout=timeout))

        return _execute

    @classmethod
    def _build_external_mcp_tools(
        cls,
        *,
        bundle: Any | None,
        project_root: Path,
    ) -> tuple[list[LlmToolSpec], dict[str, Any], list[dict[str, Any]]]:
        if bundle is None:
            return [], {}, []

        mcp_cfg = load_mcp_config(project_root=project_root)
        servers_by_id: dict[str, Any] = {}
        for server in list(getattr(bundle, "mcp_servers", []) or []):
            sid = str(getattr(server, "id", "") or "").strip()
            if sid:
                servers_by_id[sid] = server

        extra_specs: list[LlmToolSpec] = []
        external_executors: dict[str, Any] = {}
        warnings: list[dict[str, Any]] = []
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
                warnings.append(
                    {
                        "code": "mcp_binding_missing",
                        "tool": runtime_name,
                        "message": "MCP proxy tool is missing binding metadata.",
                    }
                )
                continue

            server_spec = servers_by_id.get(server_id)
            server_name = str(getattr(server_spec, "name", "") or "").strip()
            if not server_name:
                warnings.append(
                    {
                        "code": "mcp_server_missing",
                        "tool": runtime_name,
                        "server_id": server_id,
                        "message": "MCP server not present in resolved bundle.",
                    }
                )
                continue

            cfg_server = mcp_cfg.servers.get(server_name)
            if cfg_server is None or not bool(cfg_server.enabled) or not str(cfg_server.command or "").strip():
                warnings.append(
                    {
                        "code": "mcp_server_not_configured",
                        "tool": runtime_name,
                        "server_name": server_name,
                        "message": "MCP server is not configured/enabled in .aura/config/mcp.json.",
                    }
                )
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
            external_executors[runtime_name] = cls._make_mcp_stdio_executor(
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

        return extra_specs, external_executors, warnings

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        default_max_tool_calls = 10

        bundle = None
        resolved_agent_id: str | None = None
        agent_card_text: str | None = None

        agent_id_raw = str(args.get("agent_id") or "").strip()
        preset_name = str(args.get("preset") or "").strip()

        if agent_id_raw:
            if self.spec_resolver is None:
                raise ValueError("agent_id is not available in this runtime (spec resolver missing).")
            try:
                bundle = self.spec_resolver.resolve_subagent(agent_id_raw)
            except SpecResolutionError as e:
                raise ValueError(f"Invalid agent_id: {e}") from e
            resolved_agent_id = bundle.agent.id
            preset_name = bundle.agent.execution.preset_name or preset_name

        if not preset_name:
            raise ValueError("Missing either 'agent_id' or 'preset'.")

        preset = get_preset(preset_name)
        if preset is None:
            raise ValueError(f"Unknown preset: {preset_name!r}")

        # Compatibility path: map legacy preset names back to a registered agent id when possible.
        if bundle is None and self.spec_resolver is not None:
            try:
                bundle = self.spec_resolver.resolve_subagent(preset_name, strict=False)
            except Exception:
                bundle = None
            else:
                resolved_agent_id = bundle.agent.id

        if bundle is not None:
            metadata = bundle.agent.metadata if isinstance(bundle.agent.metadata, dict) else {}
            prompt_ref = metadata.get("prompt_ref")
            if isinstance(prompt_ref, str):
                agent_card_text = self._load_agent_card_text(
                    project_root=Path(project_root),
                    prompt_ref=prompt_ref,
                )

        task = str(args.get("task") or "").strip()
        if not task:
            raise ValueError("Missing or invalid 'task' (expected non-empty string).")

        allowlist = args.get("tool_allowlist")
        if allowlist is None:
            allowlist_patterns: list[str] = []
            if bundle is not None:
                allowlist_patterns = bundle.runnable_tool_runtime_names()
                if not allowlist_patterns:
                    allowlist_patterns = list(bundle.agent.execution.default_allowlist)
            if not allowlist_patterns:
                allowlist_patterns = list(preset.default_allowlist)
        else:
            if not isinstance(allowlist, list) or any(not isinstance(x, str) for x in allowlist):
                raise ValueError("Invalid 'tool_allowlist' (expected list of strings).")
            allowlist_patterns = [x.strip() for x in allowlist if x.strip()]

        max_turns = args.get("max_turns")
        if max_turns is None:
            max_turns_int = preset.limits.max_turns
            if bundle is not None and bundle.agent.execution.default_max_turns is not None:
                max_turns_int = int(bundle.agent.execution.default_max_turns)
        else:
            if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
                raise ValueError("Invalid 'max_turns' (expected integer >= 1).")
            max_turns_int = min(int(max_turns), 50)

        max_tool_calls = args.get("max_tool_calls")
        if max_tool_calls is None:
            default_budget = preset.limits.max_tool_calls
            if bundle is not None and bundle.agent.execution.default_max_tool_calls is not None:
                default_budget = int(bundle.agent.execution.default_max_tool_calls)
            max_tool_calls_int = min(default_budget, default_max_tool_calls)
        else:
            if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 1:
                raise ValueError("Invalid 'max_tool_calls' (expected integer >= 1).")
            max_tool_calls_int = min(int(max_tool_calls), 200)

        work_spec_raw = args.get("work_spec")
        work_spec: WorkSpec | None = None
        if work_spec_raw is not None and not isinstance(work_spec_raw, dict):
            raise ValueError("Invalid 'work_spec' (expected object).")
        if isinstance(work_spec_raw, dict):
            try:
                work_spec = WorkSpec.model_validate(work_spec_raw)
            except Exception as e:
                raise ValueError(f"Invalid 'work_spec': {e}") from e

        workspace_info: dict[str, Any] | None = None
        exec_ctx = context
        resolved_context = None
        if self.workspace_manager is not None:
            ws_raw = args.get("workspace")
            bind_session_default = True

            if isinstance(ws_raw, dict):
                ws_id = str(ws_raw.get("workspace_id") or "").strip() or None
                wb_id = str(ws_raw.get("workbench_id") or "").strip() or None
                bind_session = bool(ws_raw.get("bind_session", bind_session_default))
                lease_seconds = int(ws_raw.get("lease_seconds") or 900)
                role_raw = str(ws_raw.get("role") or WorkbenchRole.WORKER.value).strip()
                try:
                    role = WorkbenchRole(role_raw)
                except ValueError as e:
                    raise ValueError(f"Invalid workspace.role: {role_raw!r}") from e

                caller_kind = str((context.caller_kind if context is not None else None) or "llm").strip().lower() or "llm"
                caller_role = str((context.workspace_role if context is not None else None) or "").strip().lower()

                if wb_id or ws_id:
                    if wb_id:
                        resolved_context = self.workspace_manager.resolve_context(
                            workspace_id=ws_id,
                            workbench_id=wb_id,
                            session_id=(context.session_id if bind_session and context is not None else None),
                        )
                    elif ws_id and not wb_id:
                        if caller_kind != "system" and caller_role != WorkbenchRole.INTEGRATOR.value:
                            raise PermissionError(
                                "Auto-provisioning a workspace workbench requires integrator/system caller context."
                            )

                        operator_role = (
                            WorkbenchRole.INTEGRATOR.value if caller_kind == "system" else WorkbenchRole(caller_role).value
                        )
                        provision = self._execute_internal_workspace_tool(
                            tool_name="workspace__provision_workbench",
                            arguments={
                                "workspace_id": ws_id,
                                "agent_id": (resolved_agent_id or f"preset:{preset.name}"),
                                "instance_id": (context.tool_execution_id if context is not None else new_id("inst")),
                                "role": role.value,
                                "bind_session": bind_session,
                                "lease_seconds": lease_seconds,
                                "operator_role": operator_role,
                            },
                            context=context,
                        )
                        wb_raw = provision.get("workbench")
                        if not isinstance(wb_raw, dict):
                            raise RuntimeError("workspace__provision_workbench returned invalid workbench payload.")
                        wb_id_created = str(wb_raw.get("workbench_id") or "").strip()
                        ws_id_created = str(wb_raw.get("workspace_id") or "").strip()
                        if not wb_id_created or not ws_id_created:
                            raise RuntimeError("workspace__provision_workbench missing workspace/workbench identifiers.")
                        resolved_context = self.workspace_manager.resolve_context(
                            workspace_id=ws_id_created,
                            workbench_id=wb_id_created,
                        )

            if resolved_context is None and context is not None:
                if context.workspace_id and context.workbench_id:
                    try:
                        resolved_context = self.workspace_manager.resolve_context(
                            workspace_id=context.workspace_id,
                            workbench_id=context.workbench_id,
                            session_id=context.session_id or None,
                        )
                    except WorkspaceManagerError:
                        resolved_context = None

        inferred_workspace_root: str | None = None
        if resolved_context is not None:
            inferred_workspace_root = resolved_context.workbench.worktree_path
        elif context is not None and isinstance(context.worktree_path, str) and context.worktree_path.strip():
            inferred_workspace_root = context.worktree_path.strip()

        if work_spec is None:
            if inferred_workspace_root:
                work_spec = self._build_minimal_work_spec(goal=task, workspace_root=inferred_workspace_root)
            else:
                raise ValueError(
                    "Missing required 'work_spec'. Provide work_spec or run in a bound workspace/workbench context."
                )

        if resolved_context is not None:
            scope = work_spec.resource_scope.model_copy(
                update={
                    "workspace_roots": [resolved_context.workbench.worktree_path],
                }
            )
            work_spec = work_spec.model_copy(update={"resource_scope": scope})
            workspace_info = {
                "workspace_id": resolved_context.workspace.workspace_id,
                "workbench_id": resolved_context.workbench.workbench_id,
                "worktree_path": resolved_context.workbench.worktree_path,
                "workspace_role": resolved_context.workbench.role.value,
            }
            if context is not None:
                exec_ctx = replace(
                    context,
                    workspace_id=resolved_context.workspace.workspace_id,
                    workbench_id=resolved_context.workbench.workbench_id,
                    worktree_path=resolved_context.workbench.worktree_path,
                    workspace_role=resolved_context.workbench.role.value,
                )

        extra_tool_specs, external_tool_executors, mcp_warnings = self._build_external_mcp_tools(
            bundle=bundle,
            project_root=Path(project_root),
        )

        out = run_subagent(
            agent_id=resolved_agent_id,
            preset=preset,
            task=task,
            extra_context=args.get("context"),
            agent_card_text=agent_card_text,
            work_spec=work_spec,
            tool_allowlist=allowlist_patterns,
            max_turns=max_turns_int,
            max_tool_calls=max_tool_calls_int,
            model_router=self.model_router,
            tool_registry=self.tool_registry,
            tool_runtime=self.tool_runtime,
            artifact_store=self.artifact_store,
            project_root=project_root,
            exec_context=exec_ctx,
            extra_tool_specs=extra_tool_specs,
            external_tool_executors=external_tool_executors,
        )
        if mcp_warnings:
            merged = list(out.get("warnings") or [])
            merged.extend(mcp_warnings)
            out["warnings"] = merged
        if workspace_info is not None and isinstance(out, dict):
            out["workspace"] = workspace_info
        return out
