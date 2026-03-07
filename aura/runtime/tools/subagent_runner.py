from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..capability import CapabilityBuilder
from ..context import ContextBuilder
from ..ids import new_id
from ..llm.router import ModelRouter
from ..llm.types import ToolSpec as LlmToolSpec
from ..mcp.config import load_mcp_config, mcp_stdio_errlog_context
from ..models import WorkSpec
from ..registry import SpecResolutionError, SpecResolver
from ..sandbox import SandboxManager
from ..stores import ArtifactStore
from ..subagents.presets import get_preset, preset_input_schema
from ..subagents.runner import run_subagent
from .registry import ToolRegistry
from .runtime import ToolExecutionContext, ToolRuntime


@dataclass(slots=True)
class SubagentRunTool:
    model_router: ModelRouter
    tool_registry: ToolRegistry
    tool_runtime: ToolRuntime
    artifact_store: ArtifactStore
    spec_resolver: SpecResolver | None = None
    capability_builder: CapabilityBuilder | None = None
    context_builder: ContextBuilder | None = None
    sandbox_manager: SandboxManager | None = None

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
        project_root: Path,
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
                with mcp_stdio_errlog_context(project_root=project_root, server_name=server_name):
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
                project_root=project_root,
            )
            seen_names.add(runtime_name)

        return extra_specs, external_executors, warnings

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        default_max_tool_calls = 10

        bundle = None
        resolved_agent_id: str | None = None
        agent_card_text: str | None = None
        capability_surface = None
        capability_warnings: list[dict[str, Any]] = []

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
            if self.capability_builder is not None:
                role_hint = (
                    str(
                        (context.role if context is not None and isinstance(context.role, str) else None) or "worker"
                    )
                    .strip()
                    .lower()
                    or "worker"
                )
                try:
                    capability_surface = self.capability_builder.build_from_bundle(bundle=bundle, role=role_hint)
                except Exception as exc:
                    capability_warnings.append(
                        {
                            "code": "capability_surface_build_failed",
                            "message": str(exc),
                        }
                    )

        task = str(args.get("task") or "").strip()
        if not task:
            raise ValueError("Missing or invalid 'task' (expected non-empty string).")

        allowlist = args.get("tool_allowlist")
        if allowlist is None:
            allowlist_patterns: list[str] = []
            if capability_surface is not None and capability_surface.tool_allowlist:
                allowlist_patterns = list(capability_surface.tool_allowlist)
            elif bundle is not None:
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
            if capability_surface is not None:
                max_turns_int = int(capability_surface.max_turns)
            else:
                max_turns_int = preset.limits.max_turns
                if bundle is not None and bundle.agent.execution.default_max_turns is not None:
                    max_turns_int = int(bundle.agent.execution.default_max_turns)
        else:
            if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
                raise ValueError("Invalid 'max_turns' (expected integer >= 1).")
            max_turns_int = min(int(max_turns), 50)

        max_tool_calls = args.get("max_tool_calls")
        if max_tool_calls is None:
            if capability_surface is not None:
                default_budget = int(capability_surface.max_tool_calls)
            else:
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

        exec_ctx = context
        inferred_workspace_root: str | None = None
        if context is not None and isinstance(context.worktree_path, str) and context.worktree_path.strip():
            inferred_workspace_root = context.worktree_path.strip()

        if work_spec is None:
            if inferred_workspace_root:
                work_spec = self._build_minimal_work_spec(goal=task, workspace_root=inferred_workspace_root)
            else:
                raise ValueError(
                    "Missing required 'work_spec'. Provide work_spec or run with context.worktree_path."
                )

        if capability_surface is not None:
            extra_tool_specs = list(capability_surface.tool_specs)
            external_tool_executors = dict(capability_surface.external_executors)
            mcp_warnings: list[dict[str, Any]] = []
            for item in capability_surface.warnings:
                if not isinstance(item, str) or not item.strip():
                    continue
                mcp_warnings.append({"code": "capability_warning", "message": item.strip()})
        else:
            extra_tool_specs, external_tool_executors, mcp_warnings = self._build_external_mcp_tools(
                bundle=bundle,
                project_root=Path(project_root),
            )
        if capability_warnings:
            mcp_warnings.extend(capability_warnings)

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
        return out
