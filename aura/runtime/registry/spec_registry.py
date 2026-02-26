from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..mcp.config import McpConfig, McpServerConfig
from ..models.agent_spec import AgentExecutionMode, AgentExecutionSpec, AgentSpec
from ..models.mcp_spec import (
    McpAuthSpec,
    McpAuthType,
    McpHealthcheck,
    McpProvidedToolSpec,
    McpServerSpec,
    McpTransport,
)
from ..models.skill_spec import SkillEntrySpec, SkillEntryType, SkillOutputContract, SkillSpec
from ..models.spec_common import SpecLifecycle
from ..models.tool_spec import (
    McpToolBinding,
    SideEffectLevel,
    ToolAccessPolicy,
    ToolEffectProfile,
    ToolEntrypoint,
    ToolEntrypointType,
    ToolKind,
    ToolRuntimePolicy,
    ToolSpec,
)

if TYPE_CHECKING:
    from ..skills import SkillStore
    from ..tools.registry import ToolRegistry


class SpecRegistryError(RuntimeError):
    pass


def _slug_token(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    if not text:
        return "unknown"
    return text[:80]


def make_tool_spec_id(*, namespace: str, runtime_name: str) -> str:
    return f"tool.{namespace}.{_slug_token(runtime_name)}.v1"


def make_skill_spec_id(*, namespace: str, skill_name: str) -> str:
    return f"skill.{namespace}.{_slug_token(skill_name)}.v1"


def make_agent_spec_id(*, namespace: str, agent_name: str) -> str:
    return f"agent.{namespace}.{_slug_token(agent_name)}.v1"


def make_mcp_spec_id(*, namespace: str, server_name: str) -> str:
    return f"mcp.{namespace}.{_slug_token(server_name)}.v1"


class SpecRegistry:
    """
    Unified in-memory registry for agent/skill/tool/MCP specs.
    """

    def __init__(self, *, project_root: Path) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._agents: dict[str, AgentSpec] = {}
        self._skills: dict[str, SkillSpec] = {}
        self._tools: dict[str, ToolSpec] = {}
        self._mcp_servers: dict[str, McpServerSpec] = {}

        self._agent_aliases: dict[str, str] = {}
        self._agent_declared_aliases: dict[str, list[str]] = {}
        self._tool_runtime_names: dict[str, str] = {}
        self._skill_names: dict[str, str] = {}
        self._mcp_names: dict[str, str] = {}

    @property
    def project_root(self) -> Path:
        return self._project_root

    def clear(self) -> None:
        self._agents.clear()
        self._skills.clear()
        self._tools.clear()
        self._mcp_servers.clear()
        self._agent_aliases.clear()
        self._agent_declared_aliases.clear()
        self._tool_runtime_names.clear()
        self._skill_names.clear()
        self._mcp_names.clear()

    @staticmethod
    def _normalize_aliases(aliases: list[str] | None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for alias in aliases or []:
            text = str(alias or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def register_agent(self, spec: AgentSpec, *, aliases: list[str] | None = None) -> None:
        if spec.id in self._agents:
            raise SpecRegistryError(f"Duplicate agent spec id: {spec.id}")
        self._agents[spec.id] = spec
        self._agent_declared_aliases[spec.id] = self._normalize_aliases(aliases)
        self._rebuild_agent_alias_index()

    def upsert_agent(self, spec: AgentSpec, *, aliases: list[str] | None = None) -> None:
        self._agents[spec.id] = spec
        if aliases is None:
            self._agent_declared_aliases.setdefault(spec.id, [])
        else:
            self._agent_declared_aliases[spec.id] = self._normalize_aliases(aliases)
        self._rebuild_agent_alias_index()

    def register_skill(self, spec: SkillSpec) -> None:
        if spec.id in self._skills:
            raise SpecRegistryError(f"Duplicate skill spec id: {spec.id}")
        self._skills[spec.id] = spec
        self._rebuild_skill_name_index()

    def upsert_skill(self, spec: SkillSpec) -> None:
        self._skills[spec.id] = spec
        self._rebuild_skill_name_index()

    def register_tool(self, spec: ToolSpec) -> None:
        if spec.id in self._tools:
            raise SpecRegistryError(f"Duplicate tool spec id: {spec.id}")
        self._tools[spec.id] = spec
        self._rebuild_tool_name_index()

    def upsert_tool(self, spec: ToolSpec) -> None:
        self._tools[spec.id] = spec
        self._rebuild_tool_name_index()

    def register_mcp_server(self, spec: McpServerSpec) -> None:
        if spec.id in self._mcp_servers:
            raise SpecRegistryError(f"Duplicate MCP server spec id: {spec.id}")
        self._mcp_servers[spec.id] = spec
        self._rebuild_mcp_name_index()

    def upsert_mcp_server(self, spec: McpServerSpec) -> None:
        self._mcp_servers[spec.id] = spec
        self._rebuild_mcp_name_index()

    def list_agents(self) -> list[AgentSpec]:
        return [self._agents[k] for k in sorted(self._agents)]

    def list_skills(self) -> list[SkillSpec]:
        return [self._skills[k] for k in sorted(self._skills)]

    def list_tools(self) -> list[ToolSpec]:
        return [self._tools[k] for k in sorted(self._tools)]

    def list_mcp_servers(self) -> list[McpServerSpec]:
        return [self._mcp_servers[k] for k in sorted(self._mcp_servers)]

    def get_agent(self, spec_id: str) -> AgentSpec | None:
        return self._agents.get(spec_id)

    def get_skill(self, spec_id: str) -> SkillSpec | None:
        return self._skills.get(spec_id)

    def get_tool(self, spec_id: str) -> ToolSpec | None:
        return self._tools.get(spec_id)

    def get_mcp_server(self, spec_id: str) -> McpServerSpec | None:
        return self._mcp_servers.get(spec_id)

    def resolve_agent_id(self, identifier: str) -> str | None:
        key = str(identifier or "").strip()
        if not key:
            return None
        if key in self._agents:
            return key
        return self._agent_aliases.get(key)

    def resolve_tool_id_by_runtime_name(self, runtime_name: str) -> str | None:
        return self._tool_runtime_names.get(str(runtime_name or "").strip())

    def resolve_skill_id_by_name(self, skill_name: str) -> str | None:
        return self._skill_names.get(str(skill_name or "").strip())

    def resolve_mcp_id_by_name(self, server_name: str) -> str | None:
        return self._mcp_names.get(str(server_name or "").strip())

    def _rebuild_tool_name_index(self) -> None:
        # Tool identifiers can collide (same alias across multiple market entries, or a market
        # template shadowing a runtime tool). Prefer runtime-provided specs over market.
        def _origin_rank(spec: ToolSpec) -> int:
            meta = spec.metadata if isinstance(spec.metadata, dict) else {}
            origin = str(meta.get("origin") or "").strip()
            if origin.startswith("runtime.tool_registry"):
                return 3
            if origin.startswith("mcp.runtime"):
                return 2
            if origin.startswith("market.catalog.custom"):
                return 1
            if origin.startswith("market.catalog.builtin"):
                return 0
            return 0

        def _status_rank(spec: ToolSpec) -> int:
            if spec.status is SpecLifecycle.ACTIVE:
                return 2
            if spec.status is SpecLifecycle.DRAFT:
                return 1
            return 0

        self._tool_runtime_names = {}
        chosen: dict[str, tuple[tuple[int, int, str], str]] = {}

        def _consider(key: str, spec: ToolSpec) -> None:
            if not key:
                return
            score = (_origin_rank(spec), _status_rank(spec), spec.id)
            existing = chosen.get(key)
            if existing is None or score > existing[0]:
                chosen[key] = (score, spec.id)

        for spec in self._tools.values():
            runtime_name = spec.runtime_name if isinstance(spec.runtime_name, str) else None
            if runtime_name:
                _consider(runtime_name.strip(), spec)
            _consider(str(spec.name or "").strip(), spec)
            for alias in spec.aliases:
                _consider(str(alias or "").strip(), spec)

        self._tool_runtime_names = {key: spec_id for key, (_, spec_id) in chosen.items()}

    def _rebuild_agent_alias_index(self) -> None:
        self._agent_aliases = {}
        for spec_id, spec in self._agents.items():
            self._agent_aliases[spec_id] = spec_id
            self._agent_aliases[spec.name] = spec_id
            for alias in self._agent_declared_aliases.get(spec_id, []):
                self._agent_aliases[alias] = spec_id

    def _rebuild_skill_name_index(self) -> None:
        # Prefer local skill store entries over market catalog templates when names collide.
        def _origin_rank(spec: SkillSpec) -> int:
            meta = spec.metadata if isinstance(spec.metadata, dict) else {}
            origin = str(meta.get("origin") or "").strip()
            if origin.startswith("skill_store"):
                return 2
            if origin.startswith("market.catalog.custom"):
                return 1
            if origin.startswith("market.catalog.builtin"):
                return 0
            return 0

        def _status_rank(spec: SkillSpec) -> int:
            if spec.status is SpecLifecycle.ACTIVE:
                return 2
            if spec.status is SpecLifecycle.DRAFT:
                return 1
            return 0

        chosen: dict[str, tuple[tuple[int, int, str], str]] = {}
        for spec_id, spec in self._skills.items():
            name = spec.name
            score = (_origin_rank(spec), _status_rank(spec), spec_id)
            existing = chosen.get(name)
            if existing is None or score > existing[0]:
                chosen[name] = (score, spec_id)

        self._skill_names = {name: spec_id for name, (_, spec_id) in chosen.items()}

    def _rebuild_mcp_name_index(self) -> None:
        # Multiple MCP specs can share the same `name` when we merge:
        # - runtime config (`mcp.json`)
        # - runtime-discovered servers (`mcp.runtime`)
        # - builtin/custom market catalog entries
        #
        # Name resolution must prefer user/runtime config over market templates.
        def _origin_rank(spec: McpServerSpec) -> int:
            meta = spec.metadata if isinstance(spec.metadata, dict) else {}
            origin = str(meta.get("origin") or "").strip()
            if origin.startswith("mcp.config"):
                return 3
            if origin.startswith("mcp.runtime"):
                return 2
            if origin.startswith("market.catalog.custom"):
                return 1
            if origin.startswith("market.catalog.builtin"):
                return 0
            return 0

        def _status_rank(spec: McpServerSpec) -> int:
            # Prefer active > draft > deprecated.
            if spec.status is SpecLifecycle.ACTIVE:
                return 2
            if spec.status is SpecLifecycle.DRAFT:
                return 1
            return 0

        chosen: dict[str, tuple[tuple[int, int, int, str], str]] = {}
        for spec_id, spec in self._mcp_servers.items():
            name = spec.name
            score = (
                _origin_rank(spec),
                1 if bool(spec.enabled) else 0,
                _status_rank(spec),
                spec_id,
            )
            existing = chosen.get(name)
            if existing is None or score > existing[0]:
                chosen[name] = (score, spec_id)

        self._mcp_names = {name: spec_id for name, (_, spec_id) in chosen.items()}

    def refresh_from_runtime(
        self,
        *,
        tool_registry: "ToolRegistry | None",
        skill_store: "SkillStore | None",
        mcp_config: McpConfig | None,
        include_builtin_subagents: bool = True,
    ) -> None:
        self.clear()

        if tool_registry is not None:
            self._load_tool_specs(tool_registry=tool_registry)
        if skill_store is not None:
            self._load_skill_specs(skill_store=skill_store)
        if mcp_config is not None:
            self._load_mcp_specs(mcp_config=mcp_config)
        if include_builtin_subagents:
            self._load_builtin_subagent_agents()
        self._load_market_catalog_specs()

    def _load_tool_specs(self, *, tool_registry: "ToolRegistry") -> None:
        for tool in tool_registry.list_specs():
            runtime_name = str(tool.name or "").strip()
            if not runtime_name:
                continue
            spec_id = make_tool_spec_id(namespace="local", runtime_name=runtime_name)

            side_effect_level = SideEffectLevel.NONE
            approval_required = False
            runtime = ToolRuntimePolicy(timeout_sec=30)
            if runtime_name in {"project__apply_edits", "project__patch", "project__apply_patch", "shell__run", "browser__run"}:
                side_effect_level = SideEffectLevel.HIGH
                approval_required = runtime_name in {"shell__run", "browser__run", "project__apply_patch", "project__patch"}
                if runtime_name == "browser__run":
                    side_effect_level = SideEffectLevel.LOW
            elif runtime_name.startswith("spec__"):
                side_effect_level = SideEffectLevel.LOW
                approval_required = runtime_name in {"spec__apply", "spec__seal"}

            self.register_tool(
                ToolSpec(
                    id=spec_id,
                    name=runtime_name,
                    vendor="aura",
                    status=SpecLifecycle.ACTIVE,
                    description=str(tool.description or "").strip() or None,
                    kind=ToolKind.LOCAL,
                    runtime_name=runtime_name,
                    aliases=[runtime_name],
                    entrypoint=ToolEntrypoint(type=ToolEntrypointType.PYTHON_CALLABLE, ref=runtime_name),
                    params_schema=(
                        dict(tool.input_schema) if isinstance(tool.input_schema, dict) else {"type": "object", "properties": {}}
                    ),
                    effects=ToolEffectProfile(side_effect_level=side_effect_level, idempotent=side_effect_level is SideEffectLevel.NONE),
                    runtime=runtime,
                    policy=ToolAccessPolicy(approval_required=approval_required),
                    metadata={"origin": "runtime.tool_registry"},
                )
            )

    def _load_skill_specs(self, *, skill_store: "SkillStore") -> None:
        for meta in skill_store.list():
            spec_id = make_skill_spec_id(namespace="local", skill_name=meta.name)
            skill_path = None
            try:
                skill_path = str(meta.skill_md_path.relative_to(self._project_root))
            except Exception:
                skill_path = str(meta.skill_md_path)

            requires_tool_ids: list[str] = []
            for runtime_name in meta.allowed_tools or []:
                tool_id = self.resolve_tool_id_by_runtime_name(runtime_name)
                if tool_id is not None:
                    requires_tool_ids.append(tool_id)

            self.register_skill(
                SkillSpec(
                    id=spec_id,
                    name=meta.name,
                    vendor="local",
                    status=SpecLifecycle.ACTIVE,
                    description=meta.description,
                    source_path=skill_path,
                    entry=SkillEntrySpec(type=SkillEntryType.MARKDOWN, ref=skill_path or "SKILL.md"),
                    requires_tool_ids=requires_tool_ids,
                    output_contract=SkillOutputContract(format="markdown"),
                    metadata={"origin": "skill_store", "allowed_tools": list(meta.allowed_tools or [])},
                )
            )

    @staticmethod
    def _agent_aliases_from_metadata(spec: AgentSpec) -> list[str]:
        metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
        raw = metadata.get("aliases")
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            alias = str(item or "").strip()
            if not alias or alias in seen:
                continue
            seen.add(alias)
            out.append(alias)
        return out

    def _builtin_market_catalog_root(self) -> Path:
        return Path(__file__).resolve().parents[2] / "market" / "catalog"

    def _custom_market_catalog_root(self) -> Path:
        return self._project_root / ".aura" / "market" / "custom"

    @staticmethod
    def _is_within(base: Path, target: Path) -> bool:
        try:
            target.relative_to(base)
            return True
        except Exception:
            return False

    def _iter_market_catalog_roots(self) -> list[tuple[str, Path]]:
        roots: list[tuple[str, Path]] = []
        for source_name, root in (
            ("builtin", self._builtin_market_catalog_root()),
            ("custom", self._custom_market_catalog_root()),
        ):
            root = root.expanduser().resolve()
            if not root.exists() or not root.is_dir():
                continue
            roots.append((source_name, root))
        return roots

    def _iter_market_catalog_files(self, *, root: Path, kind: str) -> list[Path]:
        # `index.json` is treated as an optional manifest, not a required gate.
        # If present, we load its explicit paths first, then union with a directory scan.
        # This keeps hand-editing workflow smooth: adding a file doesn't require updating index.json.
        index_path = root / "index.json"
        indexed: list[Path] = []
        if index_path.exists():
            try:
                raw = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                raw = None
            section = raw.get("assets") if isinstance(raw, dict) and isinstance(raw.get("assets"), dict) else raw
            entries = section.get(kind) if isinstance(section, dict) else None
            if isinstance(entries, list):
                for item in entries:
                    rel = str(item or "").strip()
                    if not rel:
                        continue
                    candidate = (root / rel).expanduser().resolve()
                    if not self._is_within(root, candidate):
                        continue
                    if candidate.is_file() and candidate.suffix.lower() == ".json":
                        indexed.append(candidate)

        kind_dir = root / kind
        if not kind_dir.exists() or not kind_dir.is_dir():
            scanned: list[Path] = []
        else:
            scanned = sorted(
                path
                for path in kind_dir.rglob("*.json")
                if path.is_file() and not path.is_symlink() and self._is_within(root, path.resolve())
            )

        out: list[Path] = []
        seen: set[str] = set()
        for path in [*indexed, *scanned]:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
        return out

    @staticmethod
    def _market_catalog_rows(raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if not isinstance(raw, dict):
            return []
        if isinstance(raw.get("items"), list):
            return [item for item in raw.get("items") if isinstance(item, dict)]
        if isinstance(raw.get("data"), list):
            return [item for item in raw.get("data") if isinstance(item, dict)]
        return [raw]

    def _load_market_catalog_specs(self) -> None:
        for source_name, root in self._iter_market_catalog_roots():
            for kind in ("tools", "mcp_servers", "skills", "agents"):
                for path in self._iter_market_catalog_files(root=root, kind=kind):
                    try:
                        raw = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    rows = self._market_catalog_rows(raw)
                    if not rows:
                        continue

                    try:
                        rel_path = str(path.relative_to(root))
                    except Exception:
                        rel_path = str(path)

                    for row in rows:
                        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                        metadata = dict(metadata)
                        metadata.setdefault("origin", f"market.catalog.{source_name}")
                        metadata.setdefault("catalog_source", rel_path)
                        row_with_meta = dict(row)
                        row_with_meta["metadata"] = metadata
                        try:
                            if kind == "tools":
                                self.upsert_tool(ToolSpec.model_validate(row_with_meta))
                                continue
                            if kind == "mcp_servers":
                                self.upsert_mcp_server(McpServerSpec.model_validate(row_with_meta))
                                continue
                            if kind == "skills":
                                self.upsert_skill(SkillSpec.model_validate(row_with_meta))
                                continue
                            spec = AgentSpec.model_validate(row_with_meta)
                            aliases = self._agent_aliases_from_metadata(spec)
                            self.upsert_agent(spec, aliases=aliases)
                        except Exception:
                            continue

    def _load_mcp_specs(self, *, mcp_config: McpConfig) -> None:
        for server_name, server in sorted(mcp_config.servers.items()):
            try:
                spec = self._build_mcp_server_spec(server_name=server_name, server=server)
                self.register_mcp_server(spec)
            except Exception:
                continue

    def backfill_mcp_runtime_tools(self, *, server_name: str, tools: list[dict[str, Any]]) -> list[ToolSpec]:
        """
        Backfill MCP runtime-discovered tools into the registry.

        Input items:
        - runtime_name: actual runtime tool name exposed to model (prefixed)
        - remote_name: server-side raw tool name
        - description: optional
        - input_schema: optional dict
        """

        server_name = str(server_name or "").strip()
        if not server_name:
            return []

        server_id = self.resolve_mcp_id_by_name(server_name)
        server = self.get_mcp_server(server_id) if server_id is not None else None
        if server is None:
            server = McpServerSpec(
                id=make_mcp_spec_id(namespace="runtime", server_name=server_name),
                name=server_name,
                vendor="runtime",
                status=SpecLifecycle.ACTIVE,
                enabled=True,
                transport=McpTransport.UNKNOWN,
                auth=McpAuthSpec(type=McpAuthType.NONE, scopes=[]),
                healthcheck=McpHealthcheck(interval_sec=60, timeout_sec=5),
                metadata={"origin": "mcp.runtime"},
            )
            self.upsert_mcp_server(server)

        provided_by_remote: dict[str, McpProvidedToolSpec] = {p.remote_name: p for p in server.provides_tools}
        provided_ordered: list[McpProvidedToolSpec] = list(server.provides_tools)
        provided_idx: dict[str, int] = {p.remote_name: i for i, p in enumerate(provided_ordered)}

        server_slug = _slug_token(server.name)
        updated_tools: list[ToolSpec] = []

        for item in tools:
            if not isinstance(item, dict):
                continue
            runtime_name = str(item.get("runtime_name") or "").strip()
            remote_name = str(item.get("remote_name") or "").strip()
            if not runtime_name or not remote_name:
                continue
            description = str(item.get("description") or "").strip() or None
            input_schema = item.get("input_schema")
            if not isinstance(input_schema, dict):
                input_schema = {"type": "object", "properties": {}}

            existed = provided_by_remote.get(remote_name)
            tool_id = existed.tool_id if existed is not None and isinstance(existed.tool_id, str) and existed.tool_id else None
            if tool_id is None:
                tool_id = make_tool_spec_id(namespace=f"mcp.{server_slug}", runtime_name=remote_name)

            canonical_name = f"mcp:{server.name}:{remote_name}"
            tool_spec = ToolSpec(
                id=tool_id,
                name=canonical_name,
                vendor="mcp",
                status=SpecLifecycle.ACTIVE if server.enabled else SpecLifecycle.DEPRECATED,
                kind=ToolKind.MCP_PROXY,
                runtime_name=runtime_name,
                aliases=[runtime_name, f"{server.name}:{remote_name}", remote_name],
                description=description,
                entrypoint=ToolEntrypoint(type=ToolEntrypointType.MCP, ref=f"{server.name}:{remote_name}"),
                params_schema=input_schema,
                effects=ToolEffectProfile(side_effect_level=SideEffectLevel.HIGH, idempotent=False),
                runtime=ToolRuntimePolicy(timeout_sec=max(1, min(600, int(server.timeout_sec)))),
                policy=ToolAccessPolicy(approval_required=True, approval_level="high"),
                mcp_binding=McpToolBinding(server_id=server.id, remote_tool=remote_name),
                metadata={"origin": "mcp.runtime", "server_name": server.name},
            )
            self.upsert_tool(tool_spec)
            updated_tools.append(tool_spec)

            provided = McpProvidedToolSpec(remote_name=remote_name, tool_id=tool_id, description=description)
            if remote_name in provided_idx:
                provided_ordered[provided_idx[remote_name]] = provided
            else:
                provided_idx[remote_name] = len(provided_ordered)
                provided_ordered.append(provided)

        if updated_tools:
            updated_server = server.model_copy(
                update={
                    "status": (SpecLifecycle.ACTIVE if server.enabled else SpecLifecycle.DEPRECATED),
                    "provides_tools": provided_ordered,
                }
            )
            self.upsert_mcp_server(updated_server)

        return updated_tools

    def _build_mcp_server_spec(self, *, server_name: str, server: McpServerConfig) -> McpServerSpec:
        server_id = make_mcp_spec_id(namespace="server", server_name=server_name)
        auth = McpAuthSpec(type=McpAuthType.NONE, scopes=[])
        command = (server.command or "").strip()
        transport = McpTransport.STDIO if command else McpTransport.UNKNOWN
        return McpServerSpec(
            id=server_id,
            name=server_name,
            vendor="local",
            status=SpecLifecycle.ACTIVE if server.enabled else SpecLifecycle.DEPRECATED,
            enabled=bool(server.enabled),
            transport=transport,
            command=command or None,
            args=list(server.args or []),
            env=dict(server.env or {}),
            cwd=server.cwd,
            timeout_sec=int(max(1.0, float(server.timeout_s))),
            auth=auth,
            healthcheck=McpHealthcheck(interval_sec=60, timeout_sec=max(1, min(30, int(server.timeout_s)))),
            metadata={"origin": "mcp.config", "source": "mcp.json"},
        )

    def _load_builtin_subagent_agents(self) -> None:
        from ..subagents.presets import get_preset, list_presets

        capability_map: dict[str, list[str]] = {
            "file_ops_worker": ["file_ops", "workspace_editing"],
            "doc_worker": ["document_generation", "skill_execution"],
            "sheet_worker": ["spreadsheet_processing", "skill_execution"],
            "browser_worker": ["web_research", "browser_automation"],
            "verifier": ["verification", "qa"],
        }

        for preset_name in list_presets():
            preset = get_preset(preset_name)
            if preset is None:
                continue

            tool_ids: list[str] = []
            for runtime_name in preset.default_allowlist:
                tool_id = self.resolve_tool_id_by_runtime_name(runtime_name)
                if tool_id is not None:
                    tool_ids.append(tool_id)

            agent = AgentSpec(
                id=make_agent_spec_id(namespace="subagent", agent_name=preset.name),
                name=preset.name,
                vendor="aura",
                status=SpecLifecycle.ACTIVE,
                summary=f"Builtin subagent preset: {preset.name}",
                role="subagent",
                capabilities=capability_map.get(preset.name, []),
                tool_ids=tool_ids,
                execution=AgentExecutionSpec(
                    mode=AgentExecutionMode.SUBAGENT_PRESET,
                    preset_name=preset.name,
                    default_allowlist=list(preset.default_allowlist),
                    default_max_turns=int(preset.limits.max_turns),
                    default_max_tool_calls=int(preset.limits.max_tool_calls),
                    safe_shell_prefixes=list(preset.safe_shell_prefixes),
                    auto_approve_tools=list(preset.auto_approve_tools),
                ),
                metadata={"origin": "subagent.preset"},
            )
            self.register_agent(agent, aliases=[preset.name])

    def to_catalog_dict(self) -> dict[str, Any]:
        return {
            "counts": {
                "agents": len(self._agents),
                "skills": len(self._skills),
                "tools": len(self._tools),
                "mcp_servers": len(self._mcp_servers),
            },
            "agents": [spec.model_dump(mode="json") for spec in self.list_agents()],
            "skills": [spec.model_dump(mode="json") for spec in self.list_skills()],
            "tools": [spec.model_dump(mode="json") for spec in self.list_tools()],
            "mcp_servers": [spec.model_dump(mode="json") for spec in self.list_mcp_servers()],
        }

    def write_catalog_snapshot(self, *, path: Path | None = None) -> Path:
        output_path = path
        if output_path is None:
            output_path = self._project_root / ".aura" / "state" / "market" / "spec_catalog.json"
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_catalog_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(output_path)
        return output_path
