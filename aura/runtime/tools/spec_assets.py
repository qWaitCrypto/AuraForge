from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..registry import SpecResolver
from ..registry.spec_registry import SpecRegistry


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def _validate_limit(v: Any, *, default: int = 100, min_value: int = 1, max_value: int = 500) -> int:
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError("Invalid 'limit' (expected integer).")
    if v < min_value or v > max_value:
        raise ValueError(f"Invalid 'limit' (expected integer between {min_value} and {max_value}).")
    return v


def _matches_query(payload: dict[str, Any], query: str) -> bool:
    q = _norm(query)
    if not q:
        return True
    chunks: list[str] = []
    for key in ("id", "name", "vendor", "role", "kind", "runtime_name", "transport", "source_path", "summary"):
        v = payload.get(key)
        if isinstance(v, str):
            chunks.append(v)
    tags = payload.get("tags")
    if isinstance(tags, list):
        chunks.extend([x for x in tags if isinstance(x, str)])
    hay = " ".join(chunks).lower()
    return q in hay


def _agent_summary(spec: Any, *, include_relations: bool) -> dict[str, Any]:
    out = {
        "id": spec.id,
        "name": spec.name,
        "status": getattr(spec.status, "value", str(spec.status)),
        "vendor": spec.vendor,
        "role": spec.role,
        "capabilities": list(spec.capabilities),
        "execution_mode": spec.execution.mode.value,
        "preset_name": spec.execution.preset_name,
        "skill_count": len(spec.skill_ids),
        "tool_count": len(spec.tool_ids),
        "mcp_server_count": len(spec.mcp_server_ids),
        "tags": list(spec.tags),
    }
    if include_relations:
        out["skill_ids"] = list(spec.skill_ids)
        out["tool_ids"] = list(spec.tool_ids)
        out["mcp_server_ids"] = list(spec.mcp_server_ids)
    return out


def _skill_summary(spec: Any, *, include_relations: bool) -> dict[str, Any]:
    out = {
        "id": spec.id,
        "name": spec.name,
        "status": getattr(spec.status, "value", str(spec.status)),
        "vendor": spec.vendor,
        "source_path": spec.source_path,
        "trigger_count": len(spec.triggers),
        "requires_tool_count": len(spec.requires_tool_ids),
        "tags": list(spec.tags),
    }
    if include_relations:
        out["triggers"] = list(spec.triggers)
        out["requires_tool_ids"] = list(spec.requires_tool_ids)
        out["requires_capabilities"] = list(spec.requires_capabilities)
    return out


def _tool_summary(spec: Any, *, include_relations: bool) -> dict[str, Any]:
    out = {
        "id": spec.id,
        "name": spec.name,
        "status": getattr(spec.status, "value", str(spec.status)),
        "vendor": spec.vendor,
        "kind": spec.kind.value,
        "runtime_name": spec.runtime_name,
        "approval_required": bool(spec.policy.approval_required),
        "tags": list(spec.tags),
    }
    if include_relations and spec.mcp_binding is not None:
        out["mcp_binding"] = {
            "server_id": spec.mcp_binding.server_id,
            "remote_tool": spec.mcp_binding.remote_tool,
        }
    return out


def _mcp_summary(spec: Any, *, include_relations: bool) -> dict[str, Any]:
    out = {
        "id": spec.id,
        "name": spec.name,
        "status": getattr(spec.status, "value", str(spec.status)),
        "vendor": spec.vendor,
        "enabled": bool(spec.enabled),
        "transport": spec.transport.value,
        "tool_count": len(spec.provides_tools),
        "tags": list(spec.tags),
    }
    if include_relations:
        out["provides_tools"] = [item.model_dump(mode="json") for item in spec.provides_tools]
    return out


@dataclass(frozen=True, slots=True)
class SpecListAssetsTool:
    registry: SpecRegistry
    name: str = "spec__list_assets"
    description: str = "List normalized spec assets (agents/skills/tools/mcp_servers) for runtime inspection."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["all", "agents", "skills", "tools", "mcp_servers"],
                    "description": "Asset kind to list (default: all).",
                },
                "query": {"type": "string", "description": "Optional keyword filter (id/name/vendor/tags)."},
                "include_inactive": {"type": "boolean", "description": "Include non-active specs."},
                "include_relations": {"type": "boolean", "description": "Include relation ids/details in list output."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "description": "Max entries per kind."},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root) -> dict[str, Any]:
        del project_root
        kind = _norm(args.get("kind") or "all")
        if kind not in {"all", "agents", "skills", "tools", "mcp_servers"}:
            raise ValueError("Invalid 'kind'.")
        query = str(args.get("query") or "").strip()
        include_inactive = bool(args.get("include_inactive", False))
        include_relations = bool(args.get("include_relations", False))
        limit = _validate_limit(args.get("limit"), default=100)

        def _filter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for row in rows:
                status = _norm(row.get("status"))
                if not include_inactive and status and status != "active":
                    continue
                if not _matches_query(row, query):
                    continue
                out.append(row)
                if len(out) >= limit:
                    break
            return out

        agents = _filter([_agent_summary(a, include_relations=include_relations) for a in self.registry.list_agents()])
        skills = _filter([_skill_summary(s, include_relations=include_relations) for s in self.registry.list_skills()])
        tools = _filter([_tool_summary(t, include_relations=include_relations) for t in self.registry.list_tools()])
        mcp_servers = _filter([_mcp_summary(m, include_relations=include_relations) for m in self.registry.list_mcp_servers()])

        payload: dict[str, Any] = {
            "ok": True,
            "kind": kind,
            "query": query or None,
            "counts": {
                "agents": len(agents),
                "skills": len(skills),
                "tools": len(tools),
                "mcp_servers": len(mcp_servers),
            },
        }
        if kind == "all":
            payload["agents"] = agents
            payload["skills"] = skills
            payload["tools"] = tools
            payload["mcp_servers"] = mcp_servers
            return payload
        payload["items"] = {
            "agents": agents,
            "skills": skills,
            "tools": tools,
            "mcp_servers": mcp_servers,
        }[kind]
        return payload


@dataclass(frozen=True, slots=True)
class SpecGetAssetTool:
    registry: SpecRegistry
    resolver: SpecResolver | None = None
    name: str = "spec__get_asset"
    description: str = "Get detailed normalized spec asset by kind and id/name (read-only)."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["agents", "skills", "tools", "mcp_servers"],
                    "description": "Asset kind.",
                },
                "id": {"type": "string", "description": "Spec id or known alias/name."},
                "include_related": {"type": "boolean", "description": "Include related assets."},
            },
            "required": ["kind", "id"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root) -> dict[str, Any]:
        del project_root
        kind = _norm(args.get("kind"))
        if kind not in {"agents", "skills", "tools", "mcp_servers"}:
            raise ValueError("Invalid 'kind'.")
        identifier = str(args.get("id") or "").strip()
        if not identifier:
            raise ValueError("Missing or invalid 'id'.")
        include_related = bool(args.get("include_related", True))

        if kind == "agents":
            resolved = self.registry.resolve_agent_id(identifier)
            spec = self.registry.get_agent(resolved) if resolved else None
            if spec is None:
                raise ValueError(f"Unknown agent: {identifier!r}")
            out: dict[str, Any] = {"ok": True, "kind": kind, "item": spec.model_dump(mode="json")}
            if include_related and self.resolver is not None:
                bundle = self.resolver.resolve_agent(spec.id, strict=False)
                out["related"] = {
                    "skills": [_skill_summary(x, include_relations=False) for x in bundle.skills],
                    "tools": [_tool_summary(x, include_relations=True) for x in bundle.tools],
                    "mcp_servers": [_mcp_summary(x, include_relations=False) for x in bundle.mcp_servers],
                    "issues": [
                        {"severity": i.severity.value, "reference": i.reference, "message": i.message}
                        for i in bundle.issues
                    ],
                }
            return out

        if kind == "skills":
            resolved = self.registry.resolve_skill_id_by_name(identifier) or identifier
            spec = self.registry.get_skill(resolved)
            if spec is None:
                raise ValueError(f"Unknown skill: {identifier!r}")
            out = {"ok": True, "kind": kind, "item": spec.model_dump(mode="json")}
            if include_related:
                tools = []
                for tool_id in spec.requires_tool_ids:
                    tool = self.registry.get_tool(tool_id)
                    if tool is not None:
                        tools.append(_tool_summary(tool, include_relations=True))
                out["related"] = {"tools": tools}
            return out

        if kind == "tools":
            resolved = self.registry.resolve_tool_id_by_runtime_name(identifier) or identifier
            spec = self.registry.get_tool(resolved)
            if spec is None:
                raise ValueError(f"Unknown tool: {identifier!r}")
            out = {"ok": True, "kind": kind, "item": spec.model_dump(mode="json")}
            if include_related and spec.mcp_binding is not None:
                server = self.registry.get_mcp_server(spec.mcp_binding.server_id)
                out["related"] = {"mcp_server": (server.model_dump(mode="json") if server is not None else None)}
            return out

        # mcp_servers
        resolved = self.registry.resolve_mcp_id_by_name(identifier) or identifier
        spec = self.registry.get_mcp_server(resolved)
        if spec is None:
            raise ValueError(f"Unknown mcp server: {identifier!r}")
        out = {"ok": True, "kind": kind, "item": spec.model_dump(mode="json")}
        if include_related:
            tool_rows: list[dict[str, Any]] = []
            for item in spec.provides_tools:
                tool = self.registry.get_tool(item.tool_id) if isinstance(item.tool_id, str) and item.tool_id else None
                if tool is None:
                    tool_rows.append(item.model_dump(mode="json"))
                    continue
                tool_rows.append(_tool_summary(tool, include_relations=True))
            out["related"] = {"tools": tool_rows}
        return out
