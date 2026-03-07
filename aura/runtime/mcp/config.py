from __future__ import annotations

import functools
import json
import os
import re
import subprocess
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MCP_CONFIG_FILENAME = "mcp.json"
_ENV_REF_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")
_LOG_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_GH_TOKEN_UNSET = object()
_GH_AUTH_TOKEN_CACHE: object = _GH_TOKEN_UNSET


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """
    Minimal MCP server config (v0.3).

    Router mode uses a small set of local tools to list/call MCP tools.
    For now we support only stdio servers (spawned subprocess).
    """

    name: str
    enabled: bool
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str | None
    timeout_s: float


@dataclass(frozen=True, slots=True)
class McpConfig:
    servers: dict[str, McpServerConfig]
    source: str | None = None


def _as_dict(v: Any) -> dict[str, Any]:
    if isinstance(v, dict):
        return v
    raise ValueError("Expected object.")


def _as_str(v: Any) -> str:
    if isinstance(v, str):
        return v
    raise ValueError("Expected string.")


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    raise ValueError("Expected boolean.")


def _as_float(v: Any) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    raise ValueError("Expected number.")


def _as_str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        raise ValueError("Expected list.")
    out: list[str] = []
    for item in v:
        if not isinstance(item, str):
            raise ValueError("Expected list of strings.")
        if item:
            out.append(item)
    return out


def _as_env_dict(v: Any) -> dict[str, str]:
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError("Expected env object.")
    out: dict[str, str] = {}
    for k, val in v.items():
        if not isinstance(k, str) or not k:
            continue
        if not isinstance(val, str):
            continue
        resolved = _expand_env_refs(val)
        key = str(k or "").strip()
        if not resolved and key in {"GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN"}:
            resolved = str(os.environ.get("GITHUB_TOKEN", "") or "").strip()
            if not resolved:
                resolved = str(_gh_auth_token() or "")
        out[key] = resolved
    return out


def _expand_env_refs(value: str) -> str:
    text = str(value or "")
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        key = str(match.group(1) or "").strip()
        if not key:
            return ""
        return str(os.environ.get(key, ""))

    return _ENV_REF_RE.sub(_replace, text)


def mcp_logs_dir_for_project(project_root: Path) -> Path:
    return project_root / ".aura" / "logs"


def mcp_stderr_log_path(*, project_root: Path, server_name: str) -> Path:
    cleaned = _LOG_SAFE_RE.sub("_", str(server_name or "").strip()).strip("._")
    if not cleaned:
        cleaned = "server"
    return mcp_logs_dir_for_project(project_root) / f"mcp_{cleaned}.log"


@contextmanager
def mcp_stdio_errlog_context(*, project_root: Path, server_name: str):
    log_path = mcp_stderr_log_path(project_root=project_root, server_name=server_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import mcp.client.stdio as _mcp_stdio_mod
    except Exception:
        yield log_path
        return

    original = getattr(_mcp_stdio_mod, "stdio_client", None)
    if original is None:
        yield log_path
        return

    @functools.wraps(original)
    def _logged_stdio_client(*args: Any, **kwargs: Any):
        if kwargs.get("errlog") is not None:
            return original(*args, **kwargs)

        handle = open(log_path, "a", encoding="utf-8")
        kwargs["errlog"] = handle
        inner = original(*args, **kwargs)

        @asynccontextmanager
        async def _wrapped():
            try:
                async with inner as result:
                    yield result
            finally:
                try:
                    handle.close()
                except Exception:
                    pass

        return _wrapped()

    _mcp_stdio_mod.stdio_client = _logged_stdio_client
    try:
        yield log_path
    finally:
        _mcp_stdio_mod.stdio_client = original


def _gh_auth_token() -> str | None:
    global _GH_AUTH_TOKEN_CACHE
    if _GH_AUTH_TOKEN_CACHE is not _GH_TOKEN_UNSET:
        cached = str(_GH_AUTH_TOKEN_CACHE or "").strip()
        return cached or None

    token = ""
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"],
            text=True,
            capture_output=True,
            timeout=2,
        )
    except Exception:
        token = ""
    else:
        if proc.returncode == 0:
            token = str(proc.stdout or "").strip()
    _GH_AUTH_TOKEN_CACHE = token
    return token or None


def _load_mcp_config_dict(data: Any, *, source: str) -> McpConfig:
    root = _as_dict(data)
    servers_raw = root.get("mcpServers", {})
    if servers_raw is None:
        servers_raw = {}
    servers_obj = _as_dict(servers_raw)

    servers: dict[str, McpServerConfig] = {}
    for name, raw in servers_obj.items():
        if not isinstance(name, str) or not name.strip():
            continue
        cfg = _as_dict(raw)
        # Explicit server entries default to enabled unless disabled by config.
        enabled = _as_bool(cfg.get("enabled", True))
        command = _as_str(cfg.get("command", ""))
        args = _as_str_list(cfg.get("args", []))
        env = _as_env_dict(cfg.get("env"))
        cwd_raw = cfg.get("cwd")
        if isinstance(cwd_raw, str):
            cwd = cwd_raw.strip() or None
        else:
            cwd = None
        timeout_s = _as_float(cfg.get("timeout_s", 60))

        servers[name] = McpServerConfig(
            name=name,
            enabled=enabled,
            command=command.strip(),
            args=args,
            env=env,
            cwd=cwd,
            timeout_s=timeout_s,
        )

    return McpConfig(servers=servers, source=source)


def mcp_config_path_for_project(project_root: Path) -> Path:
    return project_root / ".aura" / "config" / MCP_CONFIG_FILENAME


def load_mcp_config(*, project_root: Path) -> McpConfig:
    path = mcp_config_path_for_project(project_root)
    if not path.exists():
        return McpConfig(servers={}, source=None)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"MCP config file is not valid JSON: {path} ({e})") from e
    return _load_mcp_config_dict(data, source=str(path))
