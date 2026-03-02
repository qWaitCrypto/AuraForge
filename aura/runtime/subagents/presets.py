from __future__ import annotations

import importlib.resources
import json
import warnings
from dataclasses import dataclass, field
from typing import Any


def load_prompt_asset(name: str) -> str:
    return (
        importlib.resources.files("aura.runtime")
        .joinpath("prompts", name)
        .read_text(encoding="utf-8", errors="replace")
    )


@dataclass(frozen=True, slots=True)
class SubagentLimits:
    max_turns: int
    max_tool_calls: int


@dataclass(frozen=True, slots=True)
class SubagentPreset:
    name: str
    prompt_asset: str
    default_allowlist: list[str]
    limits: SubagentLimits
    safe_shell_prefixes: list[str] = field(default_factory=list)
    auto_approve_tools: list[str] = field(default_factory=list)

    def load_prompt(self) -> str:
        return load_prompt_asset(self.prompt_asset)


def _load_presets() -> dict[str, SubagentPreset]:
    source = "aura/runtime/subagents/presets.json"
    try:
        raw = (
            importlib.resources.files("aura.runtime")
            .joinpath("subagents", "presets.json")
            .read_text(encoding="utf-8", errors="replace")
        )
        payload = json.loads(raw)
    except Exception as exc:
        warnings.warn(f"Failed to load {source}: {exc}", RuntimeWarning, stacklevel=2)
        return {}

    if not isinstance(payload, list):
        warnings.warn(f"Invalid preset payload in {source}: root must be a JSON array", RuntimeWarning, stacklevel=2)
        return {}

    out: dict[str, SubagentPreset] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue

        name = str(row.get("name") or "").strip()
        prompt_asset = str(row.get("prompt_asset") or "").strip()
        if not name or not prompt_asset:
            continue

        allow = row.get("default_allowlist")
        default_allowlist = [str(item).strip() for item in allow if isinstance(item, str) and str(item).strip()] if isinstance(allow, list) else []

        limits_raw = row.get("limits")
        if not isinstance(limits_raw, dict):
            continue
        try:
            limits = SubagentLimits(
                max_turns=int(limits_raw.get("max_turns")),
                max_tool_calls=int(limits_raw.get("max_tool_calls")),
            )
        except Exception:
            continue

        safe_prefixes_raw = row.get("safe_shell_prefixes")
        safe_shell_prefixes = (
            [str(item).strip() for item in safe_prefixes_raw if isinstance(item, str) and str(item).strip()]
            if isinstance(safe_prefixes_raw, list)
            else []
        )
        auto_tools_raw = row.get("auto_approve_tools")
        auto_approve_tools = (
            [str(item).strip() for item in auto_tools_raw if isinstance(item, str) and str(item).strip()]
            if isinstance(auto_tools_raw, list)
            else []
        )

        out[name] = SubagentPreset(
            name=name,
            prompt_asset=prompt_asset,
            default_allowlist=default_allowlist,
            limits=limits,
            safe_shell_prefixes=safe_shell_prefixes,
            auto_approve_tools=auto_approve_tools,
        )

    if not out:
        warnings.warn(f"No valid presets loaded from {source}", RuntimeWarning, stacklevel=2)
    return out


_PRESETS: dict[str, SubagentPreset] = _load_presets()


def get_preset(name: str) -> SubagentPreset | None:
    return _PRESETS.get(str(name or "").strip())


def list_presets() -> list[str]:
    return sorted(_PRESETS)


def preset_input_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": list_presets(),
        "description": "Subagent preset.",
    }
