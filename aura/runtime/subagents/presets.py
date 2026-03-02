from __future__ import annotations

import importlib.resources
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


_PRESETS: dict[str, SubagentPreset] = {
    "file_ops_worker": SubagentPreset(
        name="file_ops_worker",
        prompt_asset="subagent_file_ops_worker.md",
        default_allowlist=[
            "project__list_dir",
            "project__glob",
            "project__read_text",
            "project__read_text_many",
            "project__apply_edits",
            "shell__run",
            "snapshot__create",
            "snapshot__diff",
            "workspace__context",
            "workspace__register_submission",
        ],
        limits=SubagentLimits(max_turns=12, max_tool_calls=24),
    ),
    "doc_worker": SubagentPreset(
        name="doc_worker",
        prompt_asset="subagent_doc_worker.md",
        default_allowlist=[
            # Skills (docx/pdf)
            "skill__list",
            "skill__load",
            "skill__read_file",

            # Project IO
            "project__list_dir",
            "project__glob",
            "project__read_text",
            "project__read_text_many",
            "project__apply_edits",
            "project__search_text",

            # Execute skill scripts (approval-gated via ToolRuntime + approval agent)
            "shell__run",

            "snapshot__create",
            "snapshot__diff",
            "workspace__context",
            "workspace__register_submission",
        ],
        limits=SubagentLimits(max_turns=12, max_tool_calls=24),
        # Auto-approve running the *skill runner* script (see subagents/runner.py safety checks).
        safe_shell_prefixes=[
            ".aura/skills/aura-docx/scripts/run.py",
            ".aura/skills/aura-pdf/scripts/run.py",
        ],
    ),
    "sheet_worker": SubagentPreset(
        name="sheet_worker",
        prompt_asset="subagent_sheet_worker.md",
        default_allowlist=[
            # Skills (xlsx)
            "skill__list",
            "skill__load",
            "skill__read_file",

            # Project IO
            "project__list_dir",
            "project__glob",
            "project__read_text",
            "project__read_text_many",
            "project__apply_edits",
            "project__search_text",

            # Execute skill scripts (approval-gated via ToolRuntime + approval agent)
            "shell__run",

            "snapshot__create",
            "snapshot__diff",
            "workspace__context",
            "workspace__register_submission",
        ],
        limits=SubagentLimits(max_turns=12, max_tool_calls=24),
        # Auto-approve running the *skill runner* script (see subagents/runner.py safety checks).
        safe_shell_prefixes=[
            ".aura/skills/aura-xlsx/scripts/run.py",
        ],
    ),
    "browser_worker": SubagentPreset(
        name="browser_worker",
        prompt_asset="subagent_browser_worker.md",
        default_allowlist=[
            "skill__list",
            "skill__load",
            "skill__read_file",
            "browser__run",
            "workspace__context",
            "workspace__register_submission",
        ],
        limits=SubagentLimits(max_turns=20, max_tool_calls=50),
        safe_shell_prefixes=[],
        # Testing mode: allow browser automation without approval prompts.
        auto_approve_tools=["browser__run"],
    ),
    "market_worker": SubagentPreset(
        name="market_worker",
        prompt_asset="subagent_market_worker.md",
        default_allowlist=[
            # Project I/O and edits.
            "project__list_dir",
            "project__glob",
            "project__read_text",
            "project__read_text_many",
            "project__search_text",
            "project__text_stats",
            "project__apply_edits",
            "project__apply_patch",
            "project__patch",
            "project__aigc_detect",

            # Local execution and browser automation.
            "shell__run",
            "browser__run",

            # Skills.
            "skill__list",
            "skill__load",
            "skill__read_file",

            # Snapshots.
            "snapshot__create",
            "snapshot__diff",
            "snapshot__list",
            "snapshot__read_text",
            "snapshot__rollback",

            # Spec read-only querying.
            "spec__query",
            "spec__get",
            "spec__list_assets",
            "spec__get_asset",

            # Session context lookup.
            "session__search",

            # Workspace collaboration.
            "workspace__context",
            "workspace__heartbeat_workbench",
            "workspace__register_submission",
            "workspace__append_submission_evidence",
            "workspace__audit_chain",
            "workspace__list_submissions",
            "workspace__timeline",
        ],
        limits=SubagentLimits(max_turns=20, max_tool_calls=60),
        safe_shell_prefixes=[],
        auto_approve_tools=["browser__run"],
    ),
    "verifier": SubagentPreset(
        name="verifier",
        prompt_asset="subagent_verifier.md",
        default_allowlist=[
            "project__list_dir",
            "project__glob",
            "project__read_text",
            "project__read_text_many",
            "project__search_text",
            "snapshot__list",
            "spec__query",
            "spec__get",
            "snapshot__read_text",
            "snapshot__diff",
            "session__search",
            "workspace__context",
            "workspace__register_submission",
        ],
        limits=SubagentLimits(max_turns=6, max_tool_calls=12),
    ),
}


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
