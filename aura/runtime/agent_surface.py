from __future__ import annotations

from dataclasses import dataclass

from .llm.types import ToolSpec
from .skills import SkillMetadata


@dataclass(frozen=True, slots=True)
class SpecStatusSummary:
    status: str  # open|sealed|unknown
    label: str | None = None


def build_agent_surface(
    *,
    tools: list[ToolSpec],
    skills: list[SkillMetadata],
    spec: SpecStatusSummary,
    max_tool_lines: int = 40,
    max_skill_lines: int = 40,
) -> str:
    tool_names = {t.name for t in tools}
    has_tool_calling = bool(tools)
    has_skill_tools = "skill__list" in tool_names and "skill__load" in tool_names

    tool_lines = []
    for spec_item in tools[: max_tool_lines]:
        tool_lines.append(f"- {spec_item.name}: {spec_item.description}")
    if len(tools) > max_tool_lines:
        tool_lines.append(f"- ... ({len(tools) - max_tool_lines} more)")

    tool_notes: list[str] = []
    if "project__apply_edits" in tool_names:
        tool_notes.extend(
            [
                "- Prefer `project__apply_edits` for normal file edits; it uses structured JSON ops (no patch DSL).",
                "- For `update_file`/`insert_*`/`replace_substring_*`, copy exact lines/substrings via `project__read_text`/`project__search_text` (no guessing).",
            ]
        )
    if "project__patch" in tool_names:
        tool_notes.extend(
            [
                "- Use `project__patch` for patch-style edits; it accepts unified diff (`---/+++`, `@@`) like `git diff`.",
                "- Pass raw diff text (no ``` fences).",
            ]
        )

    skill_lines = []
    for meta in skills[: max_skill_lines]:
        skill_lines.append(f"- {meta.name}: {meta.description}")
    if len(skills) > max_skill_lines:
        suffix = "; call `skill__list`" if has_skill_tools else ""
        skill_lines.append(f"- ... ({len(skills) - max_skill_lines} more{suffix})")

    label = f" ({spec.label})" if spec.label else ""
    tools_section = tool_lines if tool_lines else (["- (tool calling disabled)"] if not has_tool_calling else ["- (no tools)"])
    skills_rules = (
        [
            "- Before starting a task, check available skills.",
            "- If a skill is relevant or user names it, call `skill__load` first.",
            "- If the list is truncated, call `skill__list`.",
        ]
        if has_skill_tools
        else ["- (skill tools are not available in this session)"]
    )

    return "\n".join(
        [
            "# Aura Agent Surface (v0.2)",
            "",
            "## Tools",
            *tools_section,
            *([] if not tool_notes else ["", "Notes:", *tool_notes]),
            "",
            "## Skills",
            "Rules:",
            *skills_rules,
            "",
            "<available_skills>",
            *(skill_lines if skill_lines else ["- (no skills found)"]),
            "</available_skills>",
            "",
            "## Spec",
            f"Status: {spec.status}{label}",
            "Rules:",
            "- Do not modify `spec/` via generic file tools when sealed; use spec workflow tools.",
            "- Use `spec__list_assets` / `spec__get_asset` to inspect normalized agent/skill/tool/mcp registry.",
        ]
    )
