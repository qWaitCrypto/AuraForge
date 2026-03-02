from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

from ..models.capability import AgentCapabilitySurface, ROLE_INTEGRATOR
from ..models.context import AgentContext
from ..models.sandbox import Sandbox
from ..models.signal import Signal, SignalType
from ..registry import SpecResolver


class _SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class ContextBuilder:
    """
    Assemble layered prompt context for agent sessions.
    """

    def __init__(self, *, project_root: Path, spec_resolver: SpecResolver | None = None) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._resolver = spec_resolver

    def build(
        self,
        *,
        surface: AgentCapabilitySurface,
        sandbox: Sandbox | None = None,
        signal: Signal | None = None,
        task_description: str | None = None,
        extra_context: str | None = None,
    ) -> AgentContext:
        layers: dict[str, str] = {}
        layers["system"] = self._build_system_layer()
        layers["agent_card"] = self._build_agent_card_layer(agent_id=surface.agent_id)
        layers["role_policy"] = self._build_role_policy_layer(role=surface.role)
        layers["task_brief"] = self._build_task_brief_layer(
            sandbox=sandbox,
            signal=signal,
            task_description=task_description,
            extra_context=extra_context,
        )
        layers["project_knowledge"] = self._build_project_knowledge_layer()
        layers["tool_surface"] = self._build_tool_surface_layer(surface=surface)
        system_prompt = self._assemble(layers)

        issue_key: str | None = None
        if sandbox is not None:
            issue_key = sandbox.issue_key
        elif signal is not None:
            issue_key = signal.issue_key

        sandbox_id: str | None = None
        if sandbox is not None:
            sandbox_id = sandbox.sandbox_id
        elif signal is not None:
            sandbox_id = signal.sandbox_id

        return AgentContext(
            system_prompt=system_prompt,
            layers=layers,
            agent_id=surface.agent_id,
            role=surface.role,
            issue_key=issue_key,
            sandbox_id=sandbox_id,
            trigger=self._infer_trigger(signal),
        )

    def _build_system_layer(self) -> str:
        text = self._read_prompt_asset("system_main.md")
        if text:
            return text
        return (
            "You are an Aura platform agent. Use tools to complete work, do not hallucinate, "
            "and report concrete outcomes."
        )

    def _build_agent_card_layer(self, *, agent_id: str) -> str:
        cleaned_id = str(agent_id or "").strip() or "unknown"
        if self._resolver is None:
            return f"Agent ID: {cleaned_id}"

        try:
            bundle = self._resolver.resolve_agent(cleaned_id, strict=False)
        except Exception:
            return f"Agent ID: {cleaned_id}"

        lines: list[str] = [
            f"Agent ID: {bundle.agent.id}",
            f"Name: {bundle.agent.name}",
        ]
        if isinstance(bundle.agent.summary, str) and bundle.agent.summary.strip():
            lines.append(f"Summary: {bundle.agent.summary.strip()}")
        if isinstance(bundle.agent.role, str) and bundle.agent.role.strip():
            lines.append(f"Role: {bundle.agent.role.strip()}")
        if bundle.agent.capabilities:
            lines.append("Capabilities:")
            for cap in bundle.agent.capabilities:
                if isinstance(cap, str) and cap.strip():
                    lines.append(f"- {cap.strip()}")

        metadata = bundle.agent.metadata if isinstance(bundle.agent.metadata, dict) else {}
        prompt_ref = metadata.get("prompt_ref")
        if isinstance(prompt_ref, str) and prompt_ref.strip():
            card_text = self._read_project_file(prompt_ref.strip(), max_chars=16_000)
            if card_text:
                lines.extend(["", "Agent Guidance:", card_text])

        return "\n".join(lines).strip()

    def _build_role_policy_layer(self, *, role: str) -> str:
        normalized = str(role or "").strip().lower()
        if normalized == ROLE_INTEGRATOR:
            text = self._read_prompt_asset("role_integrator.md")
            if text:
                return text
            return (
                "Integrator role: you can do worker actions and can also push branches, create/merge PRs, "
                "and advance delivery state."
            )

        text = self._read_prompt_asset("role_worker.md")
        if text:
            return text
        return (
            "Worker role: you can edit files, run tests, commit locally, and report progress. "
            "Do not push branches or create PRs."
        )

    def _build_task_brief_layer(
        self,
        *,
        sandbox: Sandbox | None,
        signal: Signal | None,
        task_description: str | None,
        extra_context: str | None,
    ) -> str:
        trigger = self._infer_trigger(signal)
        issue_key = (
            (sandbox.issue_key if sandbox is not None else None)
            or (signal.issue_key if signal is not None else None)
            or "UNKNOWN"
        )
        sandbox_id = (
            (sandbox.sandbox_id if sandbox is not None else None)
            or (signal.sandbox_id if signal is not None else None)
            or ""
        )
        worktree_path = sandbox.worktree_path if sandbox is not None else ""
        branch = sandbox.branch if sandbox is not None else ""
        brief = signal.brief if signal is not None else ""

        if trigger == "wake":
            base = self._render_prompt_asset(
                "task_wake.md",
                {
                    "issue_key": issue_key,
                    "brief": brief,
                },
            )
        elif trigger == "task_assigned":
            base = self._render_prompt_asset(
                "task_assigned.md",
                {
                    "issue_key": issue_key,
                    "sandbox_id": sandbox_id,
                    "worktree_path": worktree_path,
                    "branch": branch,
                    "brief": brief,
                },
            )
        elif trigger == "poll":
            base = self._render_prompt_asset(
                "task_poll.md",
                {
                    "issue_key": issue_key,
                    "brief": brief,
                },
            )
        else:
            base = "No explicit task signal was provided. Continue current chat objective."

        extra_lines: list[str] = []
        if isinstance(task_description, str) and task_description.strip():
            extra_lines.extend(["", "Task Description:", task_description.strip()])
        if isinstance(extra_context, str) and extra_context.strip():
            extra_lines.extend(["", "Extra Context:", extra_context.strip()])

        if extra_lines:
            base = base.rstrip() + "\n" + "\n".join(extra_lines).rstrip()
        return base.strip()

    def _build_project_knowledge_layer(self) -> str:
        candidate = self._project_root / ".aura" / "context" / "project_knowledge.md"
        if not candidate.exists() or not candidate.is_file():
            return ""
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return ""
        return text

    def _build_tool_surface_layer(self, *, surface: AgentCapabilitySurface) -> str:
        lines: list[str] = ["## Available Tools"]
        if surface.tool_specs:
            for spec in surface.tool_specs:
                desc = str(spec.description or "").strip()
                if desc:
                    lines.append(f"- {spec.name}: {desc}")
                else:
                    lines.append(f"- {spec.name}")
        else:
            for name in surface.tool_allowlist:
                lines.append(f"- {name}")

        if surface.warnings:
            lines.append("")
            lines.append("## Capability Warnings")
            for item in surface.warnings:
                if isinstance(item, str) and item.strip():
                    lines.append(f"- {item.strip()}")

        return "\n".join(lines).strip()

    def _assemble(self, layers: dict[str, str]) -> str:
        section_order = [
            ("system", "System"),
            ("agent_card", "Identity"),
            ("role_policy", "Permissions"),
            ("task_brief", "Task"),
            ("project_knowledge", "Project"),
            ("tool_surface", "Tools"),
        ]
        chunks: list[str] = []
        for key, title in section_order:
            body = str(layers.get(key) or "").strip()
            if not body:
                continue
            chunks.append(f"--- {title} ---\n{body}")
        return "\n\n".join(chunks).strip()

    def _read_prompt_asset(self, name: str) -> str:
        try:
            return (
                importlib.resources.files("aura.runtime")
                .joinpath("prompts", name)
                .read_text(encoding="utf-8", errors="replace")
                .strip()
            )
        except Exception:
            return ""

    def _render_prompt_asset(self, name: str, vars: dict[str, Any]) -> str:
        template = self._read_prompt_asset(name)
        if not template:
            return ""
        try:
            return template.format_map(_SafeDict(vars)).strip()
        except Exception:
            return template.strip()

    def _read_project_file(self, rel_path: str, *, max_chars: int) -> str | None:
        rel = str(rel_path or "").strip()
        if not rel:
            return None
        try:
            candidate = (self._project_root / rel).resolve()
        except Exception:
            return None
        if candidate != self._project_root and self._project_root not in candidate.parents:
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
    def _infer_trigger(signal: Signal | None) -> str:
        if signal is None:
            return "chat"
        if signal.signal_type is SignalType.WAKE:
            return "wake"
        if signal.signal_type is SignalType.TASK_ASSIGNED:
            return "task_assigned"
        return "poll"
