from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..context.builder import ContextBuilder
from ..engine import build_engine_for_session
from ..event_bus import EventBus
from ..event_log import EventLog, EventLogFileStore, extract_external_refs
from ..ids import new_id
from ..llm.config_io import load_model_config_layers_for_dir
from ..mcp.config import load_mcp_config
from ..models.capability import AgentCapabilitySurface
from ..models.signal import Signal, SignalType
from ..sandbox import SandboxManager
from ..signal import SignalBus, SignalStore
from ..stores import FileApprovalStore, FileArtifactStore, FileEventLogStore, FileSessionStore
from ..subagents.presets import list_presets
from ..tools.audit_tools import AuditQueryTool, AuditRefsTool
from ..tools.builtins import ProjectReadTextTool, ShellRunTool
from ..tools.runtime import ToolExecutionContext
from ..tools.signal_tools import SignalPollTool, SignalSendTool
from .phase3_e2e_demo import run_demo


def _check(name: str, ok: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _progress(message: str) -> None:
    print(f"[acceptance] {message}", file=sys.stderr, flush=True)


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def _build_engine(root: Path, *, session_suffix: str):
    artifact_store = FileArtifactStore(root / ".aura" / "artifacts")
    session_store = FileSessionStore(root / ".aura" / "sessions")
    approval_store = FileApprovalStore(root / ".aura" / "state" / "approvals")
    event_log_store = FileEventLogStore(
        root / ".aura" / "events",
        artifact_store=artifact_store,
        session_store=session_store,
    )
    event_bus = EventBus(event_log_store=event_log_store)
    model_config = load_model_config_layers_for_dir(root, require_project=True).merged()
    session_id = f"acc-{session_suffix}-{new_id('sess')}"
    session_store.create_session(
        {
            "session_id": session_id,
            "project_ref": str(root),
            "mode": "chat",
            "agent_id": "acceptance-checker",
            "role": "worker",
        }
    )
    engine = build_engine_for_session(
        project_root=root,
        session_id=session_id,
        event_bus=event_bus,
        session_store=session_store,
        event_log_store=event_log_store,
        artifact_store=artifact_store,
        approval_store=approval_store,
        model_config=model_config,
        tools_enabled=False,
    )
    return engine


async def _check_engine_tool_audit(root: Path) -> dict[str, Any]:
    engine = _build_engine(root, session_suffix="audit")
    _ = await engine.execute_tool_once(
        tool_name="project__read_text",
        arguments={"path": "aura/runtime/models/sandbox.py"},
        caller_kind="system",
        request_id=f"req_{new_id('req')}",
        turn_id=f"turn_{new_id('turn')}",
    )
    rows = engine.event_log.query(session_id=engine.session_id, tool_name="project__read_text", limit=50)
    return {"session_id": engine.session_id, "audit_rows": len(rows)}


async def _check_linear_mcp_discovery(root: Path) -> dict[str, Any]:
    engine = _build_engine(root, session_suffix="mcp")
    runtime_names = await engine.list_mcp_runtime_tool_names(server_names={"linear"})
    linear_tools = [name for name in runtime_names if re.match(r"^mcp__linear_[0-9a-f]{6}__", name)]
    return {
        "mcp_tools_total": len(runtime_names),
        "linear_tools_count": len(linear_tools),
        "linear_tools_sample": linear_tools[:5],
    }


def run_acceptance(*, project_root: Path) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    results: list[dict[str, Any]] = []

    # Phase 0 imports.
    _progress("phase0.imports")
    try:
        from aura.runtime.sandbox import SandboxManager as _SB  # noqa: F401
        from aura.runtime.event_log import EventLog as _EL  # noqa: F401
        from aura.runtime.signal import SignalBus as _SIG  # noqa: F401
        from aura.runtime.capability import CapabilityBuilder as _CAP  # noqa: F401
        from aura.runtime.context import ContextBuilder as _CTX  # noqa: F401
        results.append(_check("phase0.imports", True, "ok"))
    except Exception as exc:
        results.append(_check("phase0.imports", False, str(exc)))

    # Phase 1 event log extractor checks.
    _progress("phase1.event_log.extractors")
    commit_refs = extract_external_refs(
        tool_name="shell__run",
        tool_args={"command": "git commit -m test && git push origin main"},
        tool_result={"stdout": "[main 1234abc] msg"},
    )
    pr_refs = extract_external_refs(
        tool_name="mcp__github__create_pr",
        tool_args={},
        tool_result={"url": "https://github.com/acme/repo/pull/12"},
    )
    results.append(
        _check(
            "phase1.event_log.extractors",
            ("commit:1234abc" in commit_refs and any(ref.startswith("pr:") for ref in pr_refs)),
            {"commit_refs": commit_refs, "pr_refs": pr_refs},
        )
    )

    # Phase 1 signal checks.
    _progress("phase1.signal.send_poll_consume")
    sig_bus = SignalBus(store=SignalStore(project_root=root), event_log=EventLog(store=EventLogFileStore(project_root=root)))
    sig = sig_bus.send(from_agent="coordinator", to_agent="worker", signal_type=SignalType.WAKE, brief="bid", issue_key="ACC-SIG")
    polled = sig_bus.poll(to_agent="worker", unconsumed_only=True, limit=10)
    sig_bus.consume(sig.signal_id)
    queried = sig_bus.query(issue_key="ACC-SIG", limit=10)
    results.append(
        _check(
            "phase1.signal.send_poll_consume",
            (len(polled) >= 1 and any(item.signal_id == sig.signal_id and item.consumed for item in queried)),
            {"signal_id": sig.signal_id, "poll_count": len(polled), "query_count": len(queried)},
        )
    )

    # Phase 1 sandbox manager + CLI checks.
    _progress("phase1.sandbox.parallel_10")
    manager = SandboxManager(project_root=root)
    issue_key = f"ACC-SB-{int(time.time())}"
    sandbox_ids: list[str] = []
    try:
        for i in range(10):
            sb = manager.create(agent_id=f"agent{i+1}", issue_key=issue_key, base_branch="main")
            sandbox_ids.append(sb.sandbox_id)
        listed = manager.find_by_issue(issue_key)
        unique_paths = len({item.worktree_path for item in listed})
        results.append(
            _check(
                "phase1.sandbox.parallel_10",
                (len(listed) == 10 and unique_paths == 10),
                {"listed": len(listed), "unique_paths": unique_paths},
            )
        )
    except Exception as exc:
        results.append(_check("phase1.sandbox.parallel_10", False, str(exc)))
    finally:
        for sid in sandbox_ids:
            try:
                manager.destroy(sid)
            except Exception:
                pass

    _progress("phase1.sandbox.cli_create_list_destroy")
    cli_issue = f"ACC-CLI-{int(time.time())}"
    create = _run(["python", "-m", "aura", "sandbox", "create", "--agent-id", "cli-agent", "--issue-key", cli_issue], cwd=root)
    list_out = _run(["python", "-m", "aura", "sandbox", "list", "--issue-key", cli_issue], cwd=root)
    destroy_ok = False
    if create.returncode == 0:
        try:
            created_payload = json.loads(create.stdout)
            created_sid = str(created_payload["sandbox"]["sandbox_id"])
            destroy = _run(["python", "-m", "aura", "sandbox", "destroy", created_sid], cwd=root)
            destroy_ok = destroy.returncode == 0
        except Exception:
            destroy_ok = False
    results.append(
        _check(
            "phase1.sandbox.cli_create_list_destroy",
            (create.returncode == 0 and list_out.returncode == 0 and destroy_ok),
            {
                "create_rc": create.returncode,
                "list_rc": list_out.returncode,
                "destroy_ok": destroy_ok,
                "list_stdout": list_out.stdout.strip(),
                "create_stderr": create.stderr.strip(),
            },
        )
    )

    # Phase 2 capability/context checks.
    _progress("phase2.capability.surface")
    tool_registry = __import__("aura.runtime.tools.registry", fromlist=["ToolRegistry"]).ToolRegistry()
    event_log = EventLog(store=EventLogFileStore(project_root=root))
    signal_bus = SignalBus(store=SignalStore(project_root=root), event_log=event_log)
    tool_registry.register(AuditQueryTool(event_log=event_log))
    tool_registry.register(AuditRefsTool(event_log=event_log))
    tool_registry.register(SignalSendTool(signal_bus=signal_bus))
    tool_registry.register(SignalPollTool(signal_bus=signal_bus))
    tool_registry.register(ProjectReadTextTool())

    spec_registry = __import__("aura.runtime.registry.spec_registry", fromlist=["SpecRegistry"]).SpecRegistry(project_root=root)
    skill_store = __import__("aura.runtime.skills", fromlist=["SkillStore"]).SkillStore(project_root=root)
    spec_registry.refresh_from_runtime(
        tool_registry=tool_registry,
        skill_store=skill_store,
        mcp_config=load_mcp_config(project_root=root),
        include_builtin_subagents=True,
    )
    spec_resolver = __import__("aura.runtime.registry.resolver", fromlist=["SpecResolver"]).SpecResolver(registry=spec_registry)
    capability_builder = __import__("aura.runtime.capability.builder", fromlist=["CapabilityBuilder"]).CapabilityBuilder(
        tool_registry=tool_registry,
        spec_resolver=spec_resolver,
        project_root=root,
    )
    surface = capability_builder.build(agent_id="agent.market.fused.backend-developer.v1", role="worker")
    allow = set(surface.tool_allowlist)
    results.append(
        _check(
            "phase2.capability.surface",
            ("signal__send" in allow and "audit__query" in allow and "project__read_text" in allow),
            {"agent_id": surface.agent_id, "resolved_from": surface.resolved_from, "tool_count": len(allow)},
        )
    )

    _progress("phase2.context.wake_assigned")
    context_builder = ContextBuilder(project_root=root, spec_resolver=spec_resolver)
    wake = Signal(signal_id="sig_w", from_agent="coordinator", to_agent="a", signal_type=SignalType.WAKE, brief="bid", issue_key="ACC-CTX")
    assigned = Signal(
        signal_id="sig_a",
        from_agent="coordinator",
        to_agent="a",
        signal_type=SignalType.TASK_ASSIGNED,
        brief="execute",
        issue_key="ACC-CTX",
        sandbox_id="sb_ACC_CTX",
    )
    sandbox = manager.create(agent_id="ctx-agent", issue_key="ACC-CTX", base_branch="main")
    try:
        ctx_wake = context_builder.build(surface=surface, signal=wake)
        ctx_assigned = context_builder.build(surface=surface, signal=assigned, sandbox=sandbox)
        results.append(
            _check(
                "phase2.context.wake_assigned",
                (ctx_wake.trigger == "wake" and ctx_assigned.trigger == "task_assigned" and sandbox.worktree_path in ctx_assigned.layers["task_brief"]),
                {"wake_trigger": ctx_wake.trigger, "assigned_trigger": ctx_assigned.trigger},
            )
        )
    finally:
        try:
            manager.destroy(sandbox.sandbox_id)
        except Exception:
            pass

    # Engine signature and tool audit hook.
    _progress("phase2.engine.signature_and_auto_audit")
    import inspect
    from aura.runtime.engine_agno_async import AgnoAsyncEngine

    has_workspace_param = "workspace_manager" in inspect.signature(AgnoAsyncEngine).parameters
    results.append(_check("phase2.engine.no_workspace_manager_arg", not has_workspace_param, {"has_workspace_manager": has_workspace_param}))

    try:
        audit_detail = asyncio.run(asyncio.wait_for(_check_engine_tool_audit(root), timeout=60))
        results.append(_check("phase2.engine.auto_audit", audit_detail.get("audit_rows", 0) > 0, audit_detail))
    except Exception as exc:
        results.append(_check("phase2.engine.auto_audit", False, str(exc)))

    # Worker push guard.
    _progress("phase2.worker_push_blocked")
    worker_ctx = ToolExecutionContext(session_id="s", request_id="r", turn_id="t", tool_execution_id="x", role="worker")
    blocked = False
    try:
        ShellRunTool().execute(args={"command": "git push origin main", "cwd": "."}, project_root=root, context=worker_ctx)
    except Exception as exc:
        blocked = "git push" in str(exc).lower()
    results.append(_check("phase2.worker_push_blocked", blocked, {"blocked": blocked}))

    # Phase 3 linear mcp + e2e.
    _progress("phase3.linear_mcp.config")
    mcp_cfg = load_mcp_config(project_root=root)
    linear_cfg = mcp_cfg.servers.get("linear")
    linear_cfg_ok = bool(linear_cfg and linear_cfg.enabled and "mcp.linear.app/mcp" in " ".join(linear_cfg.args))
    results.append(_check("phase3.linear_mcp.config", linear_cfg_ok, {"linear_enabled": bool(linear_cfg.enabled) if linear_cfg else False, "args": (linear_cfg.args if linear_cfg else None)}))

    _progress("phase3.linear_mcp.discovery")
    try:
        mcp_detail = asyncio.run(asyncio.wait_for(_check_linear_mcp_discovery(root), timeout=120))
        results.append(_check("phase3.linear_mcp.discovery", mcp_detail.get("linear_tools_count", 0) > 0, mcp_detail))
    except Exception as exc:
        results.append(_check("phase3.linear_mcp.discovery", False, str(exc)))

    _progress("phase3.e2e.3_agents")
    try:
        e2e_3 = run_demo(
            project_root=root,
            issue_key=f"ACC-E2E-{int(time.time())}",
            base_branch="main",
            agents=["agent-a", "agent-b", "agent-c"],
            cleanup=True,
        )
        ok_3 = bool(e2e_3.get("ok")) and e2e_3.get("signals", {}).get("counts_by_type", {}).get("wake") == 3
        results.append(_check("phase3.e2e.3_agents", ok_3, e2e_3.get("signals")))
    except Exception as exc:
        results.append(_check("phase3.e2e.3_agents", False, str(exc)))

    _progress("phase3.e2e.10_agents_isolation")
    try:
        e2e_10 = run_demo(
            project_root=root,
            issue_key=f"ACC-E2E10-{int(time.time())}",
            base_branch="main",
            agents=[f"agent-{i}" for i in range(1, 11)],
            cleanup=True,
        )
        sandboxes = e2e_10.get("sandboxes", [])
        unique_paths = len({row.get("worktree_path") for row in sandboxes if isinstance(row, dict)})
        ok_10 = bool(e2e_10.get("ok")) and len(sandboxes) == 10 and unique_paths == 10
        results.append(_check("phase3.e2e.10_agents_isolation", ok_10, {"sandbox_count": len(sandboxes), "unique_paths": unique_paths}))
    except Exception as exc:
        results.append(_check("phase3.e2e.10_agents_isolation", False, str(exc)))

    # Final scans + indicators.
    _progress("phase3.cleanup_and_indicators")
    rg_workspace = _run(
        ["rg", "-n", "workspace__", "aura/runtime", "aura/cli.py", "-S", "--glob", "!aura/runtime/examples/**"],
        cwd=root,
    )
    rg_a2a = _run(
        ["rg", "-n", "A2A_|runtime\\.a2a|from \\.a2a|a2a_", "aura/runtime", "aura/cli.py", "-S", "--glob", "!aura/runtime/examples/**"],
        cwd=root,
    )
    results.append(_check("phase3.cleanup.workspace_ref_zero", rg_workspace.returncode == 1, {"returncode": rg_workspace.returncode}))
    results.append(_check("phase3.cleanup.a2a_ref_zero", rg_a2a.returncode == 1, {"returncode": rg_a2a.returncode}))

    core_templates = [
        "system_main.md",
        "role_worker.md",
        "role_integrator.md",
        "task_wake.md",
        "task_assigned.md",
        "task_poll.md",
    ]
    core_prompt_ok = all((root / "aura" / "runtime" / "prompts" / name).exists() for name in core_templates)
    preset_source = (root / "aura" / "runtime" / "subagents" / "presets.py").read_text(encoding="utf-8", errors="replace")
    preset_hardcoded = (
        "_PRESETS: dict[str, SubagentPreset] = {" in preset_source
        or "\"file_ops_worker\": SubagentPreset(" in preset_source
        or "\"market_worker\": SubagentPreset(" in preset_source
    )
    results.append(_check("indicators.core_prompt_templates", core_prompt_ok, {"core_templates": core_templates}))
    results.append(
        _check(
            "indicators.preset_hardcoding_zero",
            not preset_hardcoded,
            {"preset_hardcoded_pattern_found": preset_hardcoded, "preset_count": len(list_presets())},
        )
    )

    all_ok = all(item["ok"] for item in results)
    failed = [item["name"] for item in results if not item["ok"]]
    _progress(f"done ok={all_ok} failed={len(failed)}")
    return {"ok": all_ok, "failed": failed, "checks": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run roadmap acceptance checks for Phase 0-3.")
    parser.add_argument("--project-root", default=".", help="Aura project root (default: current directory).")
    args = parser.parse_args()

    result = run_acceptance(project_root=Path(args.project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
