from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from ..event_log import EventLog, EventLogFileStore
from ..ids import new_id
from ..models.event_log import LogEvent, LogEventKind
from ..models.signal import SignalType
from ..sandbox import SandboxManager
from ..signal import SignalBus, SignalStore
from ..tools.audit_tools import AuditQueryTool, AuditRefsTool


DEFAULT_AGENTS: list[str] = [
    "market_fused__python-pro",
    "market_fused__backend-developer",
    "market_fused__security-auditor",
]


def _run_git(project_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=project_root,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return (proc.stdout or "").strip()


def _current_branch(project_root: Path) -> str:
    branch = _run_git(project_root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    return branch or "main"


def _check_linear_mcp_config(project_root: Path) -> dict[str, Any]:
    path = project_root / ".aura" / "config" / "mcp.json"
    if not path.exists():
        return {"ok": False, "reason": "missing .aura/config/mcp.json"}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": f"invalid JSON: {exc}"}
    if not isinstance(raw, dict):
        return {"ok": False, "reason": "mcp.json root is not an object"}

    servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        return {"ok": False, "reason": "mcpServers missing or invalid"}

    linear = servers.get("linear")
    if not isinstance(linear, dict):
        return {"ok": False, "reason": "mcpServers.linear missing"}

    enabled = bool(linear.get("enabled"))
    command = str(linear.get("command") or "").strip()
    args = linear.get("args") if isinstance(linear.get("args"), list) else []
    endpoint = next((str(v) for v in args if isinstance(v, str) and v.startswith("http")), "")
    return {
        "ok": True,
        "enabled": enabled,
        "command": command,
        "endpoint": endpoint,
    }


def _record_linear_bid(event_log: EventLog, *, agent_id: str, issue_key: str) -> None:
    event_log.record(
        LogEvent(
            event_id=new_id("evt"),
            session_id=f"demo:{issue_key}:{agent_id}:bid",
            agent_id=agent_id,
            issue_key=issue_key,
            kind=LogEventKind.TOOL_CALL,
            tool_name="mcp__linear__create_comment",
            tool_args_summary=f"issue={issue_key}",
            tool_result_summary="bid comment posted",
            tool_ok=True,
            external_refs=[f"linear:https://linear.app/demo/{issue_key}#bid-{agent_id}"],
        )
    )


def _record_work_events(event_log: EventLog, *, winner: str, issue_key: str, sandbox_id: str) -> None:
    event_log.record(
        LogEvent(
            event_id=new_id("evt"),
            session_id=f"demo:{issue_key}:{winner}:work",
            agent_id=winner,
            sandbox_id=sandbox_id,
            issue_key=issue_key,
            kind=LogEventKind.TOOL_CALL,
            tool_name="project__apply_patch",
            tool_args_summary="1 file updated",
            tool_result_summary="patch applied",
            tool_ok=True,
        )
    )

    commit_sha = "abc1234"
    event_log.record(
        LogEvent(
            event_id=new_id("evt"),
            session_id=f"demo:{issue_key}:{winner}:work",
            agent_id=winner,
            sandbox_id=sandbox_id,
            issue_key=issue_key,
            kind=LogEventKind.TOOL_CALL,
            tool_name="shell__run",
            tool_args_summary="git commit -m 'demo'",
            tool_result_summary=f"commit {commit_sha}",
            tool_ok=True,
            external_refs=[f"commit:{commit_sha}"],
        )
    )

    event_log.record(
        LogEvent(
            event_id=new_id("evt"),
            session_id=f"demo:{issue_key}:{winner}:work",
            agent_id=winner,
            sandbox_id=sandbox_id,
            issue_key=issue_key,
            kind=LogEventKind.TOOL_CALL,
            tool_name="mcp__linear__create_comment",
            tool_args_summary=f"issue={issue_key}",
            tool_result_summary="completion comment posted",
            tool_ok=True,
            external_refs=[f"linear:https://linear.app/demo/{issue_key}#done-{winner}"],
        )
    )


def run_demo(
    *,
    project_root: Path,
    issue_key: str,
    base_branch: str | None,
    agents: list[str],
    cleanup: bool,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Not a git repository: {root}")

    event_log = EventLog(store=EventLogFileStore(project_root=root))
    signal_bus = SignalBus(store=SignalStore(project_root=root), event_log=event_log)
    sandbox_manager = SandboxManager(project_root=root)

    resolved_base = (str(base_branch or "").strip() or _current_branch(root))
    normalized_agents = [str(a).strip() for a in agents if str(a).strip()]
    if len(normalized_agents) < 3:
        raise ValueError("At least 3 agent IDs are required for the concurrency demo.")

    created_sandboxes = []
    created_ids: list[str] = []
    try:
        for agent_id in normalized_agents:
            try:
                sb = sandbox_manager.create(agent_id=agent_id, issue_key=issue_key, base_branch=resolved_base)
            except Exception as exc:
                raise RuntimeError(
                    "Sandbox creation failed. Verify git worktree is writable and base branch exists. "
                    f"agent={agent_id} issue={issue_key} base={resolved_base}"
                ) from exc
            created_sandboxes.append(sb)
            created_ids.append(sb.sandbox_id)

        by_agent = {sb.agent_id: sb for sb in created_sandboxes}
        unique_paths = {sb.worktree_path for sb in created_sandboxes}
        if len(unique_paths) != len(created_sandboxes):
            raise RuntimeError("Sandbox isolation check failed: duplicate worktree paths detected.")

        coordinator = "coordinator"
        wake_ids: list[str] = []
        for agent_id in normalized_agents:
            signal = signal_bus.send(
                from_agent=coordinator,
                to_agent=agent_id,
                signal_type=SignalType.WAKE,
                brief=f"Check issue {issue_key} and bid if relevant.",
                issue_key=issue_key,
            )
            wake_ids.append(signal.signal_id)

        for agent_id in normalized_agents:
            inbox = signal_bus.poll(to_agent=agent_id, unconsumed_only=True, limit=10)
            for signal in inbox:
                signal_bus.consume(signal.signal_id)
            _record_linear_bid(event_log, agent_id=agent_id, issue_key=issue_key)

        winner = normalized_agents[0]
        winner_sandbox = by_agent[winner]
        assignment = signal_bus.send(
            from_agent=coordinator,
            to_agent=winner,
            signal_type=SignalType.TASK_ASSIGNED,
            brief=f"You won {issue_key}; execute in assigned sandbox.",
            issue_key=issue_key,
            sandbox_id=winner_sandbox.sandbox_id,
            payload={"sandbox_id": winner_sandbox.sandbox_id},
        )
        signal_bus.consume(assignment.signal_id)

        _record_work_events(
            event_log,
            winner=winner,
            issue_key=issue_key,
            sandbox_id=winner_sandbox.sandbox_id,
        )

        completion = signal_bus.send(
            from_agent=winner,
            to_agent=coordinator,
            signal_type=SignalType.NOTIFY,
            brief=f"Completed {issue_key}.",
            issue_key=issue_key,
            sandbox_id=winner_sandbox.sandbox_id,
        )

        coordinator_inbox = signal_bus.poll(to_agent=coordinator, unconsumed_only=True, limit=20)
        for signal in coordinator_inbox:
            signal_bus.consume(signal.signal_id)

        audit_query = AuditQueryTool(event_log=event_log)
        audit_refs = AuditRefsTool(event_log=event_log)
        audit_events = audit_query.execute(args={"issue_key": issue_key, "limit": 500}, project_root=root)
        refs_all = audit_refs.execute(args={"issue_key": issue_key}, project_root=root)
        refs_linear = audit_refs.execute(args={"issue_key": issue_key, "ref_type": "linear"}, project_root=root)
        refs_commit = audit_refs.execute(args={"issue_key": issue_key, "ref_type": "commit"}, project_root=root)

        signals = signal_bus.query(issue_key=issue_key, limit=200)
        signal_counts: dict[str, int] = {}
        for signal in signals:
            key = signal.signal_type.value
            signal_counts[key] = signal_counts.get(key, 0) + 1

        return {
            "ok": True,
            "issue_key": issue_key,
            "base_branch": resolved_base,
            "agents": normalized_agents,
            "winner": winner,
            "linear_mcp_config": _check_linear_mcp_config(root),
            "sandboxes": [
                {
                    "sandbox_id": sb.sandbox_id,
                    "agent_id": sb.agent_id,
                    "worktree_path": sb.worktree_path,
                    "branch": sb.branch,
                }
                for sb in created_sandboxes
            ],
            "signals": {
                "total": len(signals),
                "counts_by_type": signal_counts,
                "wake_ids": wake_ids,
                "task_assigned_id": assignment.signal_id,
                "notify_id": completion.signal_id,
            },
            "audit": {
                "events_count": int(audit_events.get("count", 0)),
                "external_refs_count": int(refs_all.get("count", 0)),
                "linear_refs_count": int(refs_linear.get("count", 0)),
                "commit_refs_count": int(refs_commit.get("count", 0)),
            },
        }
    finally:
        if cleanup:
            for sandbox_id in created_ids:
                try:
                    sandbox_manager.destroy(sandbox_id)
                except Exception:
                    continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 E2E demo: signal bidding + assignment + sandbox + audit.")
    parser.add_argument("--project-root", default=".", help="Aura project root (default: current directory).")
    parser.add_argument("--issue-key", default="", help="Issue key for this run (default: auto-generated).")
    parser.add_argument("--base-branch", default="", help="Base branch for sandboxes (default: current branch).")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=DEFAULT_AGENTS,
        help="Agent IDs to include (need >= 3).",
    )
    parser.add_argument(
        "--keep-sandboxes",
        action="store_true",
        help="Keep created sandboxes for manual inspection.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    issue_key = str(args.issue_key or "").strip() or f"DEMO-{int(time.time())}"
    result = run_demo(
        project_root=root,
        issue_key=issue_key,
        base_branch=(str(args.base_branch or "").strip() or None),
        agents=[str(a).strip() for a in args.agents],
        cleanup=(not bool(args.keep_sandboxes)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
