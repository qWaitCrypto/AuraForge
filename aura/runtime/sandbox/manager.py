from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..models.sandbox import Sandbox
from .store import SandboxStore


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._:-]+", "-", str(value or "").strip())
    cleaned = cleaned.strip("-")
    return cleaned or "na"


class SandboxError(RuntimeError):
    pass


class SandboxGitError(SandboxError):
    pass


@dataclass(slots=True)
class SandboxManager:
    project_root: Path
    store: SandboxStore

    def __init__(self, *, project_root: Path, store: SandboxStore | None = None) -> None:
        root = project_root.expanduser().resolve()
        self.project_root = root
        self.store = store or SandboxStore(project_root=root)
        (self.project_root / ".aura" / "sandboxes").mkdir(parents=True, exist_ok=True)

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = ["git", *args]
        proc = subprocess.run(
            cmd,
            cwd=self.project_root,
            text=True,
            capture_output=True,
        )
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise SandboxGitError(f"Git command failed ({' '.join(cmd)}): {detail}")
        return proc

    def _build_sandbox_id(self, *, issue_key: str, agent_id: str) -> str:
        suffix = uuid.uuid4().hex[:4]
        return f"sb_{_slug(issue_key)}_{_slug(agent_id)}_{suffix}"

    def _build_branch(self, *, issue_key: str, agent_id: str, suffix: str) -> str:
        return f"agent/{_slug(issue_key)}/{_slug(agent_id)}/{suffix}"

    def create(
        self,
        *,
        agent_id: str,
        issue_key: str,
        base_branch: str = "main",
        sandbox_id: str | None = None,
    ) -> Sandbox:
        agent = str(agent_id or "").strip()
        issue = str(issue_key or "").strip()
        base = str(base_branch or "main").strip() or "main"
        if not agent:
            raise ValueError("agent_id must be a non-empty string.")
        if not issue:
            raise ValueError("issue_key must be a non-empty string.")

        existing_for_pair = next(
            (item for item in self.find_by_agent(agent) if item.issue_key == issue),
            None,
        )

        if isinstance(sandbox_id, str) and sandbox_id.strip():
            sid = sandbox_id.strip()
            existing = self.store.load(sid)
            if existing is not None:
                return existing
            if existing_for_pair is not None and existing_for_pair.sandbox_id != sid:
                raise SandboxError(
                    "Only one active sandbox is allowed per agent+issue. "
                    f"Existing={existing_for_pair.sandbox_id}. Destroy it before creating a new one."
                )
        else:
            sid = self._build_sandbox_id(issue_key=issue, agent_id=agent)
            if existing_for_pair is not None:
                return existing_for_pair

        suffix = sid.rsplit("_", 1)[-1] if "_" in sid else uuid.uuid4().hex[:4]
        branch = self._build_branch(issue_key=issue, agent_id=agent, suffix=_slug(suffix))

        worktree_rel_path = Path(".aura") / "sandboxes" / sid
        worktree_abs_path = (self.project_root / worktree_rel_path).resolve()

        if self.store.load(sid) is not None:
            return self.store.load(sid)  # type: ignore[return-value]

        if worktree_abs_path.exists() and any(worktree_abs_path.iterdir()):
            raise SandboxError(f"Sandbox worktree path already exists and is not empty: {worktree_abs_path}")

        self._run_git(
            "worktree",
            "add",
            str(worktree_abs_path),
            "-b",
            branch,
            base,
            check=True,
        )

        sandbox = Sandbox(
            sandbox_id=sid,
            agent_id=agent,
            issue_key=issue,
            worktree_path=worktree_rel_path.as_posix(),
            branch=branch,
            base_branch=base,
        )
        self.store.save(sandbox)
        return sandbox

    def destroy(self, sandbox_id: str) -> None:
        sid = str(sandbox_id or "").strip()
        if not sid:
            raise ValueError("sandbox_id must be a non-empty string.")

        sandbox = self.store.load(sid)
        if sandbox is None:
            raise SandboxError(f"Sandbox not found: {sid}")

        worktree_abs = (self.project_root / sandbox.worktree_path).resolve()

        if worktree_abs.exists():
            self._run_git("worktree", "remove", str(worktree_abs), "--force", check=True)

        self._run_git("branch", "-D", sandbox.branch, check=False)

        if worktree_abs.exists():
            shutil.rmtree(worktree_abs, ignore_errors=True)

        self.store.delete(sid)

    def get(self, sandbox_id: str) -> Sandbox | None:
        sid = str(sandbox_id or "").strip()
        if not sid:
            return None
        return self.store.load(sid)

    def list_active(self) -> list[Sandbox]:
        items = self.store.list_all()
        items.sort(key=lambda item: item.created_at, reverse=True)
        return items

    def find_by_agent(self, agent_id: str) -> list[Sandbox]:
        agent = str(agent_id or "").strip()
        out = [item for item in self.store.list_all() if item.agent_id == agent]
        out.sort(key=lambda item: item.created_at, reverse=True)
        return out

    def find_by_issue(self, issue_key: str) -> list[Sandbox]:
        issue = str(issue_key or "").strip()
        out = [item for item in self.store.list_all() if item.issue_key == issue]
        out.sort(key=lambda item: item.created_at, reverse=True)
        return out
