from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_CONFIG_FILENAME = "workspace.json"
DEFAULT_BASE_BRANCH = "main"
DEFAULT_PROTECTED_BRANCHES = ("main", "production")
DEFAULT_GITHUB_TOKEN_ENV = "GITHUB_TOKEN"

_REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HTTPS_REPO_RE = re.compile(r"^https?://github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", re.IGNORECASE)
_SSH_REPO_RE = re.compile(r"^git@github\.com:([^/\s]+)/([^/\s]+?)(?:\.git)?$", re.IGNORECASE)
_SSH_URL_REPO_RE = re.compile(r"^ssh://git@github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class WorkspaceBinding:
    publish_repo: str | None = None
    default_base_branch: str = DEFAULT_BASE_BRANCH
    protected_branches: tuple[str, ...] = DEFAULT_PROTECTED_BRANCHES
    github_token_env: str = DEFAULT_GITHUB_TOKEN_ENV
    source: str | None = None


def normalize_repo_ref(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None

    slug = raw[:-4] if raw.lower().endswith(".git") else raw
    slug = slug.strip().strip("/")
    if _REPO_SLUG_RE.match(slug):
        return slug

    for pattern in (_HTTPS_REPO_RE, _SSH_REPO_RE, _SSH_URL_REPO_RE):
        matched = pattern.match(raw)
        if matched is None:
            continue
        owner = str(matched.group(1) or "").strip()
        repo = str(matched.group(2) or "").strip()
        if owner and repo:
            return f"{owner}/{repo}"
    return None


def repo_match_key(value: Any) -> str:
    normalized = normalize_repo_ref(value)
    return normalized.lower() if normalized else ""


def workspace_config_path_for_project(project_root: Path) -> Path:
    return project_root / ".aura" / "config" / WORKSPACE_CONFIG_FILENAME


def infer_publish_repo_from_git_origin(*, project_root: Path, timeout_s: float = 2.0) -> str | None:
    root = project_root.expanduser().resolve()
    try:
        proc = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=max(0.1, float(timeout_s)),
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return normalize_repo_ref(proc.stdout)


def load_workspace_binding(*, project_root: Path) -> WorkspaceBinding:
    root = project_root.expanduser().resolve()
    path = workspace_config_path_for_project(root)
    if not path.exists():
        return WorkspaceBinding(source=None)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return WorkspaceBinding(source=str(path))
    if not isinstance(raw, dict):
        return WorkspaceBinding(source=str(path))

    publish_repo = normalize_repo_ref(raw.get("publish_repo"))
    base_branch = str(raw.get("default_base_branch") or "").strip() or DEFAULT_BASE_BRANCH

    raw_protected = raw.get("protected_branches")
    protected: list[str] = []
    seen: set[str] = set()
    if isinstance(raw_protected, list):
        for item in raw_protected:
            name = str(item or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            protected.append(name)
    if not protected:
        protected = list(DEFAULT_PROTECTED_BRANCHES)

    token_env = str(raw.get("github_token_env") or "").strip() or DEFAULT_GITHUB_TOKEN_ENV
    return WorkspaceBinding(
        publish_repo=publish_repo,
        default_base_branch=base_branch,
        protected_branches=tuple(protected),
        github_token_env=token_env,
        source=str(path),
    )
