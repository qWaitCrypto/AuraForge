from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    project_root: Path
    system_dir: Path
    config_dir: Path
    policy_dir: Path
    sessions_dir: Path
    events_dir: Path
    artifacts_dir: Path
    runs_dir: Path
    state_dir: Path
    index_dir: Path
    cache_dir: Path
    tmp_dir: Path

    @staticmethod
    def for_project(project_root: Path) -> "RuntimePaths":
        project_root = project_root.expanduser().resolve()
        system_dir = project_root / ".aura"
        return RuntimePaths(
            project_root=project_root,
            system_dir=system_dir,
            config_dir=system_dir / "config",
            policy_dir=system_dir / "policy",
            sessions_dir=system_dir / "sessions",
            events_dir=system_dir / "events",
            artifacts_dir=system_dir / "artifacts",
            runs_dir=system_dir / "runs",
            state_dir=system_dir / "state",
            index_dir=system_dir / "index",
            cache_dir=system_dir / "cache",
            tmp_dir=system_dir / "tmp",
        )

    @staticmethod
    def discover(start: Path | None = None) -> "RuntimePaths":
        here = (start or Path.cwd()).expanduser().resolve()
        for directory in [here, *here.parents]:
            candidate = directory / ".aura"
            if candidate.is_dir():
                return RuntimePaths.for_project(directory)
        raise FileNotFoundError("No Aura project found (missing .aura directory).")
