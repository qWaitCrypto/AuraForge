from __future__ import annotations

import json
from pathlib import Path

from ..models.sandbox import Sandbox


class SandboxStoreError(RuntimeError):
    pass


class SandboxStore:
    """File-backed JSON store for Sandbox metadata."""

    def __init__(self, *, project_root: Path) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._state_root = self._project_root / ".aura" / "state" / "sandboxes"
        self._worktree_root = self._project_root / ".aura" / "sandboxes"
        self._index_path = self._worktree_root / "_index.json"
        self._state_root.mkdir(parents=True, exist_ok=True)
        self._worktree_root.mkdir(parents=True, exist_ok=True)

    def _path(self, sandbox_id: str) -> Path:
        return self._state_root / f"{sandbox_id}.json"

    def save(self, sandbox: Sandbox) -> None:
        path = self._path(sandbox.sandbox_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(sandbox.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
        self._save_index_entry(sandbox)

    def load(self, sandbox_id: str) -> Sandbox | None:
        path = self._path(sandbox_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SandboxStoreError(f"Failed to read sandbox metadata: {path}") from exc
        try:
            return Sandbox.model_validate(raw)
        except Exception as exc:
            raise SandboxStoreError(f"Invalid sandbox metadata: {path}") from exc

    def delete(self, sandbox_id: str) -> None:
        path = self._path(sandbox_id)
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:
            raise SandboxStoreError(f"Failed to delete sandbox metadata: {path}") from exc
        self._delete_index_entry(sandbox_id)

    def list_all(self) -> list[Sandbox]:
        items: list[Sandbox] = []
        for path in sorted(self._state_root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                items.append(Sandbox.model_validate(raw))
            except Exception:
                continue
        return items

    def _load_index(self) -> dict[str, dict]:
        if not self._index_path.exists():
            return {}
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SandboxStoreError(f"Failed to read sandbox index: {self._index_path}") from exc
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            out[key] = dict(value)
        return out

    def _write_index(self, index_obj: dict[str, dict]) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(index_obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._index_path)

    def _save_index_entry(self, sandbox: Sandbox) -> None:
        index_obj = self._load_index()
        index_obj[sandbox.sandbox_id] = {
            "sandbox_id": sandbox.sandbox_id,
            "agent_id": sandbox.agent_id,
            "issue_key": sandbox.issue_key,
            "worktree_path": sandbox.worktree_path,
            "branch": sandbox.branch,
            "base_branch": sandbox.base_branch,
            "created_at": sandbox.created_at,
        }
        self._write_index(index_obj)

    def _delete_index_entry(self, sandbox_id: str) -> None:
        index_obj = self._load_index()
        if sandbox_id in index_obj:
            del index_obj[sandbox_id]
            self._write_index(index_obj)
