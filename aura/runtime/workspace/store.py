from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ids import now_ts_ms
from ..models.workspace import (
    IssueWorkspace,
    IssueWorkspaceState,
    SessionWorkspaceBinding,
    Workbench,
    WorkbenchState,
    WorkspaceSubmission,
    WorkspaceSubmissionStatus,
)


def _replace_surrogates(text: str) -> str:
    out: list[str] = []
    changed = False
    for ch in text:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            out.append("\uFFFD")
            changed = True
        else:
            out.append(ch)
    return "".join(out) if changed else text


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return _replace_surrogates(value)
    if isinstance(value, list):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            key = _replace_surrogates(k) if isinstance(k, str) else k
            out[key] = _sanitize_json_value(v)
        return out
    return value


def _safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_sanitize_json_value(obj), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        errors="backslashreplace",
    )
    tmp.replace(path)


class WorkspaceStoreError(RuntimeError):
    pass


class WorkspaceNotFoundError(WorkspaceStoreError):
    pass


class WorkspaceRevisionConflictError(WorkspaceStoreError):
    pass


class WorkspaceStore:
    def __init__(self, *, project_root: Path) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._workspaces_root = self._project_root / ".aura" / "state" / "workspaces"
        self._workbenches_root = self._project_root / ".aura" / "state" / "workbenches"
        self._sessions_root = self._workspaces_root / "sessions"
        self._submissions_path = self._project_root / ".aura" / "index" / "workspace_submissions.jsonl"
        self._timeline_path = self._project_root / ".aura" / "index" / "workspace_timeline.jsonl"
        self._heartbeats_path = self._project_root / ".aura" / "index" / "workspace_heartbeats.jsonl"
        self._claims_path = self._project_root / ".aura" / "index" / "workspace_claims.jsonl"
        self._awards_path = self._project_root / ".aura" / "index" / "workspace_awards.jsonl"

        self._workspaces_root.mkdir(parents=True, exist_ok=True)
        self._workbenches_root.mkdir(parents=True, exist_ok=True)
        self._sessions_root.mkdir(parents=True, exist_ok=True)
        self._submissions_path.parent.mkdir(parents=True, exist_ok=True)
        self._timeline_path.parent.mkdir(parents=True, exist_ok=True)
        self._heartbeats_path.parent.mkdir(parents=True, exist_ok=True)
        self._claims_path.parent.mkdir(parents=True, exist_ok=True)
        self._awards_path.parent.mkdir(parents=True, exist_ok=True)

    def _workspace_path(self, workspace_id: str) -> Path:
        return self._workspaces_root / f"{workspace_id}.json"

    def _workbench_path(self, workbench_id: str) -> Path:
        return self._workbenches_root / f"{workbench_id}.json"

    def _session_binding_path(self, session_id: str) -> Path:
        return self._sessions_root / f"{session_id}.json"

    def _load_json_dict(self, path: Path) -> dict[str, Any]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except Exception as e:
            raise WorkspaceStoreError(f"Failed to read JSON file {path}: {e}") from e
        if not isinstance(raw, dict):
            raise WorkspaceStoreError(f"Invalid JSON object in file {path}.")
        return raw

    def get_workspace(self, workspace_id: str) -> IssueWorkspace:
        path = self._workspace_path(workspace_id)
        if not path.exists():
            raise WorkspaceNotFoundError(f"Workspace not found: {workspace_id}")
        raw = self._load_json_dict(path)
        return IssueWorkspace.model_validate(raw)

    def list_workspaces(
        self,
        *,
        state: IssueWorkspaceState | None = None,
        issue_key: str | None = None,
    ) -> list[IssueWorkspace]:
        out: list[IssueWorkspace] = []
        for path in sorted(self._workspaces_root.glob("*.json")):
            try:
                item = IssueWorkspace.model_validate(self._load_json_dict(path))
            except Exception:
                continue
            if state is not None and item.state is not state:
                continue
            if issue_key is not None and item.issue_ref.key != issue_key:
                continue
            out.append(item)
        out.sort(key=lambda item: (item.updated_at or 0, item.created_at or 0), reverse=True)
        return out

    def save_workspace(self, workspace: IssueWorkspace, *, expected_revision: int | None = None) -> IssueWorkspace:
        path = self._workspace_path(workspace.workspace_id)
        now = now_ts_ms()
        expected = workspace.revision if expected_revision is None else expected_revision

        if path.exists():
            current = self.get_workspace(workspace.workspace_id)
            if expected != current.revision:
                raise WorkspaceRevisionConflictError(
                    f"Workspace revision mismatch for {workspace.workspace_id}: expected={expected}, actual={current.revision}"
                )
            next_revision = current.revision + 1
            created_at = current.created_at if current.created_at is not None else (workspace.created_at or now)
        else:
            if expected != 0:
                raise WorkspaceRevisionConflictError(
                    f"Workspace revision mismatch for create {workspace.workspace_id}: expected={expected}, actual=0"
                )
            next_revision = 1
            created_at = workspace.created_at or now

        saved = workspace.model_copy(update={"created_at": created_at, "updated_at": now, "revision": next_revision})
        _safe_write_json(path, saved.model_dump(mode="json"))
        return saved

    def get_workbench(self, workbench_id: str) -> Workbench:
        path = self._workbench_path(workbench_id)
        if not path.exists():
            raise WorkspaceNotFoundError(f"Workbench not found: {workbench_id}")
        raw = self._load_json_dict(path)
        return Workbench.model_validate(raw)

    def list_workbenches(
        self,
        *,
        workspace_id: str | None = None,
        state: WorkbenchState | None = None,
        agent_id: str | None = None,
        instance_id: str | None = None,
    ) -> list[Workbench]:
        out: list[Workbench] = []
        for path in sorted(self._workbenches_root.glob("*.json")):
            try:
                item = Workbench.model_validate(self._load_json_dict(path))
            except Exception:
                continue
            if workspace_id is not None and item.workspace_id != workspace_id:
                continue
            if state is not None and item.state is not state:
                continue
            if agent_id is not None and item.agent_id != agent_id:
                continue
            if instance_id is not None and item.instance_id != instance_id:
                continue
            out.append(item)
        out.sort(key=lambda item: (item.updated_at or 0, item.created_at or 0), reverse=True)
        return out

    def save_workbench(self, workbench: Workbench, *, expected_revision: int | None = None) -> Workbench:
        path = self._workbench_path(workbench.workbench_id)
        now = now_ts_ms()
        expected = workbench.revision if expected_revision is None else expected_revision

        if path.exists():
            current = self.get_workbench(workbench.workbench_id)
            if expected != current.revision:
                raise WorkspaceRevisionConflictError(
                    f"Workbench revision mismatch for {workbench.workbench_id}: expected={expected}, actual={current.revision}"
                )
            next_revision = current.revision + 1
            created_at = current.created_at if current.created_at is not None else (workbench.created_at or now)
        else:
            if expected != 0:
                raise WorkspaceRevisionConflictError(
                    f"Workbench revision mismatch for create {workbench.workbench_id}: expected={expected}, actual=0"
                )
            next_revision = 1
            created_at = workbench.created_at or now

        saved = workbench.model_copy(update={"created_at": created_at, "updated_at": now, "revision": next_revision})
        _safe_write_json(path, saved.model_dump(mode="json"))
        return saved

    def get_session_binding(self, session_id: str) -> SessionWorkspaceBinding:
        path = self._session_binding_path(session_id)
        if not path.exists():
            raise WorkspaceNotFoundError(f"Session workspace binding not found: {session_id}")
        raw = self._load_json_dict(path)
        return SessionWorkspaceBinding.model_validate(raw)

    def list_session_bindings(
        self,
        *,
        workspace_id: str | None = None,
        workbench_id: str | None = None,
    ) -> list[SessionWorkspaceBinding]:
        out: list[SessionWorkspaceBinding] = []
        for path in sorted(self._sessions_root.glob("*.json")):
            try:
                item = SessionWorkspaceBinding.model_validate(self._load_json_dict(path))
            except Exception:
                continue
            if workspace_id is not None and item.workspace_id != workspace_id:
                continue
            if workbench_id is not None and item.workbench_id != workbench_id:
                continue
            out.append(item)
        out.sort(key=lambda item: (item.updated_at or 0, item.created_at or 0), reverse=True)
        return out

    def save_session_binding(
        self,
        binding: SessionWorkspaceBinding,
        *,
        expected_revision: int | None = None,
    ) -> SessionWorkspaceBinding:
        path = self._session_binding_path(binding.session_id)
        now = now_ts_ms()
        expected = binding.revision if expected_revision is None else expected_revision

        if path.exists():
            current = self.get_session_binding(binding.session_id)
            if expected != current.revision:
                raise WorkspaceRevisionConflictError(
                    f"Session binding revision mismatch for {binding.session_id}: expected={expected}, actual={current.revision}"
                )
            next_revision = current.revision + 1
            created_at = current.created_at if current.created_at is not None else (binding.created_at or now)
        else:
            if expected != 0:
                raise WorkspaceRevisionConflictError(
                    f"Session binding revision mismatch for create {binding.session_id}: expected={expected}, actual=0"
                )
            next_revision = 1
            created_at = binding.created_at or now

        saved = binding.model_copy(update={"created_at": created_at, "updated_at": now, "revision": next_revision})
        _safe_write_json(path, saved.model_dump(mode="json"))
        return saved

    def clear_session_binding(self, session_id: str) -> None:
        path = self._session_binding_path(session_id)
        if path.exists():
            path.unlink()

    def append_submission(self, submission: WorkspaceSubmission) -> WorkspaceSubmission:
        existing_by_id = self._get_submission_by_id(submission.submission_id)
        if existing_by_id is not None:
            return existing_by_id
        existing = self.find_submission_by_commit(
            workspace_id=submission.workspace_id,
            workbench_id=submission.workbench_id,
            commit_sha=submission.commit_sha,
        )
        if existing is not None:
            return existing

        now = now_ts_ms()
        created_at = submission.created_at or now
        saved = submission.model_copy(update={"created_at": created_at, "updated_at": now})
        with self._submissions_path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(json.dumps(_sanitize_json_value(saved.model_dump(mode="json")), ensure_ascii=False))
            handle.write("\n")
        return saved

    def update_submission(
        self,
        submission_id: str,
        *,
        status: WorkspaceSubmissionStatus | None = None,
        tool_call_ids: list[str] | None = None,
        pr_url: str | None = None,
        ci_url: str | None = None,
        notes: str | None = None,
    ) -> WorkspaceSubmission:
        current = self.get_submission(submission_id)
        now = now_ts_ms()
        updates: dict[str, Any] = {"updated_at": now}
        if status is not None:
            updates["status"] = status
        if tool_call_ids is not None:
            merged = list(current.tool_call_ids)
            seen = set(merged)
            for raw in tool_call_ids:
                if not isinstance(raw, str) or not raw.strip():
                    raise WorkspaceStoreError("Invalid tool_call_ids: expected list of non-empty strings.")
                item = raw.strip()
                if item in seen:
                    continue
                seen.add(item)
                merged.append(item)
            updates["tool_call_ids"] = merged
        if pr_url is not None:
            updates["pr_url"] = pr_url
        if ci_url is not None:
            updates["ci_url"] = ci_url
        if notes is not None:
            updates["notes"] = notes

        payload = current.model_dump(mode="json")
        payload.update(updates)
        try:
            saved = WorkspaceSubmission.model_validate(payload)
        except Exception as e:
            raise WorkspaceStoreError(f"Invalid submission update for {submission_id}: {e}") from e
        with self._submissions_path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(json.dumps(_sanitize_json_value(saved.model_dump(mode="json")), ensure_ascii=False))
            handle.write("\n")
        return saved

    def get_submission(self, submission_id: str) -> WorkspaceSubmission:
        found = self._get_submission_by_id(submission_id)
        if found is not None:
            return found
        raise WorkspaceNotFoundError(f"Workspace submission not found: {submission_id}")

    def list_submissions(
        self,
        *,
        workspace_id: str | None = None,
        workbench_id: str | None = None,
        status: WorkspaceSubmissionStatus | None = None,
    ) -> list[WorkspaceSubmission]:
        out: list[WorkspaceSubmission] = []
        for item in self._iter_submissions_latest():
            if workspace_id is not None and item.workspace_id != workspace_id:
                continue
            if workbench_id is not None and item.workbench_id != workbench_id:
                continue
            if status is not None and item.status is not status:
                continue
            out.append(item)
        out.sort(key=lambda item: (item.updated_at or 0, item.created_at or 0), reverse=True)
        return out

    def find_submission_by_commit(
        self,
        *,
        workspace_id: str,
        workbench_id: str,
        commit_sha: str,
    ) -> WorkspaceSubmission | None:
        commit_key = commit_sha.strip().lower()
        for item in self._iter_submissions_latest():
            if item.workspace_id != workspace_id:
                continue
            if item.workbench_id != workbench_id:
                continue
            if item.commit_sha == commit_key:
                return item
        return None

    def append_timeline(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("created_at", now_ts_ms())
        with self._timeline_path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(json.dumps(_sanitize_json_value(payload), ensure_ascii=False))
            handle.write("\n")

    def list_timeline(
        self,
        *,
        workspace_id: str | None = None,
        workbench_id: str | None = None,
        submission_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self._timeline_path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self._timeline_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    continue
                if not isinstance(raw, dict):
                    continue
                if workspace_id is not None and raw.get("workspace_id") != workspace_id:
                    continue
                if workbench_id is not None and raw.get("workbench_id") != workbench_id:
                    continue
                if submission_id is not None and raw.get("submission_id") != submission_id:
                    continue
                out.append(raw)
        out.sort(key=lambda item: int(item.get("created_at") or 0), reverse=True)
        if isinstance(limit, int) and limit > 0:
            return out[:limit]
        return out

    def _iter_jsonl_dicts(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    continue
                if isinstance(raw, dict):
                    out.append(raw)
        return out

    @staticmethod
    def _record_ts(item: dict[str, Any]) -> int:
        return int(item.get("updated_at") or item.get("created_at") or 0)

    def _iter_latest_index_records(self, *, path: Path, id_field: str) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for item in self._iter_jsonl_dicts(path):
            item_id = str(item.get(id_field) or "").strip()
            if not item_id:
                continue
            cur = latest.get(item_id)
            if cur is None or self._record_ts(item) >= self._record_ts(cur):
                latest[item_id] = item
        return list(latest.values())

    def _get_latest_index_record(self, *, path: Path, id_field: str, item_id: str) -> dict[str, Any] | None:
        needle = str(item_id or "").strip()
        if not needle:
            return None
        match: dict[str, Any] | None = None
        for item in self._iter_jsonl_dicts(path):
            if str(item.get(id_field) or "").strip() != needle:
                continue
            if match is None or self._record_ts(item) >= self._record_ts(match):
                match = item
        return match

    def _append_index_record(self, *, path: Path, id_field: str, record: dict[str, Any]) -> dict[str, Any]:
        item_id = str(record.get(id_field) or "").strip()
        if not item_id:
            raise WorkspaceStoreError(f"Missing index id field: {id_field}")
        existing = self._get_latest_index_record(path=path, id_field=id_field, item_id=item_id)
        if existing is not None:
            return existing
        now = now_ts_ms()
        payload = dict(record)
        payload[id_field] = item_id
        payload.setdefault("created_at", now)
        payload.setdefault("updated_at", payload.get("created_at") or now)
        with path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(json.dumps(_sanitize_json_value(payload), ensure_ascii=False))
            handle.write("\n")
        return payload

    def _update_index_record(
        self,
        *,
        path: Path,
        id_field: str,
        item_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        current = self._get_latest_index_record(path=path, id_field=id_field, item_id=item_id)
        if current is None:
            raise WorkspaceNotFoundError(f"Record not found: {id_field}={item_id}")
        payload = dict(current)
        payload.update(dict(updates))
        payload[id_field] = str(item_id).strip()
        payload["updated_at"] = now_ts_ms()
        with path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(json.dumps(_sanitize_json_value(payload), ensure_ascii=False))
            handle.write("\n")
        return payload

    def append_heartbeat(self, heartbeat: dict[str, Any]) -> dict[str, Any]:
        return self._append_index_record(path=self._heartbeats_path, id_field="heartbeat_id", record=heartbeat)

    def get_heartbeat(self, heartbeat_id: str) -> dict[str, Any]:
        item = self._get_latest_index_record(path=self._heartbeats_path, id_field="heartbeat_id", item_id=heartbeat_id)
        if item is None:
            raise WorkspaceNotFoundError(f"Workspace heartbeat not found: {heartbeat_id}")
        return item

    def update_heartbeat(self, heartbeat_id: str, *, updates: dict[str, Any]) -> dict[str, Any]:
        return self._update_index_record(
            path=self._heartbeats_path,
            id_field="heartbeat_id",
            item_id=heartbeat_id,
            updates=updates,
        )

    def list_heartbeats(
        self,
        *,
        workspace_id: str | None = None,
        issue_key: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self._iter_latest_index_records(path=self._heartbeats_path, id_field="heartbeat_id"):
            if workspace_id is not None and str(item.get("workspace_id") or "") != workspace_id:
                continue
            if issue_key is not None and str(item.get("issue_key") or "") != issue_key:
                continue
            if status is not None and str(item.get("status") or "") != status:
                continue
            out.append(item)
        out.sort(key=self._record_ts, reverse=True)
        if isinstance(limit, int) and limit > 0:
            return out[:limit]
        return out

    def append_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        return self._append_index_record(path=self._claims_path, id_field="claim_id", record=claim)

    def get_claim(self, claim_id: str) -> dict[str, Any]:
        item = self._get_latest_index_record(path=self._claims_path, id_field="claim_id", item_id=claim_id)
        if item is None:
            raise WorkspaceNotFoundError(f"Workspace claim not found: {claim_id}")
        return item

    def update_claim(self, claim_id: str, *, updates: dict[str, Any]) -> dict[str, Any]:
        return self._update_index_record(
            path=self._claims_path,
            id_field="claim_id",
            item_id=claim_id,
            updates=updates,
        )

    def list_claims(
        self,
        *,
        workspace_id: str | None = None,
        heartbeat_id: str | None = None,
        agent_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self._iter_latest_index_records(path=self._claims_path, id_field="claim_id"):
            if workspace_id is not None and str(item.get("workspace_id") or "") != workspace_id:
                continue
            if heartbeat_id is not None and str(item.get("heartbeat_id") or "") != heartbeat_id:
                continue
            if agent_id is not None and str(item.get("agent_id") or "") != agent_id:
                continue
            if status is not None and str(item.get("status") or "") != status:
                continue
            out.append(item)
        out.sort(key=self._record_ts, reverse=True)
        if isinstance(limit, int) and limit > 0:
            return out[:limit]
        return out

    def append_award(self, award: dict[str, Any]) -> dict[str, Any]:
        return self._append_index_record(path=self._awards_path, id_field="award_id", record=award)

    def get_award(self, award_id: str) -> dict[str, Any]:
        item = self._get_latest_index_record(path=self._awards_path, id_field="award_id", item_id=award_id)
        if item is None:
            raise WorkspaceNotFoundError(f"Workspace award not found: {award_id}")
        return item

    def list_awards(
        self,
        *,
        workspace_id: str | None = None,
        heartbeat_id: str | None = None,
        claim_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self._iter_latest_index_records(path=self._awards_path, id_field="award_id"):
            if workspace_id is not None and str(item.get("workspace_id") or "") != workspace_id:
                continue
            if heartbeat_id is not None and str(item.get("heartbeat_id") or "") != heartbeat_id:
                continue
            if claim_id is not None and str(item.get("claim_id") or "") != claim_id:
                continue
            if agent_id is not None and str(item.get("agent_id") or "") != agent_id:
                continue
            out.append(item)
        out.sort(key=self._record_ts, reverse=True)
        if isinstance(limit, int) and limit > 0:
            return out[:limit]
        return out

    def _iter_submissions(self) -> list[WorkspaceSubmission]:
        if not self._submissions_path.exists():
            return []
        out: list[WorkspaceSubmission] = []
        with self._submissions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    out.append(WorkspaceSubmission.model_validate(raw))
                except Exception:
                    continue
        return out

    def _iter_submissions_latest(self) -> list[WorkspaceSubmission]:
        latest: dict[str, WorkspaceSubmission] = {}
        for item in self._iter_submissions():
            cur = latest.get(item.submission_id)
            if cur is None:
                latest[item.submission_id] = item
                continue
            cur_ts = cur.updated_at or cur.created_at or 0
            item_ts = item.updated_at or item.created_at or 0
            if item_ts >= cur_ts:
                latest[item.submission_id] = item
        return list(latest.values())

    def _get_submission_by_id(self, submission_id: str) -> WorkspaceSubmission | None:
        match: WorkspaceSubmission | None = None
        for item in self._iter_submissions():
            if item.submission_id != submission_id:
                continue
            if match is None:
                match = item
                continue
            cur_ts = match.updated_at or match.created_at or 0
            item_ts = item.updated_at or item.created_at or 0
            if item_ts >= cur_ts:
                match = item
        return match


__all__ = [
    "WorkspaceStore",
    "WorkspaceStoreError",
    "WorkspaceNotFoundError",
    "WorkspaceRevisionConflictError",
]
