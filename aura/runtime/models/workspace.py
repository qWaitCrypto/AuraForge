from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


_WORKSPACE_ID_RE = re.compile(r"^ws_[A-Za-z0-9._:-]+$")
_WORKBENCH_ID_RE = re.compile(r"^wb_[A-Za-z0-9._:-]+$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


def _clean_non_empty_str(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return cleaned


def _clean_optional_str(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_non_empty_str(value, field_name=field_name)


def _clean_string_list(values: list[str], *, field_name: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise ValueError(f"{field_name} must contain only strings.")
        item = raw.strip()
        if not item:
            raise ValueError(f"{field_name} cannot contain empty strings.")
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _validate_optional_http_url(value: str | None, *, field_name: str) -> str | None:
    cleaned = _clean_optional_str(value, field_name=field_name)
    if cleaned is None:
        return None
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        raise ValueError(f"{field_name} must start with http:// or https://.")
    return cleaned


class WorkspaceMergePolicy(StrEnum):
    PR_ONLY = "pr_only"


class WorkspacePushPolicy(StrEnum):
    INTEGRATOR_ONLY = "integrator_only"


class IssueWorkspaceState(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    INTEGRATING = "integrating"
    DONE = "done"
    BLOCKED = "blocked"
    ARCHIVED = "archived"


class WorkbenchRole(StrEnum):
    WORKER = "worker"
    INTEGRATOR = "integrator"
    REVIEWER = "reviewer"


class WorkbenchState(StrEnum):
    PROVISIONING = "provisioning"
    READY = "ready"
    RUNNING = "running"
    SUBMITTED = "submitted"
    INTEGRATED = "integrated"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"
    CLOSED = "closed"
    GC = "gc"


class WorkspaceSubmissionStatus(StrEnum):
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INTEGRATED = "integrated"


class WorkspaceIssueRef(BaseModel):
    provider: str
    id: str
    key: str
    url: str | None = None

    @field_validator("provider", "id", "key")
    @classmethod
    def _validate_required_fields(cls, v: str, info) -> str:
        return _clean_non_empty_str(v, field_name=f"issue_ref.{info.field_name}")

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str | None) -> str | None:
        return _validate_optional_http_url(v, field_name="issue_ref.url")


class WorkspaceRepoRef(BaseModel):
    provider: str
    owner: str
    repo: str

    @field_validator("provider", "owner", "repo")
    @classmethod
    def _validate_required_fields(cls, v: str, info) -> str:
        return _clean_non_empty_str(v, field_name=f"repo_ref.{info.field_name}")


class IssueWorkspace(BaseModel):
    workspace_id: str
    issue_ref: WorkspaceIssueRef
    repo_ref: WorkspaceRepoRef
    base_branch: str = "main"
    staging_enabled: bool = True
    staging_branch: str
    merge_policy: WorkspaceMergePolicy = WorkspaceMergePolicy.PR_ONLY
    push_policy: WorkspacePushPolicy = WorkspacePushPolicy.INTEGRATOR_ONLY
    state: IssueWorkspaceState = IssueWorkspaceState.DRAFT
    created_at: int | None = Field(default=None, ge=0)
    updated_at: int | None = Field(default=None, ge=0)
    revision: int = Field(default=0, ge=0)

    @field_validator("workspace_id")
    @classmethod
    def _validate_workspace_id(cls, v: str) -> str:
        cleaned = _clean_non_empty_str(v, field_name="workspace_id")
        if not _WORKSPACE_ID_RE.fullmatch(cleaned):
            raise ValueError("workspace_id must match pattern ws_<token>.")
        return cleaned

    @field_validator("base_branch", "staging_branch")
    @classmethod
    def _validate_branch_fields(cls, v: str, info) -> str:
        return _clean_non_empty_str(v, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "IssueWorkspace":
        if self.created_at is not None and self.updated_at is not None and self.updated_at < self.created_at:
            raise ValueError("updated_at must be >= created_at.")
        return self


class Workbench(BaseModel):
    workbench_id: str
    workspace_id: str
    agent_id: str
    instance_id: str
    role: WorkbenchRole = WorkbenchRole.WORKER
    worktree_path: str
    branch: str
    base_ref: str
    state: WorkbenchState = WorkbenchState.PROVISIONING
    lease_until: int | None = Field(default=None, ge=0)
    last_heartbeat_at: int | None = Field(default=None, ge=0)
    created_at: int | None = Field(default=None, ge=0)
    updated_at: int | None = Field(default=None, ge=0)
    revision: int = Field(default=0, ge=0)

    @field_validator("workbench_id")
    @classmethod
    def _validate_workbench_id(cls, v: str) -> str:
        cleaned = _clean_non_empty_str(v, field_name="workbench_id")
        if not _WORKBENCH_ID_RE.fullmatch(cleaned):
            raise ValueError("workbench_id must match pattern wb_<token>.")
        return cleaned

    @field_validator("workspace_id")
    @classmethod
    def _validate_workspace_id(cls, v: str) -> str:
        cleaned = _clean_non_empty_str(v, field_name="workspace_id")
        if not _WORKSPACE_ID_RE.fullmatch(cleaned):
            raise ValueError("workspace_id must match pattern ws_<token>.")
        return cleaned

    @field_validator("agent_id", "instance_id", "worktree_path", "branch", "base_ref")
    @classmethod
    def _validate_required_fields(cls, v: str, info) -> str:
        return _clean_non_empty_str(v, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_runtime_constraints(self) -> "Workbench":
        if self.created_at is not None and self.updated_at is not None and self.updated_at < self.created_at:
            raise ValueError("updated_at must be >= created_at.")
        if self.state is WorkbenchState.RUNNING and self.lease_until is None:
            raise ValueError("running workbench requires lease_until.")
        if self.last_heartbeat_at is not None and self.lease_until is not None and self.last_heartbeat_at > self.lease_until:
            raise ValueError("last_heartbeat_at must be <= lease_until.")
        return self


class WorkspaceSubmission(BaseModel):
    submission_id: str
    workspace_id: str
    workbench_id: str
    instance_id: str
    agent_id: str
    branch: str
    commit_sha: str
    changed_files: list[str]
    tool_call_ids: list[str]
    status: WorkspaceSubmissionStatus = WorkspaceSubmissionStatus.SUBMITTED
    pr_url: str | None = None
    ci_url: str | None = None
    notes: str | None = None
    created_at: int | None = Field(default=None, ge=0)
    updated_at: int | None = Field(default=None, ge=0)

    @field_validator("submission_id", "workspace_id", "workbench_id", "instance_id", "agent_id", "branch")
    @classmethod
    def _validate_required_fields(cls, v: str, info) -> str:
        field_name = str(info.field_name)
        cleaned = _clean_non_empty_str(v, field_name=field_name)
        if field_name == "workspace_id" and not _WORKSPACE_ID_RE.fullmatch(cleaned):
            raise ValueError("workspace_id must match pattern ws_<token>.")
        if field_name == "workbench_id" and not _WORKBENCH_ID_RE.fullmatch(cleaned):
            raise ValueError("workbench_id must match pattern wb_<token>.")
        return cleaned

    @field_validator("commit_sha")
    @classmethod
    def _validate_commit_sha(cls, v: str) -> str:
        cleaned = _clean_non_empty_str(v, field_name="commit_sha")
        if not _COMMIT_SHA_RE.fullmatch(cleaned):
            raise ValueError("commit_sha must be 7-64 hex characters.")
        return cleaned.lower()

    @field_validator("changed_files", "tool_call_ids")
    @classmethod
    def _validate_string_lists(cls, v: list[str], info) -> list[str]:
        cleaned = _clean_string_list(v, field_name=str(info.field_name))
        if info.field_name == "changed_files" and not cleaned:
            raise ValueError("changed_files must contain at least one path.")
        return cleaned

    @field_validator("pr_url")
    @classmethod
    def _validate_pr_url(cls, v: str | None) -> str | None:
        return _validate_optional_http_url(v, field_name="pr_url")

    @field_validator("ci_url")
    @classmethod
    def _validate_ci_url(cls, v: str | None) -> str | None:
        return _validate_optional_http_url(v, field_name="ci_url")

    @field_validator("notes")
    @classmethod
    def _validate_notes(cls, v: str | None) -> str | None:
        return _clean_optional_str(v, field_name="notes")

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "WorkspaceSubmission":
        if self.created_at is not None and self.updated_at is not None and self.updated_at < self.created_at:
            raise ValueError("updated_at must be >= created_at.")
        return self


class SessionWorkspaceBinding(BaseModel):
    session_id: str
    workspace_id: str
    workbench_id: str
    instance_id: str | None = None
    agent_id: str | None = None
    role: WorkbenchRole | None = None
    created_at: int | None = Field(default=None, ge=0)
    updated_at: int | None = Field(default=None, ge=0)
    revision: int = Field(default=0, ge=0)

    @field_validator("session_id", "workspace_id", "workbench_id")
    @classmethod
    def _validate_required_ids(cls, v: str, info) -> str:
        field_name = str(info.field_name)
        cleaned = _clean_non_empty_str(v, field_name=field_name)
        if field_name == "workspace_id" and not _WORKSPACE_ID_RE.fullmatch(cleaned):
            raise ValueError("workspace_id must match pattern ws_<token>.")
        if field_name == "workbench_id" and not _WORKBENCH_ID_RE.fullmatch(cleaned):
            raise ValueError("workbench_id must match pattern wb_<token>.")
        return cleaned

    @field_validator("instance_id", "agent_id")
    @classmethod
    def _validate_optional_ids(cls, v: str | None, info) -> str | None:
        return _clean_optional_str(v, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "SessionWorkspaceBinding":
        if self.created_at is not None and self.updated_at is not None and self.updated_at < self.created_at:
            raise ValueError("updated_at must be >= created_at.")
        return self
