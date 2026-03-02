from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models.workspace import (
    IssueWorkspaceState,
    WorkbenchRole,
    WorkbenchState,
    WorkspaceIssueRef,
    WorkspaceMergePolicy,
    WorkspacePushPolicy,
    WorkspaceRepoRef,
    WorkspaceSubmissionStatus,
)
from ..workspace import WorkspaceManager, WorkspaceManagerError
from .runtime import ToolExecutionContext


def _as_non_empty_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid '{field_name}' (expected non-empty string).")
    return value.strip()


def _as_string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Invalid '{field_name}' (expected list of strings).")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Invalid '{field_name}' (expected list of non-empty strings).")
        out.append(item.strip())
    return out


def _maybe_role(value: Any) -> WorkbenchRole:
    raw = str(value or WorkbenchRole.WORKER.value).strip()
    try:
        return WorkbenchRole(raw)
    except ValueError as e:
        raise ValueError(f"Invalid 'role': {raw!r}") from e


def _parse_role_or_none(value: Any, *, field_name: str) -> WorkbenchRole | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        return WorkbenchRole(raw)
    except ValueError as e:
        raise ValueError(f"Invalid '{field_name}': {raw!r}") from e


def _effective_operator_role(*, args: dict[str, Any], context: ToolExecutionContext | None) -> WorkbenchRole:
    # Security model: when runtime context carries a workspace role, that role is authoritative.
    # Tool args must not be able to override caller identity.
    if context is not None:
        ctx_role = _parse_role_or_none(context.workspace_role, field_name="context.workspace_role")
        if ctx_role is not None:
            return ctx_role

        caller_kind = str(context.caller_kind or "llm").strip().lower()
        if caller_kind == "system":
            # Executor/system paths may act on behalf of a provided operator role.
            return _parse_role_or_none(args.get("operator_role"), field_name="operator_role") or WorkbenchRole.INTEGRATOR

        # LLM path without a bound workspace role is untrusted; ignore args.operator_role.
        return WorkbenchRole.WORKER

    # Backward-compatible fallback for direct/non-runtime invocations.
    return _parse_role_or_none(args.get("operator_role"), field_name="operator_role") or WorkbenchRole.WORKER


def _require_integrator(*, args: dict[str, Any], context: ToolExecutionContext | None) -> None:
    role = _effective_operator_role(args=args, context=context)
    if role is not WorkbenchRole.INTEGRATOR:
        raise PermissionError("This workspace action requires integrator role.")


@dataclass(frozen=True, slots=True)
class WorkspaceCreateOrGetTool:
    manager: WorkspaceManager
    name: str = "workspace__create_or_get"
    description: str = (
        "Create (or return existing) issue workspace metadata. "
        "This does not execute remote actions; it initializes local workspace state."
    )
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "issue_ref": {
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string"},
                        "id": {"type": "string"},
                        "key": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["provider", "id", "key"],
                    "additionalProperties": False,
                },
                "repo_ref": {
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string"},
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                    },
                    "required": ["provider", "owner", "repo"],
                    "additionalProperties": False,
                },
                "base_branch": {"type": "string"},
                "staging_enabled": {"type": "boolean"},
                "merge_policy": {"type": "string", "enum": ["pr_only"]},
                "push_policy": {"type": "string", "enum": ["integrator_only"]},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["issue_ref", "repo_ref"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        issue_raw = args.get("issue_ref")
        repo_raw = args.get("repo_ref")
        if not isinstance(issue_raw, dict):
            raise ValueError("Missing or invalid 'issue_ref' (expected object).")
        if not isinstance(repo_raw, dict):
            raise ValueError("Missing or invalid 'repo_ref' (expected object).")

        issue_ref = WorkspaceIssueRef.model_validate(issue_raw)
        repo_ref = WorkspaceRepoRef.model_validate(repo_raw)

        merge_policy = WorkspaceMergePolicy(str(args.get("merge_policy") or WorkspaceMergePolicy.PR_ONLY.value))
        push_policy = WorkspacePushPolicy(str(args.get("push_policy") or WorkspacePushPolicy.INTEGRATOR_ONLY.value))
        base_branch = str(args.get("base_branch") or "main").strip() or "main"
        staging_enabled = bool(args.get("staging_enabled", True))

        ws = self.manager.create_or_get_workspace(
            issue_ref=issue_ref,
            repo_ref=repo_ref,
            base_branch=base_branch,
            staging_enabled=staging_enabled,
            merge_policy=merge_policy,
            push_policy=push_policy,
            workspace_id=(str(args.get("workspace_id")).strip() if isinstance(args.get("workspace_id"), str) else None),
        )
        return {"ok": True, "workspace": ws.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class WorkspaceProvisionWorkbenchTool:
    manager: WorkspaceManager
    name: str = "workspace__provision_workbench"
    description: str = (
        "Provision an isolated workbench (worktree + branch metadata) for one agent instance within a workspace."
    )
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "instance_id": {"type": "string"},
                "role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
                "bind_session": {"type": "boolean"},
                "lease_seconds": {"type": "integer", "minimum": 1},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["workspace_id", "agent_id", "instance_id"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        bind_session = bool(args.get("bind_session", True))
        session_id = context.session_id if (bind_session and context is not None and context.session_id) else None

        wb = self.manager.provision_workbench(
            workspace_id=_as_non_empty_str(args.get("workspace_id"), field_name="workspace_id"),
            agent_id=_as_non_empty_str(args.get("agent_id"), field_name="agent_id"),
            instance_id=_as_non_empty_str(args.get("instance_id"), field_name="instance_id"),
            role=_maybe_role(args.get("role")),
            bind_session_id=session_id,
            lease_seconds=int(args.get("lease_seconds") or 900),
        )
        ws = self.manager.get_workspace(workspace_id=wb.workspace_id)
        out: dict[str, Any] = {
            "ok": True,
            "workspace": ws.model_dump(mode="json"),
            "workbench": wb.model_dump(mode="json"),
            "work_spec_patch": {"resource_scope": {"workspace_roots": [wb.worktree_path]}},
        }
        if session_id:
            out["bound_session_id"] = session_id
        return out


@dataclass(frozen=True, slots=True)
class WorkspaceContextTool:
    manager: WorkspaceManager
    name: str = "workspace__context"
    description: str = "Return current workspace/workbench context for this session (or explicit ids)."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "workspace_id": {"type": "string"},
                "workbench_id": {"type": "string"},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        if context is not None and isinstance(context.session_id, str) and context.session_id.strip():
            session_id = context.session_id
        else:
            session_id = str(args.get("session_id") or "").strip() or None
        workspace_id = str(args.get("workspace_id") or "").strip() or None
        workbench_id = str(args.get("workbench_id") or "").strip() or None
        try:
            resolved = self.manager.resolve_context(
                session_id=session_id,
                workspace_id=workspace_id,
                workbench_id=workbench_id,
            )
        except WorkspaceManagerError as e:
            return {"ok": False, "error": str(e)}
        return {
            "ok": True,
            "workspace": resolved.workspace.model_dump(mode="json"),
            "workbench": resolved.workbench.model_dump(mode="json"),
            "binding": resolved.binding.model_dump(mode="json") if resolved.binding is not None else None,
            "work_spec_patch": {"resource_scope": {"workspace_roots": [resolved.workbench.worktree_path]}},
        }


@dataclass(frozen=True, slots=True)
class WorkspacePublishHeartbeatTool:
    manager: WorkspaceManager
    name: str = "workspace__publish_heartbeat"
    description: str = "Publish a task heartbeat in one workspace for agents to claim."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "summary": {"type": "string"},
                "topic": {"type": "string"},
                "issuer_agent_id": {"type": "string"},
                "issuer_instance_id": {"type": "string"},
                "claim_ttl_seconds": {"type": "integer", "minimum": 1},
                "metadata": {"type": "object"},
                "heartbeat_id": {"type": "string"},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["workspace_id", "summary"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        heartbeat = self.manager.publish_heartbeat(
            workspace_id=_as_non_empty_str(args.get("workspace_id"), field_name="workspace_id"),
            summary=_as_non_empty_str(args.get("summary"), field_name="summary"),
            topic=(str(args.get("topic")).strip() if isinstance(args.get("topic"), str) else None),
            issuer_agent_id=(str(args.get("issuer_agent_id")).strip() if isinstance(args.get("issuer_agent_id"), str) else None),
            issuer_instance_id=(str(args.get("issuer_instance_id")).strip() if isinstance(args.get("issuer_instance_id"), str) else None),
            claim_ttl_seconds=(int(args.get("claim_ttl_seconds")) if isinstance(args.get("claim_ttl_seconds"), int) and not isinstance(args.get("claim_ttl_seconds"), bool) else None),
            metadata=(dict(args.get("metadata")) if isinstance(args.get("metadata"), dict) else None),
            heartbeat_id=(str(args.get("heartbeat_id")).strip() if isinstance(args.get("heartbeat_id"), str) else None),
        )
        return {"ok": True, "heartbeat": heartbeat}


@dataclass(frozen=True, slots=True)
class WorkspaceSubmitClaimTool:
    manager: WorkspaceManager
    name: str = "workspace__submit_claim"
    description: str = "Submit an agent claim/bid for a published workspace heartbeat."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "heartbeat_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "instance_id": {"type": "string"},
                "proposal": {"type": "string"},
                "metadata": {"type": "object"},
                "claim_id": {"type": "string"},
            },
            "required": ["heartbeat_id", "proposal"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        heartbeat_id = _as_non_empty_str(args.get("heartbeat_id"), field_name="heartbeat_id")
        proposal = _as_non_empty_str(args.get("proposal"), field_name="proposal")
        agent_id = str(args.get("agent_id") or "").strip() or None
        instance_id = str(args.get("instance_id") or "").strip() or None

        if agent_id is None and context is not None and isinstance(context.session_id, str) and context.session_id.strip():
            try:
                resolved = self.manager.resolve_context(session_id=context.session_id)
                agent_id = resolved.workbench.agent_id
                if instance_id is None:
                    instance_id = resolved.workbench.instance_id
            except Exception:
                agent_id = None
        if agent_id is None:
            raise ValueError("Missing or invalid 'agent_id' (expected non-empty string).")

        claim = self.manager.submit_claim(
            heartbeat_id=heartbeat_id,
            agent_id=agent_id,
            proposal=proposal,
            instance_id=instance_id,
            metadata=(dict(args.get("metadata")) if isinstance(args.get("metadata"), dict) else None),
            claim_id=(str(args.get("claim_id")).strip() if isinstance(args.get("claim_id"), str) else None),
        )
        return {"ok": True, "claim": claim}


@dataclass(frozen=True, slots=True)
class WorkspaceAwardClaimTool:
    manager: WorkspaceManager
    name: str = "workspace__award_claim"
    description: str = "Committee/integrator awards one claim and records the decision."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "committee_agent_id": {"type": "string"},
                "notes": {"type": "string"},
                "auto_reject_other_claims": {"type": "boolean"},
                "award_id": {"type": "string"},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["claim_id"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        result = self.manager.award_claim(
            claim_id=_as_non_empty_str(args.get("claim_id"), field_name="claim_id"),
            committee_agent_id=(str(args.get("committee_agent_id")).strip() if isinstance(args.get("committee_agent_id"), str) else None),
            notes=(str(args.get("notes")).strip() if isinstance(args.get("notes"), str) else None),
            auto_reject_other_claims=bool(args.get("auto_reject_other_claims", True)),
            award_id=(str(args.get("award_id")).strip() if isinstance(args.get("award_id"), str) else None),
        )
        return {"ok": True, **result}


@dataclass(frozen=True, slots=True)
class WorkspaceWakeAwardedAgentTool:
    manager: WorkspaceManager
    name: str = "workspace__wake_awarded_agent"
    description: str = "Wake the awarded agent by provisioning/binding its workbench and returning wake context."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "award_id": {"type": "string"},
                "bind_session": {"type": "boolean"},
                "lease_seconds": {"type": "integer", "minimum": 1},
                "role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["award_id"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        bind_session = bool(args.get("bind_session", True))
        session_id = context.session_id if (bind_session and context is not None and context.session_id) else None
        role = _maybe_role(args.get("role"))
        result = self.manager.wake_awarded_agent(
            award_id=_as_non_empty_str(args.get("award_id"), field_name="award_id"),
            bind_session_id=session_id,
            lease_seconds=int(args.get("lease_seconds") or 900),
            role=role,
        )
        return {
            "ok": True,
            "award": result["award"],
            "claim": result["claim"],
            "heartbeat": result["heartbeat"],
            "workspace": result["workspace"].model_dump(mode="json"),
            "workbench": result["workbench"].model_dump(mode="json"),
            "wake_envelope": result["wake_envelope"],
            "work_spec_patch": {"resource_scope": {"workspace_roots": [result["workbench"].worktree_path]}},
            **({"bound_session_id": session_id} if session_id else {}),
        }


@dataclass(frozen=True, slots=True)
class WorkspaceListHeartbeatsTool:
    manager: WorkspaceManager
    name: str = "workspace__list_heartbeats"
    description: str = "List published workspace heartbeats with optional filters."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "issue_key": {"type": "string"},
                "status": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root, context
        items = self.manager.list_heartbeats(
            workspace_id=(str(args.get("workspace_id")).strip() if isinstance(args.get("workspace_id"), str) else None),
            issue_key=(str(args.get("issue_key")).strip() if isinstance(args.get("issue_key"), str) else None),
            status=(str(args.get("status")).strip() if isinstance(args.get("status"), str) else None),
            limit=(int(args.get("limit")) if isinstance(args.get("limit"), int) and not isinstance(args.get("limit"), bool) else None),
        )
        return {"ok": True, "count": len(items), "items": items}


@dataclass(frozen=True, slots=True)
class WorkspaceListClaimsTool:
    manager: WorkspaceManager
    name: str = "workspace__list_claims"
    description: str = "List workspace heartbeat claims with optional filters."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "heartbeat_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "status": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root, context
        items = self.manager.list_claims(
            workspace_id=(str(args.get("workspace_id")).strip() if isinstance(args.get("workspace_id"), str) else None),
            heartbeat_id=(str(args.get("heartbeat_id")).strip() if isinstance(args.get("heartbeat_id"), str) else None),
            agent_id=(str(args.get("agent_id")).strip() if isinstance(args.get("agent_id"), str) else None),
            status=(str(args.get("status")).strip() if isinstance(args.get("status"), str) else None),
            limit=(int(args.get("limit")) if isinstance(args.get("limit"), int) and not isinstance(args.get("limit"), bool) else None),
        )
        return {"ok": True, "count": len(items), "items": items}


@dataclass(frozen=True, slots=True)
class WorkspaceListAwardsTool:
    manager: WorkspaceManager
    name: str = "workspace__list_awards"
    description: str = "List workspace claim awards with optional filters."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "heartbeat_id": {"type": "string"},
                "claim_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root, context
        items = self.manager.list_awards(
            workspace_id=(str(args.get("workspace_id")).strip() if isinstance(args.get("workspace_id"), str) else None),
            heartbeat_id=(str(args.get("heartbeat_id")).strip() if isinstance(args.get("heartbeat_id"), str) else None),
            claim_id=(str(args.get("claim_id")).strip() if isinstance(args.get("claim_id"), str) else None),
            agent_id=(str(args.get("agent_id")).strip() if isinstance(args.get("agent_id"), str) else None),
            limit=(int(args.get("limit")) if isinstance(args.get("limit"), int) and not isinstance(args.get("limit"), bool) else None),
        )
        return {"ok": True, "count": len(items), "items": items}


@dataclass(frozen=True, slots=True)
class WorkspaceRegisterSubmissionTool:
    manager: WorkspaceManager
    name: str = "workspace__register_submission"
    description: str = "Register auditable delivery evidence (commit/PR/CI/tool calls) for the current workbench."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "workbench_id": {"type": "string"},
                "instance_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "branch": {"type": "string"},
                "commit_sha": {"type": "string"},
                "changed_files": {"type": "array", "items": {"type": "string"}},
                "tool_call_ids": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["submitted", "accepted", "rejected", "integrated"]},
                "auto_transition": {"type": "boolean"},
                "submission_id": {"type": "string"},
                "pr_url": {"type": "string"},
                "ci_url": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["commit_sha", "changed_files", "tool_call_ids"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        status_raw = str(args.get("status") or WorkspaceSubmissionStatus.SUBMITTED.value).strip()
        try:
            status = WorkspaceSubmissionStatus(status_raw)
        except ValueError as e:
            raise ValueError(f"Invalid 'status': {status_raw!r}") from e

        submission = self.manager.register_submission(
            session_id=(context.session_id if context is not None else None),
            workspace_id=(str(args.get("workspace_id")).strip() if isinstance(args.get("workspace_id"), str) else None),
            workbench_id=(str(args.get("workbench_id")).strip() if isinstance(args.get("workbench_id"), str) else None),
            instance_id=(str(args.get("instance_id")).strip() if isinstance(args.get("instance_id"), str) else None),
            agent_id=(str(args.get("agent_id")).strip() if isinstance(args.get("agent_id"), str) else None),
            branch=(str(args.get("branch")).strip() if isinstance(args.get("branch"), str) else None),
            commit_sha=_as_non_empty_str(args.get("commit_sha"), field_name="commit_sha"),
            changed_files=_as_string_list(args.get("changed_files"), field_name="changed_files"),
            tool_call_ids=_as_string_list(args.get("tool_call_ids"), field_name="tool_call_ids"),
            status=status,
            auto_transition=bool(args.get("auto_transition", False)),
            submission_id=(str(args.get("submission_id")).strip() if isinstance(args.get("submission_id"), str) else None),
            pr_url=(str(args.get("pr_url")).strip() if isinstance(args.get("pr_url"), str) else None),
            ci_url=(str(args.get("ci_url")).strip() if isinstance(args.get("ci_url"), str) else None),
            notes=(str(args.get("notes")).strip() if isinstance(args.get("notes"), str) else None),
        )
        return {"ok": True, "submission": submission.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class WorkspaceAcceptSubmissionTool:
    manager: WorkspaceManager
    name: str = "workspace__accept_submission"
    description: str = (
        "Apply integrator decision to a submission (accepted/rejected/integrated). "
        "By default this only records decision facts; state transitions are explicit."
    )
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "submission_id": {"type": "string"},
                "target_status": {"type": "string", "enum": ["accepted", "rejected", "integrated"]},
                "tool_call_ids": {"type": "array", "items": {"type": "string"}},
                "auto_transition": {"type": "boolean"},
                "pr_url": {"type": "string"},
                "ci_url": {"type": "string"},
                "notes": {"type": "string"},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["submission_id", "target_status"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        target_raw = _as_non_empty_str(args.get("target_status"), field_name="target_status")
        try:
            target = WorkspaceSubmissionStatus(target_raw)
        except ValueError as e:
            raise ValueError(f"Invalid 'target_status': {target_raw!r}") from e
        tool_call_ids: list[str] | None = None
        if "tool_call_ids" in args:
            tool_call_ids = _as_string_list(args.get("tool_call_ids"), field_name="tool_call_ids")
        submission = self.manager.accept_submission(
            submission_id=_as_non_empty_str(args.get("submission_id"), field_name="submission_id"),
            target_status=target,
            tool_call_ids=tool_call_ids,
            auto_transition=bool(args.get("auto_transition", False)),
            pr_url=(str(args.get("pr_url")).strip() if isinstance(args.get("pr_url"), str) else None),
            ci_url=(str(args.get("ci_url")).strip() if isinstance(args.get("ci_url"), str) else None),
            notes=(str(args.get("notes")).strip() if isinstance(args.get("notes"), str) else None),
        )
        return {"ok": True, "submission": submission.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class WorkspaceAppendSubmissionEvidenceTool:
    manager: WorkspaceManager
    name: str = "workspace__append_submission_evidence"
    description: str = "Append auditable evidence to an existing submission without changing workflow states."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "submission_id": {"type": "string"},
                "tool_call_ids": {"type": "array", "items": {"type": "string"}},
                "pr_url": {"type": "string"},
                "ci_url": {"type": "string"},
                "notes": {"type": "string"},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["submission_id"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        tool_call_ids: list[str] | None = None
        if "tool_call_ids" in args:
            tool_call_ids = _as_string_list(args.get("tool_call_ids"), field_name="tool_call_ids")
        pr_url = str(args.get("pr_url")).strip() if isinstance(args.get("pr_url"), str) else None
        ci_url = str(args.get("ci_url")).strip() if isinstance(args.get("ci_url"), str) else None
        notes = str(args.get("notes")).strip() if isinstance(args.get("notes"), str) else None

        if not tool_call_ids and pr_url is None and ci_url is None and notes is None:
            raise ValueError("At least one evidence field is required (tool_call_ids, pr_url, ci_url, notes).")

        submission = self.manager.append_submission_evidence(
            submission_id=_as_non_empty_str(args.get("submission_id"), field_name="submission_id"),
            tool_call_ids=tool_call_ids,
            pr_url=pr_url,
            ci_url=ci_url,
            notes=notes,
        )
        return {"ok": True, "submission": submission.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class WorkspaceAdvanceIssueStateTool:
    manager: WorkspaceManager
    name: str = "workspace__advance_issue_state"
    description: str = "Advance issue workspace state using enforced transition rules."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "target_state": {
                    "type": "string",
                    "enum": ["draft", "active", "integrating", "done", "blocked", "archived"],
                },
                "reason": {"type": "string"},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["workspace_id", "target_state"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        target_raw = _as_non_empty_str(args.get("target_state"), field_name="target_state")
        try:
            target = IssueWorkspaceState(target_raw)
        except ValueError as e:
            raise ValueError(f"Invalid 'target_state': {target_raw!r}") from e
        workspace = self.manager.transition_workspace_state(
            workspace_id=_as_non_empty_str(args.get("workspace_id"), field_name="workspace_id"),
            target_state=target,
            reason=(str(args.get("reason")).strip() if isinstance(args.get("reason"), str) else None),
        )
        return {"ok": True, "workspace": workspace.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class WorkspaceTransitionWorkbenchStateTool:
    manager: WorkspaceManager
    name: str = "workspace__transition_workbench_state"
    description: str = "Transition one workbench state explicitly using enforced transition rules."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workbench_id": {"type": "string"},
                "target_state": {
                    "type": "string",
                    "enum": ["provisioning", "ready", "running", "submitted", "integrated", "blocked", "abandoned", "closed", "gc"],
                },
                "reason": {"type": "string"},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["workbench_id", "target_state"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        target_raw = _as_non_empty_str(args.get("target_state"), field_name="target_state")
        try:
            target = WorkbenchState(target_raw)
        except ValueError as e:
            raise ValueError(f"Invalid 'target_state': {target_raw!r}") from e
        workbench = self.manager.transition_workbench_state(
            workbench_id=_as_non_empty_str(args.get("workbench_id"), field_name="workbench_id"),
            target_state=target,
            reason=(str(args.get("reason")).strip() if isinstance(args.get("reason"), str) else None),
        )
        return {"ok": True, "workbench": workbench.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class WorkspaceHeartbeatWorkbenchTool:
    manager: WorkspaceManager
    name: str = "workspace__heartbeat_workbench"
    description: str = "Renew workbench lease/heartbeat for long-running agent execution."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workbench_id": {"type": "string"},
                "lease_seconds": {"type": "integer", "minimum": 1},
            },
            "required": ["workbench_id"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root, context
        wb = self.manager.heartbeat_workbench(
            workbench_id=_as_non_empty_str(args.get("workbench_id"), field_name="workbench_id"),
            lease_seconds=int(args.get("lease_seconds") or 900),
        )
        return {"ok": True, "workbench": wb.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class WorkspaceRecoverExpiredWorkbenchesTool:
    manager: WorkspaceManager
    name: str = "workspace__recover_expired_workbenches"
    description: str = "Recover running workbenches whose lease has expired into ready/blocked state."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "target_state": {"type": "string", "enum": ["ready", "blocked"]},
                "now_ms": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1},
                "reason": {"type": "string"},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        target_raw = str(args.get("target_state") or WorkbenchState.READY.value).strip()
        try:
            target = WorkbenchState(target_raw)
        except ValueError as e:
            raise ValueError(f"Invalid 'target_state': {target_raw!r}") from e
        items = self.manager.recover_expired_workbenches(
            workspace_id=(str(args.get("workspace_id")).strip() if isinstance(args.get("workspace_id"), str) else None),
            target_state=target,
            now_ms=(int(args.get("now_ms")) if isinstance(args.get("now_ms"), int) and not isinstance(args.get("now_ms"), bool) else None),
            limit=(int(args.get("limit")) if isinstance(args.get("limit"), int) and not isinstance(args.get("limit"), bool) else None),
            reason=(str(args.get("reason")).strip() if isinstance(args.get("reason"), str) else None),
        )
        return {"ok": True, "count": len(items), "items": [it.model_dump(mode="json") for it in items]}


@dataclass(frozen=True, slots=True)
class WorkspaceCloseWorkbenchTool:
    manager: WorkspaceManager
    name: str = "workspace__close_workbench"
    description: str = "Close one workbench lifecycle (abandon/close path) and unbind attached sessions."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workbench_id": {"type": "string"},
                "reason": {"type": "string"},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["workbench_id"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        wb = self.manager.close_workbench(
            workbench_id=_as_non_empty_str(args.get("workbench_id"), field_name="workbench_id"),
            reason=(str(args.get("reason")).strip() if isinstance(args.get("reason"), str) else None),
        )
        return {"ok": True, "workbench": wb.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class WorkspaceGcWorkbenchTool:
    manager: WorkspaceManager
    name: str = "workspace__gc_workbench"
    description: str = "Move a closed workbench to gc and optionally delete its worktree directory."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workbench_id": {"type": "string"},
                "delete_worktree": {"type": "boolean"},
                "reason": {"type": "string"},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["workbench_id"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        wb = self.manager.gc_workbench(
            workbench_id=_as_non_empty_str(args.get("workbench_id"), field_name="workbench_id"),
            delete_worktree=bool(args.get("delete_worktree", True)),
            reason=(str(args.get("reason")).strip() if isinstance(args.get("reason"), str) else None),
        )
        return {"ok": True, "workbench": wb.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class WorkspaceCloseWorkspaceTool:
    manager: WorkspaceManager
    name: str = "workspace__close_workspace"
    description: str = "Close one issue workspace to done/blocked and optionally archive it."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "final_state": {"type": "string", "enum": ["done", "blocked"]},
                "archive": {"type": "boolean"},
                "close_workbenches": {"type": "boolean"},
                "reason": {"type": "string"},
                "operator_role": {"type": "string", "enum": ["worker", "integrator", "reviewer"]},
            },
            "required": ["workspace_id"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root
        _require_integrator(args=args, context=context)
        final_raw = str(args.get("final_state") or IssueWorkspaceState.DONE.value).strip()
        try:
            final_state = IssueWorkspaceState(final_raw)
        except ValueError as e:
            raise ValueError(f"Invalid 'final_state': {final_raw!r}") from e
        workspace = self.manager.close_workspace(
            workspace_id=_as_non_empty_str(args.get("workspace_id"), field_name="workspace_id"),
            final_state=final_state,
            archive=bool(args.get("archive", False)),
            close_workbenches=bool(args.get("close_workbenches", False)),
            reason=(str(args.get("reason")).strip() if isinstance(args.get("reason"), str) else None),
        )
        return {"ok": True, "workspace": workspace.model_dump(mode="json")}


@dataclass(frozen=True, slots=True)
class WorkspaceListWorkbenchesTool:
    manager: WorkspaceManager
    name: str = "workspace__list_workbenches"
    description: str = "List workbenches with optional filters."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "state": {
                    "type": "string",
                    "enum": ["provisioning", "ready", "running", "submitted", "integrated", "blocked", "abandoned", "closed", "gc"],
                },
                "agent_id": {"type": "string"},
                "instance_id": {"type": "string"},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root, context
        state_raw = args.get("state")
        state: WorkbenchState | None = None
        if isinstance(state_raw, str) and state_raw.strip():
            try:
                state = WorkbenchState(state_raw.strip())
            except ValueError as e:
                raise ValueError(f"Invalid 'state': {state_raw!r}") from e
        items = self.manager.list_workbenches(
            workspace_id=(str(args.get("workspace_id")).strip() if isinstance(args.get("workspace_id"), str) else None),
            state=state,
            agent_id=(str(args.get("agent_id")).strip() if isinstance(args.get("agent_id"), str) else None),
            instance_id=(str(args.get("instance_id")).strip() if isinstance(args.get("instance_id"), str) else None),
        )
        return {"ok": True, "count": len(items), "items": [it.model_dump(mode="json") for it in items]}


@dataclass(frozen=True, slots=True)
class WorkspaceAuditChainTool:
    manager: WorkspaceManager
    name: str = "workspace__audit_chain"
    description: str = "Return full workspace delivery audit chain (workspace/workbenches/submissions/timeline)."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "include_timeline": {"type": "boolean"},
                "timeline_limit": {"type": "integer", "minimum": 1},
            },
            "required": ["workspace_id"],
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root, context
        data = self.manager.audit_chain(
            workspace_id=_as_non_empty_str(args.get("workspace_id"), field_name="workspace_id"),
            include_timeline=bool(args.get("include_timeline", True)),
            timeline_limit=(int(args.get("timeline_limit")) if isinstance(args.get("timeline_limit"), int) and not isinstance(args.get("timeline_limit"), bool) else 500),
        )
        return {"ok": True, **data}


@dataclass(frozen=True, slots=True)
class WorkspaceListSubmissionsTool:
    manager: WorkspaceManager
    name: str = "workspace__list_submissions"
    description: str = "List workspace submissions (latest version per submission_id)."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "workbench_id": {"type": "string"},
                "status": {"type": "string", "enum": ["submitted", "accepted", "rejected", "integrated"]},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root, context
        status_raw = args.get("status")
        status: WorkspaceSubmissionStatus | None = None
        if isinstance(status_raw, str) and status_raw.strip():
            try:
                status = WorkspaceSubmissionStatus(status_raw.strip())
            except ValueError as e:
                raise ValueError(f"Invalid 'status': {status_raw!r}") from e
        items = self.manager.list_submissions(
            workspace_id=(str(args.get("workspace_id")).strip() if isinstance(args.get("workspace_id"), str) else None),
            workbench_id=(str(args.get("workbench_id")).strip() if isinstance(args.get("workbench_id"), str) else None),
            status=status,
        )
        return {"ok": True, "count": len(items), "items": [it.model_dump(mode="json") for it in items]}


@dataclass(frozen=True, slots=True)
class WorkspaceTimelineTool:
    manager: WorkspaceManager
    name: str = "workspace__timeline"
    description: str = "Read workspace timeline events for auditing."
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "workbench_id": {"type": "string"},
                "submission_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        }
    )

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        del project_root, context
        items = self.manager.list_timeline(
            workspace_id=(str(args.get("workspace_id")).strip() if isinstance(args.get("workspace_id"), str) else None),
            workbench_id=(str(args.get("workbench_id")).strip() if isinstance(args.get("workbench_id"), str) else None),
            submission_id=(str(args.get("submission_id")).strip() if isinstance(args.get("submission_id"), str) else None),
            limit=(int(args.get("limit")) if isinstance(args.get("limit"), int) and not isinstance(args.get("limit"), bool) else None),
        )
        return {"ok": True, "count": len(items), "items": items}


__all__ = [
    "WorkspaceAwardClaimTool",
    "WorkspaceAcceptSubmissionTool",
    "WorkspaceAuditChainTool",
    "WorkspaceAppendSubmissionEvidenceTool",
    "WorkspaceAdvanceIssueStateTool",
    "WorkspaceCloseWorkbenchTool",
    "WorkspaceCloseWorkspaceTool",
    "WorkspaceContextTool",
    "WorkspaceCreateOrGetTool",
    "WorkspaceGcWorkbenchTool",
    "WorkspaceHeartbeatWorkbenchTool",
    "WorkspaceListAwardsTool",
    "WorkspaceListClaimsTool",
    "WorkspaceListHeartbeatsTool",
    "WorkspaceListWorkbenchesTool",
    "WorkspaceListSubmissionsTool",
    "WorkspacePublishHeartbeatTool",
    "WorkspaceProvisionWorkbenchTool",
    "WorkspaceRecoverExpiredWorkbenchesTool",
    "WorkspaceRegisterSubmissionTool",
    "WorkspaceSubmitClaimTool",
    "WorkspaceTimelineTool",
    "WorkspaceTransitionWorkbenchStateTool",
    "WorkspaceWakeAwardedAgentTool",
]
