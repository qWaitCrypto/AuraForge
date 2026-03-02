from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ids import new_id, now_ts_ms
from ..models.workspace import (
    IssueWorkspace,
    IssueWorkspaceState,
    SessionWorkspaceBinding,
    Workbench,
    WorkbenchRole,
    WorkbenchState,
    WorkspaceIssueRef,
    WorkspaceMergePolicy,
    WorkspacePushPolicy,
    WorkspaceRepoRef,
    WorkspaceSubmission,
    WorkspaceSubmissionStatus,
)
from .store import WorkspaceNotFoundError, WorkspaceStore


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._:-]+", "-", str(value or "").strip())
    cleaned = cleaned.strip("-")
    return cleaned or "na"


@dataclass(frozen=True, slots=True)
class ResolvedWorkspaceContext:
    workspace: IssueWorkspace
    workbench: Workbench
    binding: SessionWorkspaceBinding | None


class WorkspaceManagerError(RuntimeError):
    pass


class WorkspaceGitError(WorkspaceManagerError):
    pass


_WORKSPACE_TRANSITIONS: dict[IssueWorkspaceState, set[IssueWorkspaceState]] = {
    IssueWorkspaceState.DRAFT: {IssueWorkspaceState.ACTIVE},
    IssueWorkspaceState.ACTIVE: {IssueWorkspaceState.INTEGRATING, IssueWorkspaceState.BLOCKED, IssueWorkspaceState.DONE},
    IssueWorkspaceState.INTEGRATING: {IssueWorkspaceState.ACTIVE, IssueWorkspaceState.BLOCKED, IssueWorkspaceState.DONE},
    IssueWorkspaceState.BLOCKED: {IssueWorkspaceState.ACTIVE, IssueWorkspaceState.ARCHIVED},
    IssueWorkspaceState.DONE: {IssueWorkspaceState.ARCHIVED},
    IssueWorkspaceState.ARCHIVED: set(),
}

_WORKBENCH_TRANSITIONS: dict[WorkbenchState, set[WorkbenchState]] = {
    WorkbenchState.PROVISIONING: {WorkbenchState.READY, WorkbenchState.BLOCKED, WorkbenchState.ABANDONED},
    WorkbenchState.READY: {WorkbenchState.RUNNING, WorkbenchState.SUBMITTED, WorkbenchState.BLOCKED, WorkbenchState.ABANDONED},
    WorkbenchState.RUNNING: {WorkbenchState.SUBMITTED, WorkbenchState.BLOCKED, WorkbenchState.ABANDONED, WorkbenchState.READY},
    WorkbenchState.SUBMITTED: {WorkbenchState.INTEGRATED, WorkbenchState.BLOCKED, WorkbenchState.ABANDONED, WorkbenchState.READY},
    WorkbenchState.INTEGRATED: {WorkbenchState.CLOSED},
    WorkbenchState.BLOCKED: {WorkbenchState.READY, WorkbenchState.ABANDONED, WorkbenchState.CLOSED},
    WorkbenchState.ABANDONED: {WorkbenchState.CLOSED},
    WorkbenchState.CLOSED: {WorkbenchState.GC},
    WorkbenchState.GC: set(),
}

_SUBMISSION_TRANSITIONS: dict[WorkspaceSubmissionStatus, set[WorkspaceSubmissionStatus]] = {
    WorkspaceSubmissionStatus.SUBMITTED: {
        WorkspaceSubmissionStatus.ACCEPTED,
        WorkspaceSubmissionStatus.REJECTED,
    },
    WorkspaceSubmissionStatus.ACCEPTED: {
        WorkspaceSubmissionStatus.INTEGRATED,
        WorkspaceSubmissionStatus.REJECTED,
    },
    WorkspaceSubmissionStatus.REJECTED: {
        WorkspaceSubmissionStatus.SUBMITTED,
        WorkspaceSubmissionStatus.ACCEPTED,
    },
    WorkspaceSubmissionStatus.INTEGRATED: set(),
}


@dataclass(slots=True)
class WorkspaceManager:
    project_root: Path
    store: WorkspaceStore

    def __init__(self, *, project_root: Path, store: WorkspaceStore | None = None) -> None:
        root = project_root.expanduser().resolve()
        self.project_root = root
        self.store = store or WorkspaceStore(project_root=root)

    def _append_timeline(
        self,
        *,
        event: str,
        workspace_id: str | None = None,
        workbench_id: str | None = None,
        submission_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "event": event,
            "workspace_id": workspace_id,
            "workbench_id": workbench_id,
            "submission_id": submission_id,
            "payload": dict(payload or {}),
            "created_at": now_ts_ms(),
        }
        try:
            self.store.append_timeline(data)
        except Exception:
            # Timeline append must never break primary workspace flow.
            return

    def _ensure_workspace_transition(
        self,
        *,
        current: IssueWorkspaceState,
        target: IssueWorkspaceState,
        workspace_id: str,
    ) -> None:
        if current is target:
            return
        allowed = _WORKSPACE_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise WorkspaceManagerError(
                f"Invalid workspace state transition for {workspace_id}: {current.value} -> {target.value}"
            )

    def _ensure_workbench_transition(
        self,
        *,
        current: WorkbenchState,
        target: WorkbenchState,
        workbench_id: str,
    ) -> None:
        if current is target:
            return
        allowed = _WORKBENCH_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise WorkspaceManagerError(
                f"Invalid workbench state transition for {workbench_id}: {current.value} -> {target.value}"
            )

    def _ensure_submission_transition(
        self,
        *,
        current: WorkspaceSubmissionStatus,
        target: WorkspaceSubmissionStatus,
        submission_id: str,
    ) -> None:
        if current is target:
            return
        allowed = _SUBMISSION_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise WorkspaceManagerError(
                f"Invalid submission state transition for {submission_id}: {current.value} -> {target.value}"
            )

    def create_or_get_workspace(
        self,
        *,
        issue_ref: WorkspaceIssueRef,
        repo_ref: WorkspaceRepoRef,
        base_branch: str = "main",
        staging_enabled: bool = True,
        merge_policy: WorkspaceMergePolicy = WorkspaceMergePolicy.PR_ONLY,
        push_policy: WorkspacePushPolicy = WorkspacePushPolicy.INTEGRATOR_ONLY,
        workspace_id: str | None = None,
    ) -> IssueWorkspace:
        issue_key = _slug(issue_ref.key)
        ws_id = workspace_id.strip() if isinstance(workspace_id, str) and workspace_id.strip() else f"ws_{issue_key}"
        try:
            return self.store.get_workspace(ws_id)
        except WorkspaceNotFoundError:
            pass

        staging_branch = f"staging/{issue_key}" if staging_enabled else base_branch.strip()
        now = now_ts_ms()
        ws = IssueWorkspace(
            workspace_id=ws_id,
            issue_ref=issue_ref,
            repo_ref=repo_ref,
            base_branch=base_branch.strip() or "main",
            staging_enabled=bool(staging_enabled),
            staging_branch=staging_branch,
            merge_policy=merge_policy,
            push_policy=push_policy,
            state=IssueWorkspaceState.DRAFT,
            created_at=now,
            updated_at=now,
            revision=0,
        )
        saved = self.store.save_workspace(ws, expected_revision=0)
        self._append_timeline(
            event="workspace_created",
            workspace_id=saved.workspace_id,
            payload={
                "issue_key": saved.issue_ref.key,
                "base_branch": saved.base_branch,
                "staging_branch": saved.staging_branch,
                "staging_enabled": saved.staging_enabled,
            },
        )
        return saved

    def get_workspace(self, *, workspace_id: str) -> IssueWorkspace:
        return self.store.get_workspace(workspace_id)

    def transition_workspace_state(
        self,
        *,
        workspace_id: str,
        target_state: IssueWorkspaceState,
        reason: str | None = None,
    ) -> IssueWorkspace:
        current = self.store.get_workspace(workspace_id)
        self._ensure_workspace_transition(current=current.state, target=target_state, workspace_id=current.workspace_id)
        if current.state is target_state:
            return current
        saved = self.store.save_workspace(
            current.model_copy(update={"state": target_state}),
            expected_revision=current.revision,
        )
        self._append_timeline(
            event="workspace_state_changed",
            workspace_id=saved.workspace_id,
            payload={
                "from_state": current.state.value,
                "to_state": target_state.value,
                "reason": reason,
            },
        )
        return saved

    def transition_workbench_state(
        self,
        *,
        workbench_id: str,
        target_state: WorkbenchState,
        reason: str | None = None,
    ) -> Workbench:
        current = self.store.get_workbench(workbench_id)
        self._ensure_workbench_transition(current=current.state, target=target_state, workbench_id=current.workbench_id)
        if current.state is target_state:
            return current
        saved = self.store.save_workbench(
            current.model_copy(update={"state": target_state}),
            expected_revision=current.revision,
        )
        self._append_timeline(
            event="workbench_state_changed",
            workspace_id=saved.workspace_id,
            workbench_id=saved.workbench_id,
            payload={
                "from_state": current.state.value,
                "to_state": target_state.value,
                "reason": reason,
            },
        )
        return saved

    def get_session_binding(self, *, session_id: str) -> SessionWorkspaceBinding | None:
        try:
            return self.store.get_session_binding(session_id)
        except WorkspaceNotFoundError:
            return None

    def bind_session(
        self,
        *,
        session_id: str,
        workspace_id: str,
        workbench_id: str,
        role: WorkbenchRole | None = None,
        agent_id: str | None = None,
        instance_id: str | None = None,
    ) -> SessionWorkspaceBinding:
        now = now_ts_ms()
        existing = self.get_session_binding(session_id=session_id)
        if existing is None:
            binding = SessionWorkspaceBinding(
                session_id=session_id,
                workspace_id=workspace_id,
                workbench_id=workbench_id,
                role=role,
                agent_id=agent_id,
                instance_id=instance_id,
                created_at=now,
                updated_at=now,
                revision=0,
            )
            return self.store.save_session_binding(binding, expected_revision=0)
        next_binding = existing.model_copy(
            update={
                "workspace_id": workspace_id,
                "workbench_id": workbench_id,
                "role": role if role is not None else existing.role,
                "agent_id": agent_id if agent_id is not None else existing.agent_id,
                "instance_id": instance_id if instance_id is not None else existing.instance_id,
            }
        )
        return self.store.save_session_binding(next_binding, expected_revision=existing.revision)

    def clear_session_binding(self, *, session_id: str) -> None:
        self.store.clear_session_binding(session_id)

    def resolve_context(
        self,
        *,
        session_id: str | None = None,
        workspace_id: str | None = None,
        workbench_id: str | None = None,
    ) -> ResolvedWorkspaceContext:
        binding: SessionWorkspaceBinding | None = None
        if session_id:
            binding = self.get_session_binding(session_id=session_id)
        target_workspace_id = workspace_id
        target_workbench_id = workbench_id
        if binding is not None:
            if target_workspace_id is not None and target_workspace_id != binding.workspace_id:
                raise WorkspaceManagerError(
                    "Explicit workspace_id does not match the session-bound workspace context."
                )
            if target_workbench_id is not None and target_workbench_id != binding.workbench_id:
                raise WorkspaceManagerError(
                    "Explicit workbench_id does not match the session-bound workspace context."
                )
            target_workspace_id = binding.workspace_id
            target_workbench_id = binding.workbench_id
        preloaded_workbench: Workbench | None = None
        if target_workspace_id is None and target_workbench_id is not None:
            preloaded_workbench = self.store.get_workbench(target_workbench_id)
            target_workspace_id = preloaded_workbench.workspace_id
        if not target_workspace_id or not target_workbench_id:
            raise WorkspaceManagerError("Workspace context is not bound for this session.")

        workspace = self.store.get_workspace(target_workspace_id)
        workbench = preloaded_workbench or self.store.get_workbench(target_workbench_id)
        if workbench.workspace_id != workspace.workspace_id:
            raise WorkspaceManagerError("Workbench/workspace mismatch.")
        return ResolvedWorkspaceContext(workspace=workspace, workbench=workbench, binding=binding)

    def provision_workbench(
        self,
        *,
        workspace_id: str,
        agent_id: str,
        instance_id: str,
        role: WorkbenchRole = WorkbenchRole.WORKER,
        bind_session_id: str | None = None,
        lease_seconds: int = 900,
    ) -> Workbench:
        workspace = self.store.get_workspace(workspace_id)
        existing = self.store.list_workbenches(workspace_id=workspace_id, agent_id=agent_id, instance_id=instance_id)
        if existing:
            wb = existing[0]
            if bind_session_id:
                self.bind_session(
                    session_id=bind_session_id,
                    workspace_id=workspace.workspace_id,
                    workbench_id=wb.workbench_id,
                    role=wb.role,
                    agent_id=wb.agent_id,
                    instance_id=wb.instance_id,
                )
            return wb

        issue_key = _slug(workspace.issue_ref.key)
        agent_slug = _slug(agent_id)
        seq = len(self.store.list_workbenches(workspace_id=workspace_id, agent_id=agent_id)) + 1
        workbench_id = f"wb_{issue_key}_{agent_slug}_{seq:04d}"
        instance_short = _slug(instance_id)[:8] or "inst"
        branch = f"agent__{issue_key}__{agent_slug}__{instance_short}"
        base_ref = workspace.staging_branch if workspace.staging_enabled else workspace.base_branch
        worktree_rel = f".aura/tmp/worktrees/{workspace.workspace_id}/{workbench_id}"

        now = now_ts_ms()
        wb = Workbench(
            workbench_id=workbench_id,
            workspace_id=workspace.workspace_id,
            agent_id=agent_id,
            instance_id=instance_id,
            role=role,
            worktree_path=worktree_rel,
            branch=branch,
            base_ref=base_ref,
            state=WorkbenchState.PROVISIONING,
            lease_until=now + max(1, int(lease_seconds)) * 1000,
            last_heartbeat_at=now,
            created_at=now,
            updated_at=now,
            revision=0,
        )
        wb = self.store.save_workbench(wb, expected_revision=0)
        self._append_timeline(
            event="workbench_created",
            workspace_id=wb.workspace_id,
            workbench_id=wb.workbench_id,
            payload={
                "agent_id": wb.agent_id,
                "instance_id": wb.instance_id,
                "branch": wb.branch,
                "worktree_path": wb.worktree_path,
            },
        )

        try:
            actual_branch = self._ensure_worktree(
                worktree_rel=worktree_rel,
                branch=branch,
                base_ref=base_ref,
                fallback_ref=workspace.base_branch,
            )
            wb = self.store.save_workbench(
                wb.model_copy(
                    update={
                        "branch": actual_branch,
                        "last_heartbeat_at": now_ts_ms(),
                        "lease_until": now_ts_ms() + max(1, int(lease_seconds)) * 1000,
                    }
                ),
                expected_revision=wb.revision,
            )
            wb = self.transition_workbench_state(
                workbench_id=wb.workbench_id,
                target_state=WorkbenchState.READY,
                reason="Provisioned worktree and branch.",
            )
        except Exception:
            wb = self.store.save_workbench(
                wb.model_copy(update={"last_heartbeat_at": now_ts_ms()}),
                expected_revision=wb.revision,
            )
            wb = self.transition_workbench_state(
                workbench_id=wb.workbench_id,
                target_state=WorkbenchState.BLOCKED,
                reason="Failed to provision worktree/branch.",
            )
            raise

        if workspace.state is IssueWorkspaceState.DRAFT:
            self.transition_workspace_state(
                workspace_id=workspace.workspace_id,
                target_state=IssueWorkspaceState.ACTIVE,
                reason="First workbench provisioned.",
            )

        if bind_session_id:
            self.bind_session(
                session_id=bind_session_id,
                workspace_id=workspace.workspace_id,
                workbench_id=wb.workbench_id,
                role=wb.role,
                agent_id=wb.agent_id,
                instance_id=wb.instance_id,
            )
        return wb

    def heartbeat_workbench(self, *, workbench_id: str, lease_seconds: int = 900) -> Workbench:
        wb = self.store.get_workbench(workbench_id)
        now = now_ts_ms()
        target_state = WorkbenchState.RUNNING if wb.state in {WorkbenchState.READY, WorkbenchState.RUNNING} else wb.state
        if target_state is not wb.state:
            wb = self.transition_workbench_state(
                workbench_id=wb.workbench_id,
                target_state=target_state,
                reason="Heartbeat promoted workbench to running.",
            )
        saved = self.store.save_workbench(
            wb.model_copy(
                update={
                    "last_heartbeat_at": now,
                    "lease_until": now + max(1, int(lease_seconds)) * 1000,
                }
            ),
            expected_revision=wb.revision,
        )
        self._append_timeline(
            event="workbench_heartbeat",
            workspace_id=saved.workspace_id,
            workbench_id=saved.workbench_id,
            payload={"lease_until": saved.lease_until},
        )
        return saved

    def publish_heartbeat(
        self,
        *,
        workspace_id: str,
        summary: str,
        topic: str | None = None,
        issuer_agent_id: str | None = None,
        issuer_instance_id: str | None = None,
        claim_ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
        heartbeat_id: str | None = None,
    ) -> dict[str, Any]:
        workspace = self.store.get_workspace(workspace_id)
        summary_clean = str(summary or "").strip()
        if not summary_clean:
            raise WorkspaceManagerError("summary must be a non-empty string.")
        topic_clean = str(topic or "").strip() or None
        now = now_ts_ms()
        ttl = int(claim_ttl_seconds) if isinstance(claim_ttl_seconds, int) and claim_ttl_seconds > 0 else None
        expires_at = now + (ttl * 1000) if ttl is not None else None

        payload: dict[str, Any] = {
            "heartbeat_id": (str(heartbeat_id).strip() if isinstance(heartbeat_id, str) and heartbeat_id.strip() else new_id("hb")),
            "workspace_id": workspace.workspace_id,
            "issue_key": workspace.issue_ref.key,
            "topic": topic_clean,
            "summary": summary_clean,
            "status": "open",
            "issuer_agent_id": (str(issuer_agent_id).strip() if isinstance(issuer_agent_id, str) else None),
            "issuer_instance_id": (str(issuer_instance_id).strip() if isinstance(issuer_instance_id, str) else None),
            "claim_ttl_seconds": ttl,
            "expires_at": expires_at,
            "metadata": dict(metadata or {}),
            "created_at": now,
            "updated_at": now,
        }
        saved = self.store.append_heartbeat(payload)
        self._append_timeline(
            event="heartbeat_published",
            workspace_id=workspace.workspace_id,
            payload={
                "heartbeat_id": saved.get("heartbeat_id"),
                "topic": saved.get("topic"),
                "summary": saved.get("summary"),
                "expires_at": saved.get("expires_at"),
            },
        )
        return saved

    def submit_claim(
        self,
        *,
        heartbeat_id: str,
        agent_id: str,
        proposal: str,
        instance_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        claim_id: str | None = None,
    ) -> dict[str, Any]:
        hb_id = str(heartbeat_id or "").strip()
        if not hb_id:
            raise WorkspaceManagerError("heartbeat_id must be a non-empty string.")
        agent = str(agent_id or "").strip()
        if not agent:
            raise WorkspaceManagerError("agent_id must be a non-empty string.")
        proposal_clean = str(proposal or "").strip()
        if not proposal_clean:
            raise WorkspaceManagerError("proposal must be a non-empty string.")

        heartbeat = self.store.get_heartbeat(hb_id)
        now = now_ts_ms()
        status = str(heartbeat.get("status") or "").strip().lower() or "open"
        if status != "open":
            raise WorkspaceManagerError(f"Heartbeat {hb_id} is not open (status={status}).")
        expires_at = heartbeat.get("expires_at")
        if isinstance(expires_at, int) and expires_at > 0 and now > expires_at:
            heartbeat = self.store.update_heartbeat(hb_id, updates={"status": "expired"})
            self._append_timeline(
                event="heartbeat_expired",
                workspace_id=str(heartbeat.get("workspace_id") or ""),
                payload={"heartbeat_id": hb_id, "expires_at": expires_at},
            )
            raise WorkspaceManagerError(f"Heartbeat {hb_id} is expired.")

        claim_payload: dict[str, Any] = {
            "claim_id": (str(claim_id).strip() if isinstance(claim_id, str) and claim_id.strip() else new_id("claim")),
            "workspace_id": str(heartbeat.get("workspace_id") or "").strip(),
            "heartbeat_id": hb_id,
            "issue_key": str(heartbeat.get("issue_key") or "").strip() or None,
            "agent_id": agent,
            "instance_id": (str(instance_id).strip() if isinstance(instance_id, str) and instance_id.strip() else None),
            "proposal": proposal_clean,
            "status": "submitted",
            "metadata": dict(metadata or {}),
            "created_at": now,
            "updated_at": now,
        }
        saved = self.store.append_claim(claim_payload)
        self._append_timeline(
            event="claim_submitted",
            workspace_id=str(saved.get("workspace_id") or ""),
            payload={
                "claim_id": saved.get("claim_id"),
                "heartbeat_id": saved.get("heartbeat_id"),
                "agent_id": saved.get("agent_id"),
            },
        )
        return saved

    def award_claim(
        self,
        *,
        claim_id: str,
        committee_agent_id: str | None = None,
        notes: str | None = None,
        auto_reject_other_claims: bool = True,
        award_id: str | None = None,
    ) -> dict[str, Any]:
        claim_key = str(claim_id or "").strip()
        if not claim_key:
            raise WorkspaceManagerError("claim_id must be a non-empty string.")

        claim = self.store.get_claim(claim_key)
        existing_awards = self.store.list_awards(claim_id=claim_key, limit=1)
        if existing_awards:
            heartbeat = self.store.get_heartbeat(str(claim.get("heartbeat_id") or ""))
            return {"award": existing_awards[0], "claim": claim, "heartbeat": heartbeat, "rejected_claim_ids": []}

        now = now_ts_ms()
        claim_status = str(claim.get("status") or "").strip().lower() or "submitted"
        if claim_status in {"rejected", "withdrawn"}:
            raise WorkspaceManagerError(f"Claim {claim_key} cannot be awarded (status={claim_status}).")

        heartbeat_id = str(claim.get("heartbeat_id") or "").strip()
        if not heartbeat_id:
            raise WorkspaceManagerError(f"Claim {claim_key} is missing heartbeat_id.")
        heartbeat = self.store.get_heartbeat(heartbeat_id)

        hb_status = str(heartbeat.get("status") or "").strip().lower() or "open"
        if hb_status not in {"open", "awarded"}:
            raise WorkspaceManagerError(f"Heartbeat {heartbeat_id} cannot accept awards (status={hb_status}).")

        expires_at = heartbeat.get("expires_at")
        if isinstance(expires_at, int) and expires_at > 0 and now > expires_at and hb_status != "awarded":
            heartbeat = self.store.update_heartbeat(heartbeat_id, updates={"status": "expired"})
            self._append_timeline(
                event="heartbeat_expired",
                workspace_id=str(heartbeat.get("workspace_id") or ""),
                payload={"heartbeat_id": heartbeat_id, "expires_at": expires_at},
            )
            raise WorkspaceManagerError(f"Heartbeat {heartbeat_id} is expired.")

        claim = self.store.update_claim(claim_key, updates={"status": "awarded"})
        heartbeat = self.store.update_heartbeat(
            heartbeat_id,
            updates={
                "status": "awarded",
                "awarded_claim_id": claim_key,
                "awarded_agent_id": str(claim.get("agent_id") or "").strip() or None,
            },
        )

        rejected_claim_ids: list[str] = []
        if auto_reject_other_claims:
            peers = self.store.list_claims(heartbeat_id=heartbeat_id)
            for peer in peers:
                peer_id = str(peer.get("claim_id") or "").strip()
                if not peer_id or peer_id == claim_key:
                    continue
                peer_status = str(peer.get("status") or "").strip().lower()
                if peer_status not in {"submitted", "shortlisted"}:
                    continue
                self.store.update_claim(peer_id, updates={"status": "rejected"})
                rejected_claim_ids.append(peer_id)
                self._append_timeline(
                    event="claim_rejected",
                    workspace_id=str(peer.get("workspace_id") or ""),
                    payload={
                        "claim_id": peer_id,
                        "heartbeat_id": heartbeat_id,
                        "reason": f"Awarded claim {claim_key}.",
                    },
                )

        award_payload: dict[str, Any] = {
            "award_id": (str(award_id).strip() if isinstance(award_id, str) and award_id.strip() else new_id("award")),
            "workspace_id": str(claim.get("workspace_id") or "").strip(),
            "heartbeat_id": heartbeat_id,
            "claim_id": claim_key,
            "agent_id": str(claim.get("agent_id") or "").strip(),
            "instance_id": str(claim.get("instance_id") or "").strip() or None,
            "committee_agent_id": (str(committee_agent_id).strip() if isinstance(committee_agent_id, str) and committee_agent_id.strip() else None),
            "notes": (str(notes).strip() if isinstance(notes, str) and notes.strip() else None),
            "created_at": now,
            "updated_at": now,
        }
        award = self.store.append_award(award_payload)
        self._append_timeline(
            event="claim_awarded",
            workspace_id=str(award.get("workspace_id") or ""),
            payload={
                "award_id": award.get("award_id"),
                "heartbeat_id": award.get("heartbeat_id"),
                "claim_id": award.get("claim_id"),
                "agent_id": award.get("agent_id"),
                "auto_reject_other_claims": bool(auto_reject_other_claims),
                "rejected_claim_ids": rejected_claim_ids,
            },
        )
        return {"award": award, "claim": claim, "heartbeat": heartbeat, "rejected_claim_ids": rejected_claim_ids}

    def wake_awarded_agent(
        self,
        *,
        award_id: str,
        bind_session_id: str | None = None,
        lease_seconds: int = 900,
        role: WorkbenchRole = WorkbenchRole.WORKER,
    ) -> dict[str, Any]:
        award_key = str(award_id or "").strip()
        if not award_key:
            raise WorkspaceManagerError("award_id must be a non-empty string.")
        award = self.store.get_award(award_key)

        claim_id = str(award.get("claim_id") or "").strip()
        if not claim_id:
            raise WorkspaceManagerError(f"Award {award_key} is missing claim_id.")
        claim = self.store.get_claim(claim_id)
        heartbeat_id = str(award.get("heartbeat_id") or "").strip()
        if not heartbeat_id:
            raise WorkspaceManagerError(f"Award {award_key} is missing heartbeat_id.")
        heartbeat = self.store.get_heartbeat(heartbeat_id)

        workspace_id = str(award.get("workspace_id") or claim.get("workspace_id") or heartbeat.get("workspace_id") or "").strip()
        if not workspace_id:
            raise WorkspaceManagerError(f"Award {award_key} is missing workspace_id.")
        workspace = self.store.get_workspace(workspace_id)

        agent_id = str(claim.get("agent_id") or "").strip()
        if not agent_id:
            raise WorkspaceManagerError(f"Claim {claim_id} is missing agent_id.")
        instance_id = str(claim.get("instance_id") or "").strip() or new_id("inst")

        wb = self.provision_workbench(
            workspace_id=workspace.workspace_id,
            agent_id=agent_id,
            instance_id=instance_id,
            role=role,
            bind_session_id=(str(bind_session_id).strip() if isinstance(bind_session_id, str) and bind_session_id.strip() else None),
            lease_seconds=lease_seconds,
        )

        wake_envelope = {
            "workspace_id": workspace.workspace_id,
            "issue_ref": workspace.issue_ref.model_dump(mode="json"),
            "repo_ref": workspace.repo_ref.model_dump(mode="json"),
            "workbench_id": wb.workbench_id,
            "branch": wb.branch,
            "worktree_path": wb.worktree_path,
            "agent_id": wb.agent_id,
            "instance_id": wb.instance_id,
            "claim_id": claim_id,
            "claim_proposal": claim.get("proposal"),
            "award_id": award_key,
            "merge_policy": workspace.merge_policy.value,
            "push_policy": workspace.push_policy.value,
        }
        self._append_timeline(
            event="awarded_agent_woken",
            workspace_id=workspace.workspace_id,
            workbench_id=wb.workbench_id,
            payload={
                "award_id": award_key,
                "claim_id": claim_id,
                "heartbeat_id": heartbeat_id,
                "agent_id": wb.agent_id,
                "instance_id": wb.instance_id,
            },
        )
        return {
            "award": award,
            "claim": claim,
            "heartbeat": heartbeat,
            "workspace": workspace,
            "workbench": wb,
            "wake_envelope": wake_envelope,
        }

    def register_submission(
        self,
        *,
        commit_sha: str,
        changed_files: list[str],
        tool_call_ids: list[str],
        session_id: str | None = None,
        workspace_id: str | None = None,
        workbench_id: str | None = None,
        instance_id: str | None = None,
        agent_id: str | None = None,
        branch: str | None = None,
        status: WorkspaceSubmissionStatus = WorkspaceSubmissionStatus.SUBMITTED,
        auto_transition: bool = False,
        submission_id: str | None = None,
        pr_url: str | None = None,
        ci_url: str | None = None,
        notes: str | None = None,
    ) -> WorkspaceSubmission:
        resolved = self.resolve_context(session_id=session_id, workspace_id=workspace_id, workbench_id=workbench_id)
        wb = resolved.workbench
        sub = WorkspaceSubmission(
            submission_id=submission_id.strip() if isinstance(submission_id, str) and submission_id.strip() else new_id("subm"),
            workspace_id=resolved.workspace.workspace_id,
            workbench_id=wb.workbench_id,
            instance_id=(instance_id or wb.instance_id).strip(),
            agent_id=(agent_id or wb.agent_id).strip(),
            branch=(branch or wb.branch).strip(),
            commit_sha=commit_sha,
            changed_files=changed_files,
            tool_call_ids=tool_call_ids,
            status=status,
            pr_url=pr_url,
            ci_url=ci_url,
            notes=notes,
            created_at=now_ts_ms(),
        )
        saved = self.store.append_submission(sub)
        self._append_timeline(
            event="submission_registered",
            workspace_id=saved.workspace_id,
            workbench_id=saved.workbench_id,
            submission_id=saved.submission_id,
            payload={
                "status": saved.status.value,
                "commit_sha": saved.commit_sha,
                "branch": saved.branch,
                "auto_transition": bool(auto_transition),
            },
        )

        if auto_transition:
            wb = self.store.get_workbench(saved.workbench_id)
            ws = self.store.get_workspace(saved.workspace_id)
            if status in {WorkspaceSubmissionStatus.SUBMITTED, WorkspaceSubmissionStatus.ACCEPTED}:
                if wb.state is not WorkbenchState.SUBMITTED:
                    wb = self.transition_workbench_state(
                        workbench_id=wb.workbench_id,
                        target_state=WorkbenchState.SUBMITTED,
                        reason=f"Submission {saved.submission_id} registered.",
                    )
                if ws.state is IssueWorkspaceState.ACTIVE:
                    ws = self.transition_workspace_state(
                        workspace_id=ws.workspace_id,
                        target_state=IssueWorkspaceState.INTEGRATING,
                        reason=f"Submission {saved.submission_id} entered integration flow.",
                    )
            elif status is WorkspaceSubmissionStatus.INTEGRATED:
                if wb.state is not WorkbenchState.SUBMITTED:
                    wb = self.transition_workbench_state(
                        workbench_id=wb.workbench_id,
                        target_state=WorkbenchState.SUBMITTED,
                        reason=f"Submission {saved.submission_id} preparing integrate transition.",
                    )
                wb = self.transition_workbench_state(
                    workbench_id=wb.workbench_id,
                    target_state=WorkbenchState.INTEGRATED,
                    reason=f"Submission {saved.submission_id} integrated.",
                )
                if ws.state is IssueWorkspaceState.ACTIVE:
                    self.transition_workspace_state(
                        workspace_id=ws.workspace_id,
                        target_state=IssueWorkspaceState.INTEGRATING,
                        reason=f"Submission {saved.submission_id} integrated.",
                    )
            elif status is WorkspaceSubmissionStatus.REJECTED:
                if wb.state in {WorkbenchState.SUBMITTED, WorkbenchState.RUNNING}:
                    self.transition_workbench_state(
                        workbench_id=wb.workbench_id,
                        target_state=WorkbenchState.BLOCKED,
                        reason=f"Submission {saved.submission_id} rejected.",
                    )
                if ws.state is IssueWorkspaceState.INTEGRATING:
                    self.transition_workspace_state(
                        workspace_id=ws.workspace_id,
                        target_state=IssueWorkspaceState.ACTIVE,
                        reason=f"Submission {saved.submission_id} rejected.",
                    )
        return saved

    def accept_submission(
        self,
        *,
        submission_id: str,
        target_status: WorkspaceSubmissionStatus,
        tool_call_ids: list[str] | None = None,
        auto_transition: bool = False,
        pr_url: str | None = None,
        ci_url: str | None = None,
        notes: str | None = None,
    ) -> WorkspaceSubmission:
        if target_status not in {
            WorkspaceSubmissionStatus.ACCEPTED,
            WorkspaceSubmissionStatus.REJECTED,
            WorkspaceSubmissionStatus.INTEGRATED,
        }:
            raise WorkspaceManagerError(f"Unsupported target submission status: {target_status.value}")

        current = self.store.get_submission(submission_id)
        self._ensure_submission_transition(
            current=current.status,
            target=target_status,
            submission_id=current.submission_id,
        )
        added_tool_call_ids: list[str] = []
        if tool_call_ids:
            existing = set(current.tool_call_ids)
            for raw in tool_call_ids:
                item = str(raw or "").strip()
                if not item or item in existing:
                    continue
                existing.add(item)
                added_tool_call_ids.append(item)
        updated = self.store.update_submission(
            submission_id=current.submission_id,
            status=target_status,
            tool_call_ids=tool_call_ids,
            pr_url=pr_url,
            ci_url=ci_url,
            notes=notes,
        )
        self._append_timeline(
            event="submission_status_changed",
            workspace_id=updated.workspace_id,
            workbench_id=updated.workbench_id,
            submission_id=updated.submission_id,
            payload={
                "from_status": current.status.value,
                "to_status": target_status.value,
                "pr_url": updated.pr_url,
                "ci_url": updated.ci_url,
                "tool_call_ids_added": added_tool_call_ids,
                "auto_transition": bool(auto_transition),
            },
        )

        if auto_transition:
            wb = self.store.get_workbench(updated.workbench_id)
            ws = self.store.get_workspace(updated.workspace_id)
            if target_status is WorkspaceSubmissionStatus.ACCEPTED:
                if wb.state is not WorkbenchState.SUBMITTED:
                    self.transition_workbench_state(
                        workbench_id=wb.workbench_id,
                        target_state=WorkbenchState.SUBMITTED,
                        reason=f"Submission {updated.submission_id} accepted.",
                    )
                if ws.state is IssueWorkspaceState.ACTIVE:
                    self.transition_workspace_state(
                        workspace_id=ws.workspace_id,
                        target_state=IssueWorkspaceState.INTEGRATING,
                        reason=f"Submission {updated.submission_id} accepted.",
                    )
            elif target_status is WorkspaceSubmissionStatus.REJECTED:
                if wb.state in {WorkbenchState.SUBMITTED, WorkbenchState.RUNNING}:
                    self.transition_workbench_state(
                        workbench_id=wb.workbench_id,
                        target_state=WorkbenchState.BLOCKED,
                        reason=f"Submission {updated.submission_id} rejected.",
                    )
                if ws.state is IssueWorkspaceState.INTEGRATING:
                    self.transition_workspace_state(
                        workspace_id=ws.workspace_id,
                        target_state=IssueWorkspaceState.ACTIVE,
                        reason=f"Submission {updated.submission_id} rejected.",
                    )
            elif target_status is WorkspaceSubmissionStatus.INTEGRATED:
                if wb.state is not WorkbenchState.SUBMITTED:
                    wb = self.transition_workbench_state(
                        workbench_id=wb.workbench_id,
                        target_state=WorkbenchState.SUBMITTED,
                        reason=f"Submission {updated.submission_id} integration finalized.",
                    )
                self.transition_workbench_state(
                    workbench_id=wb.workbench_id,
                    target_state=WorkbenchState.INTEGRATED,
                    reason=f"Submission {updated.submission_id} integration finalized.",
                )
                if ws.state is IssueWorkspaceState.ACTIVE:
                    self.transition_workspace_state(
                        workspace_id=ws.workspace_id,
                        target_state=IssueWorkspaceState.INTEGRATING,
                        reason=f"Submission {updated.submission_id} integration finalized.",
                )
        return updated

    def append_submission_evidence(
        self,
        *,
        submission_id: str,
        tool_call_ids: list[str] | None = None,
        pr_url: str | None = None,
        ci_url: str | None = None,
        notes: str | None = None,
    ) -> WorkspaceSubmission:
        if not tool_call_ids and pr_url is None and ci_url is None and notes is None:
            raise WorkspaceManagerError(
                "At least one evidence field is required (tool_call_ids, pr_url, ci_url, notes)."
            )
        current = self.store.get_submission(submission_id)

        added_tool_call_ids: list[str] = []
        if tool_call_ids:
            existing = set(current.tool_call_ids)
            for raw in tool_call_ids:
                item = str(raw or "").strip()
                if not item or item in existing:
                    continue
                existing.add(item)
                added_tool_call_ids.append(item)

        updated = self.store.update_submission(
            submission_id=current.submission_id,
            tool_call_ids=tool_call_ids,
            pr_url=pr_url,
            ci_url=ci_url,
            notes=notes,
        )
        self._append_timeline(
            event="submission_evidence_appended",
            workspace_id=updated.workspace_id,
            workbench_id=updated.workbench_id,
            submission_id=updated.submission_id,
            payload={
                "tool_call_ids_added": added_tool_call_ids,
                "pr_url": updated.pr_url,
                "ci_url": updated.ci_url,
            },
        )
        return updated

    def list_submissions(
        self,
        *,
        workspace_id: str | None = None,
        workbench_id: str | None = None,
        status: WorkspaceSubmissionStatus | None = None,
    ) -> list[WorkspaceSubmission]:
        return self.store.list_submissions(
            workspace_id=workspace_id,
            workbench_id=workbench_id,
            status=status,
        )

    def list_workbenches(
        self,
        *,
        workspace_id: str | None = None,
        state: WorkbenchState | None = None,
        agent_id: str | None = None,
        instance_id: str | None = None,
    ) -> list[Workbench]:
        return self.store.list_workbenches(
            workspace_id=workspace_id,
            state=state,
            agent_id=agent_id,
            instance_id=instance_id,
        )

    def list_heartbeats(
        self,
        *,
        workspace_id: str | None = None,
        issue_key: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.store.list_heartbeats(
            workspace_id=workspace_id,
            issue_key=issue_key,
            status=(str(status).strip().lower() if isinstance(status, str) and status.strip() else None),
            limit=limit,
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
        return self.store.list_claims(
            workspace_id=workspace_id,
            heartbeat_id=heartbeat_id,
            agent_id=agent_id,
            status=(str(status).strip().lower() if isinstance(status, str) and status.strip() else None),
            limit=limit,
        )

    def list_awards(
        self,
        *,
        workspace_id: str | None = None,
        heartbeat_id: str | None = None,
        claim_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.store.list_awards(
            workspace_id=workspace_id,
            heartbeat_id=heartbeat_id,
            claim_id=claim_id,
            agent_id=agent_id,
            limit=limit,
        )

    def list_timeline(
        self,
        *,
        workspace_id: str | None = None,
        workbench_id: str | None = None,
        submission_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.store.list_timeline(
            workspace_id=workspace_id,
            workbench_id=workbench_id,
            submission_id=submission_id,
            limit=limit,
        )

    def audit_chain(
        self,
        *,
        workspace_id: str,
        include_timeline: bool = True,
        timeline_limit: int | None = 500,
    ) -> dict[str, Any]:
        workspace = self.store.get_workspace(workspace_id)
        workbenches = self.store.list_workbenches(workspace_id=workspace.workspace_id)
        submissions = self.store.list_submissions(workspace_id=workspace.workspace_id)
        timeline: list[dict[str, Any]] = []
        if include_timeline:
            timeline = self.store.list_timeline(workspace_id=workspace.workspace_id, limit=timeline_limit)

        submission_ids_by_workbench: dict[str, list[str]] = {}
        for sub in submissions:
            bucket = submission_ids_by_workbench.setdefault(sub.workbench_id, [])
            if sub.submission_id not in bucket:
                bucket.append(sub.submission_id)

        return {
            "workspace": workspace.model_dump(mode="json"),
            "workbenches": [wb.model_dump(mode="json") for wb in workbenches],
            "submissions": [sub.model_dump(mode="json") for sub in submissions],
            "timeline": timeline,
            "summary": {
                "workspace_state": workspace.state.value,
                "workbench_count": len(workbenches),
                "submission_count": len(submissions),
                "timeline_count": len(timeline),
                "submission_ids_by_workbench": submission_ids_by_workbench,
            },
        }

    def recover_expired_workbenches(
        self,
        *,
        workspace_id: str | None = None,
        target_state: WorkbenchState = WorkbenchState.READY,
        now_ms: int | None = None,
        limit: int | None = None,
        reason: str | None = None,
    ) -> list[Workbench]:
        if target_state not in {WorkbenchState.READY, WorkbenchState.BLOCKED}:
            raise WorkspaceManagerError("target_state must be one of: ready, blocked.")

        now = int(now_ms) if isinstance(now_ms, int) and now_ms >= 0 else now_ts_ms()
        recovered: list[Workbench] = []
        running = self.store.list_workbenches(workspace_id=workspace_id, state=WorkbenchState.RUNNING)
        running.sort(key=lambda item: int(item.lease_until or 0))

        for wb in running:
            lease_until = int(wb.lease_until or 0)
            if lease_until > now:
                continue
            moved = self.transition_workbench_state(
                workbench_id=wb.workbench_id,
                target_state=target_state,
                reason=reason or f"Lease expired at {lease_until}.",
            )
            for binding in self.store.list_session_bindings(workbench_id=moved.workbench_id):
                self.store.clear_session_binding(binding.session_id)
                self._append_timeline(
                    event="workbench_session_unbound",
                    workspace_id=moved.workspace_id,
                    workbench_id=moved.workbench_id,
                    payload={"session_id": binding.session_id},
                )
            self._append_timeline(
                event="workbench_recovered_from_lease_expiry",
                workspace_id=moved.workspace_id,
                workbench_id=moved.workbench_id,
                payload={
                    "lease_until": lease_until,
                    "target_state": target_state.value,
                },
            )
            recovered.append(moved)
            if isinstance(limit, int) and limit > 0 and len(recovered) >= limit:
                break
        return recovered

    def close_workbench(
        self,
        *,
        workbench_id: str,
        reason: str | None = None,
    ) -> Workbench:
        wb = self.store.get_workbench(workbench_id)
        if wb.state is WorkbenchState.GC:
            raise WorkspaceManagerError(f"Cannot close gc workbench: {workbench_id}")
        if wb.state is WorkbenchState.CLOSED:
            return wb

        close_reason = reason or "Workbench closed."
        if wb.state is WorkbenchState.INTEGRATED:
            wb = self.transition_workbench_state(
                workbench_id=wb.workbench_id,
                target_state=WorkbenchState.CLOSED,
                reason=close_reason,
            )
        else:
            if wb.state is not WorkbenchState.ABANDONED:
                wb = self.transition_workbench_state(
                    workbench_id=wb.workbench_id,
                    target_state=WorkbenchState.ABANDONED,
                    reason=close_reason,
                )
            wb = self.transition_workbench_state(
                workbench_id=wb.workbench_id,
                target_state=WorkbenchState.CLOSED,
                reason=close_reason,
            )

        for binding in self.store.list_session_bindings(workbench_id=wb.workbench_id):
            self.store.clear_session_binding(binding.session_id)
            self._append_timeline(
                event="workbench_session_unbound",
                workspace_id=wb.workspace_id,
                workbench_id=wb.workbench_id,
                payload={"session_id": binding.session_id},
            )
        self._append_timeline(
            event="workbench_closed",
            workspace_id=wb.workspace_id,
            workbench_id=wb.workbench_id,
            payload={"reason": close_reason},
        )
        return wb

    def close_workspace(
        self,
        *,
        workspace_id: str,
        final_state: IssueWorkspaceState = IssueWorkspaceState.DONE,
        archive: bool = False,
        close_workbenches: bool = False,
        reason: str | None = None,
    ) -> IssueWorkspace:
        if final_state not in {IssueWorkspaceState.DONE, IssueWorkspaceState.BLOCKED}:
            raise WorkspaceManagerError("final_state must be one of: done, blocked.")

        ws = self.store.get_workspace(workspace_id)
        close_reason = reason or "Workspace closed."
        workbenches = self.store.list_workbenches(workspace_id=ws.workspace_id)

        if close_workbenches:
            for wb in workbenches:
                if wb.state in {WorkbenchState.CLOSED, WorkbenchState.GC}:
                    continue
                self.close_workbench(workbench_id=wb.workbench_id, reason=close_reason)
        else:
            remaining = [wb.workbench_id for wb in workbenches if wb.state not in {WorkbenchState.CLOSED, WorkbenchState.GC}]
            if remaining:
                raise WorkspaceManagerError(
                    f"Cannot close workspace {ws.workspace_id} with active workbenches: {', '.join(remaining)}. "
                    "Set close_workbenches=true to close them automatically."
                )

        if ws.state is IssueWorkspaceState.DRAFT:
            ws = self.transition_workspace_state(
                workspace_id=ws.workspace_id,
                target_state=IssueWorkspaceState.ACTIVE,
                reason="Workspace closure bootstrap from draft.",
            )

        if ws.state is not final_state:
            ws = self.transition_workspace_state(
                workspace_id=ws.workspace_id,
                target_state=final_state,
                reason=close_reason,
            )
        if archive and ws.state is not IssueWorkspaceState.ARCHIVED:
            ws = self.transition_workspace_state(
                workspace_id=ws.workspace_id,
                target_state=IssueWorkspaceState.ARCHIVED,
                reason=close_reason,
            )
        self._append_timeline(
            event="workspace_closed",
            workspace_id=ws.workspace_id,
            payload={
                "final_state": ws.state.value,
                "archived": bool(archive and ws.state is IssueWorkspaceState.ARCHIVED),
                "close_workbenches": bool(close_workbenches),
                "reason": close_reason,
            },
        )
        return ws

    def gc_workbench(
        self,
        *,
        workbench_id: str,
        delete_worktree: bool = True,
        reason: str | None = None,
    ) -> Workbench:
        wb = self.store.get_workbench(workbench_id)
        if wb.state not in {WorkbenchState.CLOSED, WorkbenchState.GC}:
            raise WorkspaceManagerError(
                f"Workbench {workbench_id} must be closed before gc (current={wb.state.value})."
            )
        if wb.state is WorkbenchState.CLOSED:
            wb = self.transition_workbench_state(
                workbench_id=wb.workbench_id,
                target_state=WorkbenchState.GC,
                reason=reason or "Workbench gc requested.",
            )
        removed = False
        if delete_worktree:
            removed = self._delete_worktree_path(wb.worktree_path)
        self._append_timeline(
            event="workbench_gc",
            workspace_id=wb.workspace_id,
            workbench_id=wb.workbench_id,
            payload={
                "delete_worktree": bool(delete_worktree),
                "worktree_removed": removed,
                "reason": reason,
            },
        )
        return wb

    def _delete_worktree_path(self, worktree_rel: str) -> bool:
        rel = str(worktree_rel or "").strip()
        if not rel:
            return False
        rel_path = Path(rel)
        if rel_path.is_absolute():
            raise WorkspaceManagerError("worktree_path must be project-relative for gc.")

        allowed_root = (self.project_root / ".aura" / "tmp" / "worktrees").resolve()
        target = (self.project_root / rel_path).resolve()
        if target == allowed_root or allowed_root not in target.parents:
            raise WorkspaceManagerError("Refusing to delete path outside .aura/tmp/worktrees.")
        if not target.exists():
            return False
        shutil.rmtree(target)

        # Best-effort cleanup of empty parent dirs under the allowed root.
        parent = target.parent
        while parent != allowed_root and parent.exists():
            try:
                if any(parent.iterdir()):
                    break
                parent.rmdir()
                parent = parent.parent
            except Exception:
                break
        return True

    def _ensure_worktree(self, *, worktree_rel: str, branch: str, base_ref: str, fallback_ref: str) -> str:
        worktree_abs = (self.project_root / worktree_rel).resolve()
        worktree_abs.parent.mkdir(parents=True, exist_ok=True)
        if (worktree_abs / ".git").exists():
            return branch
        if not self._is_git_repo():
            worktree_abs.mkdir(parents=True, exist_ok=True)
            return branch

        if worktree_abs.exists() and any(worktree_abs.iterdir()):
            raise WorkspaceGitError(f"Refusing to attach worktree to non-empty path: {worktree_rel}")

        selected_ref = base_ref if self._git_ref_exists(base_ref) else (fallback_ref if self._git_ref_exists(fallback_ref) else "HEAD")
        candidates = [branch]
        candidates.append(f"wb__{_slug(branch)}")

        last_error: Exception | None = None
        for cand in candidates:
            try:
                if self._git_branch_exists(cand):
                    self._git(["worktree", "add", "--force", str(worktree_abs), cand])
                else:
                    self._git(["worktree", "add", "--force", "-b", cand, str(worktree_abs), selected_ref])
                return cand
            except Exception as e:
                last_error = e
                try:
                    if worktree_abs.exists() and not any(worktree_abs.iterdir()):
                        worktree_abs.rmdir()
                except Exception:
                    pass
                continue
        assert last_error is not None
        raise last_error

    def _is_git_repo(self) -> bool:
        return self._git(["rev-parse", "--is-inside-work-tree"], check=False).returncode == 0

    def _git_branch_exists(self, branch: str) -> bool:
        return self._git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], check=False).returncode == 0

    def _git_ref_exists(self, ref: str) -> bool:
        return self._git(["rev-parse", "--verify", "--quiet", ref], check=False).returncode == 0

    def _git(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.project_root,
            text=True,
            capture_output=True,
        )
        if check and proc.returncode != 0:
            cmd = " ".join(["git", *args])
            raise WorkspaceGitError(f"Command failed: {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        return proc


__all__ = [
    "ResolvedWorkspaceContext",
    "WorkspaceGitError",
    "WorkspaceManager",
    "WorkspaceManagerError",
]
