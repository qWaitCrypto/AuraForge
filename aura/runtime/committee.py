from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .bidding import BiddingConfig, BiddingService
from .ids import new_id, now_ts_ms
from .models.bidding import BiddingPhase
from .models.notification import NotificationType
from .models.sandbox import Sandbox
from .models.signal import Signal, SignalType
from .notifications import NotificationStore
from .sandbox import SandboxManager
from .signal import SignalBus

COMMITTEE_AGENT_ID = "committee"
PROJECT_REQUEST_TYPE = "project_request"
COMMITTEE_DECOMPOSITION_MODE = "coordinator_mvp"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = _clean_text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _slug_token(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", str(raw or "").strip()).strip("-")
    if not cleaned:
        return "TASK"
    return cleaned.upper()[:24]


def is_project_request_signal(signal: Signal) -> bool:
    if signal.signal_type is not SignalType.WAKE:
        return False
    payload = signal.payload if isinstance(signal.payload, dict) else {}
    return _clean_text(payload.get("type")).lower() == PROJECT_REQUEST_TYPE


class CommitteeTask(BaseModel):
    issue_key: str
    title: str
    description: str
    required_capabilities: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    candidate_agents: list[str] = Field(default_factory=list)
    issue_key_is_placeholder: bool = False
    issue_key_source: str = "payload"


class CommitteeRequestStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"


class CommitteeRequest(BaseModel):
    request_id: str
    source_signal_id: str | None = None
    from_agent: str | None = None
    goal: str
    context: str
    constraints: list[str] = Field(default_factory=list)
    priority: str = "medium"
    references: list[str] = Field(default_factory=list)
    status: CommitteeRequestStatus = CommitteeRequestStatus.PENDING
    tasks: list[CommitteeTask] = Field(default_factory=list)
    wake_signal_ids: list[str] = Field(default_factory=list)
    created_at: int = Field(default_factory=now_ts_ms)
    updated_at: int = Field(default_factory=now_ts_ms)


@dataclass(slots=True)
class CommitteeStore:
    project_root: Path
    _root: Path = field(init=False, repr=False)
    _requests_path: Path = field(init=False, repr=False)
    _activity_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root = self.project_root.expanduser().resolve() / ".aura" / "state" / "committee"
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._requests_path = root / "pending_requests.jsonl"
        self._activity_path = root / "activity_log.jsonl"

    def append_request(self, request: CommitteeRequest) -> None:
        with self._requests_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")

    def upsert_request(self, request: CommitteeRequest) -> None:
        rows = self.list_requests(limit=0)
        replaced = False
        for idx, row in enumerate(rows):
            if row.request_id == request.request_id:
                rows[idx] = request
                replaced = True
                break
        if not replaced:
            rows.append(request)
        rows.sort(key=lambda item: item.created_at)
        tmp = self._requests_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(self._requests_path)

    def list_requests(
        self,
        *,
        limit: int = 100,
        status: CommitteeRequestStatus | str | None = None,
    ) -> list[CommitteeRequest]:
        if not self._requests_path.exists():
            return []
        out: list[CommitteeRequest] = []
        if isinstance(status, CommitteeRequestStatus):
            target_status = status.value
        elif isinstance(status, str):
            target_status = _clean_text(status).lower()
        else:
            target_status = ""
        for line in self._requests_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = CommitteeRequest.model_validate_json(line)
            except Exception:
                continue
            if target_status and item.status.value != target_status:
                continue
            out.append(item)
        out.sort(key=lambda item: item.created_at)
        if limit > 0 and len(out) > limit:
            return out[-limit:]
        return out

    def get_request(self, request_id: str) -> CommitteeRequest | None:
        target = _clean_text(request_id)
        if not target:
            return None
        for item in reversed(self.list_requests(limit=0)):
            if item.request_id == target:
                return item
        return None

    def append_activity(self, *, event: str, payload: dict[str, Any]) -> None:
        row = {
            "event": _clean_text(event) or "committee_event",
            "ts_ms": now_ts_ms(),
            "payload": dict(payload or {}),
        }
        with self._activity_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(slots=True)
class CommitteeCoordinator:
    project_root: Path
    signal_bus: SignalBus
    store: CommitteeStore | None = None
    default_candidates: tuple[str, ...] = ("market_worker",)
    bidding: BiddingService | None = None
    sandbox_manager: Any | None = None
    notifications: NotificationStore | None = None
    bid_evaluation_mode: str = "heuristic_mvp"
    completion_verification_mode: str = "rules_mvp"

    def __post_init__(self) -> None:
        self.project_root = self.project_root.expanduser().resolve()
        if self.store is None:
            self.store = CommitteeStore(self.project_root)
        if self.bidding is None:
            self.bidding = BiddingService(project_root=self.project_root, config=BiddingConfig())
        configured_mode = _clean_text(self.bid_evaluation_mode).lower()
        bidding_mode = _clean_text(getattr(self.bidding, "evaluation_mode", "")).lower()
        if not configured_mode:
            self.bid_evaluation_mode = bidding_mode or "heuristic_mvp"
        elif configured_mode == "heuristic_mvp" and bidding_mode and bidding_mode != "heuristic_mvp":
            # Keep coordinator default aligned with non-default bidding evaluator mode.
            self.bid_evaluation_mode = bidding_mode
        self.completion_verification_mode = _clean_text(self.completion_verification_mode).lower() or "rules_mvp"
        if self.sandbox_manager is None:
            self.sandbox_manager = SandboxManager(project_root=self.project_root)
        if self.notifications is None:
            self.notifications = NotificationStore(project_root=self.project_root)

    def handle_signal(self, signal: Signal) -> dict[str, Any]:
        if signal.to_agent != COMMITTEE_AGENT_ID:
            return {"handled": False, "reason": "not_for_committee"}
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        payload_type = _clean_text(payload.get("type")).lower()
        if is_project_request_signal(signal):
            return self._handle_project_request(signal)
        if signal.signal_type is SignalType.NOTIFY and payload_type == "bid_comments":
            return self._handle_bid_comments(signal)
        if signal.signal_type is SignalType.NOTIFY and payload_type in {"check_bids", "bid_check"}:
            return self._handle_bid_check(signal)
        if signal.signal_type is SignalType.NOTIFY and self._is_task_completed_signal(signal):
            return self._handle_task_completed(signal)
        return {"handled": False, "reason": "unsupported_signal"}

    def _handle_project_request(self, signal: Signal) -> dict[str, Any]:
        # Design choice (MVP): coordinator path owns project_request handling end-to-end.
        # AgentRunner short-circuits when `handled=True`, so this signal is not processed again
        # by the LLM chat loop in the same turn.
        request = self._request_from_signal(signal)
        self.store.upsert_request(request)
        self.store.append_activity(
            event="project_request_received",
            payload={
                "request_id": request.request_id,
                "signal_id": signal.signal_id,
                "from_agent": signal.from_agent,
                "task_count": len(request.tasks),
            },
        )
        wake_ids = self._dispatch_wakes(request=request)
        status = CommitteeRequestStatus.DISPATCHED if wake_ids else CommitteeRequestStatus.QUEUED
        request = request.model_copy(update={"status": status, "wake_signal_ids": wake_ids, "updated_at": now_ts_ms()})
        self.store.upsert_request(request)
        self.store.append_activity(
            event="project_request_dispatched",
            payload={
                "request_id": request.request_id,
                "status": request.status.value,
                "wake_count": len(wake_ids),
                "issue_keys": [item.issue_key for item in request.tasks],
                "decomposition_mode": COMMITTEE_DECOMPOSITION_MODE,
                "placeholder_issue_keys": [item.issue_key for item in request.tasks if item.issue_key_is_placeholder],
            },
        )
        return {
            "handled": True,
            "request_id": request.request_id,
            "status": request.status.value,
            "task_count": len(request.tasks),
            "wake_count": len(wake_ids),
            "wake_signal_ids": wake_ids,
            "decomposition_mode": COMMITTEE_DECOMPOSITION_MODE,
        }

    def _handle_bid_comments(self, signal: Signal) -> dict[str, Any]:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        issue_key = _clean_text(payload.get("issue_key")) or _clean_text(signal.issue_key)
        comments = payload.get("comments")
        if not issue_key:
            return {"handled": False, "reason": "missing_issue_key"}
        if not isinstance(comments, list):
            return {"handled": False, "reason": "missing_comments"}

        collected = self.collect_bids(issue_key=issue_key, comments=comments)
        auto_eval = bool(payload.get("auto_evaluate", True))
        evaluated: dict[str, Any] | None = None
        if auto_eval:
            evaluated = self.evaluate_bids(
                issue_key=issue_key,
                base_branch=str(payload.get("base_branch") or "main").strip() or "main",
            )
        return {
            "handled": True,
            "kind": "bid_comments",
            "issue_key": issue_key,
            "collected": collected,
            "evaluated": evaluated,
        }

    def _handle_bid_check(self, signal: Signal) -> dict[str, Any]:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        comments_by_issue_raw = payload.get("comments_by_issue")
        comments_by_issue = comments_by_issue_raw if isinstance(comments_by_issue_raw, dict) else {}
        auto_evaluate = bool(payload.get("auto_evaluate", True))
        force_evaluate = bool(payload.get("force_evaluate", False))

        explicit_issue_keys = _clean_list(payload.get("issue_keys"))
        issue_keys: list[str] = []
        if explicit_issue_keys:
            issue_keys = explicit_issue_keys
        else:
            seen: set[str] = set()
            for phase in (BiddingPhase.BIDDING, BiddingPhase.REBID, BiddingPhase.EVALUATING):
                for row in self.bidding.store.list(phase=phase, limit=0):
                    if row.issue_key in seen:
                        continue
                    seen.add(row.issue_key)
                    issue_keys.append(row.issue_key)

        processed: list[dict[str, Any]] = []
        for issue_key in issue_keys:
            raw_comments = comments_by_issue.get(issue_key, [])
            comments = raw_comments if isinstance(raw_comments, list) else []
            collected = self.collect_bids(issue_key=issue_key, comments=comments)
            evaluated: dict[str, Any] | None = None
            should_evaluate = auto_evaluate and (
                force_evaluate or bool(comments) or collected.get("phase") == BiddingPhase.EVALUATING.value
            )
            if should_evaluate:
                evaluated = self.evaluate_bids(issue_key=issue_key, base_branch=str(payload.get("base_branch") or "main"))
            processed.append(
                {
                    "issue_key": issue_key,
                    "comment_count": len(comments),
                    "collected": collected,
                    "evaluated": evaluated,
                }
            )

        self.store.append_activity(
            event="bid_check_processed",
            payload={
                "signal_id": signal.signal_id,
                "issues": [item["issue_key"] for item in processed],
                "auto_evaluate": auto_evaluate,
            },
        )
        return {
            "handled": True,
            "kind": "bid_check",
            "processed_count": len(processed),
            "issues": processed,
        }

    @staticmethod
    def _is_task_completed_signal(signal: Signal) -> bool:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        payload_type = _clean_text(payload.get("type")).lower()
        if payload_type not in {"task_completed", "work_completed", "completion"}:
            return False
        issue_key = _clean_text(payload.get("issue_key")) or _clean_text(signal.issue_key)
        return bool(issue_key)

    def _verify_task_completed(self, *, signal: Signal, payload: dict[str, Any], issue_key: str) -> dict[str, Any]:
        worker_agent = _clean_text(signal.from_agent) or "worker"
        verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
        decision = _clean_text(verification.get("decision")).lower()
        summary = _clean_text(verification.get("summary")) or _clean_text(payload.get("summary"))
        missing_criteria = _clean_list(verification.get("missing_criteria"))
        required_revisions = _clean_list(verification.get("required_revisions"))

        acceptance_criteria = _clean_list(payload.get("acceptance_criteria"))
        audit_summary = _clean_text(payload.get("audit_summary")) or _clean_text(payload.get("audit_evidence"))
        snapshot_summary = _clean_text(payload.get("snapshot_summary")) or _clean_text(payload.get("snapshot_diff"))

        # Worker decision flags are ignored. Committee must verify independently.
        if decision not in {"accept", "reject"}:
            if required_revisions or missing_criteria:
                decision = "reject"
            elif acceptance_criteria and (not audit_summary or not snapshot_summary):
                if not audit_summary:
                    required_revisions.append("Attach audit evidence for verification.")
                if not snapshot_summary:
                    required_revisions.append("Attach snapshot diff evidence for verification.")
                decision = "reject"
            else:
                decision = "accept"

        if decision == "accept" and (required_revisions or missing_criteria):
            decision = "reject"

        if not summary:
            if decision == "accept":
                summary = f"{issue_key} completed by {worker_agent}."
            else:
                summary = "Committee verification requires revisions."

        return {
            "decision": decision,
            "summary": summary,
            "missing_criteria": missing_criteria,
            "required_revisions": required_revisions,
            "acceptance_criteria": acceptance_criteria,
            "audit_summary": audit_summary,
            "snapshot_summary": snapshot_summary,
            "mode": self.completion_verification_mode,
        }

    def _sandbox_worktree_is_clean(self, *, sandbox_id: str) -> tuple[bool, str | None]:
        manager = self.sandbox_manager
        if manager is None:
            return False, "sandbox_manager_unavailable"

        custom_is_clean = getattr(manager, "is_clean", None)
        if callable(custom_is_clean):
            try:
                is_clean = bool(custom_is_clean(sandbox_id))
                return is_clean, None if is_clean else "sandbox_dirty"
            except Exception as exc:
                return False, f"sandbox_clean_probe_failed:{exc}"

        getter = getattr(manager, "get", None)
        if not callable(getter):
            return False, "sandbox_state_unknown"
        try:
            sandbox = getter(sandbox_id)
        except Exception as exc:
            return False, f"sandbox_lookup_failed:{exc}"
        if not isinstance(sandbox, Sandbox):
            return False, "sandbox_not_found"

        worktree_abs = (self.project_root / sandbox.worktree_path).resolve()
        if not worktree_abs.exists():
            return False, "sandbox_worktree_missing"

        probe = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree_abs,
            text=True,
            capture_output=True,
        )
        if probe.returncode != 0:
            detail = _clean_text(probe.stderr or probe.stdout)
            return False, f"sandbox_clean_probe_failed:{detail or probe.returncode}"
        if _clean_text(probe.stdout):
            return False, "sandbox_dirty"
        return True, None

    def _cleanup_sandbox_if_safe(self, *, signal: Signal, payload: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        sandbox_id = _clean_text(signal.sandbox_id)
        if not sandbox_id:
            return False, "missing_sandbox_id", None
        if not bool(payload.get("pr_merged")) and not bool(payload.get("cleanup_force")):
            return False, "pr_not_merged", None
        if not bool(payload.get("cleanup_force")):
            clean, reason = self._sandbox_worktree_is_clean(sandbox_id=sandbox_id)
            if not clean:
                return False, reason or "sandbox_not_clean", None
        if self.sandbox_manager is None:
            return False, "sandbox_manager_unavailable", None
        try:
            self.sandbox_manager.destroy(sandbox_id)
            return True, None, None
        except Exception as exc:
            return False, None, str(exc)

    def _handle_task_completed(self, signal: Signal) -> dict[str, Any]:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        issue_key = _clean_text(payload.get("issue_key")) or _clean_text(signal.issue_key)
        if not issue_key:
            return {"handled": False, "reason": "missing_issue_key"}
        worker_agent = _clean_text(signal.from_agent) or "worker"
        verification = self._verify_task_completed(signal=signal, payload=payload, issue_key=issue_key)
        accept = verification.get("decision") == "accept"

        if accept:
            summary = _clean_text(verification.get("summary")) or f"{issue_key} completed by {worker_agent}."
            notification = self.notifications.create(
                notification_type=NotificationType.TASK_COMPLETED,
                title=f"{issue_key} completed",
                summary=summary,
                issue_key=issue_key,
                pr_url=_clean_text(payload.get("pr_url")) or None,
                details={
                    "agent_id": worker_agent,
                    "sandbox_id": signal.sandbox_id,
                    "signal_id": signal.signal_id,
                    "verification_mode": verification.get("mode"),
                },
            )
            user_signal = self.signal_bus.send(
                from_agent=COMMITTEE_AGENT_ID,
                to_agent="super_agent",
                signal_type=SignalType.NOTIFY,
                brief=f"Task completed: {issue_key}"[:200],
                issue_key=issue_key,
                payload={
                    "type": "task_completed",
                    "issue_key": issue_key,
                    "agent_id": worker_agent,
                    "summary": summary,
                    "notification_id": notification.notification_id,
                    "pr_url": _clean_text(payload.get("pr_url")) or None,
                    "verification_mode": verification.get("mode"),
                },
            )

            # TODO(phase-05): finalize push/PR/merge workflow before defaulting to cleanup.
            cleanup_error: str | None = None
            cleanup_cleaned = False
            cleanup_skipped_reason: str | None = None
            if bool(payload.get("cleanup_sandbox")):
                cleanup_cleaned, cleanup_skipped_reason, cleanup_error = self._cleanup_sandbox_if_safe(
                    signal=signal,
                    payload=payload,
                )
            self.store.append_activity(
                event="completion_accepted",
                payload={
                    "issue_key": issue_key,
                    "agent_id": worker_agent,
                    "notification_id": notification.notification_id,
                    "notify_signal_id": user_signal.signal_id,
                    "verification_mode": verification.get("mode"),
                    "verification_summary": summary,
                    "cleanup_requested": bool(payload.get("cleanup_sandbox")),
                    "cleanup_cleaned": cleanup_cleaned,
                    "cleanup_skipped_reason": cleanup_skipped_reason,
                    "cleanup_error": cleanup_error,
                },
            )
            return {
                "handled": True,
                "kind": "task_completed",
                "decision": "accept",
                "issue_key": issue_key,
                "notification_id": notification.notification_id,
                "notify_signal_id": user_signal.signal_id,
                "verification_mode": verification.get("mode"),
                "cleanup_cleaned": cleanup_cleaned,
                "cleanup_skipped_reason": cleanup_skipped_reason,
                "cleanup_error": cleanup_error,
            }

        missing_criteria = _clean_list(verification.get("missing_criteria"))
        required_revisions = _clean_list(verification.get("required_revisions"))
        feedback_items = list(required_revisions)
        feedback_items.extend([f"Missing acceptance criterion: {item}" for item in missing_criteria])
        feedback = "; ".join([item for item in feedback_items if item]) or _clean_text(verification.get("summary"))
        if not feedback:
            feedback = _clean_text(payload.get("feedback")) or _clean_text(payload.get("reason")) or "Revisions required."
        revision_signal = self.signal_bus.send(
            from_agent=COMMITTEE_AGENT_ID,
            to_agent=worker_agent,
            signal_type=SignalType.WAKE,
            brief=f"Revisions needed for {issue_key}"[:200],
            issue_key=issue_key,
            sandbox_id=signal.sandbox_id,
            payload={
                "type": "revision_request",
                "issue_key": issue_key,
                "feedback": feedback,
                "original_signal_id": signal.signal_id,
            },
        )
        notification = self.notifications.create(
            notification_type=NotificationType.REVIEW_NEEDED,
            title=f"{issue_key} needs revisions",
            summary=feedback,
            issue_key=issue_key,
            details={
                "agent_id": worker_agent,
                "sandbox_id": signal.sandbox_id,
                "revision_signal_id": revision_signal.signal_id,
            },
        )
        self.store.append_activity(
            event="completion_rejected",
            payload={
                "issue_key": issue_key,
                "agent_id": worker_agent,
                "feedback": feedback,
                "verification_mode": verification.get("mode"),
                "missing_criteria": missing_criteria,
                "required_revisions": required_revisions,
                "revision_signal_id": revision_signal.signal_id,
                "notification_id": notification.notification_id,
            },
        )
        return {
            "handled": True,
            "kind": "task_completed",
            "decision": "reject",
            "issue_key": issue_key,
            "feedback": feedback,
            "revision_signal_id": revision_signal.signal_id,
            "notification_id": notification.notification_id,
            "verification_mode": verification.get("mode"),
        }

    def _request_from_signal(self, signal: Signal) -> CommitteeRequest:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        goal = _clean_text(payload.get("goal")) or _clean_text(signal.brief) or "Project request"
        context = _clean_text(payload.get("context")) or _clean_text(signal.brief)
        priority = _clean_text(payload.get("priority")).lower() or "medium"
        if priority not in {"high", "medium", "low"}:
            priority = "medium"

        request_id = _clean_text(payload.get("request_id"))
        if not request_id:
            request_id = f"req_{new_id('committee')}"

        tasks = self._tasks_from_payload(payload=payload, request_id=request_id, goal=goal, context=context)
        return CommitteeRequest(
            request_id=request_id,
            source_signal_id=signal.signal_id,
            from_agent=signal.from_agent,
            goal=goal,
            context=context,
            constraints=_clean_list(payload.get("constraints")),
            priority=priority,
            references=_clean_list(payload.get("references")),
            tasks=tasks,
        )

    def _tasks_from_payload(
        self,
        *,
        payload: dict[str, Any],
        request_id: str,
        goal: str,
        context: str,
    ) -> list[CommitteeTask]:
        raw_tasks = payload.get("tasks")
        candidates_global = _clean_list(payload.get("candidate_agents"))
        out: list[CommitteeTask] = []
        if isinstance(raw_tasks, list):
            for idx, raw in enumerate(raw_tasks):
                if not isinstance(raw, dict):
                    continue
                issue_key = _clean_text(raw.get("issue_key")) or self._issue_key_for(request_id=request_id, index=idx, goal=goal)
                title = _clean_text(raw.get("title")) or f"Task {idx + 1}"
                description = _clean_text(raw.get("description")) or _clean_text(raw.get("brief")) or context
                required = _clean_list(raw.get("required_capabilities") or raw.get("capabilities"))
                acceptance = _clean_list(raw.get("acceptance_criteria"))
                task_candidates = _clean_list(raw.get("candidate_agents") or raw.get("candidates")) or list(candidates_global)
                out.append(
                    CommitteeTask(
                        issue_key=issue_key,
                        title=title,
                        description=description or title,
                        required_capabilities=required,
                        acceptance_criteria=acceptance,
                        candidate_agents=task_candidates,
                        issue_key_is_placeholder=(not _clean_text(raw.get("issue_key"))),
                        issue_key_source="payload" if _clean_text(raw.get("issue_key")) else "generated",
                    )
                )
        if out:
            return out

        return [
            CommitteeTask(
                issue_key=self._issue_key_for(request_id=request_id, index=0, goal=goal),
                title=goal[:120] or "Project task",
                description=context or goal,
                required_capabilities=_clean_list(payload.get("required_capabilities")),
                acceptance_criteria=_clean_list(payload.get("acceptance_criteria")),
                candidate_agents=list(candidates_global),
                issue_key_is_placeholder=True,
                issue_key_source="generated",
            )
        ]

    def _issue_key_for(self, *, request_id: str, index: int, goal: str) -> str:
        suffix = _slug_token(request_id)[-8:]
        goal_token = _slug_token(goal)[:8]
        # Placeholder key for local flow; Linear-backed integration should replace this with real issue keys.
        return f"PLH-{goal_token}-{suffix}-{index + 1}"

    def _dispatch_wakes(self, *, request: CommitteeRequest) -> list[str]:
        wake_ids: list[str] = []
        for task in request.tasks:
            candidates = [c for c in task.candidate_agents if c and c != COMMITTEE_AGENT_ID]
            if not candidates:
                candidates = [c for c in self.default_candidates if c and c != COMMITTEE_AGENT_ID]
            for candidate in candidates:
                brief = f"Bid requested for {task.issue_key}: {task.title}"
                signal = self.signal_bus.send(
                    from_agent=COMMITTEE_AGENT_ID,
                    to_agent=candidate,
                    signal_type=SignalType.WAKE,
                    brief=brief[:200],
                    issue_key=task.issue_key,
                    payload={
                        "type": "committee_task",
                        "request_id": request.request_id,
                        "goal": request.goal,
                        "context": request.context,
                        "priority": request.priority,
                        "constraints": list(request.constraints),
                        "references": list(request.references),
                        "task": task.model_dump(mode="json"),
                        "issue_key_is_placeholder": bool(task.issue_key_is_placeholder),
                    },
                )
                wake_ids.append(signal.signal_id)
            self.bidding.open(issue_key=task.issue_key, candidates=list(candidates))
        return wake_ids

    def collect_bids(self, *, issue_key: str, comments: list[Any]) -> dict[str, Any]:
        record = self.bidding.collect(issue_key=issue_key, comments=comments)
        self.store.append_activity(
            event="bids_collected",
            payload={
                "issue_key": issue_key,
                "bid_count": len(record.bids),
                "phase": record.phase.value,
            },
        )
        return {
            "ok": True,
            "issue_key": issue_key,
            "phase": record.phase.value,
            "bid_count": len(record.bids),
            "bids": [item.model_dump(mode="json") for item in record.bids],
        }

    def evaluate_bids(self, *, issue_key: str, base_branch: str = "main") -> dict[str, Any]:
        record, decision = self.bidding.evaluate(issue_key=issue_key)
        payload: dict[str, Any] = {
            "ok": True,
            "issue_key": issue_key,
            "phase": record.phase.value,
            "action": decision.action,
            "selected_agent": decision.selected_agent,
            "reason": decision.reason,
            "evaluation_mode": self.bid_evaluation_mode,
        }

        if decision.action == "assign" and decision.selected_agent:
            sandbox: Sandbox | None = None
            sandbox_error: str | None = None
            try:
                if self.sandbox_manager is not None:
                    sandbox = self.sandbox_manager.create(
                        agent_id=decision.selected_agent,
                        issue_key=issue_key,
                        base_branch=str(base_branch or "main").strip() or "main",
                    )
            except Exception as exc:
                sandbox_error = str(exc)
                sandbox = None
            assign_signal = self.signal_bus.send(
                from_agent=COMMITTEE_AGENT_ID,
                to_agent=decision.selected_agent,
                signal_type=SignalType.TASK_ASSIGNED,
                brief=f"Task assigned for {issue_key}",
                issue_key=issue_key,
                sandbox_id=sandbox.sandbox_id if sandbox is not None else None,
                payload={
                    "type": "task_assigned",
                    "issue_key": issue_key,
                    "selected_agent": decision.selected_agent,
                    "bidding_round": int(record.round),
                },
            )
            assigned = self.bidding.mark_assigned(issue_key=issue_key, selected_agent=decision.selected_agent)
            self.store.append_activity(
                event="bid_assigned",
                payload={
                    "issue_key": issue_key,
                    "selected_agent": decision.selected_agent,
                    "signal_id": assign_signal.signal_id,
                    "sandbox_id": sandbox.sandbox_id if sandbox is not None else None,
                    "sandbox_error": sandbox_error,
                    "evaluation_mode": self.bid_evaluation_mode,
                },
            )
            payload.update(
                {
                    "phase": assigned.phase.value,
                    "task_assigned_signal_id": assign_signal.signal_id,
                    "sandbox_id": sandbox.sandbox_id if sandbox is not None else None,
                    "sandbox_error": sandbox_error,
                }
            )
            return payload

        if decision.action == "rebid":
            wake_ids: list[str] = []
            for candidate in record.candidates:
                signal = self.signal_bus.send(
                    from_agent=COMMITTEE_AGENT_ID,
                    to_agent=candidate,
                    signal_type=SignalType.WAKE,
                    brief=f"Rebid requested for {issue_key}",
                    issue_key=issue_key,
                    payload={"type": "committee_rebid", "issue_key": issue_key, "round": int(record.round)},
                )
                wake_ids.append(signal.signal_id)
            record = record.model_copy(update={"phase": BiddingPhase.BIDDING, "wake_sent_at": now_ts_ms()})
            self.bidding.store.save(record)
            self.store.append_activity(
                event="bid_rebid",
                payload={"issue_key": issue_key, "round": int(record.round), "wake_count": len(wake_ids)},
            )
            payload.update({"phase": record.phase.value, "rebid_wake_signal_ids": wake_ids})
            return payload

        self.store.append_activity(
            event="bid_evaluated",
            payload={
                "issue_key": issue_key,
                "action": decision.action,
                "reason": decision.reason,
                "evaluation_mode": self.bid_evaluation_mode,
            },
        )
        return payload
