from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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


class CommitteeRequest(BaseModel):
    request_id: str
    source_signal_id: str | None = None
    from_agent: str | None = None
    goal: str
    context: str
    constraints: list[str] = Field(default_factory=list)
    priority: str = "medium"
    references: list[str] = Field(default_factory=list)
    status: str = "pending"
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

    def list_requests(self, *, limit: int = 100, status: str | None = None) -> list[CommitteeRequest]:
        if not self._requests_path.exists():
            return []
        out: list[CommitteeRequest] = []
        target_status = _clean_text(status).lower() if isinstance(status, str) else ""
        for line in self._requests_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = CommitteeRequest.model_validate_json(line)
            except Exception:
                continue
            if target_status and _clean_text(item.status).lower() != target_status:
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

    def __post_init__(self) -> None:
        self.project_root = self.project_root.expanduser().resolve()
        if self.store is None:
            self.store = CommitteeStore(self.project_root)
        if self.bidding is None:
            self.bidding = BiddingService(project_root=self.project_root, config=BiddingConfig())
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
        if signal.signal_type is SignalType.NOTIFY and self._is_task_completed_signal(signal):
            return self._handle_task_completed(signal)
        return {"handled": False, "reason": "unsupported_signal"}

    def _handle_project_request(self, signal: Signal) -> dict[str, Any]:
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
        status = "dispatched" if wake_ids else "queued"
        request = request.model_copy(update={"status": status, "wake_signal_ids": wake_ids, "updated_at": now_ts_ms()})
        self.store.upsert_request(request)
        self.store.append_activity(
            event="project_request_dispatched",
            payload={
                "request_id": request.request_id,
                "status": request.status,
                "wake_count": len(wake_ids),
                "issue_keys": [item.issue_key for item in request.tasks],
            },
        )
        return {
            "handled": True,
            "request_id": request.request_id,
            "status": request.status,
            "task_count": len(request.tasks),
            "wake_count": len(wake_ids),
            "wake_signal_ids": wake_ids,
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

    @staticmethod
    def _is_task_completed_signal(signal: Signal) -> bool:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        payload_type = _clean_text(payload.get("type")).lower()
        if payload_type in {"task_completed", "work_completed", "completion"}:
            return True
        brief = _clean_text(signal.brief).lower()
        return "task_completed" in brief or "completed" in brief

    def _handle_task_completed(self, signal: Signal) -> dict[str, Any]:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        issue_key = _clean_text(payload.get("issue_key")) or _clean_text(signal.issue_key)
        if not issue_key:
            return {"handled": False, "reason": "missing_issue_key"}
        worker_agent = _clean_text(signal.from_agent) or "worker"

        decision = payload.get("decision")
        accept: bool
        if isinstance(decision, str) and decision.strip():
            accept = decision.strip().lower() in {"accept", "approved", "pass"}
        elif isinstance(payload.get("accept"), bool):
            accept = bool(payload.get("accept"))
        else:
            accept = True

        if accept:
            summary = _clean_text(payload.get("summary")) or f"{issue_key} completed by {worker_agent}."
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
                },
            )

            cleanup_error: str | None = None
            if bool(payload.get("cleanup_sandbox")) and isinstance(signal.sandbox_id, str) and signal.sandbox_id.strip():
                try:
                    if self.sandbox_manager is not None:
                        self.sandbox_manager.destroy(signal.sandbox_id.strip())
                except Exception as exc:
                    cleanup_error = str(exc)
            self.store.append_activity(
                event="completion_accepted",
                payload={
                    "issue_key": issue_key,
                    "agent_id": worker_agent,
                    "notification_id": notification.notification_id,
                    "notify_signal_id": user_signal.signal_id,
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
                "cleanup_error": cleanup_error,
            }

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
            )
        ]

    def _issue_key_for(self, *, request_id: str, index: int, goal: str) -> str:
        suffix = _slug_token(request_id)[-8:]
        goal_token = _slug_token(goal)[:8]
        return f"AUTO-{goal_token}-{suffix}-{index + 1}"

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
            payload={"issue_key": issue_key, "action": decision.action, "reason": decision.reason},
        )
        return payload
