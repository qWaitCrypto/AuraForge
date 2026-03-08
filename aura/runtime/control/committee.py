from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .bidding import BiddingConfig, BiddingService
from ..ids import new_id, now_ts_ms
from .notifications import NotificationStore
from ..models.notification import NotificationType
from ..sandbox import SandboxManager
from ..models.signal import Signal, SignalType
from ..signal import SignalBus

COMMITTEE_AGENT_ID = "committee"
PROJECT_REQUEST_TYPE = "project_request"
COMMITTEE_DECOMPOSITION_MODE = "thin_router"


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


_JSON_OBJECT_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


def _parse_json_object(raw: Any) -> dict[str, Any] | None:
    text = _clean_text(raw)
    if not text:
        return None

    candidates = [text]
    match = _JSON_OBJECT_BLOCK_RE.search(text)
    if match is not None:
        candidates.insert(0, match.group(1))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


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
    linear_project: str | None = None
    publish_repo: str | None = None
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
    """
    Thin router for Committee agent.

    This layer intentionally avoids workflow decisions and external side effects.
    It only prepares context, records auditable state, and captures run outcomes.
    """

    project_root: Path
    signal_bus: SignalBus
    store: CommitteeStore | None = None
    default_candidates: tuple[str, ...] = ("market_worker",)
    bidding: BiddingService | None = None
    dispatcher: Any | None = None
    sandbox_manager: Any | None = None
    notifications: NotificationStore | None = None
    bid_evaluation_mode: str = "llm_delegated"
    bid_llm_evaluator: Any | None = None
    completion_verification_mode: str = "llm_delegated"
    issue_creator: Any | None = None
    linear_comments_reader: Any | None = None
    linear_mcp_server: str = "linear"
    linear_team_id: str | None = None
    delivery_publisher: Any | None = None
    auto_publish_on_accept: bool = False
    publish_repo_env: str = "AURAFORGE_PUBLISH_REPO"
    coordinator_mode: str = COMMITTEE_DECOMPOSITION_MODE
    max_seed_tasks: int = 6

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
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None and running_loop.is_running():
            raise RuntimeError("handle_signal() cannot run inside an active event loop; use ahandle_signal().")
        return asyncio.run(self.ahandle_signal(signal))

    async def ahandle_signal(self, signal: Signal) -> dict[str, Any]:
        # Compat API: supports standalone/non-runner callers that still invoke coordinator entry directly.
        # In thin-router mode, we intentionally prepare context only and let Engine perform all decisions.
        prepared = self.prepare_context(signal=signal, session_id=None)
        return {
            "handled": False,
            "mode": _clean_text(self.coordinator_mode).lower() or COMMITTEE_DECOMPOSITION_MODE,
            **prepared,
        }

    @staticmethod
    def _signal_kind(signal: Signal) -> str:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        payload_type = _clean_text(payload.get("type")).lower()
        if is_project_request_signal(signal):
            return PROJECT_REQUEST_TYPE
        if signal.signal_type is SignalType.NOTIFY and payload_type in {"bid_check", "check_bids", "bid_comments"}:
            return "bid_check"
        if signal.signal_type is SignalType.NOTIFY and payload_type in {"task_completed", "work_completed", "completion"}:
            return "task_completed"
        return f"{signal.signal_type.value}:{payload_type or 'default'}"

    def prepare_context(self, *, signal: Signal, session_id: str | None) -> dict[str, Any]:
        if signal.to_agent != COMMITTEE_AGENT_ID:
            return {"prepared": False, "reason": "not_for_committee"}

        kind = self._signal_kind(signal)
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        summary: dict[str, Any] = {
            "prepared": True,
            "kind": kind,
            "signal_id": signal.signal_id,
            "signal_type": signal.signal_type.value,
        }

        if kind == PROJECT_REQUEST_TYPE:
            request = self._request_from_signal(signal, allow_fallback_tasks=False)
            request = request.model_copy(update={"status": CommitteeRequestStatus.PENDING, "updated_at": now_ts_ms()})
            self.store.upsert_request(request)
            self.store.append_activity(
                event="project_request_received",
                payload={
                    "mode": _clean_text(self.coordinator_mode).lower() or COMMITTEE_DECOMPOSITION_MODE,
                    "request_id": request.request_id,
                    "signal_id": signal.signal_id,
                    "session_id": session_id,
                    "from_agent": signal.from_agent,
                    "goal": request.goal,
                    "task_count": len(request.tasks),
                },
            )
            summary.update({"request_id": request.request_id, "task_count": len(request.tasks)})
            return summary

        self.store.append_activity(
            event="committee_signal_received",
            payload={
                "mode": _clean_text(self.coordinator_mode).lower() or COMMITTEE_DECOMPOSITION_MODE,
                "kind": kind,
                "signal_id": signal.signal_id,
                "session_id": session_id,
                "issue_key": signal.issue_key,
                "payload_type": _clean_text(payload.get("type")).lower() or None,
            },
        )

        if kind == "bid_check":
            summary.update(self._handle_bid_check(signal=signal, session_id=session_id))
        return summary

    def post_process(
        self,
        *,
        signal: Signal,
        session_id: str | None,
        run_status: str,
        run_id: str,
        error: str | None,
        assistant_text: str | None = None,
    ) -> None:
        kind = self._signal_kind(signal)
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        verification = _parse_json_object(assistant_text) if kind == "task_completed" else None
        verify_decision = _clean_text((verification or {}).get("decision")).lower() or None
        notification_id: str | None = None

        if (
            kind == "task_completed"
            and _clean_text(run_status).lower() == "completed"
            and verify_decision == "accept"
            and self.notifications is not None
        ):
            summary_text = (
                _clean_text((verification or {}).get("summary"))
                or _clean_text(payload.get("summary"))
                or f"{_clean_text(signal.issue_key) or 'task'} completed"
            )
            notification = self.notifications.create(
                notification_type=NotificationType.TASK_COMPLETED,
                title=f"{_clean_text(signal.issue_key) or 'Task'} completed",
                summary=summary_text,
                issue_key=_clean_text(signal.issue_key) or None,
                details={
                    "signal_id": signal.signal_id,
                    "run_id": _clean_text(run_id) or None,
                    "worker": _clean_text(signal.from_agent) or None,
                    "verification": verification,
                },
            )
            notification_id = notification.notification_id
            if self.delivery_publisher is not None and self.auto_publish_on_accept:
                try:
                    self.delivery_publisher(notification=notification, signal=signal, verification=verification)
                except TypeError:
                    try:
                        self.delivery_publisher(notification)
                    except Exception:
                        pass
                except Exception:
                    pass

        self.store.append_activity(
            event="committee_signal_completed",
            payload={
                "mode": _clean_text(self.coordinator_mode).lower() or COMMITTEE_DECOMPOSITION_MODE,
                "kind": kind,
                "signal_id": signal.signal_id,
                "session_id": session_id,
                "run_id": _clean_text(run_id) or None,
                "run_status": _clean_text(run_status) or None,
                "error": _clean_text(error) or None,
                "verify_decision": verify_decision,
                "notification_id": notification_id,
            },
        )

    def _handle_bid_check(self, *, signal: Signal, session_id: str | None) -> dict[str, Any]:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        issue_key = _clean_text(signal.issue_key) or _clean_text(payload.get("issue_key"))
        if not issue_key or self.bidding is None:
            return {"auto_action": "skipped"}

        comments = self._load_bid_comments(signal=signal)
        candidates = self._candidate_agents_for_issue(signal=signal, issue_key=issue_key)
        record = self.bidding.get(issue_key)
        if record is None:
            record = self.bidding.open(issue_key=issue_key, candidates=candidates)

        collected = self.bidding.collect(issue_key=issue_key, comments=comments, candidates=candidates)
        record, decision = self.bidding.evaluate(issue_key=issue_key)

        result: dict[str, Any] = {
            "auto_action": decision.action,
            "issue_key": issue_key,
            "bid_count": len(collected.bids),
        }

        if decision.action == "assign" and decision.selected_agent:
            if self.dispatcher is None:
                result["auto_action"] = "assign_pending_dispatch"
            else:
                from .dispatcher import DispatchRequest

                dispatch_result = self.dispatcher.dispatch(
                    DispatchRequest(
                        issue_key=issue_key,
                        brief=_clean_text(payload.get("brief")) or f"Committee assigned {issue_key}",
                        signal_type=SignalType.TASK_ASSIGNED,
                        agent_id=decision.selected_agent,
                        base_branch=_clean_text(payload.get("base_branch")) or "main",
                        payload={
                            "type": "committee_task",
                            "issue_key": issue_key,
                            "source_signal_id": signal.signal_id,
                            "selected_agent": decision.selected_agent,
                            "reason": decision.reason,
                            "rejection_reasons": decision.rejection_reasons or {},
                        },
                    )
                )
                result.update({
                    "selected_agent": decision.selected_agent,
                    "dispatch_ok": bool(dispatch_result.dispatched),
                })
                self.store.append_activity(
                    event="bidding_assigned",
                    payload={
                        "signal_id": signal.signal_id,
                        "session_id": session_id,
                        "issue_key": issue_key,
                        "selected_agent": decision.selected_agent,
                        "dispatch_ok": bool(dispatch_result.dispatched),
                        "dispatch_rejection": dispatch_result.rejection_reason,
                    },
                )
                if dispatch_result.dispatched:
                    self.bidding.mark_assigned(issue_key=issue_key, selected_agent=decision.selected_agent)
                else:
                    result["dispatch_rejection"] = dispatch_result.rejection_reason
            return result

        if decision.action == "rebid":
            rebid_candidates = candidates or list(record.candidates)
            wake_signal_ids: list[str] = []
            for agent_id in rebid_candidates:
                wake = self.signal_bus.send(
                    from_agent=COMMITTEE_AGENT_ID,
                    to_agent=agent_id,
                    signal_type=SignalType.WAKE,
                    brief=f"Rebid requested for {issue_key}",
                    issue_key=issue_key,
                    payload={
                        "type": "committee_task",
                        "issue_key": issue_key,
                        "rebid": True,
                        "round": int(getattr(record, 'round', 1) or 1),
                    },
                )
                wake_signal_ids.append(wake.signal_id)
            self._persist_wake_signal_ids(issue_key=issue_key, wake_signal_ids=wake_signal_ids)
            self.store.append_activity(
                event="bidding_rebid_requested",
                payload={
                    "signal_id": signal.signal_id,
                    "session_id": session_id,
                    "issue_key": issue_key,
                    "candidates": rebid_candidates,
                    "wake_signal_ids": wake_signal_ids,
                },
            )
        return result

    def _load_bid_comments(self, *, signal: Signal) -> list[Any]:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        comments = payload.get("comments")
        if isinstance(comments, list):
            return list(comments)

        reader = self.linear_comments_reader
        if reader is None:
            return []

        issue_key = _clean_text(signal.issue_key) or _clean_text(payload.get("issue_key"))
        if not issue_key:
            return []

        attempts = (
            lambda: reader(issue_key=issue_key, payload=payload),
            lambda: reader(issue_key),
            lambda: reader(signal),
        )
        for call in attempts:
            try:
                loaded = call()
            except TypeError:
                continue
            except Exception as exc:
                self.store.append_activity(
                    event="bid_comment_reader_failed",
                    payload={
                        "signal_id": signal.signal_id,
                        "issue_key": issue_key,
                        "error": str(exc),
                    },
                )
                return []
            if isinstance(loaded, list):
                return list(loaded)
        return []

    def _requests_for_issue(self, *, issue_key: str) -> list[CommitteeRequest]:
        matched: list[CommitteeRequest] = []
        for request in self.store.list_requests(limit=0):
            for task in list(getattr(request, "tasks", []) or []):
                if _clean_text(getattr(task, "issue_key", "")) != issue_key:
                    continue
                matched.append(request)
                break
        return matched

    def _persist_wake_signal_ids(self, *, issue_key: str, wake_signal_ids: list[str]) -> None:
        signal_ids = _clean_list(wake_signal_ids)
        if not signal_ids:
            return
        for request in self._requests_for_issue(issue_key=issue_key):
            merged = _clean_list([*list(getattr(request, "wake_signal_ids", []) or []), *signal_ids])
            if merged == list(getattr(request, "wake_signal_ids", []) or []):
                continue
            self.store.upsert_request(
                request.model_copy(
                    update={
                        "wake_signal_ids": merged,
                        "updated_at": now_ts_ms(),
                    }
                )
            )

    def _candidate_agents_for_issue(self, *, signal: Signal, issue_key: str) -> list[str]:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        ordered: list[str] = []
        seen: set[str] = set()

        def _add(raw: Any) -> None:
            item = _clean_text(raw)
            if not item or item == COMMITTEE_AGENT_ID or item in seen:
                return
            seen.add(item)
            ordered.append(item)

        for agent_id in _clean_list(payload.get("candidate_agents")):
            _add(agent_id)

        record = self.bidding.get(issue_key) if self.bidding is not None else None
        for agent_id in list(getattr(record, "candidates", []) or []):
            _add(agent_id)

        requests = self._requests_for_issue(issue_key=issue_key)
        wake_signal_ids: list[str] = []
        wake_candidates_found = False
        for request in requests:
            for task in list(getattr(request, "tasks", []) or []):
                if _clean_text(getattr(task, "issue_key", "")) != issue_key:
                    continue
                for agent_id in list(getattr(task, "candidate_agents", []) or []):
                    _add(agent_id)
            wake_signal_ids.extend(list(getattr(request, "wake_signal_ids", []) or []))

        for wake_signal_id in _clean_list(wake_signal_ids):
            wake = self.signal_bus.find_signal(wake_signal_id)
            if wake is None or wake.signal_type is not SignalType.WAKE:
                continue
            if _clean_text(wake.issue_key) != issue_key:
                continue
            if _clean_text(wake.from_agent) != COMMITTEE_AGENT_ID:
                continue
            _add(wake.to_agent)
            wake_candidates_found = True

        if not wake_candidates_found:
            fallback_wakes = self.signal_bus.query(
                from_agent=COMMITTEE_AGENT_ID,
                signal_type=SignalType.WAKE,
                issue_key=issue_key,
                include_archive=True,
                limit=64,
            )
            fallback_ids: list[str] = []
            for wake in fallback_wakes:
                if _clean_text(wake.to_agent) == COMMITTEE_AGENT_ID:
                    continue
                _add(wake.to_agent)
                fallback_ids.append(wake.signal_id)
            self._persist_wake_signal_ids(issue_key=issue_key, wake_signal_ids=fallback_ids)

        return ordered

    def _request_from_signal(self, signal: Signal, *, allow_fallback_tasks: bool = True) -> CommitteeRequest:
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        goal = _clean_text(payload.get("goal")) or _clean_text(signal.brief) or "Project request"
        context = _clean_text(payload.get("context")) or _clean_text(signal.brief)
        priority = _clean_text(payload.get("priority")).lower() or "medium"
        if priority not in {"high", "medium", "low"}:
            priority = "medium"

        request_id = _clean_text(payload.get("request_id"))
        if not request_id:
            request_id = f"req_{new_id('committee')}"

        linear_project = (
            _clean_text(payload.get("linear_project"))
            or _clean_text(payload.get("project"))
            or _clean_text(payload.get("project_name"))
            or None
        )
        publish_repo = (
            _clean_text(payload.get("publish_repo"))
            or _clean_text(payload.get("github_repo"))
            or _clean_text(payload.get("repo"))
            or None
        )

        tasks = self._tasks_from_payload(
            payload=payload,
            request_id=request_id,
            goal=goal,
            context=context,
            allow_fallback=allow_fallback_tasks,
        )
        return CommitteeRequest(
            request_id=request_id,
            source_signal_id=signal.signal_id,
            from_agent=signal.from_agent,
            goal=goal,
            context=context,
            constraints=_clean_list(payload.get("constraints")),
            priority=priority,
            linear_project=linear_project,
            publish_repo=publish_repo,
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
        allow_fallback: bool = True,
    ) -> list[CommitteeTask]:
        # Task list in request payload is treated as user/operator hints only.
        # In agentic flow, Committee LLM can adopt, modify, or replace these hints.
        raw_tasks = payload.get("tasks")
        candidates_global = _clean_list(payload.get("candidate_agents"))
        out: list[CommitteeTask] = []

        if isinstance(raw_tasks, list):
            for idx, raw in enumerate(raw_tasks):
                if idx >= max(1, int(self.max_seed_tasks)):
                    break
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
        if not allow_fallback:
            return []

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
        return f"PLH-{goal_token}-{suffix}-{index + 1}"
