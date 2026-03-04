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
from .ids import new_id, now_ts_ms
from .notifications import NotificationStore
from .sandbox import SandboxManager
from .models.signal import Signal, SignalType
from .signal import SignalBus

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
        return summary

    def post_process(
        self,
        *,
        signal: Signal,
        session_id: str | None,
        run_status: str,
        run_id: str,
        error: str | None,
    ) -> None:
        self.store.append_activity(
            event="committee_signal_completed",
            payload={
                "mode": _clean_text(self.coordinator_mode).lower() or COMMITTEE_DECOMPOSITION_MODE,
                "kind": self._signal_kind(signal),
                "signal_id": signal.signal_id,
                "session_id": session_id,
                "run_id": _clean_text(run_id) or None,
                "run_status": _clean_text(run_status) or None,
                "error": _clean_text(error) or None,
            },
        )

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
