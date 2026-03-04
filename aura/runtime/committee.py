from __future__ import annotations

import asyncio
import importlib.resources
import inspect
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from .bidding import BidRanker, BiddingConfig, BiddingDecision, BiddingService
from .ids import new_id, now_ts_ms
from .mcp.config import McpServerConfig, load_mcp_config
from .models.bidding import BidEntry, BiddingPhase, BiddingRecord
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


def _to_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def _priority_to_linear(value: str) -> int | None:
    normalized = _clean_text(value).lower()
    if normalized == "high":
        return 2
    if normalized == "medium":
        return 3
    if normalized == "low":
        return 4
    return None


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
    linear_team_id: str | None = None
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
    bid_llm_evaluator: Callable[..., dict[str, Any]] | None = None
    completion_verification_mode: str = "rules_mvp"
    issue_creator: Callable[..., Any] | None = None
    linear_comments_reader: Callable[..., Any] | None = None
    linear_mcp_server: str = "linear"
    linear_team_id: str | None = None
    delivery_publisher: Callable[..., dict[str, Any]] | None = None
    auto_publish_on_accept: bool = False

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
        self.auto_publish_on_accept = _to_bool(
            self.auto_publish_on_accept,
            default=_to_bool(os.getenv("AURAFORGE_AUTO_PUBLISH_ON_ACCEPT"), default=False),
        )
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
        if signal.to_agent != COMMITTEE_AGENT_ID:
            return {"handled": False, "reason": "not_for_committee"}
        payload = signal.payload if isinstance(signal.payload, dict) else {}
        payload_type = _clean_text(payload.get("type")).lower()
        if is_project_request_signal(signal):
            return await self._handle_project_request(signal)
        if signal.signal_type is SignalType.NOTIFY and payload_type == "bid_comments":
            return await self._handle_bid_comments(signal)
        if signal.signal_type is SignalType.NOTIFY and payload_type in {"check_bids", "bid_check"}:
            return await self._handle_bid_check(signal)
        if signal.signal_type is SignalType.NOTIFY and self._is_task_completed_signal(signal):
            return self._handle_task_completed(signal)
        return {"handled": False, "reason": "unsupported_signal"}

    async def _handle_project_request(self, signal: Signal) -> dict[str, Any]:
        # Design choice (MVP): coordinator path owns project_request handling end-to-end.
        # AgentRunner short-circuits when `handled=True`, so this signal is not processed again
        # by the LLM chat loop in the same turn.
        request = self._request_from_signal(signal)
        request = await self._materialize_issue_keys(request=request)
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

    async def _handle_bid_comments(self, signal: Signal) -> dict[str, Any]:
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

    async def _handle_bid_check(self, signal: Signal) -> dict[str, Any]:
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
            if not comments and _to_bool(payload.get("fetch_linear_comments"), default=True):
                comments = await self._fetch_linear_comments_for_issue(issue_key=issue_key)
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

    def _compose_linear_issue_description(self, *, request: CommitteeRequest, task: CommitteeTask) -> str:
        lines: list[str] = []
        lines.append(task.description or task.title)
        if task.acceptance_criteria:
            lines.append("")
            lines.append("Acceptance criteria:")
            for criterion in task.acceptance_criteria:
                lines.append(f"- {criterion}")
        if task.required_capabilities:
            lines.append("")
            lines.append("Required capabilities:")
            for capability in task.required_capabilities:
                lines.append(f"- {capability}")
        if request.constraints:
            lines.append("")
            lines.append("Constraints:")
            for constraint in request.constraints:
                lines.append(f"- {constraint}")
        if request.references:
            lines.append("")
            lines.append("References:")
            for ref in request.references:
                lines.append(f"- {ref}")
        return "\n".join(lines).strip()

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _linear_server_config(self) -> tuple[str, McpServerConfig] | None:
        try:
            cfg = load_mcp_config(project_root=self.project_root)
        except Exception:
            return None
        preferred = _clean_text(self.linear_mcp_server)
        if preferred:
            server = cfg.servers.get(preferred)
            if server is not None and server.enabled and _clean_text(server.command):
                return preferred, server
        candidates: list[tuple[str, McpServerConfig]] = []
        for name, server in sorted(cfg.servers.items()):
            if not server.enabled or not _clean_text(server.command):
                continue
            haystack = " ".join([name, server.command, " ".join(server.args)]).lower()
            if "linear" in haystack:
                candidates.append((name, server))
        if candidates:
            return candidates[0]
        return None

    @staticmethod
    def _tool_name_variants(name: str) -> set[str]:
        base = _clean_text(name).lower()
        if not base:
            return set()
        return {
            base,
            base.replace("-", "_"),
            base.replace("_", "-"),
        }

    @staticmethod
    def _extract_remote_name(*, runtime_name: str, fn_name: str, prefix: str) -> str:
        if fn_name.startswith(prefix):
            return fn_name[len(prefix) :]
        if runtime_name.startswith(prefix):
            return runtime_name[len(prefix) :]
        if fn_name.startswith("mcp__"):
            parts = fn_name.split("__", 2)
            if len(parts) == 3:
                return parts[2]
        if runtime_name.startswith("mcp__"):
            parts = runtime_name.split("__", 2)
            if len(parts) == 3:
                return parts[2]
        return fn_name or runtime_name

    async def _linear_mcp_call(self, *, tool_names: tuple[str, ...], arguments: dict[str, Any]) -> Any:
        selected = self._linear_server_config()
        if selected is None:
            return None
        server_name, server = selected

        try:
            from agno.tools.mcp.mcp import MCPTools
            from mcp import StdioServerParameters
            from mcp.client.stdio import get_default_environment
            from agno.run import RunContext
            from agno.tools.function import FunctionCall
            from agno.tools.function import ToolResult as AgnoToolResult
        except Exception:
            return None

        wanted: set[str] = set()
        for item in tool_names:
            wanted.update(self._tool_name_variants(item))
        if not wanted:
            return None

        prefix = f"mcp__{server_name}__"
        params = StdioServerParameters(
            command=server.command,
            args=list(server.args),
            env={**get_default_environment(), **dict(server.env)},
            cwd=server.cwd,
        )
        toolkit = MCPTools(
            server_params=params,
            transport="stdio",
            timeout_seconds=int(max(1.0, float(server.timeout_s))),
            tool_name_prefix=prefix,
        )
        async with toolkit as entered:
            async_functions = entered.get_async_functions()
            target = None
            for runtime_name, fn in async_functions.items():
                fn_name = _clean_text(getattr(fn, "name", runtime_name) or runtime_name)
                remote_name = self._extract_remote_name(runtime_name=str(runtime_name), fn_name=fn_name, prefix=prefix)
                variants = set()
                variants.update(self._tool_name_variants(str(runtime_name)))
                variants.update(self._tool_name_variants(fn_name))
                variants.update(self._tool_name_variants(remote_name))
                if variants & wanted:
                    target = fn
                    break
            if target is None:
                return None

            try:
                target._run_context = RunContext(
                    run_id=f"committee_{new_id('run')}",
                    session_id=f"committee_{new_id('session')}",
                    metadata={},
                )
            except Exception:
                pass

            call = FunctionCall(
                function=target,
                arguments=dict(arguments or {}),
                call_id=f"call_{new_id('mcp')}",
            )
            try:
                result = await call.aexecute()
            except Exception:
                return None
            if _clean_text(getattr(result, "status", "")).lower() != "success":
                return None
            raw = getattr(result, "result", None)
            if isinstance(raw, AgnoToolResult):
                return {"content": raw.content}
            return raw

    @staticmethod
    def _parse_json_text(text: str) -> Any:
        raw = _clean_text(text)
        if not raw:
            return None
        if raw.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
            if match:
                raw = _clean_text(match.group(1))
        if not raw or raw[0] not in {"{", "["}:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    @classmethod
    def _normalize_mcp_payload(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if len(value) == 1 and "content" in value and isinstance(value.get("content"), str):
                parsed = cls._parse_json_text(value.get("content", ""))
                if parsed is not None:
                    return parsed
            return value
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            parsed = cls._parse_json_text(value)
            if parsed is not None:
                return parsed
        return value

    @classmethod
    def _extract_rows(cls, value: Any) -> list[dict[str, Any]]:
        normalized = cls._normalize_mcp_payload(value)
        if isinstance(normalized, list):
            return [item for item in normalized if isinstance(item, dict)]
        if isinstance(normalized, dict):
            rows: list[dict[str, Any]] = []
            for key in ("items", "nodes", "results", "data", "issues", "teams", "comments"):
                child = normalized.get(key)
                if isinstance(child, list):
                    rows.extend([item for item in child if isinstance(item, dict)])
                elif isinstance(child, dict):
                    rows.extend(cls._extract_rows(child))
            if rows:
                return rows
            return [normalized]
        return []

    @staticmethod
    def _extract_identifier(value: Any) -> str:
        if isinstance(value, str):
            text = _clean_text(value)
            if re.match(r"^[A-Za-z][A-Za-z0-9]+-\d+$", text):
                return text.upper()
            match = re.search(r"\b([A-Za-z][A-Za-z0-9]+-\d+)\b", text)
            if match:
                return _clean_text(match.group(1)).upper()
            return ""
        if isinstance(value, dict):
            for key in ("identifier", "issue_key", "issueKey", "key", "id", "url", "content"):
                found = CommitteeCoordinator._extract_identifier(value.get(key))
                if found:
                    return found
            for child in value.values():
                found = CommitteeCoordinator._extract_identifier(child)
                if found:
                    return found
            return ""
        if isinstance(value, list):
            for item in value:
                found = CommitteeCoordinator._extract_identifier(item)
                if found:
                    return found
            return ""
        return ""

    def _resolve_linear_team_hint(self, *, request: CommitteeRequest) -> str:
        if _clean_text(request.linear_team_id):
            return _clean_text(request.linear_team_id)
        if _clean_text(self.linear_team_id):
            return _clean_text(self.linear_team_id)
        for item in request.references:
            text = _clean_text(item)
            if not text:
                continue
            if text.lower().startswith("linear_team_id:"):
                return _clean_text(text.split(":", 1)[1])
        return ""

    async def _resolve_linear_team_for_create(self, *, request: CommitteeRequest) -> str:
        hinted = self._resolve_linear_team_hint(request=request)
        if hinted:
            return hinted
        raw = await self._linear_mcp_call(
            tool_names=("list_teams",),
            arguments={"includeArchived": False, "limit": 50},
        )
        rows = self._extract_rows(raw)
        if not rows:
            return ""
        selected: dict[str, Any] | None = None
        for row in rows:
            if _to_bool(row.get("isDefault"), default=False):
                selected = row
                break
        if selected is None:
            selected = rows[0]
        for key in ("key", "id", "name", "slug"):
            text = _clean_text(selected.get(key))
            if text:
                return text
        return ""

    async def _create_linear_issue_key(self, *, request: CommitteeRequest, task: CommitteeTask) -> str:
        args: dict[str, Any] = {
            "title": (task.title or "Committee task")[:250],
            "description": self._compose_linear_issue_description(request=request, task=task),
        }
        team = await self._resolve_linear_team_for_create(request=request)
        if team:
            args["team"] = team
        priority = _priority_to_linear(request.priority)
        if isinstance(priority, int):
            args["priority"] = priority
        raw = await self._linear_mcp_call(tool_names=("save_issue", "create_issue"), arguments=args)
        return self._extract_identifier(raw)

    async def _find_linear_issue_id(self, *, issue_key: str) -> str:
        query = _clean_text(issue_key)
        if not query:
            return ""
        raw = await self._linear_mcp_call(
            tool_names=("list_issues",),
            arguments={"query": query, "limit": 20},
        )
        rows = self._extract_rows(raw)
        normalized = query.upper()
        fallback_id = ""
        for row in rows:
            identifier = _clean_text(row.get("identifier") or row.get("issue_key") or row.get("key")).upper()
            issue_id = _clean_text(row.get("id"))
            if issue_id and not fallback_id:
                fallback_id = issue_id
            if issue_id and identifier == normalized:
                return issue_id
        if fallback_id:
            return fallback_id

        direct = await self._linear_mcp_call(
            tool_names=("get_issue",),
            arguments={"id": query},
        )
        direct_rows = self._extract_rows(direct)
        for row in direct_rows:
            issue_id = _clean_text(row.get("id"))
            if issue_id:
                return issue_id
        return ""

    async def _fetch_linear_comments_for_issue(self, *, issue_key: str) -> list[dict[str, Any]]:
        reader = self.linear_comments_reader
        if callable(reader):
            try:
                rows = await self._maybe_await(reader(issue_key=issue_key, coordinator=self))
            except TypeError:
                try:
                    rows = await self._maybe_await(reader(issue_key))
                except Exception:
                    rows = []
            except Exception:
                rows = []
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict) or isinstance(item, str)]

        issue_id = await self._find_linear_issue_id(issue_key=issue_key)
        if not issue_id:
            return []
        raw = await self._linear_mcp_call(
            tool_names=("list_comments",),
            arguments={"issueId": issue_id},
        )
        comments_nodes = self._extract_rows(raw)
        out: list[dict[str, Any]] = []
        for row in comments_nodes:
            if not isinstance(row, dict):
                continue
            body = _clean_text(row.get("body") or row.get("content") or row.get("text"))
            if not body:
                continue
            out.append(
                {
                    "id": _clean_text(row.get("id")) or None,
                    "body": body,
                    "url": _clean_text(row.get("url")) or None,
                }
            )
        return out

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

    def _run_git(self, *, cwd: Path, args: list[str]) -> tuple[bool, str]:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            detail = _clean_text(proc.stderr or proc.stdout)
            return False, detail or str(proc.returncode)
        return True, _clean_text(proc.stdout)

    def _run_cmd(self, *, cwd: Path, argv: list[str]) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError:
            return False, f"{argv[0]} not found"
        if proc.returncode != 0:
            detail = _clean_text(proc.stderr or proc.stdout)
            return False, detail or str(proc.returncode)
        return True, _clean_text(proc.stdout)

    @staticmethod
    def _first_url(text: str) -> str:
        match = re.search(r"https?://[^\s)]+", str(text or ""))
        if match is None:
            return ""
        return _clean_text(match.group(0))

    def _github_create_pr(
        self,
        *,
        worktree: Path,
        head_branch: str,
        base_branch: str,
        issue_key: str,
        summary: str,
    ) -> tuple[str | None, str | None]:
        title = f"[{issue_key}] Delivery from Committee"
        body_lines = [
            f"Automated delivery for `{issue_key}`.",
            "",
            "Summary:",
            summary or "Completed by worker automation.",
        ]
        body = "\n".join(body_lines)
        ok, auth_out = self._run_cmd(cwd=worktree, argv=["gh", "auth", "status", "-h", "github.com"])
        if not ok:
            return None, f"gh_auth_unavailable:{auth_out}"

        ok, create_out = self._run_cmd(
            cwd=worktree,
            argv=[
                "gh",
                "pr",
                "create",
                "--head",
                head_branch,
                "--base",
                base_branch,
                "--title",
                title[:250],
                "--body",
                body,
            ],
        )
        if not ok:
            if "already exists" not in create_out.lower():
                return None, f"gh_pr_create_failed:{create_out}"
            lookup_ok, lookup_out = self._run_cmd(
                cwd=worktree,
                argv=["gh", "pr", "list", "--head", head_branch, "--state", "open", "--json", "url", "--jq", ".[0].url"],
            )
            if not lookup_ok:
                return None, f"gh_pr_exists_but_lookup_failed:{lookup_out}"
            pr_url = self._first_url(lookup_out)
            if not pr_url:
                return None, "gh_pr_exists_but_missing_url"
            return pr_url, None

        pr_url = self._first_url(create_out)
        if not pr_url:
            return None, "gh_pr_missing_url"
        return pr_url, None

    def _publish_delivery(
        self,
        *,
        issue_key: str,
        signal: Signal,
        payload: dict[str, Any],
        summary: str,
    ) -> dict[str, Any]:
        publisher = self.delivery_publisher
        if callable(publisher):
            try:
                result = publisher(issue_key=issue_key, signal=signal, payload=payload, summary=summary, coordinator=self)
            except TypeError:
                try:
                    result = publisher(issue_key, signal, payload, summary)
                except Exception as exc:
                    return {"ok": False, "error": f"delivery_publisher_failed:{exc}"}
            except Exception as exc:
                return {"ok": False, "error": f"delivery_publisher_failed:{exc}"}
            return result if isinstance(result, dict) else {"ok": False, "error": "delivery_publisher_invalid_result"}

        auto_publish = _to_bool(payload.get("auto_publish"), default=self.auto_publish_on_accept)
        if not auto_publish:
            return {"ok": False, "reason": "auto_publish_disabled"}

        sandbox_id = _clean_text(signal.sandbox_id)
        if not sandbox_id:
            return {"ok": False, "error": "missing_sandbox_id"}
        manager = self.sandbox_manager
        if manager is None:
            return {"ok": False, "error": "sandbox_manager_unavailable"}
        getter = getattr(manager, "get", None)
        if not callable(getter):
            return {"ok": False, "error": "sandbox_lookup_unavailable"}
        try:
            sandbox = getter(sandbox_id)
        except Exception as exc:
            return {"ok": False, "error": f"sandbox_lookup_failed:{exc}"}
        if not isinstance(sandbox, Sandbox):
            return {"ok": False, "error": "sandbox_not_found"}

        worktree_abs = (self.project_root / sandbox.worktree_path).resolve()
        if not worktree_abs.exists():
            return {"ok": False, "error": "sandbox_worktree_missing"}

        pushed = False
        push_error: str | None = None
        ok, out = self._run_git(cwd=worktree_abs, args=["push", "-u", "origin", sandbox.branch])
        if ok:
            pushed = True
        else:
            push_error = out

        pr_url = _clean_text(payload.get("pr_url")) or ""
        pr_error: str | None = None
        if pushed and not pr_url:
            base_branch = _clean_text(payload.get("base_branch")) or _clean_text(sandbox.base_branch) or "main"
            created_url, err = self._github_create_pr(
                worktree=worktree_abs,
                head_branch=sandbox.branch,
                base_branch=base_branch,
                issue_key=issue_key,
                summary=summary,
            )
            pr_url = _clean_text(created_url)
            pr_error = err

        return {
            "ok": bool(pushed),
            "pushed": pushed,
            "push_error": push_error,
            "pr_url": pr_url or None,
            "pr_error": pr_error,
            "branch": sandbox.branch,
            "sandbox_id": sandbox.sandbox_id,
        }

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
            publish = self._publish_delivery(issue_key=issue_key, signal=signal, payload=payload, summary=summary)
            pr_url = _clean_text(payload.get("pr_url")) or _clean_text(publish.get("pr_url")) or None
            notification = self.notifications.create(
                notification_type=NotificationType.TASK_COMPLETED,
                title=f"{issue_key} completed",
                summary=summary,
                issue_key=issue_key,
                pr_url=pr_url,
                details={
                    "agent_id": worker_agent,
                    "sandbox_id": signal.sandbox_id,
                    "signal_id": signal.signal_id,
                    "verification_mode": verification.get("mode"),
                    "delivery": publish,
                },
            )
            pr_notification_id: str | None = None
            if pr_url:
                pr_note = self.notifications.create(
                    notification_type=NotificationType.PR_CREATED,
                    title=f"{issue_key} PR created",
                    summary=f"PR opened for {issue_key}.",
                    issue_key=issue_key,
                    pr_url=pr_url,
                    details={"agent_id": worker_agent, "sandbox_id": signal.sandbox_id},
                )
                pr_notification_id = pr_note.notification_id
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
                    "pr_notification_id": pr_notification_id,
                    "pr_url": pr_url,
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
                    "pr_notification_id": pr_notification_id,
                    "notify_signal_id": user_signal.signal_id,
                    "verification_mode": verification.get("mode"),
                    "verification_summary": summary,
                    "cleanup_requested": bool(payload.get("cleanup_sandbox")),
                    "cleanup_cleaned": cleanup_cleaned,
                    "cleanup_skipped_reason": cleanup_skipped_reason,
                    "cleanup_error": cleanup_error,
                    "delivery": publish,
                },
            )
            return {
                "handled": True,
                "kind": "task_completed",
                "decision": "accept",
                "issue_key": issue_key,
                "notification_id": notification.notification_id,
                "pr_notification_id": pr_notification_id,
                "notify_signal_id": user_signal.signal_id,
                "verification_mode": verification.get("mode"),
                "cleanup_cleaned": cleanup_cleaned,
                "cleanup_skipped_reason": cleanup_skipped_reason,
                "cleanup_error": cleanup_error,
                "pr_url": pr_url,
                "delivery": publish,
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

    async def _create_issue_key(self, *, request: CommitteeRequest, task: CommitteeTask) -> tuple[str, str]:
        creator = self.issue_creator
        if callable(creator):
            try:
                created = await self._maybe_await(creator(request=request, task=task, coordinator=self))
            except TypeError:
                try:
                    created = await self._maybe_await(creator(request, task))
                except Exception:
                    created = None
            except Exception:
                created = None
            created_key = _clean_text(created)
            if created_key:
                return created_key, "linear_mcp"
        created_key = await self._create_linear_issue_key(request=request, task=task)
        if created_key:
            return created_key, "linear_mcp"
        return task.issue_key, task.issue_key_source

    async def _materialize_issue_keys(self, *, request: CommitteeRequest) -> CommitteeRequest:
        updated_tasks: list[CommitteeTask] = []
        replaced: list[dict[str, str]] = []
        for task in request.tasks:
            if not task.issue_key_is_placeholder:
                updated_tasks.append(task)
                continue
            issue_key, source = await self._create_issue_key(request=request, task=task)
            issue_key_value = _clean_text(issue_key) or task.issue_key
            is_placeholder = issue_key_value == task.issue_key and bool(task.issue_key_is_placeholder)
            next_task = task.model_copy(
                update={
                    "issue_key": issue_key_value,
                    "issue_key_is_placeholder": is_placeholder,
                    "issue_key_source": source,
                }
            )
            if next_task.issue_key != task.issue_key:
                replaced.append({"from": task.issue_key, "to": next_task.issue_key, "source": source})
            updated_tasks.append(next_task)

        if replaced:
            self.store.append_activity(
                event="issue_keys_materialized",
                payload={
                    "request_id": request.request_id,
                    "replacements": replaced,
                },
            )
        return request.model_copy(update={"tasks": updated_tasks, "updated_at": now_ts_ms()})

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

        linear_team_id = (
            _clean_text(payload.get("linear_team_id"))
            or _clean_text(payload.get("team_id"))
            or _clean_text(payload.get("teamId"))
            or _clean_text(payload.get("linear_team"))
            or None
        )

        tasks = self._tasks_from_payload(payload=payload, request_id=request_id, goal=goal, context=context)
        return CommitteeRequest(
            request_id=request_id,
            source_signal_id=signal.signal_id,
            from_agent=signal.from_agent,
            goal=goal,
            context=context,
            constraints=_clean_list(payload.get("constraints")),
            priority=priority,
            linear_team_id=linear_team_id,
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

    @staticmethod
    def _read_prompt_asset(name: str) -> str:
        try:
            return (
                importlib.resources.files("aura.runtime")
                .joinpath("prompts", name)
                .read_text(encoding="utf-8", errors="replace")
                .strip()
            )
        except Exception:
            return ""

    @classmethod
    def _render_prompt_asset(cls, name: str, vars: dict[str, Any]) -> str:
        template = cls._read_prompt_asset(name)
        if not template:
            return ""
        rendered = template
        for key, value in vars.items():
            rendered = rendered.replace("{" + str(key) + "}", str(value))
        return rendered.strip()

    def _build_bid_eval_prompt(self, *, issue_key: str, bids: list[BidEntry]) -> str:
        serialized_bids = [item.model_dump(mode="json") for item in bids]
        bids_json = json.dumps(serialized_bids, ensure_ascii=False, indent=2, sort_keys=True)
        template = self._render_prompt_asset(
            "committee_bid_eval.md",
            {
                "issue_key": issue_key,
                "bids_json": bids_json,
            },
        )
        if template:
            return template
        return f"Issue key: {issue_key}\n\nBids JSON:\n{bids_json}"

    @staticmethod
    def _ranked_agents_from_eval_result(result: dict[str, Any], bids: list[BidEntry]) -> tuple[list[str], str, dict[str, str]]:
        by_agent = {item.agent_id: item for item in bids}
        ranked: list[str] = []

        selected_agent = _clean_text(result.get("selected_agent"))
        if selected_agent and selected_agent in by_agent:
            ranked.append(selected_agent)

        runner_up = _clean_text(result.get("runner_up"))
        if runner_up and runner_up in by_agent and runner_up not in ranked:
            ranked.append(runner_up)

        raw_ranked = result.get("ranked_agent_ids")
        if isinstance(raw_ranked, list):
            for item in raw_ranked:
                candidate = _clean_text(item)
                if candidate and candidate in by_agent and candidate not in ranked:
                    ranked.append(candidate)

        for bid in bids:
            if bid.agent_id not in ranked:
                ranked.append(bid.agent_id)

        reason = _clean_text(result.get("reason")) or "llm_bid_eval"
        raw_rejections = result.get("rejection_reasons")
        rejections: dict[str, str] = {}
        if isinstance(raw_rejections, dict):
            for agent_id, value in raw_rejections.items():
                key = _clean_text(agent_id)
                note = _clean_text(value)
                if key and key in by_agent and note:
                    rejections[key] = note
        return ranked, reason, rejections

    def _build_llm_ranker(self, *, meta: dict[str, Any]) -> BidRanker:
        def _rank(issue_key: str, bids: list[BidEntry], record: BiddingRecord) -> list[BidEntry]:
            del record
            prompt = self._build_bid_eval_prompt(issue_key=issue_key, bids=bids)
            meta["prompt_chars"] = len(prompt)
            evaluator = self.bid_llm_evaluator
            if not callable(evaluator):
                meta["fallback_reason"] = "llm_evaluator_unavailable"
                return []

            payload_bids = [item.model_dump(mode="json") for item in bids]
            try:
                raw = evaluator(prompt=prompt, issue_key=issue_key, bids=payload_bids)
            except TypeError:
                try:
                    raw = evaluator(prompt)
                except Exception as exc:
                    meta["fallback_reason"] = f"llm_eval_error:{exc}"
                    return []
            except Exception as exc:
                meta["fallback_reason"] = f"llm_eval_error:{exc}"
                return []

            if not isinstance(raw, dict):
                meta["fallback_reason"] = "llm_eval_invalid_payload"
                return []

            ranked_agents, reason, rejections = self._ranked_agents_from_eval_result(raw, bids)
            by_agent = {item.agent_id: item for item in bids}
            ranked_bids: list[BidEntry] = [by_agent[agent_id] for agent_id in ranked_agents if agent_id in by_agent]
            if not ranked_bids:
                meta["fallback_reason"] = "llm_eval_no_valid_winner"
                return []
            meta["llm_reason"] = reason
            if rejections:
                meta["llm_rejections"] = rejections
            return ranked_bids

        return _rank

    def evaluate_bids(self, *, issue_key: str, base_branch: str = "main") -> dict[str, Any]:
        llm_meta: dict[str, Any] = {}
        use_llm = self.bid_evaluation_mode.startswith("llm")
        if use_llm:
            ranker = self._build_llm_ranker(meta=llm_meta)
            record, decision = self.bidding.evaluate_with_ranker(
                issue_key=issue_key,
                rank_bids=ranker,
                evaluation_mode=self.bid_evaluation_mode,
            )
        else:
            record, decision = self.bidding.evaluate(issue_key=issue_key)

        if decision.action == "assign" and decision.selected_agent:
            llm_reason = _clean_text(llm_meta.get("llm_reason"))
            if llm_reason:
                decision = BiddingDecision(
                    action=decision.action,
                    selected_agent=decision.selected_agent,
                    reason=llm_reason,
                    rejection_reasons=decision.rejection_reasons,
                )
            llm_rejections = llm_meta.get("llm_rejections")
            if isinstance(llm_rejections, dict) and llm_rejections:
                updated = record.model_copy(update={"rejection_reasons": llm_rejections, "updated_at": now_ts_ms()})
                self.bidding.store.save(updated)
                record = updated
                decision = BiddingDecision(
                    action=decision.action,
                    selected_agent=decision.selected_agent,
                    reason=decision.reason,
                    rejection_reasons=llm_rejections,
                )

        payload: dict[str, Any] = {
            "ok": True,
            "issue_key": issue_key,
            "phase": record.phase.value,
            "action": decision.action,
            "selected_agent": decision.selected_agent,
            "reason": decision.reason,
            "evaluation_mode": self.bid_evaluation_mode,
        }
        if llm_meta:
            payload["llm_meta"] = dict(llm_meta)

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
