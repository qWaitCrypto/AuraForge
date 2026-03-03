from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models.signal import SignalType
from ..sandbox import SandboxManager
from ..signal import SignalBus
from .policy import PolicyGate

_DEFAULT_AGENT_IDS = ["market_worker"]
_DEFAULT_FANOUT = 1


@dataclass(frozen=True, slots=True)
class DispatchRequest:
    issue_key: str
    brief: str
    signal_type: SignalType = SignalType.WAKE
    agent_id: str | None = None
    base_branch: str = "main"
    payload: dict | None = None


@dataclass(frozen=True, slots=True)
class DispatchResult:
    dispatched: bool
    signal_id: str | None = None
    sandbox_id: str | None = None
    agent_id: str | None = None
    dispatched_agents: list[str] = field(default_factory=list)
    rejection_reason: str | None = None


class Dispatcher:
    def __init__(
        self,
        *,
        project_root: Path,
        signal_bus: SignalBus,
        sandbox_manager: SandboxManager,
        policy_gate: PolicyGate,
    ) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._signal_bus = signal_bus
        self._sandbox_manager = sandbox_manager
        self._policy_gate = policy_gate
        self._routing_path = self._project_root / ".aura" / "config" / "routing.json"
        self._routing_path.parent.mkdir(parents=True, exist_ok=True)

    def load_routing(self) -> dict[str, Any]:
        default = {
            "routes": [],
            "default_agent_ids": list(_DEFAULT_AGENT_IDS),
            "default_fanout": _DEFAULT_FANOUT,
        }
        if not self._routing_path.exists():
            return default
        try:
            raw = json.loads(self._routing_path.read_text(encoding="utf-8"))
        except Exception:
            return default
        if not isinstance(raw, dict):
            return default
        routes_raw = raw.get("routes")
        routes = routes_raw if isinstance(routes_raw, list) else []
        default_agent_ids = _normalize_agent_ids(raw.get("default_agent_ids")) or list(_DEFAULT_AGENT_IDS)
        default_fanout = _coerce_positive_int(raw.get("default_fanout"), default=_DEFAULT_FANOUT)
        return {
            "routes": routes,
            "default_agent_ids": default_agent_ids,
            "default_fanout": default_fanout,
        }

    def route(self, issue_key: str, labels: list[str] | None = None) -> list[str]:
        issue = str(issue_key or "").strip()
        normalized_labels = {str(item or "").strip() for item in (labels or []) if str(item or "").strip()}
        routing = self.load_routing()
        default_fanout = _coerce_positive_int(routing.get("default_fanout"), default=_DEFAULT_FANOUT)

        for raw in routing.get("routes", []):
            if not isinstance(raw, dict):
                continue
            match = raw.get("match")
            if not _match_route(match=match, issue_key=issue, labels=normalized_labels):
                continue
            agent_ids = _normalize_agent_ids(raw.get("agent_ids"))
            if not agent_ids:
                continue
            fanout = _coerce_positive_int(raw.get("fanout"), default=default_fanout)
            return agent_ids[:fanout]

        default_agents = _normalize_agent_ids(routing.get("default_agent_ids")) or list(_DEFAULT_AGENT_IDS)
        return default_agents[:default_fanout]

    def dispatch(self, request: DispatchRequest) -> DispatchResult:
        issue = str(request.issue_key or "").strip()
        brief = str(request.brief or "").strip()
        base_branch = str(request.base_branch or "main").strip() or "main"
        payload = request.payload if isinstance(request.payload, dict) else None
        if not issue:
            return DispatchResult(dispatched=False, rejection_reason="dispatch:issue_key_required")
        if not brief:
            return DispatchResult(dispatched=False, rejection_reason="dispatch:brief_required")

        if request.signal_type is SignalType.WAKE:
            labels = _extract_labels(payload)
            if isinstance(request.agent_id, str) and request.agent_id.strip():
                candidates = [request.agent_id.strip()]
            else:
                candidates = self.route(issue, labels=labels)

            dispatched_agents: list[str] = []
            first_signal_id: str | None = None
            rejection_reason: str | None = None
            for candidate in candidates:
                check = self._policy_gate.check(agent_id=candidate, issue_key=issue)
                if not check.allowed:
                    if rejection_reason is None:
                        rejection_reason = check.reason
                    continue
                try:
                    signal = self._signal_bus.send(
                        from_agent="control.dispatcher",
                        to_agent=candidate,
                        signal_type=SignalType.WAKE,
                        brief=brief,
                        issue_key=issue,
                        payload=payload,
                    )
                except Exception as exc:
                    if rejection_reason is None:
                        rejection_reason = f"dispatch:send_failed:{exc}"
                    continue
                dispatched_agents.append(candidate)
                if first_signal_id is None:
                    first_signal_id = signal.signal_id

            if not dispatched_agents:
                return DispatchResult(
                    dispatched=False,
                    dispatched_agents=[],
                    rejection_reason=rejection_reason or "dispatch:no_eligible_agents",
                )
            return DispatchResult(
                dispatched=True,
                signal_id=first_signal_id,
                dispatched_agents=dispatched_agents,
            )

        if isinstance(request.agent_id, str) and request.agent_id.strip():
            target_agent = request.agent_id.strip()
        else:
            routed = self.route(issue, labels=_extract_labels(payload))
            target_agent = routed[0] if routed else ""
        if not target_agent:
            return DispatchResult(dispatched=False, rejection_reason="dispatch:agent_not_found")

        check = self._policy_gate.check(agent_id=target_agent, issue_key=issue)
        if not check.allowed:
            return DispatchResult(dispatched=False, agent_id=target_agent, rejection_reason=check.reason)

        sandbox = None
        if request.signal_type is SignalType.TASK_ASSIGNED:
            try:
                sandbox = self._sandbox_manager.create(
                    agent_id=target_agent,
                    issue_key=issue,
                    base_branch=base_branch,
                )
            except Exception as exc:
                return DispatchResult(
                    dispatched=False,
                    agent_id=target_agent,
                    rejection_reason=f"dispatch:sandbox_create_failed:{exc}",
                )
        try:
            signal = self._signal_bus.send(
                from_agent="control.dispatcher",
                to_agent=target_agent,
                signal_type=request.signal_type,
                brief=brief,
                issue_key=issue,
                sandbox_id=sandbox.sandbox_id if sandbox is not None else None,
                payload=payload,
            )
        except Exception as exc:
            if sandbox is not None:
                try:
                    self._sandbox_manager.destroy(sandbox.sandbox_id)
                except Exception:
                    pass
            return DispatchResult(
                dispatched=False,
                agent_id=target_agent,
                rejection_reason=f"dispatch:send_failed:{exc}",
            )

        return DispatchResult(
            dispatched=True,
            signal_id=signal.signal_id,
            sandbox_id=sandbox.sandbox_id if sandbox is not None else None,
            agent_id=target_agent,
            dispatched_agents=[target_agent],
        )

    def dispatch_from_linear_event(self, event: dict[str, Any]) -> DispatchResult:
        payload = event if isinstance(event, dict) else {}
        issue_key = _extract_first_text(
            payload,
            [
                ("issue_key",),
                ("issue", "identifier"),
                ("data", "identifier"),
                ("data", "issue", "identifier"),
                ("identifier",),
            ],
        )
        brief = _extract_first_text(
            payload,
            [
                ("brief",),
                ("title",),
                ("issue", "title"),
                ("data", "title"),
                ("data", "issue", "title"),
            ],
        ) or "wake"
        labels = _extract_labels(payload)
        request = DispatchRequest(
            issue_key=issue_key or "",
            brief=brief,
            signal_type=SignalType.WAKE,
            agent_id=None,
            payload={"event": payload, "labels": labels},
        )
        return self.dispatch(request)


def _normalize_agent_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in value:
        cleaned = str(raw or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _coerce_positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default


def _match_route(*, match: Any, issue_key: str, labels: set[str]) -> bool:
    if not isinstance(match, dict):
        return False
    for key, raw_expected in match.items():
        expected = str(raw_expected or "").strip()
        if not expected:
            return False
        if key == "label":
            if expected not in labels:
                return False
        elif key == "project_prefix":
            if not issue_key.startswith(expected):
                return False
        elif key == "issue_key":
            if issue_key != expected:
                return False
        else:
            # Unknown match key: fail-safe reject.
            return False
    return True


def _extract_labels(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    candidates: list[Any] = []
    candidates.append(payload.get("labels"))
    data = payload.get("data")
    if isinstance(data, dict):
        candidates.append(data.get("labels"))
        issue_data = data.get("issue")
        if isinstance(issue_data, dict):
            candidates.append(issue_data.get("labels"))
    issue_top = payload.get("issue")
    if isinstance(issue_top, dict):
        candidates.append(issue_top.get("labels"))

    out: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str):
                label = item.strip()
            elif isinstance(item, dict):
                label = str(item.get("name") or item.get("label") or "").strip()
            else:
                label = ""
            if not label or label in seen:
                continue
            seen.add(label)
            out.append(label)
    return out


def _extract_first_text(payload: dict[str, Any], paths: list[tuple[str, ...]]) -> str | None:
    for path in paths:
        current: Any = payload
        for part in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
        if isinstance(current, str) and current.strip():
            return current.strip()
    return None
