from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from ..ids import now_ts_ms
from .agent_status import AgentStatusTracker, AgentState


class ControlPolicy(BaseModel):
    max_total_active_sandboxes: int = Field(default=20, ge=0)
    max_agent_active_issues: int = Field(default=3, ge=0)
    max_sandboxes_per_issue: int = Field(default=2, ge=0)
    signal_unresponded_timeout_ms: int = Field(default=1_800_000, ge=1)
    sandbox_idle_timeout_ms: int = Field(default=3_600_000, ge=1)
    agent_cooldown_after_failure_ms: int = Field(default=300_000, ge=1)
    max_failures_per_agent_24h: int = Field(default=5, ge=0)


@dataclass(frozen=True, slots=True)
class PolicyCheckResult:
    allowed: bool
    reason: str | None = None


class PolicyGate:
    def __init__(
        self,
        *,
        project_root: Path,
        status_tracker: AgentStatusTracker,
    ) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._status_tracker = status_tracker
        self._policy_path = self._project_root / ".aura" / "config" / "policy.json"
        self._policy_path.parent.mkdir(parents=True, exist_ok=True)

    def load_policy(self) -> ControlPolicy:
        if not self._policy_path.exists():
            return ControlPolicy()
        try:
            raw = json.loads(self._policy_path.read_text(encoding="utf-8"))
        except Exception:
            return ControlPolicy()
        if not isinstance(raw, dict):
            return ControlPolicy()
        merged = ControlPolicy().model_dump(mode="python")
        merged.update(raw)
        try:
            return ControlPolicy.model_validate(merged)
        except Exception:
            return ControlPolicy()

    def save_policy(self, policy: ControlPolicy) -> None:
        payload = policy.model_dump(mode="json")
        self._policy_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._policy_path.with_suffix(self._policy_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._policy_path)

    def check(self, *, agent_id: str, issue_key: str) -> PolicyCheckResult:
        agent = str(agent_id or "").strip()
        issue = str(issue_key or "").strip()
        if not agent:
            return PolicyCheckResult(allowed=False, reason="policy:agent_id_required")
        if not issue:
            return PolicyCheckResult(allowed=False, reason="policy:issue_key_required")

        policy = self.load_policy()
        now = now_ts_ms()
        agent_record = self._status_tracker.refresh(
            agent,
            sandbox_idle_timeout_ms=policy.sandbox_idle_timeout_ms,
        )
        all_by_agent = {item.agent_id: item for item in self._status_tracker.list_all()}
        all_by_agent[agent_record.agent_id] = agent_record
        all_records = list(all_by_agent.values())

        if (
            agent_record.state is AgentState.COOLING
            and isinstance(agent_record.cooling_until_ts, int)
            and agent_record.cooling_until_ts > now
        ):
            return PolicyCheckResult(
                allowed=False,
                reason=f"policy:agent_cooling:{agent_record.cooling_until_ts}",
            )

        agent_active_issues = len({key for key in agent_record.active_issue_keys if key.strip()})
        if agent_active_issues >= policy.max_agent_active_issues:
            return PolicyCheckResult(
                allowed=False,
                reason=f"policy:max_agent_active_issues:{agent_active_issues}/{policy.max_agent_active_issues}",
            )

        total_active_sandboxes = sum(len(item.active_sandbox_ids) for item in all_records)
        if total_active_sandboxes >= policy.max_total_active_sandboxes:
            return PolicyCheckResult(
                allowed=False,
                reason=f"policy:max_total_active_sandboxes:{total_active_sandboxes}/{policy.max_total_active_sandboxes}",
            )

        issue_active_count = sum(1 for item in all_records if issue in item.active_issue_keys)
        if issue_active_count >= policy.max_sandboxes_per_issue:
            return PolicyCheckResult(
                allowed=False,
                reason=f"policy:max_sandboxes_per_issue:{issue_active_count}/{policy.max_sandboxes_per_issue}",
            )

        if (
            policy.max_failures_per_agent_24h > 0
            and agent_record.failure_count_24h >= policy.max_failures_per_agent_24h
        ):
            return PolicyCheckResult(
                allowed=False,
                reason=(
                    "policy:max_failures_per_agent_24h:"
                    f"{agent_record.failure_count_24h}/{policy.max_failures_per_agent_24h}"
                ),
            )

        return PolicyCheckResult(allowed=True, reason=None)
