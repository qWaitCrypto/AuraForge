from __future__ import annotations

from pathlib import Path

from aura.runtime.bidding import BiddingConfig, BiddingService, parse_bid_from_comment
from aura.runtime.committee import CommitteeCoordinator
from aura.runtime.models.sandbox import Sandbox
from aura.runtime.models.signal import SignalType
from aura.runtime.signal import SignalBus, SignalStore


def _bid_comment(*, agent_id: str, confidence: str = "high", turns: int = 8) -> str:
    return (
        "## 📋 BID\n"
        "```json\n"
        "{\n"
        f'  "agent_id": "{agent_id}",\n'
        f'  "confidence": "{confidence}",\n'
        '  "approach": "Implement scoped changes with tests.",\n'
        '  "deliverables": ["src/auth.py", "tests/test_auth.py"],\n'
        '  "estimated_files": 2,\n'
        f'  "estimated_turns": {turns},\n'
        '  "risks": ["migration window"],\n'
        '  "questions": [],\n'
        '  "relevant_experience": ["auth", "python"]\n'
        "}\n"
        "```\n"
    )


class _FakeSandboxManager:
    def __init__(self) -> None:
        self.created: list[tuple[str, str, str]] = []

    def create(self, *, agent_id: str, issue_key: str, base_branch: str = "main") -> Sandbox:
        self.created.append((agent_id, issue_key, base_branch))
        return Sandbox(
            sandbox_id=f"sb_{issue_key}_{agent_id}",
            agent_id=agent_id,
            issue_key=issue_key,
            worktree_path=f".aura/sandboxes/sb_{issue_key}_{agent_id}",
            branch=f"agent/{issue_key}/{agent_id}/test",
            base_branch=base_branch,
        )


def test_parse_bid_from_comment_strict_json_block() -> None:
    parsed = parse_bid_from_comment(_bid_comment(agent_id="agent.alpha"))
    assert parsed is not None
    assert parsed["agent_id"] == "agent.alpha"
    assert parsed["confidence"] == "high"
    assert parsed["deliverables"] == ["src/auth.py", "tests/test_auth.py"]

    assert parse_bid_from_comment("no bid here") is None


def test_bidding_service_open_collect_and_evaluate_assign(tmp_path: Path) -> None:
    service = BiddingService(project_root=tmp_path, config=BiddingConfig(bidding_timeout_s=60, min_bids=1, max_bids=5))
    issue_key = "AUTO-AUTH-1"
    _ = service.open(issue_key=issue_key, candidates=["agent.alpha", "agent.beta"])
    collected = service.collect(
        issue_key=issue_key,
        comments=[
            {"id": "c1", "body": _bid_comment(agent_id="agent.alpha", confidence="high", turns=6)},
            {"id": "c2", "body": _bid_comment(agent_id="agent.beta", confidence="medium", turns=12)},
        ],
    )
    assert len(collected.bids) == 2

    record, decision = service.evaluate(issue_key=issue_key)
    assert record.phase.value == "evaluating"
    assert decision.action == "assign"
    assert decision.selected_agent == "agent.alpha"

    assigned = service.mark_assigned(issue_key=issue_key, selected_agent=decision.selected_agent or "")
    assert assigned.phase.value == "assigned"


def test_committee_evaluate_bids_sends_task_assigned_signal(tmp_path: Path) -> None:
    bus = SignalBus(store=SignalStore(project_root=tmp_path))
    fake_sandbox = _FakeSandboxManager()
    coordinator = CommitteeCoordinator(
        project_root=tmp_path,
        signal_bus=bus,
        sandbox_manager=fake_sandbox,
    )

    issue_key = "AUTO-DASH-1"
    _ = coordinator.bidding.open(issue_key=issue_key, candidates=["agent.ui"])
    _ = coordinator.collect_bids(
        issue_key=issue_key,
        comments=[{"id": "c1", "body": _bid_comment(agent_id="agent.ui", confidence="high", turns=5)}],
    )
    result = coordinator.evaluate_bids(issue_key=issue_key, base_branch="main")

    assert result["action"] == "assign"
    assert result["selected_agent"] == "agent.ui"
    assert result["sandbox_id"] == "sb_AUTO-DASH-1_agent.ui"
    assert fake_sandbox.created == [("agent.ui", issue_key, "main")]

    assigned_signals = bus.query(
        to_agent="agent.ui",
        signal_type=SignalType.TASK_ASSIGNED,
        issue_key=issue_key,
        include_archive=True,
        limit=0,
    )
    assert len(assigned_signals) == 1
    assert assigned_signals[0].sandbox_id == "sb_AUTO-DASH-1_agent.ui"


def test_bidding_service_uses_custom_ranker(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def _custom_ranker(issue_key: str, bids, record):
        seen["issue_key"] = issue_key
        seen["round"] = record.round
        seen["bid_count"] = len(bids)
        return sorted(bids, key=lambda item: item.agent_id, reverse=True)

    service = BiddingService(
        project_root=tmp_path,
        config=BiddingConfig(bidding_timeout_s=60, min_bids=1, max_bids=5),
        rank_bids=_custom_ranker,
        evaluation_mode="llm_stub",
    )
    issue_key = "AUTO-RANK-1"
    _ = service.open(issue_key=issue_key, candidates=["agent.alpha", "agent.beta"])
    _ = service.collect(
        issue_key=issue_key,
        comments=[
            {"id": "c1", "body": _bid_comment(agent_id="agent.alpha", confidence="high", turns=2)},
            {"id": "c2", "body": _bid_comment(agent_id="agent.beta", confidence="low", turns=20)},
        ],
    )

    _, decision = service.evaluate(issue_key=issue_key)
    assert decision.action == "assign"
    assert decision.selected_agent == "agent.beta"
    assert service.evaluation_mode == "llm_stub"
    assert seen == {"issue_key": issue_key, "round": 1, "bid_count": 2}


def test_committee_evaluation_mode_follows_bidding_service(tmp_path: Path) -> None:
    bus = SignalBus(store=SignalStore(project_root=tmp_path))

    def _custom_ranker(issue_key: str, bids, record):
        del issue_key, record
        return list(bids)

    bidding = BiddingService(
        project_root=tmp_path,
        config=BiddingConfig(bidding_timeout_s=60, min_bids=1, max_bids=5),
        rank_bids=_custom_ranker,
        evaluation_mode="llm_stub",
    )
    coordinator = CommitteeCoordinator(
        project_root=tmp_path,
        signal_bus=bus,
        bidding=bidding,
        sandbox_manager=_FakeSandboxManager(),
    )

    issue_key = "AUTO-EVAL-1"
    _ = coordinator.bidding.open(issue_key=issue_key, candidates=["agent.eval"])
    _ = coordinator.collect_bids(
        issue_key=issue_key,
        comments=[{"id": "c1", "body": _bid_comment(agent_id="agent.eval", confidence="high", turns=4)}],
    )

    result = coordinator.evaluate_bids(issue_key=issue_key)
    assert result["action"] == "assign"
    assert result["evaluation_mode"] == "llm_stub"
