from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .ids import now_ts_ms
from .models.bidding import BidEntry, BiddingPhase, BiddingRecord

_BID_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


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


def _issue_token(issue_key: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", _clean_text(issue_key))
    return cleaned or "unknown_issue"


@dataclass(frozen=True, slots=True)
class BiddingConfig:
    bidding_timeout_s: int = 600
    min_bids: int = 1
    max_bids: int = 5
    max_rebid_rounds: int = 2
    rebid_expand_pool: bool = True


@dataclass(frozen=True, slots=True)
class BiddingDecision:
    action: str  # wait | assign | rebid | failed
    selected_agent: str | None = None
    reason: str | None = None
    rejection_reasons: dict[str, str] | None = None


BidRanker = Callable[[str, list[BidEntry], BiddingRecord], list[BidEntry]]


class BiddingStore:
    def __init__(self, *, project_root: Path) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._root = self._project_root / ".aura" / "state" / "bidding"
        self._archive = self._root / "archive"
        self._root.mkdir(parents=True, exist_ok=True)
        self._archive.mkdir(parents=True, exist_ok=True)

    def _path(self, issue_key: str) -> Path:
        return self._root / f"{_issue_token(issue_key)}.json"

    def save(self, record: BiddingRecord) -> None:
        path = self._path(record.issue_key)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)

    def get(self, issue_key: str) -> BiddingRecord | None:
        path = self._path(issue_key)
        if not path.exists():
            return None
        try:
            return BiddingRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list(self, *, phase: BiddingPhase | None = None, limit: int = 200) -> list[BiddingRecord]:
        out: list[BiddingRecord] = []
        for path in sorted(self._root.glob("*.json")):
            if path.name == "archive":
                continue
            try:
                item = BiddingRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if phase is not None and item.phase is not phase:
                continue
            out.append(item)
        out.sort(key=lambda item: item.updated_at)
        if limit > 0 and len(out) > limit:
            return out[-limit:]
        return out

    def archive_record(self, issue_key: str) -> None:
        path = self._path(issue_key)
        if not path.exists():
            return
        target = self._archive / path.name
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        tmp.replace(target)
        path.unlink(missing_ok=True)


def parse_bid_from_comment(comment_body: str) -> dict[str, Any] | None:
    body = _clean_text(comment_body)
    if not body:
        return None
    match = _BID_JSON_BLOCK_RE.search(body)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(1))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    required = {"agent_id", "confidence", "approach", "deliverables"}
    if not required.issubset(raw.keys()):
        return None

    agent_id = _clean_text(raw.get("agent_id"))
    confidence = _clean_text(raw.get("confidence")).lower()
    approach = _clean_text(raw.get("approach"))
    deliverables = _clean_list(raw.get("deliverables"))
    if not agent_id or confidence not in {"high", "medium", "low"} or not approach or not deliverables:
        return None

    estimated_files = raw.get("estimated_files")
    estimated_turns = raw.get("estimated_turns")
    return {
        "agent_id": agent_id,
        "confidence": confidence,
        "approach": approach,
        "deliverables": deliverables,
        "estimated_files": int(estimated_files) if isinstance(estimated_files, int) and estimated_files >= 0 else 0,
        "estimated_turns": int(estimated_turns) if isinstance(estimated_turns, int) and estimated_turns >= 0 else 0,
        "risks": _clean_list(raw.get("risks")),
        "questions": _clean_list(raw.get("questions")),
        "relevant_experience": _clean_list(raw.get("relevant_experience")),
    }


def _extract_comment_fields(comment: Any) -> tuple[str, str | None, str | None]:
    if isinstance(comment, str):
        return comment, None, None
    if not isinstance(comment, dict):
        return "", None, None
    body = ""
    # Linear comment payloads expose Markdown in `body`; `content/text` are compatibility fallbacks.
    for key in ("body", "content", "text"):
        candidate = comment.get(key)
        if isinstance(candidate, str) and candidate.strip():
            body = candidate
            break
    comment_id = _clean_text(comment.get("id") or comment.get("comment_id")) or None
    comment_ref = _clean_text(comment.get("url") or comment.get("ref")) or None
    return body, comment_id, comment_ref


def _score_bid(entry: BidEntry) -> int:
    # MVP heuristic only. Keep deterministic ranking now and replace with LLM bid-eval later.
    confidence = {"high": 3, "medium": 2, "low": 1}.get(entry.confidence, 0)
    deliverables = min(10, len(entry.deliverables))
    approach_bonus = 1 if len(entry.approach) >= 32 else 0
    turn_penalty = min(50, max(0, int(entry.estimated_turns)))
    file_penalty = min(20, max(0, int(entry.estimated_files // 4)))
    return confidence * 100 + deliverables * 5 + approach_bonus * 10 - turn_penalty - file_penalty


def _heuristic_rank_bids(issue_key: str, bids: list[BidEntry], record: BiddingRecord) -> list[BidEntry]:
    del issue_key, record
    return sorted(bids, key=_score_bid, reverse=True)


class BiddingService:
    def __init__(
        self,
        *,
        project_root: Path,
        store: BiddingStore | None = None,
        config: BiddingConfig | None = None,
        rank_bids: BidRanker | None = None,
        evaluation_mode: str = "heuristic_mvp",
    ) -> None:
        self._project_root = project_root.expanduser().resolve()
        self.store = store or BiddingStore(project_root=self._project_root)
        self.config = config or BiddingConfig()
        self._rank_bids = rank_bids or _heuristic_rank_bids
        self.evaluation_mode = _clean_text(evaluation_mode).lower() or "heuristic_mvp"

    def open(self, *, issue_key: str, candidates: list[str], now_ms: int | None = None) -> BiddingRecord:
        now = int(now_ms or now_ts_ms())
        record = BiddingRecord(
            issue_key=_clean_text(issue_key),
            phase=BiddingPhase.BIDDING,
            candidates=_clean_list(candidates),
            wake_sent_at=now,
            bidding_deadline=now + int(self.config.bidding_timeout_s * 1000),
            updated_at=now,
        )
        self.store.save(record)
        return record

    def get(self, issue_key: str) -> BiddingRecord | None:
        return self.store.get(issue_key)

    def collect(self, *, issue_key: str, comments: list[Any], now_ms: int | None = None) -> BiddingRecord:
        record = self.store.get(issue_key)
        if record is None:
            record = self.open(issue_key=issue_key, candidates=[], now_ms=now_ms)

        # Intentional: one active bid per agent per issue; later bids replace earlier bids from same agent.
        merged: dict[str, BidEntry] = {entry.agent_id: entry for entry in record.bids}
        for idx, comment in enumerate(comments):
            body, comment_id, comment_ref = _extract_comment_fields(comment)
            parsed = parse_bid_from_comment(body)
            if parsed is None:
                continue
            parsed["linear_comment_id"] = comment_id or f"comment_{idx + 1}"
            parsed["comment_ref"] = comment_ref
            merged[parsed["agent_id"]] = BidEntry.model_validate(parsed)

        bids = sorted(merged.values(), key=lambda item: item.received_at)
        phase = record.phase
        if len(bids) >= int(self.config.max_bids):
            phase = BiddingPhase.EVALUATING
        updated = record.model_copy(update={"bids": bids, "phase": phase, "updated_at": int(now_ms or now_ts_ms())})
        self.store.save(updated)
        return updated

    def evaluate(self, *, issue_key: str, now_ms: int | None = None) -> tuple[BiddingRecord, BiddingDecision]:
        record = self.store.get(issue_key)
        if record is None:
            raise ValueError(f"No bidding record for issue: {issue_key}")

        now = int(now_ms or now_ts_ms())
        bids = list(record.bids)
        if record.phase is BiddingPhase.ASSIGNED and record.selected_agent:
            return record, BiddingDecision(action="assign", selected_agent=record.selected_agent, reason="already_assigned")

        if len(bids) < int(self.config.min_bids):
            deadline = int(record.bidding_deadline or 0)
            if deadline > 0 and now < deadline:
                pending = record.model_copy(update={"phase": BiddingPhase.BIDDING, "updated_at": now})
                self.store.save(pending)
                return pending, BiddingDecision(action="wait", reason="waiting_for_more_bids")
            if int(record.round) <= int(self.config.max_rebid_rounds):
                # Intentional: keep prior valid bids across rebid rounds so existing candidates remain eligible.
                rebid = record.model_copy(
                    update={
                        "phase": BiddingPhase.REBID,
                        "round": int(record.round) + 1,
                        "updated_at": now,
                        "bidding_deadline": now + int(self.config.bidding_timeout_s * 1000),
                    }
                )
                self.store.save(rebid)
                return rebid, BiddingDecision(action="rebid", reason="insufficient_bids")
            failed = record.model_copy(update={"phase": BiddingPhase.FAILED, "updated_at": now})
            self.store.save(failed)
            return failed, BiddingDecision(action="failed", reason="insufficient_bids_after_retries")

        ranked = list(self._rank_bids(issue_key, bids, record))
        if not ranked:
            ranked = _heuristic_rank_bids(issue_key, bids, record)
        winner = ranked[0]
        rejection: dict[str, str] = {}
        for entry in ranked[1:]:
            rejection[entry.agent_id] = "Lower fit score than selected bid."

        evaluated = record.model_copy(
            update={
                "phase": BiddingPhase.EVALUATING,
                "selected_agent": winner.agent_id,
                "rejection_reasons": rejection,
                "updated_at": now,
            }
        )
        self.store.save(evaluated)
        return evaluated, BiddingDecision(
            action="assign",
            selected_agent=winner.agent_id,
            reason="highest_evaluation_score",
            rejection_reasons=rejection,
        )

    def evaluate_with_ranker(
        self,
        *,
        issue_key: str,
        rank_bids: BidRanker,
        evaluation_mode: str | None = None,
        now_ms: int | None = None,
    ) -> tuple[BiddingRecord, BiddingDecision]:
        previous_ranker = self._rank_bids
        previous_mode = self.evaluation_mode
        self._rank_bids = rank_bids
        if isinstance(evaluation_mode, str) and evaluation_mode.strip():
            self.evaluation_mode = _clean_text(evaluation_mode).lower()
        try:
            return self.evaluate(issue_key=issue_key, now_ms=now_ms)
        finally:
            self._rank_bids = previous_ranker
            self.evaluation_mode = previous_mode

    def mark_assigned(self, *, issue_key: str, selected_agent: str, now_ms: int | None = None) -> BiddingRecord:
        record = self.store.get(issue_key)
        if record is None:
            raise ValueError(f"No bidding record for issue: {issue_key}")
        now = int(now_ms or now_ts_ms())
        updated = record.model_copy(
            update={
                "phase": BiddingPhase.ASSIGNED,
                "selected_agent": _clean_text(selected_agent),
                "assigned_at": now,
                "updated_at": now,
            }
        )
        self.store.save(updated)
        return updated
