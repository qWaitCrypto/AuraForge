from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from ..ids import now_ts_ms


class BiddingPhase(StrEnum):
    OPEN = "open"
    BIDDING = "bidding"
    EVALUATING = "evaluating"
    ASSIGNED = "assigned"
    REBID = "rebid"
    FAILED = "failed"


class BidEntry(BaseModel):
    agent_id: str
    confidence: str
    approach: str
    deliverables: list[str]
    estimated_files: int = Field(default=0, ge=0)
    estimated_turns: int = Field(default=0, ge=0)
    risks: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    relevant_experience: list[str] = Field(default_factory=list)
    linear_comment_id: str | None = None
    comment_ref: str | None = None
    received_at: int = Field(default_factory=now_ts_ms)

    @field_validator("agent_id", "approach")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("field must be non-empty")
        return cleaned

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: str) -> str:
        cleaned = str(value or "").strip().lower()
        if cleaned not in {"high", "medium", "low"}:
            raise ValueError("confidence must be one of high|medium|low")
        return cleaned

    @field_validator("deliverables", "risks", "questions", "relevant_experience")
    @classmethod
    def _normalize_list(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            item = str(raw or "").strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out


class BiddingRecord(BaseModel):
    issue_key: str
    phase: BiddingPhase = BiddingPhase.OPEN
    candidates: list[str] = Field(default_factory=list)
    bids: list[BidEntry] = Field(default_factory=list)
    selected_agent: str | None = None
    rejection_reasons: dict[str, str] = Field(default_factory=dict)
    round: int = 1
    wake_sent_at: int | None = None
    bidding_deadline: int | None = None
    assigned_at: int | None = None
    created_at: int = Field(default_factory=now_ts_ms)
    updated_at: int = Field(default_factory=now_ts_ms)

    @field_validator("issue_key")
    @classmethod
    def _validate_issue_key(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("issue_key must be non-empty")
        return cleaned

    @field_validator("candidates")
    @classmethod
    def _normalize_candidates(cls, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            item = str(raw or "").strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out
