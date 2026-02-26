from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..protocol import ArtifactRef


class EnvelopeType(StrEnum):
    REQUEST = "REQUEST"
    ACK = "ACK"
    PROGRESS = "PROGRESS"
    DONE = "DONE"
    BLOCKER = "BLOCKER"
    CANCEL = "CANCEL"
    ERROR = "ERROR"


class MailboxStatus(StrEnum):
    ENQUEUED = "ENQUEUED"
    LOCKED = "LOCKED"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
    DEADLETTER = "DEADLETTER"


@dataclass(frozen=True, slots=True)
class Envelope:
    """
    A2A-lite message envelope.

    Note: `to_agent_id` is a mailbox address. It can represent either:
    - a specific agent instance mailbox (precise routing), or
    - a shared mailbox (multiple consumers compete to lock messages).
    """

    msg_id: str
    idempotency_key: str
    thread_id: str

    from_agent_id: str
    to_agent_id: str
    type: EnvelopeType
    created_at_ms: int

    payload: dict[str, Any] = field(default_factory=dict)
    refs: list[ArtifactRef] = field(default_factory=list)

    task_id: str | None = None
    issue_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "msg_id": self.msg_id,
            "idempotency_key": self.idempotency_key,
            "thread_id": self.thread_id,
            "from_agent_id": self.from_agent_id,
            "to_agent_id": self.to_agent_id,
            "type": self.type.value,
            "created_at_ms": int(self.created_at_ms),
            "payload": dict(self.payload or {}),
            "refs": [r.to_dict() for r in self.refs],
        }
        if self.task_id is not None:
            out["task_id"] = self.task_id
        if self.issue_id is not None:
            out["issue_id"] = self.issue_id
        return out

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Envelope":
        refs_raw = raw.get("refs") or []
        refs: list[ArtifactRef] = []
        if isinstance(refs_raw, list):
            for item in refs_raw:
                if isinstance(item, dict):
                    refs.append(ArtifactRef.from_dict(item))

        return Envelope(
            msg_id=str(raw["msg_id"]),
            idempotency_key=str(raw.get("idempotency_key") or raw["msg_id"]),
            thread_id=str(raw["thread_id"]),
            from_agent_id=str(raw["from_agent_id"]),
            to_agent_id=str(raw["to_agent_id"]),
            type=EnvelopeType(str(raw["type"])),
            created_at_ms=int(raw["created_at_ms"]),
            payload=dict(raw.get("payload") or {}),
            refs=refs,
            task_id=str(raw["task_id"]) if raw.get("task_id") is not None else None,
            issue_id=str(raw["issue_id"]) if raw.get("issue_id") is not None else None,
        )


def dumps_refs(refs: list[ArtifactRef]) -> str:
    return json.dumps([r.to_dict() for r in refs], ensure_ascii=False, sort_keys=True)


def loads_refs(text: str) -> list[ArtifactRef]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    out: list[ArtifactRef] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                out.append(ArtifactRef.from_dict(item))
            except Exception:
                continue
    return out

