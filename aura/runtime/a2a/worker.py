from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ..error_codes import ErrorCode
from ..ids import new_id, now_ts_ms
from ..protocol import ArtifactRef
from .protocol import Envelope, EnvelopeType
from .runtime import A2ARuntime


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    mailbox_id: str
    consumer_id: str
    idle_timeout_s: float = 120.0
    poll_interval_ms: int = 250
    lease_ms: int = 60_000


class Worker:
    """
    Minimal worker that handles REQUEST envelopes by sending ACK + DONE.

    This is a bootstrap implementation to validate end-to-end mailbox execution.
    More complex workers can be plugged in later (policy packs, tool runtime, etc).
    """

    def __init__(self, *, runtime: A2ARuntime, config: WorkerConfig) -> None:
        self._rt = runtime
        self._cfg = config

    def run(self) -> int:
        last_activity = time.monotonic()

        while True:
            self._rt.mailbox.sweep_expired_locks(limit=1000)

            claim = self._rt.mailbox.claim_next(
                mailbox_id=self._cfg.mailbox_id,
                consumer_id=self._cfg.consumer_id,
                lease_ms=self._cfg.lease_ms,
            )
            if claim is None:
                if time.monotonic() - last_activity >= float(self._cfg.idle_timeout_s):
                    return 0
                time.sleep(max(0.01, int(self._cfg.poll_interval_ms)) / 1000.0)
                continue

            last_activity = time.monotonic()

            try:
                self._handle_message(claim.envelope, lock_token=claim.lock_token)
            except Exception as e:
                # Last-resort: fail and allow retry/deadletter.
                self._rt.mailbox.fail(
                    msg_id=claim.envelope.msg_id,
                    lock_token=claim.lock_token,
                    error=str(e),
                    error_code=ErrorCode.UNKNOWN,
                )

    def _handle_message(self, envelope: Envelope, *, lock_token: str) -> None:
        if envelope.type != EnvelopeType.REQUEST:
            # For now, treat non-REQUEST messages as no-ops.
            self._rt.mailbox.complete(msg_id=envelope.msg_id, lock_token=lock_token)
            return

        # ACK
        self._send(
            to_agent_id=envelope.from_agent_id,
            idempotency_key=f"{envelope.msg_id}:ACK",
            thread_id=envelope.thread_id,
            task_id=envelope.task_id,
            issue_id=envelope.issue_id,
            type=EnvelopeType.ACK,
            payload={
                "ack_of": envelope.msg_id,
                "accepted": True,
                "consumer_id": self._cfg.consumer_id,
            },
            refs=[],
        )

        # DONE (always produces an artifact ref so downstream can verify without hidden context).
        result = self._compute_done_payload(envelope)
        ref = self._put_result_artifact(envelope=envelope, result=result)
        self._send(
            to_agent_id=envelope.from_agent_id,
            idempotency_key=f"{envelope.msg_id}:DONE",
            thread_id=envelope.thread_id,
            task_id=envelope.task_id,
            issue_id=envelope.issue_id,
            type=EnvelopeType.DONE,
            payload=result,
            refs=[ref],
        )

        self._rt.mailbox.complete(msg_id=envelope.msg_id, lock_token=lock_token)

    def _send(
        self,
        *,
        to_agent_id: str,
        idempotency_key: str,
        thread_id: str,
        task_id: str | None,
        issue_id: str | None,
        type: EnvelopeType,
        payload: dict[str, Any],
        refs: list[ArtifactRef],
    ) -> None:
        now = now_ts_ms()
        env = Envelope(
            msg_id=new_id("msg"),
            idempotency_key=str(idempotency_key),
            thread_id=str(thread_id),
            task_id=task_id,
            issue_id=issue_id,
            from_agent_id=str(self._cfg.mailbox_id),
            to_agent_id=str(to_agent_id),
            type=type,
            created_at_ms=now,
            payload=dict(payload or {}),
            refs=list(refs or []),
        )
        self._rt.mailbox.enqueue(env)

    @staticmethod
    def _compute_done_payload(envelope: Envelope) -> dict[str, Any]:
        return {
            "done_of": envelope.msg_id,
            "status": "ok",
            "echo": envelope.payload,
        }

    def _put_result_artifact(self, *, envelope: Envelope, result: dict[str, Any]) -> ArtifactRef:
        content = json.dumps(
            {
                "msg_id": envelope.msg_id,
                "thread_id": envelope.thread_id,
                "from_agent_id": envelope.from_agent_id,
                "to_agent_id": envelope.to_agent_id,
                "payload": envelope.payload,
                "result": result,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        return self._rt.artifact_store.put(
            content,
            kind="a2a_worker_output",
            meta={
                "summary": f"A2A DONE for msg_id={envelope.msg_id}",
            },
        )

