from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..error_codes import ErrorCode
from ..event_bus import EventBus
from ..ids import new_id, now_ts_ms
from ..protocol import Event, EventKind
from .protocol import Envelope, EnvelopeType, MailboxStatus, loads_refs


SCHEMA_VERSION = 1
A2A_EVENT_SCHEMA_VERSION = "a2a-lite/0.1"


@dataclass(frozen=True, slots=True)
class EnqueueOutcome:
    msg_id: str
    thread_id: str
    deduped: bool


@dataclass(frozen=True, slots=True)
class ClaimOutcome:
    envelope: Envelope
    locked_by: str
    lock_token: str
    lease_expires_at_ms: int
    attempts: int
    max_attempts: int


class MailboxStore:
    def __init__(
        self,
        *,
        db_path: Path,
        event_bus: EventBus | None = None,
        event_session_id: str = "a2a",
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_bus = event_bus
        self._event_session_id = event_session_id
        self._ensure_schema()

    def enqueue(
        self,
        envelope: Envelope,
        *,
        max_attempts: int = 8,
    ) -> EnqueueOutcome:
        now = now_ts_ms()
        payload_json = json.dumps(envelope.payload or {}, ensure_ascii=False, sort_keys=True)
        refs_json = json.dumps([r.to_dict() for r in envelope.refs], ensure_ascii=False, sort_keys=True)

        with self._connect() as conn:
            self._begin_immediate(conn)
            try:
                conn.execute(
                    """
                    INSERT INTO envelopes (
                      msg_id,
                      idempotency_key,
                      thread_id,
                      task_id,
                      issue_id,
                      from_agent_id,
                      to_agent_id,
                      type,
                      created_at_ms,
                      payload_json,
                      refs_json,
                      status,
                      attempts,
                      max_attempts,
                      next_visible_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        envelope.msg_id,
                        envelope.idempotency_key,
                        envelope.thread_id,
                        envelope.task_id,
                        envelope.issue_id,
                        envelope.from_agent_id,
                        envelope.to_agent_id,
                        envelope.type.value,
                        int(envelope.created_at_ms),
                        payload_json,
                        refs_json,
                        MailboxStatus.ENQUEUED.value,
                        0,
                        int(max_attempts),
                        int(now),
                    ),
                )
            except sqlite3.IntegrityError:
                # Idempotency or duplicate msg_id: return existing msg_id for this idempotency_key.
                row = conn.execute(
                    """
                    SELECT msg_id, thread_id
                    FROM envelopes
                    WHERE to_agent_id = ? AND idempotency_key = ?
                    """,
                    (envelope.to_agent_id, envelope.idempotency_key),
                ).fetchone()
                if row and isinstance(row[0], str) and row[0].strip():
                    msg_id = str(row[0])
                    thread_id = str(row[1] or "")
                    conn.commit()
                    self._emit(
                        kind=EventKind.A2A_MESSAGE_DEDUPED.value,
                        request_id=msg_id,
                        payload={
                            "msg_id": msg_id,
                            "thread_id": thread_id,
                            "to_agent_id": envelope.to_agent_id,
                            "idempotency_key": envelope.idempotency_key,
                        },
                    )
                    return EnqueueOutcome(msg_id=msg_id, thread_id=thread_id, deduped=True)
                raise

            conn.commit()

        self._emit(
            kind=EventKind.A2A_MESSAGE_ENQUEUED.value,
            request_id=envelope.msg_id,
            payload={
                "msg_id": envelope.msg_id,
                "thread_id": envelope.thread_id,
                "from_agent_id": envelope.from_agent_id,
                "to_agent_id": envelope.to_agent_id,
                "type": envelope.type.value,
                "idempotency_key": envelope.idempotency_key,
            },
        )
        return EnqueueOutcome(msg_id=envelope.msg_id, thread_id=envelope.thread_id, deduped=False)

    def sweep_expired_locks(self, *, limit: int = 1000) -> int:
        now = now_ts_ms()
        with self._connect() as conn:
            self._begin_immediate(conn)
            cur = conn.execute(
                """
                UPDATE envelopes
                SET status = ?,
                    locked_by = NULL,
                    lock_token = NULL,
                    lease_expires_at_ms = NULL,
                    next_visible_at_ms = ?
                WHERE msg_id IN (
                  SELECT msg_id
                  FROM envelopes
                  WHERE status = ?
                    AND lease_expires_at_ms IS NOT NULL
                    AND lease_expires_at_ms <= ?
                  LIMIT ?
                )
                """,
                (
                    MailboxStatus.ENQUEUED.value,
                    int(now),
                    MailboxStatus.LOCKED.value,
                    int(now),
                    int(limit),
                ),
            )
            changed = int(cur.rowcount or 0)
            conn.commit()

        if changed:
            self._emit(
                kind=EventKind.A2A_LOCK_RECOVERED.value,
                request_id=None,
                payload={"recovered": changed},
            )
        return changed

    def claim_next(
        self,
        *,
        mailbox_id: str,
        consumer_id: str,
        lease_ms: int = 60_000,
    ) -> ClaimOutcome | None:
        now = now_ts_ms()
        lease_expires = now + int(lease_ms)

        with self._connect() as conn:
            self._begin_immediate(conn)

            # Enforce per-consumer serial: at most one active lock per consumer.
            row = conn.execute(
                """
                SELECT 1
                FROM envelopes
                WHERE status = ?
                  AND locked_by = ?
                  AND lease_expires_at_ms IS NOT NULL
                  AND lease_expires_at_ms > ?
                LIMIT 1
                """,
                (
                    MailboxStatus.LOCKED.value,
                    consumer_id,
                    int(now),
                ),
            ).fetchone()
            if row is not None:
                conn.commit()
                return None

            # Best-effort lock recovery (kept in-transaction for correctness).
            conn.execute(
                """
                UPDATE envelopes
                SET status = ?,
                    locked_by = NULL,
                    lock_token = NULL,
                    lease_expires_at_ms = NULL,
                    next_visible_at_ms = ?
                WHERE status = ?
                  AND lease_expires_at_ms IS NOT NULL
                  AND lease_expires_at_ms <= ?
                """,
                (
                    MailboxStatus.ENQUEUED.value,
                    int(now),
                    MailboxStatus.LOCKED.value,
                    int(now),
                ),
            )

            candidate = conn.execute(
                """
                SELECT msg_id
                FROM envelopes
                WHERE to_agent_id = ?
                  AND status IN (?, ?)
                  AND next_visible_at_ms <= ?
                ORDER BY created_at_ms ASC
                LIMIT 1
                """,
                (
                    mailbox_id,
                    MailboxStatus.ENQUEUED.value,
                    MailboxStatus.FAILED.value,
                    int(now),
                ),
            ).fetchone()
            if not candidate:
                conn.commit()
                return None

            msg_id = str(candidate[0])
            lock_token = new_id("lock")
            cur = conn.execute(
                """
                UPDATE envelopes
                SET status = ?,
                    locked_by = ?,
                    lock_token = ?,
                    lease_expires_at_ms = ?
                WHERE msg_id = ?
                  AND status IN (?, ?)
                """,
                (
                    MailboxStatus.LOCKED.value,
                    consumer_id,
                    lock_token,
                    int(lease_expires),
                    msg_id,
                    MailboxStatus.ENQUEUED.value,
                    MailboxStatus.FAILED.value,
                ),
            )
            if int(cur.rowcount or 0) != 1:
                conn.commit()
                return None

            row = conn.execute(
                """
                SELECT
                  msg_id,
                  idempotency_key,
                  thread_id,
                  task_id,
                  issue_id,
                  from_agent_id,
                  to_agent_id,
                  type,
                  created_at_ms,
                  payload_json,
                  refs_json,
                  attempts,
                  max_attempts
                FROM envelopes
                WHERE msg_id = ?
                """,
                (msg_id,),
            ).fetchone()
            conn.commit()

        if not row:
            return None

        envelope = Envelope(
            msg_id=str(row[0]),
            idempotency_key=str(row[1]),
            thread_id=str(row[2]),
            task_id=str(row[3]) if row[3] is not None else None,
            issue_id=str(row[4]) if row[4] is not None else None,
            from_agent_id=str(row[5]),
            to_agent_id=str(row[6]),
            type=EnvelopeType(str(row[7])),
            created_at_ms=int(row[8]),
            payload=json.loads(row[9]) if row[9] else {},
            refs=loads_refs(str(row[10]) if row[10] else "[]"),
        )
        attempts = int(row[11] or 0)
        max_attempts = int(row[12] or 0)

        self._emit(
            kind=EventKind.A2A_MESSAGE_LOCKED.value,
            request_id=envelope.msg_id,
            payload={
                "msg_id": envelope.msg_id,
                "thread_id": envelope.thread_id,
                "from_agent_id": envelope.from_agent_id,
                "to_agent_id": envelope.to_agent_id,
                "locked_by": consumer_id,
                "lease_expires_at_ms": int(lease_expires),
                "attempts": attempts,
                "max_attempts": max_attempts,
            },
        )

        return ClaimOutcome(
            envelope=envelope,
            locked_by=consumer_id,
            lock_token=lock_token,
            lease_expires_at_ms=int(lease_expires),
            attempts=attempts,
            max_attempts=max_attempts,
        )

    def complete(self, *, msg_id: str, lock_token: str) -> bool:
        now = now_ts_ms()
        with self._connect() as conn:
            self._begin_immediate(conn)
            row = conn.execute(
                """
                SELECT thread_id, to_agent_id
                FROM envelopes
                WHERE msg_id = ?
                  AND status = ?
                  AND lock_token = ?
                """,
                (
                    msg_id,
                    MailboxStatus.LOCKED.value,
                    lock_token,
                ),
            ).fetchone()
            if not row:
                conn.commit()
                return False
            thread_id = str(row[0] or "")
            to_agent_id = str(row[1] or "")
            cur = conn.execute(
                """
                UPDATE envelopes
                SET status = ?,
                    processed_at_ms = ?,
                    locked_by = NULL,
                    lock_token = NULL,
                    lease_expires_at_ms = NULL
                WHERE msg_id = ?
                  AND status = ?
                  AND lock_token = ?
                """,
                (
                    MailboxStatus.PROCESSED.value,
                    int(now),
                    msg_id,
                    MailboxStatus.LOCKED.value,
                    lock_token,
                ),
            )
            changed = int(cur.rowcount or 0)
            conn.commit()

        if changed:
            self._emit(
                kind=EventKind.A2A_MESSAGE_PROCESSED.value,
                request_id=msg_id,
                payload={
                    "msg_id": msg_id,
                    "thread_id": thread_id,
                    "to_agent_id": to_agent_id,
                },
            )
        return changed == 1

    def fail(
        self,
        *,
        msg_id: str,
        lock_token: str,
        error: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN,
    ) -> MailboxStatus | None:
        now = now_ts_ms()
        with self._connect() as conn:
            self._begin_immediate(conn)
            row = conn.execute(
                """
                SELECT attempts, max_attempts, thread_id, to_agent_id
                FROM envelopes
                WHERE msg_id = ?
                  AND status = ?
                  AND lock_token = ?
                """,
                (
                    msg_id,
                    MailboxStatus.LOCKED.value,
                    lock_token,
                ),
            ).fetchone()
            if not row:
                conn.commit()
                return None

            attempts = int(row[0] or 0) + 1
            max_attempts = int(row[1] or 8)
            thread_id = str(row[2] or "")
            to_agent_id = str(row[3] or "")

            if attempts >= max_attempts:
                new_status = MailboxStatus.DEADLETTER
                next_visible = now
            else:
                new_status = MailboxStatus.FAILED
                next_visible = now + _compute_backoff_ms(attempts)

            conn.execute(
                """
                UPDATE envelopes
                SET status = ?,
                    attempts = ?,
                    next_visible_at_ms = ?,
                    last_error_code = ?,
                    last_error = ?,
                    locked_by = NULL,
                    lock_token = NULL,
                    lease_expires_at_ms = NULL
                WHERE msg_id = ?
                  AND lock_token = ?
                """,
                (
                    new_status.value,
                    int(attempts),
                    int(next_visible),
                    str(error_code.value),
                    str(error),
                    msg_id,
                    lock_token,
                ),
            )
            conn.commit()

        kind = (
            EventKind.A2A_MESSAGE_DEADLETTERED.value
            if new_status is MailboxStatus.DEADLETTER
            else EventKind.A2A_MESSAGE_FAILED.value
        )
        self._emit(
            kind=kind,
            request_id=msg_id,
            payload={
                "msg_id": msg_id,
                "thread_id": thread_id,
                "to_agent_id": to_agent_id,
                "status": new_status.value,
                "attempts": attempts,
                "max_attempts": max_attempts,
                "next_visible_at_ms": int(next_visible),
                "error_code": str(error_code.value),
                "error": str(error),
            },
        )
        return new_status

    def replay_deadletter(self, *, msg_id: str) -> bool:
        now = now_ts_ms()
        with self._connect() as conn:
            self._begin_immediate(conn)
            row = conn.execute(
                """
                SELECT thread_id, to_agent_id
                FROM envelopes
                WHERE msg_id = ?
                  AND status = ?
                """,
                (
                    msg_id,
                    MailboxStatus.DEADLETTER.value,
                ),
            ).fetchone()
            if not row:
                conn.commit()
                return False
            thread_id = str(row[0] or "")
            to_agent_id = str(row[1] or "")
            cur = conn.execute(
                """
                UPDATE envelopes
                SET status = ?,
                    attempts = 0,
                    next_visible_at_ms = ?,
                    locked_by = NULL,
                    lock_token = NULL,
                    lease_expires_at_ms = NULL
                WHERE msg_id = ?
                  AND status = ?
                """,
                (
                    MailboxStatus.ENQUEUED.value,
                    int(now),
                    msg_id,
                    MailboxStatus.DEADLETTER.value,
                ),
            )
            changed = int(cur.rowcount or 0)
            conn.commit()

        if changed:
            self._emit(
                kind=EventKind.A2A_MESSAGE_REPLAYED.value,
                request_id=msg_id,
                payload={
                    "msg_id": msg_id,
                    "thread_id": thread_id,
                    "to_agent_id": to_agent_id,
                },
            )
        return changed == 1

    def stats(self, *, mailbox_id: str | None = None) -> dict[str, Any]:
        where = ""
        params: tuple[Any, ...] = ()
        if mailbox_id is not None:
            where = "WHERE to_agent_id = ?"
            params = (mailbox_id,)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT status, COUNT(*) AS n
                FROM envelopes
                {where}
                GROUP BY status
                """,
                params,
            ).fetchall()

        counts: dict[str, int] = {s.value: 0 for s in MailboxStatus}
        for status, n in rows or []:
            if isinstance(status, str):
                counts[status] = int(n or 0)

        return {
            "mailbox_id": mailbox_id,
            "counts": counts,
        }

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cur = conn.execute("PRAGMA user_version").fetchone()
            user_version = int(cur[0]) if cur else 0
            if user_version == SCHEMA_VERSION:
                return
            if user_version not in {0, SCHEMA_VERSION}:
                raise RuntimeError(f"Unsupported mailbox schema version: {user_version}")

            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")

            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS envelopes (
                  msg_id TEXT PRIMARY KEY,
                  idempotency_key TEXT NOT NULL,
                  thread_id TEXT NOT NULL,
                  task_id TEXT,
                  issue_id TEXT,
                  from_agent_id TEXT NOT NULL,
                  to_agent_id TEXT NOT NULL,
                  type TEXT NOT NULL,
                  created_at_ms INTEGER NOT NULL,
                  payload_json TEXT NOT NULL,
                  refs_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  max_attempts INTEGER NOT NULL DEFAULT 8,
                  next_visible_at_ms INTEGER NOT NULL,
                  locked_by TEXT,
                  lock_token TEXT,
                  lease_expires_at_ms INTEGER,
                  processed_at_ms INTEGER,
                  last_error_code TEXT,
                  last_error TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS ux_envelopes_to_idempotency
                  ON envelopes(to_agent_id, idempotency_key);

                CREATE INDEX IF NOT EXISTS idx_envelopes_to_status_visible
                  ON envelopes(to_agent_id, status, next_visible_at_ms, created_at_ms);

                CREATE INDEX IF NOT EXISTS idx_envelopes_lock_expiry
                  ON envelopes(status, lease_expires_at_ms);

                CREATE INDEX IF NOT EXISTS idx_envelopes_locked_by
                  ON envelopes(locked_by, status, lease_expires_at_ms);
                """
            )
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=5.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        # Apply runtime pragmas on every connection for stable concurrency behavior.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        return conn

    @staticmethod
    def _begin_immediate(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")

    def _emit(self, *, kind: str, payload: dict[str, Any], request_id: str | None) -> None:
        if self._event_bus is None:
            return
        event = Event(
            kind=kind,
            payload=dict(payload or {}),
            session_id=self._event_session_id,
            event_id=new_id("evt"),
            timestamp=now_ts_ms(),
            request_id=request_id,
            # A2A events are observability-only; do not overload Aura turn semantics.
            turn_id=None,
            schema_version=A2A_EVENT_SCHEMA_VERSION,
        )
        try:
            self._event_bus.publish(event)
        except Exception:
            # Event emission must not break mailbox correctness.
            return


def _compute_backoff_ms(attempt: int) -> int:
    # Exponential backoff with small jitter; cap at 60s.
    exp = min(60_000, int(2**max(0, attempt)) * 1000)
    return exp + random.randint(0, 250)
