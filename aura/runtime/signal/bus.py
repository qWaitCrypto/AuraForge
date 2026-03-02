from __future__ import annotations

from dataclasses import dataclass

from ..ids import new_id
from ..models.event_log import LogEvent, LogEventKind
from ..models.signal import Signal, SignalType
from ..event_log.logger import EventLog
from .store import SignalStore


@dataclass(slots=True)
class SignalBus:
    store: SignalStore
    event_log: EventLog | None = None

    def send(
        self,
        *,
        from_agent: str,
        to_agent: str,
        signal_type: SignalType,
        brief: str,
        issue_key: str | None = None,
        sandbox_id: str | None = None,
        payload: dict | None = None,
    ) -> Signal:
        signal = Signal(
            signal_id=new_id("sig"),
            from_agent=from_agent,
            to_agent=to_agent,
            signal_type=signal_type,
            brief=brief,
            issue_key=issue_key,
            sandbox_id=sandbox_id,
            payload=payload,
        )
        self.store.append(signal)

        if self.event_log is not None:
            self.event_log.record(
                LogEvent(
                    event_id=new_id("evt"),
                    session_id=f"signal:{signal.signal_id}",
                    agent_id=from_agent,
                    sandbox_id=sandbox_id,
                    issue_key=issue_key,
                    kind=LogEventKind.SIGNAL_SENT,
                    tool_name="signal__send",
                    tool_args_summary=f"to={to_agent} type={signal_type.value}",
                    tool_result_summary=signal.brief,
                    tool_ok=True,
                    external_refs=[f"signal:{signal.signal_id}"],
                )
            )

        return signal

    def poll(self, *, to_agent: str, unconsumed_only: bool = True, limit: int = 20) -> list[Signal]:
        return self.store.read_inbox(to_agent, unconsumed_only=unconsumed_only, limit=limit)

    def consume(self, signal_id: str) -> None:
        signal = self.store.mark_consumed(signal_id)

        if self.event_log is not None:
            self.event_log.record(
                LogEvent(
                    event_id=new_id("evt"),
                    session_id=f"signal:{signal_id}",
                    agent_id=signal.to_agent,
                    sandbox_id=signal.sandbox_id,
                    issue_key=signal.issue_key,
                    kind=LogEventKind.SIGNAL_RECEIVED,
                    tool_name="signal__poll",
                    tool_args_summary=f"signal_id={signal_id}",
                    tool_result_summary="consumed",
                    tool_ok=True,
                    external_refs=[f"signal:{signal_id}"],
                )
            )

    def query(
        self,
        *,
        from_agent: str | None = None,
        to_agent: str | None = None,
        signal_type: SignalType | None = None,
        issue_key: str | None = None,
        since_ms: int | None = None,
        limit: int = 100,
    ) -> list[Signal]:
        return self.store.query_all(
            from_agent=from_agent,
            to_agent=to_agent,
            signal_type=signal_type,
            issue_key=issue_key,
            since_ms=since_ms,
            limit=limit,
        )
