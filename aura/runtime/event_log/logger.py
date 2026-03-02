from __future__ import annotations

from ..models.event_log import LogEvent, LogEventKind
from .store import EventLogFileStore


class EventLog:
    """Append-only audit log facade."""

    def __init__(self, *, store: EventLogFileStore) -> None:
        self._store = store

    def record(self, event: LogEvent) -> None:
        self._store.append(event)

    def query(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        sandbox_id: str | None = None,
        issue_key: str | None = None,
        tool_name: str | None = None,
        kind: LogEventKind | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int = 500,
    ) -> list[LogEvent]:
        events = self._store.read(session_id=session_id, since_ms=since_ms, until_ms=until_ms)

        out: list[LogEvent] = []
        for event in events:
            if agent_id is not None and event.agent_id != agent_id:
                continue
            if sandbox_id is not None and event.sandbox_id != sandbox_id:
                continue
            if issue_key is not None and event.issue_key != issue_key:
                continue
            if tool_name is not None and event.tool_name != tool_name:
                continue
            if kind is not None and event.kind is not kind:
                continue
            out.append(event)

        if limit > 0 and len(out) > limit:
            out = out[-limit:]
        return out

    def query_external_refs(
        self,
        *,
        agent_id: str | None = None,
        issue_key: str | None = None,
        ref_prefix: str | None = None,
        since_ms: int | None = None,
    ) -> list[str]:
        events = self.query(agent_id=agent_id, issue_key=issue_key, since_ms=since_ms, limit=10000)

        refs: list[str] = []
        seen: set[str] = set()
        for event in events:
            for ref in event.external_refs:
                if ref_prefix is not None and not ref.startswith(ref_prefix):
                    continue
                if ref in seen:
                    continue
                seen.add(ref)
                refs.append(ref)
        return refs
