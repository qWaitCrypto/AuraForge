from __future__ import annotations

from dataclasses import dataclass

from ..event_bus import EventBus
from ..project import RuntimePaths
from ..stores import FileArtifactStore, FileEventLogStore, FileSessionStore
from .mailbox import MailboxStore


@dataclass(frozen=True, slots=True)
class A2ARuntime:
    paths: RuntimePaths
    event_bus: EventBus
    session_store: FileSessionStore
    event_log_store: FileEventLogStore
    artifact_store: FileArtifactStore
    mailbox: MailboxStore


def build_a2a_runtime(*, paths: RuntimePaths, session_id: str = "a2a") -> A2ARuntime:
    artifact_store = FileArtifactStore(paths.artifacts_dir)
    session_store = FileSessionStore(paths.sessions_dir)
    event_log_store = FileEventLogStore(
        paths.events_dir, artifact_store=artifact_store, session_store=session_store
    )
    event_bus = EventBus(event_log_store=event_log_store)

    # Ensure the synthetic A2A session exists so debug export/validate can work.
    try:
        session_store.get_session(session_id)
    except FileNotFoundError:
        session_store.create_session(
            {
                "session_id": session_id,
                "project_ref": str(paths.project_root),
                "mode": "a2a",
            }
        )

    mailbox = MailboxStore(
        db_path=paths.state_dir / "a2a_mailbox.sqlite3",
        event_bus=event_bus,
        event_session_id=session_id,
    )
    return A2ARuntime(
        paths=paths,
        event_bus=event_bus,
        session_store=session_store,
        event_log_store=event_log_store,
        artifact_store=artifact_store,
        mailbox=mailbox,
    )

