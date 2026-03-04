from __future__ import annotations

from pathlib import Path

from aura.cli import EXIT_OK, main
from aura.runtime.committee import CommitteeCoordinator
from aura.runtime.models.notification import NotificationType
from aura.runtime.models.sandbox import Sandbox
from aura.runtime.models.signal import SignalType
from aura.runtime.notifications import NotificationStore
from aura.runtime.signal import SignalBus, SignalStore


class _FakeSandboxManager:
    def __init__(self) -> None:
        self.destroyed: list[str] = []

    def create(self, *, agent_id: str, issue_key: str, base_branch: str = "main") -> Sandbox:
        return Sandbox(
            sandbox_id=f"sb_{issue_key}_{agent_id}",
            agent_id=agent_id,
            issue_key=issue_key,
            worktree_path=f".aura/sandboxes/sb_{issue_key}_{agent_id}",
            branch=f"agent/{issue_key}/{agent_id}/test",
            base_branch=base_branch,
        )

    def destroy(self, sandbox_id: str) -> None:
        self.destroyed.append(sandbox_id)


def test_notification_store_create_list_mark_read(tmp_path: Path) -> None:
    store = NotificationStore(project_root=tmp_path)
    created = store.create(
        notification_type=NotificationType.TASK_COMPLETED,
        title="AUTO-1 completed",
        summary="All checks passed.",
        issue_key="AUTO-1",
    )
    items = store.list(unread_only=True)
    assert len(items) == 1
    assert items[0].notification_id == created.notification_id
    assert items[0].read is False

    updated = store.mark_read(created.notification_id)
    assert updated is not None
    assert updated.read is True
    assert store.list(unread_only=True) == []


def test_committee_task_completed_accept_notifies_user(tmp_path: Path) -> None:
    bus = SignalBus(store=SignalStore(project_root=tmp_path))
    fake_sandbox = _FakeSandboxManager()
    notifications = NotificationStore(project_root=tmp_path)
    coordinator = CommitteeCoordinator(
        project_root=tmp_path,
        signal_bus=bus,
        sandbox_manager=fake_sandbox,
        notifications=notifications,
    )

    signal = bus.send(
        from_agent="agent.worker",
        to_agent="committee",
        signal_type=SignalType.NOTIFY,
        brief="task_completed",
        issue_key="AUTO-ACCEPT-1",
        sandbox_id="sb_AUTO-ACCEPT-1_agent.worker",
        payload={
            "type": "task_completed",
            "summary": "Implemented and validated.",
            "accept": True,
            "cleanup_sandbox": True,
        },
    )
    result = coordinator.handle_signal(signal)

    assert result["handled"] is True
    assert result["decision"] == "accept"
    assert fake_sandbox.destroyed == ["sb_AUTO-ACCEPT-1_agent.worker"]

    user_signals = bus.query(
        to_agent="super_agent",
        signal_type=SignalType.NOTIFY,
        issue_key="AUTO-ACCEPT-1",
        include_archive=True,
        limit=0,
    )
    assert len(user_signals) == 1
    items = notifications.list(unread_only=True)
    assert len(items) == 1
    assert items[0].notification_type is NotificationType.TASK_COMPLETED


def test_committee_task_completed_reject_wakes_worker(tmp_path: Path) -> None:
    bus = SignalBus(store=SignalStore(project_root=tmp_path))
    coordinator = CommitteeCoordinator(project_root=tmp_path, signal_bus=bus, sandbox_manager=_FakeSandboxManager())

    signal = bus.send(
        from_agent="agent.worker",
        to_agent="committee",
        signal_type=SignalType.NOTIFY,
        brief="task_completed",
        issue_key="AUTO-REJECT-1",
        sandbox_id="sb_AUTO-REJECT-1_agent.worker",
        payload={
            "type": "task_completed",
            "accept": False,
            "feedback": "Missing regression tests for edge cases.",
        },
    )
    result = coordinator.handle_signal(signal)

    assert result["handled"] is True
    assert result["decision"] == "reject"

    wakes = bus.query(
        to_agent="agent.worker",
        signal_type=SignalType.WAKE,
        issue_key="AUTO-REJECT-1",
        include_archive=True,
        limit=0,
    )
    assert len(wakes) == 1
    assert wakes[0].payload is not None
    assert wakes[0].payload.get("type") == "revision_request"

    notifs = coordinator.notifications.list(unread_only=True)
    assert len(notifs) == 1
    assert notifs[0].notification_type is NotificationType.REVIEW_NEEDED


def test_cli_notifications_json_and_mark_read(tmp_path: Path, monkeypatch, capsys) -> None:
    assert main(["init", str(tmp_path)]) == EXIT_OK
    monkeypatch.chdir(tmp_path)

    store = NotificationStore(project_root=tmp_path)
    created = store.create(
        notification_type=NotificationType.TASK_COMPLETED,
        title="AUTO-CLI-1 completed",
        summary="Done",
        issue_key="AUTO-CLI-1",
    )

    rc_json = main(["notifications", "--json"])
    out_json = capsys.readouterr().out
    assert rc_json == EXIT_OK
    assert created.notification_id in out_json
    assert '"count": 1' in out_json

    rc_mark = main(["notifications", "--mark-read", created.notification_id])
    out_mark = capsys.readouterr().out
    assert rc_mark == EXIT_OK
    assert '"ok": true' in out_mark.lower()

    rc_unread = main(["notifications", "--json", "--unread"])
    out_unread = capsys.readouterr().out
    assert rc_unread == EXIT_OK
    assert '"count": 0' in out_unread
