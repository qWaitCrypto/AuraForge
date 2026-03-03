from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..event_log import EventLog, EventLogFileStore
from ..sandbox import SandboxManager
from ..signal import SignalBus, SignalStore
from .agent_status import AgentStatusTracker
from .dispatcher import DispatchRequest, DispatchResult, Dispatcher
from .policy import ControlPolicy, PolicyCheckResult, PolicyGate


@dataclass(slots=True)
class ControlPlane:
    event_log: EventLog
    signal_bus: SignalBus
    sandbox_manager: SandboxManager
    status_tracker: AgentStatusTracker
    policy_gate: PolicyGate
    dispatcher: Dispatcher


def build_control_plane(*, project_root: Path) -> ControlPlane:
    root = project_root.expanduser().resolve()
    event_log = EventLog(store=EventLogFileStore(project_root=root))
    signal_bus = SignalBus(store=SignalStore(project_root=root), event_log=event_log)
    sandbox_manager = SandboxManager(project_root=root)
    status_tracker = AgentStatusTracker(
        project_root=root,
        event_log=event_log,
        signal_bus=signal_bus,
        sandbox_manager=sandbox_manager,
    )
    policy_gate = PolicyGate(
        project_root=root,
        status_tracker=status_tracker,
    )
    dispatcher = Dispatcher(
        project_root=root,
        signal_bus=signal_bus,
        sandbox_manager=sandbox_manager,
        policy_gate=policy_gate,
    )
    return ControlPlane(
        event_log=event_log,
        signal_bus=signal_bus,
        sandbox_manager=sandbox_manager,
        status_tracker=status_tracker,
        policy_gate=policy_gate,
        dispatcher=dispatcher,
    )


__all__ = [
    "AgentStatusTracker",
    "ControlPlane",
    "ControlPolicy",
    "DispatchRequest",
    "DispatchResult",
    "Dispatcher",
    "PolicyCheckResult",
    "PolicyGate",
    "build_control_plane",
]
