from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..event_log import EventLog, EventLogFileStore
from ..sandbox import SandboxManager
from ..signal import SignalBus, SignalStore
from .agent_status import AgentStatusTracker
from .circuit_breaker import (
    BreakerConfig,
    BreakerRecord,
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
    get_shared_circuit_breaker,
)
from .dashboard import AgentRow, DashboardAggregator, DashboardSnapshot, IssueRow, SystemSummary
from .dispatcher import DispatchRequest, DispatchResult, Dispatcher
from .health_probe import HealthProbe, ProbeIssue, ProbeIssueKind, ProbeReport
from .policy import ControlPolicy, PolicyCheckResult, PolicyGate
from .recovery import RecoveryAction, RecoveryManager, RecoveryRecord


@dataclass(slots=True)
class ControlPlane:
    event_log: EventLog
    signal_bus: SignalBus
    sandbox_manager: SandboxManager
    status_tracker: AgentStatusTracker
    policy_gate: PolicyGate
    dispatcher: Dispatcher
    health_probe: HealthProbe
    recovery_manager: RecoveryManager
    circuit_breaker: CircuitBreaker
    dashboard: DashboardAggregator


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
    health_probe = HealthProbe(
        project_root=root,
        status_tracker=status_tracker,
        signal_bus=signal_bus,
        policy_gate=policy_gate,
    )
    recovery_manager = RecoveryManager(
        project_root=root,
        sandbox_manager=sandbox_manager,
        signal_bus=signal_bus,
        status_tracker=status_tracker,
        policy_gate=policy_gate,
        event_log=event_log,
    )
    circuit_breaker = get_shared_circuit_breaker(project_root=root)
    dashboard = DashboardAggregator(
        project_root=root,
        status_tracker=status_tracker,
        signal_bus=signal_bus,
        sandbox_manager=sandbox_manager,
        event_log=event_log,
    )
    return ControlPlane(
        event_log=event_log,
        signal_bus=signal_bus,
        sandbox_manager=sandbox_manager,
        status_tracker=status_tracker,
        policy_gate=policy_gate,
        dispatcher=dispatcher,
        health_probe=health_probe,
        recovery_manager=recovery_manager,
        circuit_breaker=circuit_breaker,
        dashboard=dashboard,
    )


__all__ = [
    "AgentStatusTracker",
    "AgentRow",
    "BreakerConfig",
    "BreakerRecord",
    "BreakerState",
    "CircuitBreaker",
    "CircuitOpenError",
    "ControlPlane",
    "ControlPolicy",
    "DashboardAggregator",
    "DashboardSnapshot",
    "DispatchRequest",
    "DispatchResult",
    "Dispatcher",
    "HealthProbe",
    "IssueRow",
    "PolicyCheckResult",
    "PolicyGate",
    "ProbeIssue",
    "ProbeIssueKind",
    "ProbeReport",
    "RecoveryAction",
    "RecoveryManager",
    "RecoveryRecord",
    "SystemSummary",
    "build_control_plane",
]
