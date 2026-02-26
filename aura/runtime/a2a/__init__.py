"""A2A-lite: local agent-to-agent mailbox primitives."""

from .activator import Activator, ActivatorConfig
from .mailbox import MailboxStore
from .protocol import Envelope, EnvelopeType, MailboxStatus
from .runtime import A2ARuntime, build_a2a_runtime
from .worker import Worker, WorkerConfig

__all__ = [
    "A2ARuntime",
    "Activator",
    "ActivatorConfig",
    "Envelope",
    "EnvelopeType",
    "MailboxStatus",
    "MailboxStore",
    "Worker",
    "WorkerConfig",
    "build_a2a_runtime",
]
