from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from ..ids import new_id
from .runtime import A2ARuntime


@dataclass(frozen=True, slots=True)
class ActivatorConfig:
    mailbox_id: str
    consumer_prefix: str = "worker"
    max_instances: int = 4
    poll_interval_ms: int = 500
    worker_idle_timeout_s: float = 30.0
    worker_lease_ms: int = 60_000


class Activator:
    """
    Minimal activator that spawns "sleeping" worker processes on demand.

    The worker exits on idle timeout, so the system naturally scales down to zero.
    """

    def __init__(self, *, runtime: A2ARuntime, config: ActivatorConfig) -> None:
        self._rt = runtime
        self._cfg = config
        self._children: list[subprocess.Popen] = []
        self._stop = False

    def run(self) -> int:
        self._install_signal_handlers()

        while not self._stop:
            self._reap_children()
            self._rt.mailbox.sweep_expired_locks(limit=1000)

            stats = self._rt.mailbox.stats(mailbox_id=self._cfg.mailbox_id)
            counts = stats.get("counts") if isinstance(stats, dict) else None
            enq = int(counts.get("ENQUEUED", 0)) if isinstance(counts, dict) else 0
            failed = int(counts.get("FAILED", 0)) if isinstance(counts, dict) else 0
            pending = max(0, enq + failed)

            capacity = max(0, int(self._cfg.max_instances) - len(self._children))
            if pending > 0 and capacity > 0:
                spawn = min(capacity, pending)
                for _ in range(spawn):
                    self._spawn_worker()

            time.sleep(max(0.05, int(self._cfg.poll_interval_ms)) / 1000.0)

        self._terminate_children()
        return 0

    def _spawn_worker(self) -> None:
        consumer_id = f"{self._cfg.consumer_prefix}.{new_id('inst')}"
        cmd = [
            sys.executable,
            "-m",
            "aura",
            "a2a",
            "worker",
            "--mailbox",
            self._cfg.mailbox_id,
            "--consumer",
            consumer_id,
            "--idle-timeout-s",
            str(self._cfg.worker_idle_timeout_s),
            "--lease-ms",
            str(self._cfg.worker_lease_ms),
        ]
        env = os.environ.copy()
        p = subprocess.Popen(cmd, env=env)
        self._children.append(p)

    def _reap_children(self) -> None:
        alive: list[subprocess.Popen] = []
        for p in self._children:
            if p.poll() is None:
                alive.append(p)
        self._children = alive

    def _terminate_children(self) -> None:
        for p in self._children:
            try:
                p.terminate()
            except Exception:
                pass
        deadline = time.time() + 5.0
        for p in self._children:
            while time.time() < deadline:
                if p.poll() is not None:
                    break
                time.sleep(0.05)
        for p in self._children:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        self._children = []

    def _install_signal_handlers(self) -> None:
        def _handler(_sig, _frame) -> None:
            self._stop = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except Exception:
                continue

