from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_runner import AgentRunner, RunnerConfig, load_runner_sessions_snapshot
from .control import ControlPlane, build_control_plane
from .ids import now_ts_ms


@dataclass(frozen=True, slots=True)
class ControlHubConfig:
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    probe_interval_s: float = 60.0
    recovery_interval_s: float = 120.0


class ControlHub:
    """
    Runtime daemon that keeps AgentRunner + health/recovery loops alive.
    """

    def __init__(self, *, project_root: Path, config: ControlHubConfig | None = None) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._config = config or ControlHubConfig()

        self._control: ControlPlane = build_control_plane(project_root=self._project_root)
        self._runner = AgentRunner(
            project_root=self._project_root,
            signal_bus=self._control.signal_bus,
            config=self._config.runner,
        )

        self._pid_path = self._project_root / ".aura" / "control_hub.pid"
        self._stop_path = self._project_root / ".aura" / "control_hub.stop"
        self._pid_path.parent.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._runner_task: asyncio.Task[None] | None = None
        self._probe_task: asyncio.Task[None] | None = None
        self._recovery_task: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Task[None] | None = None

    @property
    def control_plane(self) -> ControlPlane:
        return self._control

    @property
    def runner(self) -> AgentRunner:
        return self._runner

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._remove_stop_file()
        self.write_pid_file()

        self._runner_task = asyncio.create_task(self._runner.start(), name="control_hub:runner")
        self._probe_task = asyncio.create_task(self._probe_loop(), name="control_hub:probe")
        self._recovery_task = asyncio.create_task(self._recovery_loop(), name="control_hub:recovery")
        self._stop_task = asyncio.create_task(self._stop_watcher_loop(), name="control_hub:stop")

        try:
            await asyncio.gather(self._runner_task, self._probe_task, self._recovery_task, self._stop_task)
        finally:
            self.remove_pid_file()
            self._remove_stop_file()
            self._running = False

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        await self._runner.stop()

        current = asyncio.current_task()
        tasks = [self._probe_task, self._recovery_task, self._runner_task, self._stop_task]
        wait_tasks: list[asyncio.Task[None]] = []
        for task in tasks:
            if task is None or task is current:
                continue
            if not task.done():
                task.cancel()
            wait_tasks.append(task)

        if wait_tasks:
            await asyncio.gather(*wait_tasks, return_exceptions=True)
        self.remove_pid_file()
        self._remove_stop_file()

    async def _probe_loop(self) -> None:
        interval = max(1.0, float(self._config.probe_interval_s))
        while self._running:
            try:
                _ = self._control.health_probe.probe()
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def _recovery_loop(self) -> None:
        interval = max(1.0, float(self._config.recovery_interval_s))
        while self._running:
            try:
                report = self._control.health_probe.probe()
                _ = self._control.recovery_manager.auto_recover(report)
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def _stop_watcher_loop(self) -> None:
        while self._running:
            if self._stop_path.exists():
                await self.stop()
                return
            await asyncio.sleep(0.5)

    def write_pid_file(self) -> None:
        payload = {
            "pid": os.getpid(),
            "started_at": now_ts_ms(),
            "project_root": str(self._project_root),
        }
        tmp = self._pid_path.with_suffix(".pid.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self._pid_path)

    def remove_pid_file(self) -> None:
        try:
            if self._pid_path.exists():
                self._pid_path.unlink()
        except Exception:
            return

    def _remove_stop_file(self) -> None:
        try:
            if self._stop_path.exists():
                self._stop_path.unlink()
        except Exception:
            return

    @staticmethod
    def _pid_file(project_root: Path) -> Path:
        root = project_root.expanduser().resolve()
        return root / ".aura" / "control_hub.pid"

    @staticmethod
    def read_pid_info(project_root: Path) -> dict[str, Any] | None:
        path = ControlHub._pid_file(project_root)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def is_running(project_root: Path) -> bool:
        info = ControlHub.read_pid_info(project_root)
        if not isinstance(info, dict):
            return False
        pid = info.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False
        return True

    @staticmethod
    def stop_running(project_root: Path) -> bool:
        root = project_root.expanduser().resolve()
        info = ControlHub.read_pid_info(root)
        if not isinstance(info, dict):
            return False
        pid = info.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        stop_path = root / ".aura" / "control_hub.stop"
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stop_path.write_text("stop\n", encoding="utf-8")
        except Exception:
            pass
        if os.name == "nt":
            return True
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except Exception:
            return False
        return True

    @staticmethod
    def status_snapshot(project_root: Path) -> dict[str, Any]:
        root = project_root.expanduser().resolve()
        pid_info = ControlHub.read_pid_info(root)
        return {
            "running": ControlHub.is_running(root),
            "pid": pid_info.get("pid") if isinstance(pid_info, dict) else None,
            "started_at": pid_info.get("started_at") if isinstance(pid_info, dict) else None,
            "runner": load_runner_sessions_snapshot(project_root=root),
        }
