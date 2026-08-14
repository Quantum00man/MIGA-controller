import json
import shutil
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import config
from app.core.experiment_manager import ExperimentManager
from app.models.schemas import ScanConfig


class ScheduleManager:
    """Persistent, server-owned experiment queue."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, manager: ExperimentManager | None = None):
        if getattr(self, "initialized", False):
            return
        self.initialized = True
        self.manager = manager or ExperimentManager()
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._state = self._load()
        self._thread = threading.Thread(target=self._run, name="miga-scheduler", daemon=True)
        self._thread.start()

    @staticmethod
    def _default_state() -> Dict[str, Any]:
        return {
            "active": False, "stopRequested": False, "waiting": False,
            "activeTaskIndex": -1, "activeTaskId": None,
            "currentTaskStep": 0, "currentTaskTotalSteps": 0,
            "currentTaskStartedAtMs": None, "waitUntilMs": None,
            "scheduleStartedAtMs": None, "completedTaskIds": [],
            "timingMode": "sequential", "sequentialGapSec": 0,
            "tasks": [], "statusMessage": "IDLE", "error": None,
        }

    def _load(self) -> Dict[str, Any]:
        state = self._default_state()
        try:
            raw = json.loads(Path(config.SCHEDULE_STATE_PATH).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                state.update(raw)
        except Exception:
            pass
        # A process cannot resume a scan that was already in memory. Retry that task.
        if state.get("active"):
            state["waiting"] = False
            state["currentTaskStartedAtMs"] = None
        return state

    def _save_locked(self) -> None:
        path = Path(config.SCHEDULE_STATE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            result = deepcopy(self._state)
        # Sequence source can be large and is private execution input, not status.
        result["tasks"] = [
            {key: value for key, value in task.items() if key != "sequence_snapshot"}
            for task in result.get("tasks", [])
        ]
        result["currentTaskStep"] = int(getattr(self.manager.status, "current_step", 0) or 0)
        return result

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("No scheduled tasks configured")
        normalized: List[Dict[str, Any]] = []
        for index, task in enumerate(tasks):
            if not isinstance(task, dict) or not isinstance(task.get("sequence_snapshot"), str):
                raise ValueError(f"Task {index + 1} has no sequence snapshot")
            config_payload = ScanConfig(**(task.get("config") or {})).dict()
            if config_payload.get("mode") == "ac_stark":
                raise ValueError("AC Stark Centering is available in Live Mode only")
            normalized.append({
                "id": str(task.get("id") or f"task_{index + 1}"),
                "name": str(task.get("name") or f"Task {index + 1}"),
                "config": config_payload,
                "sequence_snapshot": task["sequence_snapshot"],
                "sequence_file_name": str(task.get("sequence_file_name") or task.get("sequence_name") or "sequence.mot"),
                "scheduledAtMs": task.get("scheduledAtMs"),
                "estimated_points": int(task.get("estimated_points") or 0),
            })
        timing = str(payload.get("timingMode") or "sequential").lower()
        if timing not in {"sequential", "specific"}:
            raise ValueError("Invalid timing mode")
        with self._lock:
            if self._state.get("active"):
                raise ValueError("A scheduled queue is already active")
            if self.manager.get_active_mode():
                raise ValueError("Hardware is currently busy")
            self._state = self._default_state()
            self._state.update({
                "active": True, "tasks": normalized, "timingMode": timing,
                "sequentialGapSec": max(0.0, float(payload.get("sequentialGapSec") or 0)),
                "scheduleStartedAtMs": int(time.time() * 1000), "statusMessage": "SCHEDULE READY",
            })
            self._save_locked()
        self._wake.set()
        return self.get_status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._state.get("active"):
                return self.get_status()
            self._state["stopRequested"] = True
            self._state["statusMessage"] = "STOPPING"
            self._save_locked()
        self._wake.set()
        if self.manager.get_active_mode() == "scan":
            self.manager.stop_scan()
        return self.get_status()

    def _set(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)
            self._save_locked()

    def _should_stop(self) -> bool:
        with self._lock:
            return bool(self._state.get("stopRequested"))

    def _wait_until(self, target_ms: int) -> bool:
        self._set(waiting=True, waitUntilMs=target_ms, statusMessage="WAITING")
        while int(time.time() * 1000) < target_ms:
            if self._should_stop():
                return False
            self._wake.wait(min(1.0, max(0.05, (target_ms - int(time.time() * 1000)) / 1000)))
            self._wake.clear()
        self._set(waiting=False, waitUntilMs=None)
        return not self._should_stop()

    def _install_sequence(self, task: Dict[str, Any]) -> str:
        target = Path(config.SEQUENCE_TEMPLATE_PATH_WIN if config.IS_WINDOWS else config.SEQUENCE_TEMPLATE_PATH_LINUX)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".scheduled.tmp")
        tmp.write_text(task["sequence_snapshot"], encoding="utf-8")
        shutil.move(str(tmp), str(target))
        return task["sequence_file_name"]

    def _run(self) -> None:
        while True:
            self._wake.wait(1.0)
            self._wake.clear()
            with self._lock:
                if not self._state.get("active"):
                    continue
                tasks = deepcopy(self._state.get("tasks") or [])
                completed = set(self._state.get("completedTaskIds") or [])
                timing = self._state.get("timingMode")
                gap_sec = float(self._state.get("sequentialGapSec") or 0)
            try:
                for index, task in enumerate(tasks):
                    if task["id"] in completed:
                        continue
                    if self._should_stop():
                        break
                    self._set(activeTaskIndex=index, activeTaskId=task["id"], currentTaskStep=0,
                              currentTaskTotalSteps=task.get("estimated_points", 0))
                    target_ms = None
                    if timing == "specific":
                        try:
                            target_ms = int(task.get("scheduledAtMs"))
                        except (TypeError, ValueError):
                            raise ValueError(f"Task start time missing: {task['name']}")
                    elif completed and gap_sec > 0:
                        target_ms = int(time.time() * 1000 + gap_sec * 1000)
                    if target_ms and target_ms > int(time.time() * 1000) and not self._wait_until(target_ms):
                        break
                    sequence_name = self._install_sequence(task)
                    scan_config = deepcopy(task["config"])
                    scan_config["sequence_name"] = sequence_name
                    self._set(currentTaskStartedAtMs=int(time.time() * 1000), statusMessage=f"TASK {index + 1}/{len(tasks)}")
                    result = self.manager.start_scan(scan_config)
                    if result.get("status") != "success":
                        raise RuntimeError(result.get("message") or "Task start failed")
                    while self.manager.status.is_running:
                        if self._should_stop():
                            self.manager.stop_scan()
                        time.sleep(0.5)
                    if self._should_stop():
                        break
                    completed.add(task["id"])
                    self._set(completedTaskIds=list(completed), currentTaskStep=task.get("estimated_points", 0))
                if self._should_stop():
                    self._set(active=False, waiting=False, waitUntilMs=None, statusMessage="STOPPED")
                else:
                    self._set(active=False, waiting=False, waitUntilMs=None, statusMessage="IDLE")
            except Exception as exc:
                self._set(active=False, waiting=False, waitUntilMs=None, statusMessage="ERROR", error=str(exc))
