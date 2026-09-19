import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import config
from app.core.experiment_manager import ExperimentManager
from app.models.schemas import ScanConfig, SyncStartRequest


class TaskStartError(RuntimeError):
    """A task could not be accepted by its controller, before acquisition began."""


class ScheduleManager:
    """Persistent, server-owned experiment queue."""

    _instance = None
    TASK_START_RETRY_COUNT = 3
    TASK_START_RETRY_DELAY_S = 5

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, manager: ExperimentManager | None = None, sync_manager: Any | None = None):
        if getattr(self, "initialized", False):
            if manager is not None:
                self.manager = manager
            if sync_manager is not None:
                self.sync_manager = sync_manager
            return
        self.initialized = True
        self.manager = manager or ExperimentManager()
        self.sync_manager = sync_manager
        self._lock = threading.RLock()
        self._log_lock = threading.RLock()
        self._wake = threading.Event()
        self._state = self._load()
        if str(self._state.get("error") or "").startswith("Scheduled queue interrupted by controller restart"):
            self._save_locked()
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
            "scheduleId": None,
            "timingMode": "sequential", "sequentialGapSec": 0,
            "tasks": [], "statusMessage": "IDLE", "error": None, "errorAtMs": None,
        }

    def _load(self) -> Dict[str, Any]:
        state = self._default_state()
        try:
            raw = json.loads(Path(config.SCHEDULE_STATE_PATH).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                state.update(raw)
        except Exception:
            pass
        # Hardware and remote SYNC state cannot be reconstructed safely after a
        # controller restart. Keep the queue for inspection, but never retry an
        # in-flight task automatically.
        if state.get("active"):
            interrupted_task = state.get("activeTaskId")
            detail = f" while running task {interrupted_task}" if interrupted_task else ""
            state["active"] = False
            state["stopRequested"] = False
            state["waiting"] = False
            state["waitUntilMs"] = None
            state["currentTaskStartedAtMs"] = None
            state["statusMessage"] = "ERROR"
            state["error"] = f"Scheduled queue interrupted by controller restart{detail}; review and start it again manually"
            state["errorAtMs"] = int(time.time() * 1000)
        return state

    def _match_current_sync_slave(self, queued_slave: Dict[str, Any]) -> Dict[str, Any]:
        """Rebase a persisted queue entry onto the controller's current node id.

        Browser queues and generated mid-fringe queues may outlive a Settings edit,
        which creates a new internal slave id even when the physical controller URL
        is unchanged. The URL is the stable identity for execution; calibration and
        sequence snapshots remain those captured by the queued task.
        """
        try:
            configured = self.manager.get_settings().get("sync_slaves") or []
        except (AttributeError, TypeError):
            return queued_slave

        enabled = [item for item in configured if isinstance(item, dict) and item.get("enabled", True)]
        queued_id = str(queued_slave.get("node_id") or "").strip()

        def normalize_url(value: Any) -> str:
            return str(value or "").strip().rstrip("/").lower()

        match = next((item for item in enabled if str(item.get("id") or item.get("node_id") or "").strip() == queued_id), None)
        if match is None:
            queued_url = normalize_url(queued_slave.get("base_url"))
            url_matches = [item for item in enabled if queued_url and normalize_url(item.get("base_url")) == queued_url]
            if len(url_matches) == 1:
                match = url_matches[0]
        if match is None:
            return queued_slave

        current_id = str(match.get("id") or match.get("node_id") or queued_id).strip()
        return {
            **queued_slave,
            "node_id": current_id,
            "name": str(match.get("name") or queued_slave.get("name") or current_id),
            "base_url": str(match.get("base_url") or queued_slave.get("base_url") or "").rstrip("/"),
            "enabled": True,
        }

    def _save_locked(self) -> None:
        path = Path(config.SCHEDULE_STATE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _public_task_details(task: Dict[str, Any] | None) -> Dict[str, Any]:
        """Return task-level audit fields without sequence or per-shot contents."""
        task = task if isinstance(task, dict) else {}
        config_payload = task.get("config") if isinstance(task.get("config"), dict) else {}
        return {
            "task_id": task.get("id"),
            "task_name": task.get("name"),
            "task_mode": task.get("execution_mode"),
            "scan_mode": config_payload.get("mode"),
            "estimated_points": task.get("estimated_points"),
        }

    def _log_event(self, event: str, *, task: Dict[str, Any] | None = None, detail: str = "", **extra: Any) -> None:
        """Append one durable schedule-level event. Logging must never stop a run."""
        try:
            now = datetime.now().astimezone()
            with self._lock:
                schedule_id = self._state.get("scheduleId")
            record = {
                "timestamp_ms": int(now.timestamp() * 1000),
                "timestamp": now.isoformat(timespec="milliseconds"),
                "event": str(event),
                "schedule_id": schedule_id,
                **self._public_task_details(task),
            }
            if detail:
                record["detail"] = str(detail)
            record.update({key: value for key, value in extra.items() if value is not None})
            log_dir = Path(config.SCHEDULE_LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"{now.date().isoformat()}.jsonl"
            if not hasattr(self, "_log_lock"):
                self._log_lock = threading.RLock()
            with self._log_lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception as exc:
            print(f"[Schedule] Unable to write audit log: {exc}")

    def get_logs(self, date: str = "", errors_only: bool = False, limit: int = 500) -> Dict[str, Any]:
        """Return recent events from a daily JSONL log without exposing task snapshots."""
        log_dir = Path(config.SCHEDULE_LOG_DIR)
        available_dates = sorted(
            (path.stem for path in log_dir.glob("????-??-??.jsonl")), reverse=True
        ) if log_dir.is_dir() else []
        selected_date = str(date or (available_dates[0] if available_dates else "")).strip()
        if selected_date and (len(selected_date) != 10 or selected_date not in available_dates):
            raise ValueError("Schedule log date was not found")
        records: List[Dict[str, Any]] = []
        if selected_date:
            path = log_dir / f"{selected_date}.jsonl"
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line)
                    if isinstance(item, dict) and (not errors_only or item.get("event") in {"task_start_failed", "task_failed", "schedule_failed"}):
                        records.append(item)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Unable to read schedule log: {exc}") from exc
        safe_limit = max(1, min(int(limit), 5000))
        return {"dates": available_dates, "date": selected_date, "records": list(reversed(records[-safe_limit:]))}

    def get_log_download(self, date: str) -> tuple[bytes, str]:
        selected_date = str(date or "").strip()
        try:
            datetime.strptime(selected_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("A schedule log date is required")
        path = Path(config.SCHEDULE_LOG_DIR) / f"{selected_date}.jsonl"
        if not path.is_file():
            raise ValueError("Schedule log date was not found")
        return path.read_bytes(), path.name

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            result = deepcopy(self._state)
        # Sequence source can be large and is private execution input, not status.
        public_tasks = []
        for task in result.get("tasks", []):
            public_task = {key: value for key, value in task.items() if key != "sequence_snapshot"}
            sync_payload = public_task.get("sync")
            if isinstance(sync_payload, dict):
                public_task["sync"] = {
                    **sync_payload,
                    "slaves": [
                        {
                            key: value
                            for key, value in slave.items()
                            if key not in {"sequence_content", "sequence_content_base64"}
                        }
                        for slave in sync_payload.get("slaves", [])
                        if isinstance(slave, dict)
                    ],
                }
            public_tasks.append(public_task)
        result["tasks"] = public_tasks
        active_task = next(
            (task for task in self._state.get("tasks", []) if task.get("id") == result.get("activeTaskId")),
            None,
        )
        if (
            active_task
            and active_task.get("execution_mode") == "sync"
            and self.sync_manager is not None
            and not result.get("waiting")
        ):
            result["currentTaskStep"] = int(self.sync_manager.status().get("master_step") or 0)
        else:
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
            if config_payload.get("parameter_source") == "markers":
                raise ValueError("Auto Markers are available in Live only")
            if config_payload.get("mode") == "ac_stark":
                raise ValueError("AC Stark Centering is available in Live Mode only")
            execution_mode = str(task.get("execution_mode") or "scan").strip().lower()
            if execution_mode not in {"scan", "sync"}:
                raise ValueError(f"Task {index + 1} has an invalid execution mode")
            normalized_task = {
                "id": str(task.get("id") or f"task_{index + 1}"),
                "name": str(task.get("name") or f"Task {index + 1}"),
                "execution_mode": execution_mode,
                "config": config_payload,
                "sequence_snapshot": task["sequence_snapshot"],
                "temporary_sequence": bool(task.get("temporary_sequence", False)),
                "note": str(task.get("note") or ""),
                "mid_fringe_p0_us2": task.get("mid_fringe_p0_us2"),
                "sequence_file_name": str(task.get("sequence_file_name") or task.get("sequence_name") or "sequence.mot"),
                "scheduledAtMs": task.get("scheduledAtMs"),
                "estimated_points": int(task.get("estimated_points") or 0),
            }
            if execution_mode == "sync":
                if self.sync_manager is None:
                    raise ValueError("SYNC scheduling is unavailable")
                if config_payload.get("mode") == "lock_in":
                    raise ValueError("Lock-in Measurement is not available in SYNC mode")
                raw_sync = task.get("sync") if isinstance(task.get("sync"), dict) else {}
                sync_payload = SyncStartRequest(
                    scan_config=config_payload,
                    master_delay_ms=raw_sync.get("master_delay_ms", 100),
                    slaves=raw_sync.get("slaves") or [],
                ).dict()
                if not sync_payload["slaves"]:
                    raise ValueError(f"Task {index + 1} has no enabled Sync slave")
                sync_payload["slaves"] = [
                    self._match_current_sync_slave(slave) for slave in sync_payload["slaves"]
                ]
                for slave in sync_payload["slaves"]:
                    if not str(slave.get("sequence_content") or slave.get("sequence_content_base64") or "").strip():
                        raise ValueError(f"Task {index + 1} has no sequence for Sync slave {slave.get('name') or slave.get('node_id')}")
                normalized_task["sync"] = {
                    "master_delay_ms": sync_payload["master_delay_ms"],
                    "slaves": sync_payload["slaves"],
                }
            normalized.append(normalized_task)
        timing = str(payload.get("timingMode") or "sequential").lower()
        if timing not in {"sequential", "specific"}:
            raise ValueError("Invalid timing mode")
        with self._lock:
            if self._state.get("active"):
                raise ValueError("A scheduled queue is already active")
            if self.manager.get_active_mode():
                raise ValueError("Hardware is currently busy")
            if self.sync_manager is not None and self.sync_manager.status().get("active"):
                raise ValueError("A Sync run is currently active")
            self._state = self._default_state()
            self._state.update({
                "active": True, "tasks": normalized, "timingMode": timing,
                "sequentialGapSec": max(0.0, float(payload.get("sequentialGapSec") or 0)),
                "scheduleStartedAtMs": int(time.time() * 1000), "scheduleId": uuid4().hex,
                "statusMessage": "SCHEDULE READY",
            })
            self._save_locked()
        self._log_event("schedule_started", task_count=len(normalized), timing_mode=timing,
                        sequential_gap_sec=self._state.get("sequentialGapSec"))
        self._wake.set()
        return self.get_status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._state.get("active"):
                return self.get_status()
            self._state["stopRequested"] = True
            self._state["statusMessage"] = "STOPPING"
            self._save_locked()
        self._log_event("schedule_stop_requested", detail="Stop requested by user")
        self._wake.set()
        if self.sync_manager is not None and self.sync_manager.status().get("active"):
            try:
                self.sync_manager.stop_master("Scheduled queue stop requested")
            except ValueError:
                pass
        elif self.manager.get_active_mode() == "scan":
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

    def _execute_task(self, task: Dict[str, Any]) -> None:
        temporary_path = None
        try:
            try:
                sequence_name = task["sequence_file_name"]
                if task.get("temporary_sequence"):
                    file_descriptor, raw_path = tempfile.mkstemp(prefix="miga_scheduled_", suffix=".mot")
                    os.close(file_descriptor)
                    temporary_path = Path(raw_path)
                    temporary_path.write_text(task["sequence_snapshot"], encoding="utf-8")
                else:
                    sequence_name = self._install_sequence(task)
                scan_config = deepcopy(task["config"])
                scan_config["sequence_name"] = sequence_name
                if temporary_path is not None:
                    scan_config["_template_path_override"] = str(temporary_path)
                if task.get("execution_mode") == "sync":
                    if self.sync_manager is None:
                        raise TaskStartError("SYNC scheduling is unavailable")
                    sync_payload = deepcopy(task.get("sync") or {})
                    sync_payload["scan_config"] = scan_config
                    self.sync_manager.start_master(sync_payload)
                else:
                    result = self.manager.start_scan(scan_config)
                    if result.get("status") != "success":
                        raise TaskStartError(result.get("message") or "Task start failed")
            except TaskStartError:
                raise
            except Exception as exc:
                raise TaskStartError(str(exc) or "Task start failed") from exc

            if task.get("execution_mode") == "sync":
                stop_sent = False
                while self.sync_manager.status().get("active"):
                    if self._should_stop() and not stop_sent:
                        try:
                            self.sync_manager.stop_master("Scheduled queue stop requested")
                        except ValueError:
                            pass
                        stop_sent = True
                    time.sleep(0.5)
                sync_status = self.sync_manager.status()
                if sync_status.get("status") == "error":
                    raise RuntimeError(sync_status.get("message") or "SYNC task failed")
                return
            while self.manager.status.is_running:
                if self._should_stop():
                    self.manager.stop_scan()
                time.sleep(0.5)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _execute_task_with_start_retries(self, task: Dict[str, Any], task_index: int, task_count: int) -> bool:
        """Retry only controller acceptance failures; never replay an active task."""
        total_attempts = self.TASK_START_RETRY_COUNT + 1
        for attempt in range(1, total_attempts + 1):
            self._set(
                waiting=False,
                waitUntilMs=None,
                currentTaskStartedAtMs=int(time.time() * 1000),
                statusMessage=(f"TASK {task_index}/{task_count}" if attempt == 1 else
                               f"TASK {task_index}/{task_count} · START RETRY {attempt - 1}/{self.TASK_START_RETRY_COUNT}"),
            )
            try:
                self._execute_task(task)
                return True
            except TaskStartError as exc:
                self._log_event(
                    "task_start_failed",
                    task=task,
                    detail=str(exc),
                    task_index=task_index,
                    attempt_number=attempt,
                    total_attempt_limit=total_attempts,
                )
                if attempt >= total_attempts:
                    raise RuntimeError(
                        f"Task could not start after {total_attempts} attempts: {exc}"
                    ) from exc
                retry_number = attempt
                self._log_event(
                    "task_start_retry",
                    task=task,
                    detail=str(exc),
                    task_index=task_index,
                    retry_number=retry_number,
                    retry_limit=self.TASK_START_RETRY_COUNT,
                    retry_delay_s=self.TASK_START_RETRY_DELAY_S,
                )
                retry_at_ms = int(time.time() * 1000 + self.TASK_START_RETRY_DELAY_S * 1000)
                self._set(
                    statusMessage=(f"TASK {task_index}/{task_count} · START FAILED; "
                                   f"RETRY {retry_number}/{self.TASK_START_RETRY_COUNT} IN {self.TASK_START_RETRY_DELAY_S}S"),
                )
                if not self._wait_until(retry_at_ms):
                    return False
        return False

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
                    if target_ms and target_ms > int(time.time() * 1000):
                        self._log_event("task_wait_started", task=task, scheduled_for_ms=target_ms)
                        if not self._wait_until(target_ms):
                            break
                    if target_ms:
                        self._log_event("task_wait_finished", task=task, scheduled_for_ms=target_ms)
                    self._log_event("task_started", task=task, task_index=index + 1, task_count=len(tasks))
                    try:
                        if not self._execute_task_with_start_retries(task, index + 1, len(tasks)):
                            break
                    except Exception as exc:
                        self._log_event("task_failed", task=task, detail=str(exc), task_index=index + 1)
                        raise
                    if self._should_stop():
                        self._log_event("task_stopped", task=task, detail="Schedule stop requested", task_index=index + 1)
                        break
                    completed.add(task["id"])
                    self._set(completedTaskIds=list(completed), currentTaskStep=task.get("estimated_points", 0))
                    self._log_event("task_completed", task=task, task_index=index + 1)
                if self._should_stop():
                    self._set(active=False, waiting=False, waitUntilMs=None, statusMessage="STOPPED")
                    self._log_event("schedule_stopped", detail="Schedule stopped before all tasks completed")
                else:
                    self._set(active=False, waiting=False, waitUntilMs=None, statusMessage="IDLE")
                    self._log_event("schedule_completed", completed_task_count=len(completed), task_count=len(tasks))
            except Exception as exc:
                self._set(
                    active=False,
                    waiting=False,
                    waitUntilMs=None,
                    statusMessage="ERROR",
                    error=str(exc),
                    errorAtMs=int(time.time() * 1000),
                )
                self._log_event("schedule_failed", detail=str(exc))
