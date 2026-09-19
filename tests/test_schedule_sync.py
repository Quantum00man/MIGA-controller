import threading
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.schedule_manager import ScheduleManager
from app.models.schemas import ScanConfig


class FakeExperimentManager:
    def __init__(self):
        self.status = SimpleNamespace(current_step=0, is_running=False)
        self.started = []
        self.stopped = False
        self.settings = {"sync_slaves": []}

    def get_settings(self):
        return self.settings

    def get_active_mode(self):
        return None

    def start_scan(self, config):
        self.started.append(config)
        return {"status": "success"}

    def stop_scan(self):
        self.stopped = True


class FakeSyncManager:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.runtime = {"active": False, "status": "done", "master_step": 0}

    def status(self):
        return dict(self.runtime)

    def start_master(self, payload):
        self.started.append(payload)
        return self.status()

    def stop_master(self, reason):
        self.stopped.append(reason)
        self.runtime["active"] = False
        self.runtime["status"] = "stopped"
        return self.status()


def make_scheduler():
    scheduler = object.__new__(ScheduleManager)
    scheduler.manager = FakeExperimentManager()
    scheduler.sync_manager = FakeSyncManager()
    scheduler._lock = threading.RLock()
    scheduler._wake = threading.Event()
    scheduler._state = scheduler._default_state()
    scheduler._save_locked = lambda: None
    return scheduler


def sync_task():
    return {
        "id": "sync_1",
        "name": "Morning SYNC",
        "execution_mode": "sync",
        "config": ScanConfig().dict(),
        "sequence_snapshot": "master sequence",
        "sequence_file_name": "master.mot",
        "estimated_points": 11,
        "sync": {
            "master_delay_ms": 125,
            "slaves": [
                {
                    "node_id": "slave_1",
                    "name": "Slave 1",
                    "base_url": "http://127.0.0.1:9001",
                    "sequence_name": "slave.mot",
                    "sequence_content_base64": "c2xhdmUgc2VxdWVuY2U=",
                    "phase_calibration": {"name": "Slave fringe", "reference_t2_us2": 12.5},
                    "enabled": True,
                }
            ],
        },
    }


def test_start_accepts_sync_task_and_redacts_sequence_content():
    scheduler = make_scheduler()

    status = scheduler.start({"timingMode": "sequential", "tasks": [sync_task()]})

    assert status["active"] is True
    stored = scheduler._state["tasks"][0]
    assert stored["execution_mode"] == "sync"
    assert stored["sync"]["master_delay_ms"] == 125
    assert stored["sync"]["slaves"][0]["sequence_content_base64"]
    assert stored["sync"]["slaves"][0]["phase_calibration"]["name"] == "Slave fringe"
    public_slave = status["tasks"][0]["sync"]["slaves"][0]
    assert "sequence_content" not in public_slave
    assert "sequence_content_base64" not in public_slave
    assert "sequence_snapshot" not in status["tasks"][0]


def test_start_rebases_persisted_sync_slave_id_using_current_url():
    scheduler = make_scheduler()
    scheduler.manager.settings["sync_slaves"] = [{
        "id": "slave_current",
        "name": "MIGA21",
        "base_url": "http://127.0.0.1:9001/",
        "enabled": True,
    }]
    task = sync_task()
    task["sync"]["slaves"][0]["node_id"] = "slave_historical"

    scheduler.start({"timingMode": "sequential", "tasks": [task]})

    stored_slave = scheduler._state["tasks"][0]["sync"]["slaves"][0]
    assert stored_slave["node_id"] == "slave_current"
    assert stored_slave["base_url"] == "http://127.0.0.1:9001"
    assert stored_slave["phase_calibration"]["name"] == "Slave fringe"


def test_execute_sync_task_uses_sync_manager_with_master_sequence_name():
    scheduler = make_scheduler()
    task = sync_task()
    scheduler._install_sequence = lambda queued_task: queued_task["sequence_file_name"]

    scheduler._execute_task(task)

    assert scheduler.manager.started == []
    assert len(scheduler.sync_manager.started) == 1
    payload = scheduler.sync_manager.started[0]
    assert payload["scan_config"]["sequence_name"] == "master.mot"
    assert payload["master_delay_ms"] == 125
    assert payload["slaves"][0]["node_id"] == "slave_1"
    assert payload["slaves"][0]["phase_calibration"]["reference_t2_us2"] == 12.5


def test_scheduled_sync_transfer_function_preserves_scan_plan_fields():
    scheduler = make_scheduler()
    task = sync_task()
    task["config"].update({
        "mode": "transfer_function",
        "transfer_frequency_start_hz": 100.0,
        "transfer_frequency_stop_hz": 300.0,
        "transfer_frequency_step_hz": 100.0,
        "transfer_repeats": 4,
        "transfer_phase_degrees": [0.0, 90.0],
        "transfer_phase_scan_mode": "frequency_interleaved",
    })

    scheduler.start({"timingMode": "sequential", "tasks": [task]})
    stored = scheduler._state["tasks"][0]
    scheduler._install_sequence = lambda queued_task: queued_task["sequence_file_name"]
    scheduler._execute_task(stored)

    payload = scheduler.sync_manager.started[0]
    config = payload["scan_config"]
    assert config["mode"] == "transfer_function"
    assert config["transfer_frequency_start_hz"] == 100.0
    assert config["transfer_frequency_stop_hz"] == 300.0
    assert config["transfer_frequency_step_hz"] == 100.0
    assert config["transfer_repeats"] == 4
    assert config["transfer_phase_degrees"] == [0.0, 90.0]
    assert config["transfer_phase_scan_mode"] == "frequency_interleaved"


def test_existing_schedule_task_defaults_to_regular_scan():
    scheduler = make_scheduler()
    task = sync_task()
    task.pop("execution_mode")
    task.pop("sync")

    scheduler.start({"timingMode": "sequential", "tasks": [task]})
    stored = scheduler._state["tasks"][0]
    scheduler._install_sequence = lambda queued_task: queued_task["sequence_file_name"]
    scheduler._execute_task(stored)

    assert stored["execution_mode"] == "scan"
    assert scheduler.sync_manager.started == []
    assert scheduler.manager.started[0]["sequence_name"] == "master.mot"


def test_temporary_sequence_does_not_install_over_active_template():
    scheduler = make_scheduler()
    task = sync_task()
    task["execution_mode"] = "scan"
    task.pop("sync")
    task["temporary_sequence"] = True
    scheduler._install_sequence = lambda queued_task: (_ for _ in ()).throw(
        AssertionError("temporary task must not overwrite the active template")
    )

    scheduler.start({"timingMode": "sequential", "tasks": [task]})
    stored = scheduler._state["tasks"][0]
    scheduler._execute_task(stored)

    override = scheduler.manager.started[0]["_template_path_override"]
    assert scheduler.manager.started[0]["sequence_name"] == "master.mot"
    assert not Path(override).exists()


def test_persisted_active_schedule_is_interrupted_instead_of_retried_after_restart():
    with tempfile.TemporaryDirectory() as temporary:
        state_path = Path(temporary) / "schedule_state.json"
        state_path.write_text(json.dumps({
            "active": True,
            "waiting": True,
            "activeTaskId": "sync_1",
            "tasks": [sync_task()],
            "statusMessage": "TASK 1/1",
        }), encoding="utf-8")
        scheduler = object.__new__(ScheduleManager)
        with patch("app.core.schedule_manager.config.SCHEDULE_STATE_PATH", state_path):
            state = scheduler._load()

    assert state["active"] is False
    assert state["waiting"] is False
    assert state["statusMessage"] == "ERROR"
    assert "controller restart" in state["error"]
    assert state["errorAtMs"] is not None
    assert state["tasks"][0]["id"] == "sync_1"


def test_schedule_logs_are_persistent_and_exclude_sequence_contents():
    scheduler = make_scheduler()
    task = sync_task()
    task["sequence_snapshot"] = "private master sequence"
    task["sync"]["slaves"][0]["sequence_content_base64"] = "private slave sequence"
    scheduler._state["scheduleId"] = "audit-test"

    with tempfile.TemporaryDirectory() as temporary:
        log_dir = Path(temporary) / "schedule_logs"
        with patch("app.core.schedule_manager.config.SCHEDULE_LOG_DIR", log_dir):
            scheduler._log_event("task_started", task=task, task_index=1, task_count=3)
            scheduler._log_event("task_failed", task=task, detail="Slave controller disconnected")
            logs = scheduler.get_logs(errors_only=True)
            payload, filename = scheduler.get_log_download(logs["date"])

    assert len(logs["dates"]) == 1
    assert len(logs["records"]) == 1
    assert logs["records"][0]["event"] == "task_failed"
    assert logs["records"][0]["detail"] == "Slave controller disconnected"
    assert "sequence_snapshot" not in logs["records"][0]
    assert b"private master sequence" not in payload
    assert b"private slave sequence" not in payload
    assert filename.endswith(".jsonl")


def test_task_start_retries_three_times_before_the_fourth_attempt_succeeds():
    scheduler = make_scheduler()
    task = sync_task()
    task["execution_mode"] = "scan"
    task.pop("sync")
    scheduler._install_sequence = lambda queued_task: queued_task["sequence_file_name"]
    responses = iter([
        {"status": "error", "message": "controller busy"},
        {"status": "error", "message": "controller busy"},
        {"status": "error", "message": "controller busy"},
        {"status": "success"},
    ])
    start_calls = []
    def start_scan(config):
        start_calls.append(config)
        return next(responses)
    scheduler.manager.start_scan = start_scan
    scheduler._wait_until = lambda _target_ms: True
    events = []
    scheduler._log_event = lambda event, **kwargs: events.append((event, kwargs))

    assert scheduler._execute_task_with_start_retries(task, 1, 1) is True
    assert len(start_calls) == 4
    assert len([event for event, _kwargs in events if event == "task_start_failed"]) == 3
    assert len([event for event, _kwargs in events if event == "task_start_retry"]) == 3
    assert events[-1][1]["retry_delay_s"] == 5
