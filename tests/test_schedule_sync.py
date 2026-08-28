import threading
from pathlib import Path
from types import SimpleNamespace

from app.core.schedule_manager import ScheduleManager
from app.models.schemas import ScanConfig


class FakeExperimentManager:
    def __init__(self):
        self.status = SimpleNamespace(current_step=0, is_running=False)
        self.started = []
        self.stopped = False

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
