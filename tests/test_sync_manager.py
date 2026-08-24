import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.experiment_manager import ExperimentManager
from app.core.sync_manager import SyncManager


class FakeResponse:
    def __init__(self, payload=None):
        self.payload = payload or {"ready": True}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeManager:
    def __init__(self, root, role="master"):
        self.settings = {
            "sync_role": role,
            "sync_node_name": "Node A",
            "sync_shared_token": "secret",
            "sync_allowed_master_ip": "",
            "sync_slaves": [],
        }
        self.status = SimpleNamespace(is_running=False, current_step=0, total_steps=0, message="IDLE")
        self.data_manager = SimpleNamespace(current_run_dir=Path(root), current_run_id_str="run01_20260824")
        self.listeners = []
        self.published = []
        self.started = []
        self.events = []

    def add_data_listener(self, listener):
        self.listeners.append(listener)

    def get_settings(self):
        return self.settings

    def get_active_mode(self):
        return "scan" if self.status.is_running else None

    def build_scan_parameter_plan(self, config):
        return [[3], [1], [3], [2]]

    def start_scan(self, config, parameters_override=None):
        self.events.append("master_start" if config.get("sync_role") == "master" else "slave_start")
        self.started.append((config, parameters_override))
        self.status.is_running = True
        self.status.total_steps = len(parameters_override or [])
        return {"status": "success", "message": "started"}

    def stop_scan(self):
        self.status.is_running = False
        return {"status": "success", "message": "stopping"}

    def publish_data(self, payload, notify_listeners=True):
        self.published.append(payload)


class SyncManagerTests(unittest.TestCase):
    def test_sync_parameter_plan_accepts_link_formula_mode(self):
        manager = ExperimentManager.__new__(ExperimentManager)
        manager._generate_parameters = lambda config: [[1, 2], [2, 4]]

        plan = manager.build_scan_parameter_plan({"mode": "link", "link_formulas": ["P0 * 2"]})

        self.assertEqual(plan, [[1, 2], [2, 4]])

    def test_master_sends_one_exact_shot_plan_and_starts_slave_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeManager(tmp)
            sync = SyncManager(manager)
            calls = []

            def post(url, **kwargs):
                calls.append((url, kwargs.get("json")))
                if url.endswith("/start"):
                    manager.events.append("slave_start")
                return FakeResponse()

            payload = {
                "scan_config": {"mode": "standard", "start": 1, "stop": 3, "step": 1},
                "master_delay_ms": 0,
                "slaves": [{
                    "node_id": "slave_b", "name": "Node B", "base_url": "192.168.1.20:8000",
                    "sequence_name": "slave.mot", "sequence_content": "+10us WAIT = OFF (1)\n",
                    "enabled": True,
                }],
            }

            with patch("app.core.sync_manager.requests.post", side_effect=post), patch.object(sync, "_monitor_master"):
                result = sync.start_master(payload)

            prepare = next(body for url, body in calls if url.endswith("/prepare"))
            self.assertEqual(prepare["shot_plan"], [[3], [1], [3], [2]])
            self.assertEqual(manager.events, ["slave_start", "master_start"])
            self.assertEqual(result["expected_shots"], 4)
            master_plan = manager.started[0][1]
            self.assertEqual([item["metadata"]["sync_shot_index"] for item in master_plan], [0, 1, 2, 3])
            self.assertEqual([item["metadata"]["sync_p0"] for item in master_plan], [3, 1, 3, 2])

    def test_fast_first_master_result_is_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeManager(tmp)
            sync = SyncManager(manager)

            def start_scan(config, parameters_override=None):
                manager.started.append((config, parameters_override))
                manager.status.is_running = True
                manager.status.total_steps = len(parameters_override or [])
                first = parameters_override[0]["metadata"]
                sync._capture_local_result({
                    **first,
                    "stream_type": "scan_point",
                    "intf_p1": 0.5,
                })
                return {"status": "success", "message": "started"}

            manager.start_scan = start_scan
            payload = {
                "scan_config": {"mode": "standard"},
                "master_delay_ms": 0,
                "slaves": [{
                    "node_id": "slave_b", "name": "Node B", "base_url": "192.168.1.20:8000",
                    "sequence_name": "slave.mot", "sequence_content": "+10us WAIT = OFF (1)\n",
                    "enabled": True,
                }],
            }

            with patch("app.core.sync_manager.requests.post", return_value=FakeResponse()), patch.object(sync, "_monitor_master"):
                sync.start_master(payload)

            self.assertIn(0, sync._master_results)

    def test_slave_runs_same_shot_count_without_applying_master_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeManager(tmp, role="slave")
            sync = SyncManager(manager)
            prepared = sync.prepare_node({
                "sync_run_id": "sync_test",
                "master_node_id": "master",
                "scan_config": {"mode": "standard"},
                "shot_plan": [[10], [20]],
                "sequence_name": "slave.mot",
                "sequence_content": "+10us WAIT = OFF (1)\n",
            })

            result = sync.start_node(prepared["sync_run_id"])

            self.assertEqual(result["shot_count"], 2)
            parameters = manager.started[0][1]
            self.assertEqual([item["sequence_parameters"] for item in parameters], [[], []])
            self.assertEqual([item["metadata"]["sync_p0"] for item in parameters], [10, 20])

    def test_slave_sequence_upload_preserves_non_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeManager(tmp, role="slave")
            sync = SyncManager(manager)
            source = b"\xb5s marker\r\n"

            with patch("app.core.sync_manager.config.BASE_DIR", Path(tmp)):
                sync.prepare_node({
                    "sync_run_id": "sync_binary",
                    "master_node_id": "master",
                    "scan_config": {"mode": "standard"},
                    "shot_plan": [[1]],
                    "sequence_name": "slave.mot",
                    "sequence_content_base64": base64.b64encode(source).decode("ascii"),
                })

            saved_path = Path(sync._prepared["sync_binary"]["sequence_path"])
            self.assertEqual(saved_path.read_bytes(), source)

    def test_slave_disconnect_stops_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeManager(tmp)
            manager.status.is_running = True
            sync = SyncManager(manager)
            sync._runtime.update({
                "active": True,
                "sync_run_id": "sync_test",
                "status": "running",
                "expected_shots": 4,
                "slaves": [{
                    "node_id": "slave_b", "name": "Node B",
                    "base_url": "http://192.168.1.20:8000", "cursor": 0,
                }],
            })

            with patch("app.core.sync_manager.requests.get", side_effect=ConnectionError("offline")), patch(
                "app.core.sync_manager.requests.post", return_value=FakeResponse()
            ):
                sync._monitor_master()

            self.assertFalse(manager.status.is_running)
            self.assertFalse(sync.status()["active"])
            self.assertEqual(sync.status()["status"], "error")
            self.assertIn("disconnected", sync.status()["message"])

    def test_stop_orders_slave_delay_then_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeManager(tmp)
            manager.status.is_running = True
            sync = SyncManager(manager)
            sync._runtime.update({
                "active": True,
                "sync_run_id": "sync_test",
                "status": "running",
                "master_delay_ms": 25,
                "slaves": [{
                    "node_id": "slave_b", "name": "Node B",
                    "base_url": "http://192.168.1.20:8000",
                }],
            })
            events = []
            manager.stop_scan = lambda: events.append("master_stop") or {"status": "success"}

            with patch(
                "app.core.sync_manager.requests.post",
                side_effect=lambda *args, **kwargs: events.append("slave_stop") or FakeResponse(),
            ), patch("app.core.sync_manager.time.sleep", side_effect=lambda seconds: events.append("delay")):
                sync.stop_master()

            self.assertEqual(events, ["slave_stop", "delay", "master_stop"])
            self.assertTrue(sync.status()["stop_requested"])

    def test_pairing_skips_missing_or_error_shots(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeManager(tmp)
            sync = SyncManager(manager)
            sync._runtime.update({"active": True, "sync_run_id": "sync_test", "slaves": [], "expected_shots": 2})
            sync._slave_results = {"slave_b": {
                0: {"sync_shot_index": 0, "sync_p0": 4, "intf_p1": 0.4},
                1: {"sync_shot_index": 1, "sync_p0": 5, "error": "missing"},
            }}

            sync._capture_local_result({
                "stream_type": "scan_point", "sync_run_id": "sync_test", "sync_role": "master",
                "sync_node_id": "Node A", "sync_shot_index": 0, "sync_p0": 4, "intf_p1": 0.5,
            })
            sync._capture_local_result({
                "stream_type": "scan_point", "sync_run_id": "sync_test", "sync_role": "master",
                "sync_node_id": "Node A", "sync_shot_index": 1, "sync_p0": 5, "intf_p1": 0.6,
            })

            pairs = [item for item in manager.published if item.get("stream_type") == "sync_pair"]
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["sync_shot_index"], 0)
            self.assertEqual(pairs[0]["master"]["intf_p1"], 0.5)
            self.assertEqual(pairs[0]["slave"]["intf_p1"], 0.4)

    def test_shared_token_and_allowed_master_ip_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeManager(tmp, role="slave")
            manager.settings["sync_allowed_master_ip"] = "192.168.1.10"
            sync = SyncManager(manager)

            sync.authorize("secret", "192.168.1.10")
            with self.assertRaises(PermissionError):
                sync.authorize("wrong", "192.168.1.10")
            with self.assertRaises(PermissionError):
                sync.authorize("secret", "192.168.1.11")


if __name__ == "__main__":
    unittest.main()
