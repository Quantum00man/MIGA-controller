import base64
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.core.experiment_manager import ExperimentManager
from app.core.data_loader import DataLoader
from app.core.sync_manager import SyncManager


class FakeResponse:
    def __init__(self, payload=None, content=b"", headers=None):
        self.payload = payload or {"ready": True}
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def iter_content(self, chunk_size=1024 * 1024):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]


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
                "scan_config": {
                    "mode": "standard", "start": 1, "stop": 3, "step": 1,
                    "interferometer_phase_calibration_override": {"name": "Master fringe"},
                },
                "master_delay_ms": 0,
                "slaves": [{
                    "node_id": "slave_b", "name": "Node B", "base_url": "192.168.1.20:8000",
                    "sequence_name": "slave.mot", "sequence_content": "+10us WAIT = OFF (1)\n",
                    "phase_calibration": {"name": "Slave fringe", "reference_t2_us2": 12.5},
                    "enabled": True,
                }],
            }

            with patch("app.core.sync_manager.requests.post", side_effect=post), patch.object(sync, "_monitor_master"):
                result = sync.start_master(payload)

            prepare = next(body for url, body in calls if url.endswith("/prepare"))
            self.assertEqual(prepare["shot_plan"], [[3], [1], [3], [2]])
            self.assertEqual(
                prepare["scan_config"]["interferometer_phase_calibration_override"]["name"],
                "Slave fringe",
            )
            self.assertEqual(manager.events, ["slave_start", "master_start"])
            self.assertEqual(result["expected_shots"], 4)
            master_plan = manager.started[0][1]
            self.assertEqual(manager.started[0][0]["interferometer_phase_calibration_override"]["name"], "Master fringe")
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
            self.assertIn("3 consecutive status failures", sync.status()["message"])
            self.assertIn("offline", sync.status()["message"])

    def test_single_status_poll_failure_is_retried_without_stopping_sync(self):
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
            attempts = 0

            def poll_status(*args, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectionError("temporary network jitter")
                manager.status.is_running = False
                manager.status.current_step = 4
                return FakeResponse({
                    "is_running": False,
                    "current_step": 4,
                    "latest_sequence": 0,
                    "results": [],
                })

            with patch("app.core.sync_manager.requests.get", side_effect=poll_status), patch(
                "app.core.sync_manager.time.sleep", return_value=None
            ), patch.object(sync, "_replicate_archives", return_value={}):
                sync._monitor_master()

            self.assertEqual(attempts, 2)
            self.assertEqual(sync.status()["status"], "done")
            self.assertEqual(sync.status()["slaves"][0]["status_poll_failures"], 0)
            self.assertNotIn("error", sync.status()["slaves"][0])

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
                0: {"sync_shot_index": 0, "sync_p0": 4, "intf_p1": 0.4, "sigma_up": 1.2,
                    "interferometer_phase": -0.12, "interferometer_phase_valid": True,
                    "interferometer_phase_calibration_id": "slave-cal",
                    "interferometer_phase_calibration_name": "Slave fringe"},
                1: {"sync_shot_index": 1, "sync_p0": 5, "error": "missing"},
            }}

            sync._capture_local_result({
                "stream_type": "scan_point", "sync_run_id": "sync_test", "sync_role": "master",
                "sync_node_id": "Node A", "sync_shot_index": 0, "sync_p0": 4,
                "sync_parameters": [4, 12], "intf_p1": 0.5,
                "sigma_up": 1.1, "interferometer_phase": 0.08,
                "interferometer_phase_valid": True,
                "interferometer_phase_calibration_id": "master-cal",
                "interferometer_phase_calibration_name": "Master fringe",
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
            self.assertEqual(pairs[0]["master"]["interferometer_phase"], 0.08)
            self.assertEqual(pairs[0]["master"]["interferometer_phase_calibration_id"], "master-cal")
            self.assertEqual(pairs[0]["slave"]["interferometer_phase"], -0.12)
            self.assertEqual(pairs[0]["slave"]["interferometer_phase_calibration_id"], "slave-cal")
            self.assertEqual(pairs[0]["sync_parameters"], [4, 12])
            self.assertEqual(pairs[0]["master"]["sync_parameters"], [4, 12])
            manifest = json.loads((Path(tmp) / "sync_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["node_results"]["master"][0]["sigma_up"], 1.1)
            self.assertEqual(manifest["node_results"]["slaves"]["slave_b"][0]["sigma_up"], 1.2)
            self.assertEqual(manifest["node_results"]["master"][0]["interferometer_phase_calibration_name"], "Master fringe")
            self.assertEqual(manifest["node_results"]["slaves"]["slave_b"][0]["interferometer_phase_calibration_name"], "Slave fringe")
            self.assertEqual(manifest["pairs"][0]["sync_parameters"], [4, 12])

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

    def test_full_archive_bundle_is_installed_under_sync_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "2026" / "08" / "24" / "run01_20260824"
            (run_dir / "waveforms").mkdir(parents=True)
            (run_dir / "config.json").write_text(json.dumps({"sync_run_id": "sync_test"}), encoding="utf-8")
            (run_dir / "results.csv").write_text("Step,Parameter_P0\n0,1\n", encoding="utf-8")
            (run_dir / "sequence.mot").write_text("+10us WAIT = OFF (1)\n", encoding="utf-8")
            np.savez_compressed(
                run_dir / "waveforms" / "step_0000.npz",
                raw_up=[1.0, 2.0], raw_dw=[3.0, 4.0], fit_up=[1.1, 2.1], fit_dw=[3.1, 4.1],
                time_axis=[0.0, 0.1], window_up=[0.0, 0.1], window_dw=[0.0, 0.1],
            )
            manager = FakeManager(run_dir)
            sync = SyncManager(manager)
            sync._runtime.update({"sync_run_id": "sync_test", "master_run_dir": str(run_dir)})

            bundle, metadata = sync.build_archive_bundle("sync_test")
            try:
                installed = sync._install_archive_bundle(run_dir, "slave_b", bundle, metadata["sha256"])
            finally:
                bundle.unlink(missing_ok=True)

            replica = run_dir / installed["path"]
            self.assertEqual((replica / "results.csv").read_text(encoding="utf-8"), "Step,Parameter_P0\n0,1\n")
            self.assertTrue((replica / "waveforms" / "step_0000.npz").is_file())
            self.assertFalse((replica / "sync_nodes").exists())

            manifest = {
                "archive_nodes": {
                    "master": {"path": "."},
                    "slave_b": {"path": installed["path"]},
                }
            }
            (run_dir / "sync_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loader = DataLoader()
            loader.base_dir = Path(tmp)
            sequence_path, _ = loader.get_archived_sequence_file(
                "2026", "08", "24", "run01_20260824", node_id="slave_b"
            )
            self.assertEqual(sequence_path.read_text(encoding="utf-8"), "+10us WAIT = OFF (1)\n")
            waveform = loader.load_waveform(
                "2026", "08", "24", "run01_20260824", 0, node_id="slave_b"
            )
            self.assertEqual(waveform["raw_up"], [1.0, 2.0])

    def test_archive_extraction_rejects_parent_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "unsafe.zip"
            target = Path(tmp) / "target"
            target.mkdir()
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escaped.txt", "unsafe")
            with self.assertRaises(ValueError):
                SyncManager._extract_archive_safely(archive_path, target)
            self.assertFalse((Path(tmp) / "escaped.txt").exists())

    def test_successful_replication_pulls_slave_and_pushes_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run01_20260824"
            (run_dir / "waveforms").mkdir(parents=True)
            (run_dir / "config.json").write_text(json.dumps({"sync_run_id": "sync_test"}), encoding="utf-8")
            (run_dir / "results.csv").write_text("master", encoding="utf-8")
            slave_zip = Path(tmp) / "slave.zip"
            with zipfile.ZipFile(slave_zip, "w") as archive:
                archive.writestr("config.json", json.dumps({"sync_run_id": "sync_test"}))
                archive.writestr("results.csv", "slave")
                archive.writestr("waveforms/step_0000.npz", "slave-waveform")
            slave_bytes = slave_zip.read_bytes()
            slave_sha = hashlib.sha256(slave_bytes).hexdigest()

            manager = FakeManager(run_dir)
            sync = SyncManager(manager)
            sync._runtime.update({
                "sync_run_id": "sync_test",
                "master_run_dir": str(run_dir),
                "slaves": [{"node_id": "slave_b", "name": "Node B", "base_url": "http://slave"}],
            })
            calls = []

            def get(url, **kwargs):
                return FakeResponse(
                    content=slave_bytes,
                    headers={"X-MIGA-Archive-SHA256": slave_sha, "X-MIGA-Archive-Run-Id": "run07_20260824"},
                )

            def post(url, **kwargs):
                calls.append(url)
                return FakeResponse(payload={"node_id": "master", "status": "installed"})

            with patch("app.core.sync_manager.requests.get", side_effect=get), patch(
                "app.core.sync_manager.requests.post", side_effect=post
            ):
                replication = sync._replicate_archives()

            self.assertEqual(replication["status"], "complete")
            self.assertEqual((run_dir / "sync_nodes" / "slave_b" / "results.csv").read_text(encoding="utf-8"), "slave")
            self.assertEqual(calls, ["http://slave/sync/node/archive/sync_test/master"])

    def test_replication_status_update_preserves_saved_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            original = {
                "runtime": {"sync_run_id": "sync_test", "status": "done"},
                "pairs": [{"sync_shot_index": 0}],
                "node_results": {"master": [{"sync_shot_index": 0}]},
            }
            (run_dir / "sync_manifest.json").write_text(json.dumps(original), encoding="utf-8")
            sync = SyncManager(FakeManager(run_dir))
            sync._update_replication_manifest(
                run_dir,
                {"sync_run_id": "sync_test", "slaves": []},
                {"status": "incomplete", "nodes": {}},
            )
            saved = json.loads((run_dir / "sync_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["pairs"], original["pairs"])
            self.assertEqual(saved["node_results"], original["node_results"])
            self.assertEqual(saved["archive_replication"]["status"], "incomplete")

    def test_legacy_master_manifest_can_start_historical_replication(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            legacy = {
                "runtime": {
                    "sync_run_id": "sync_legacy",
                    "status": "done",
                    "slaves": [{"node_id": "slave_b", "base_url": "http://slave"}],
                },
                "pairs": [{"sync_shot_index": 0}],
            }
            (run_dir / "sync_manifest.json").write_text(json.dumps(legacy), encoding="utf-8")
            sync = SyncManager(FakeManager(run_dir))
            with patch.object(sync, "_replicate_archives", return_value={"status": "complete"}) as replicate:
                result = sync.retry_archive_replication(run_dir)
            self.assertEqual(result["status"], "complete")
            passed_runtime = replicate.call_args.args[1]
            self.assertEqual(passed_runtime["sync_run_id"], "sync_legacy")
            self.assertEqual(passed_runtime["master_run_dir"], str(run_dir.resolve()))

    def test_replica_manifest_cannot_start_master_replication(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            replica = {
                "runtime": {"sync_run_id": "sync_test", "slaves": [{"node_id": "slave_b"}]},
                "archive_nodes": {"master": {"local": False, "path": "sync_nodes/master"}},
            }
            (run_dir / "sync_manifest.json").write_text(json.dumps(replica), encoding="utf-8")
            sync = SyncManager(FakeManager(run_dir, role="slave"))
            with self.assertRaises(ValueError):
                sync.retry_archive_replication(run_dir)

    def test_archive_phase_calibrations_are_loaded_from_selected_remote_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "sync_manifest.json").write_text(json.dumps({
                "runtime": {"sync_run_id": "sync_test", "slaves": [{"node_id": "slave_b", "base_url": "http://slave:8000"}]},
                "archive_nodes": {"master": {"local": True, "path": "."}, "slave_b": {"local": False, "path": "sync_nodes/slave_b"}},
            }), encoding="utf-8")
            manager = FakeManager(run_dir)
            sync = SyncManager(manager)
            remote = {"id": "slave-cal", "name": "Slave fringe"}
            with patch("app.core.sync_manager.requests.get", return_value=FakeResponse(payload={"calibrations": [remote], "active": remote})) as get:
                payload = sync.get_archive_node_phase_calibrations(run_dir, "slave_b")
            self.assertEqual(payload["source"], "remote_settings")
            self.assertEqual(payload["active"]["id"], "slave-cal")
            self.assertEqual(get.call_args.args[0], "http://slave:8000/sync/node/phase-calibrations")
            self.assertEqual(get.call_args.kwargs["headers"], {"X-MIGA-Sync-Token": "secret"})

    def test_master_distributes_versioned_phase_metadata_to_every_slave(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "sync_manifest.json").write_text(json.dumps({
                "runtime": {"sync_run_id": "sync_test", "slaves": [
                    {"node_id": "slave_b", "base_url": "http://slave-b"},
                    {"node_id": "slave_c", "base_url": "http://slave-c"},
                ]},
                "archive_nodes": {"master": {"local": True, "path": "."}},
            }), encoding="utf-8")
            sync = SyncManager(FakeManager(run_dir))
            calls = []

            def post(url, **kwargs):
                calls.append((url, kwargs.get("json")))
                return FakeResponse(payload={"installed": True, "revision": 7})

            metadata = {"version": 2, "revision": 7, "original_calibrations": {}, "overrides": {}}
            with patch("app.core.sync_manager.requests.post", side_effect=post):
                result = sync.distribute_phase_analysis_metadata(run_dir, metadata, "http://master:8000")
            self.assertEqual(result["status"], "synced")
            self.assertEqual([item[0] for item in calls], [
                "http://slave-b/sync/node/archive-phase-analysis/sync_test",
                "http://slave-c/sync/node/archive-phase-analysis/sync_test",
            ])
            self.assertTrue(all(item[1]["coordinator_url"] == "http://master:8000" for item in calls))

    def test_localhost_coordinator_url_is_advertised_as_master_lan_address(self):
        class FakeSocket:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def connect(self, address): self.address = address
            def getsockname(self): return ("192.168.10.12", 54321)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "sync_manifest.json").write_text(json.dumps({
                "runtime": {"slaves": [{"node_id": "slave_b", "base_url": "http://192.168.10.20:8000"}]},
            }), encoding="utf-8")
            sync = SyncManager(FakeManager(run_dir))
            with patch("app.core.sync_manager.socket.socket", return_value=FakeSocket()):
                advertised = sync.advertised_phase_coordinator_url(run_dir, "http://127.0.0.1:8000")
        self.assertEqual(advertised, "http://192.168.10.12:8000")

    def test_archive_replication_recovers_newer_pending_phase_metadata_from_slave(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            replica_dir = run_dir / "sync_nodes" / "slave_b"
            replica_dir.mkdir(parents=True)
            master = {"version": 2, "revision": 2, "nodes": {}, "sync_status": "synced"}
            replica = {"version": 2, "revision": 3, "nodes": {"slave_b": {"override": {"node_id": "slave_b"}}}, "sync_status": "pending"}
            (run_dir / "sync_phase_analysis.json").write_text(json.dumps(master), encoding="utf-8")
            (replica_dir / "sync_phase_analysis.json").write_text(json.dumps(replica), encoding="utf-8")
            sync = SyncManager(FakeManager(run_dir))
            merged = sync._merge_newer_replica_phase_metadata(run_dir, "sync_nodes/slave_b")
            installed = json.loads((run_dir / "sync_phase_analysis.json").read_text(encoding="utf-8"))
            self.assertTrue(merged)
            self.assertEqual(installed["revision"], 3)
            self.assertIn("slave_b", installed["nodes"])
            self.assertTrue((run_dir / "sync_phase_analysis.previous.json").is_file())


if __name__ == "__main__":
    unittest.main()
