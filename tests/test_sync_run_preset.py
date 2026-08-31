import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.experiment_manager import ExperimentManager


class SyncRunPresetTests(unittest.TestCase):
    def _manager(self, slave_ids):
        manager = object.__new__(ExperimentManager)
        manager.status = SimpleNamespace(is_running=False)
        manager.settings = {
            "sync_role": "master",
            "sync_slaves": [
                {"id": node_id, "name": node_id.upper(), "enabled": True}
                for node_id in slave_ids
            ],
        }
        return manager

    def _create_run(self, root: Path, *, include_master_sequence=True, include_slave_sequence=True):
        run_dir = root / "2026" / "08" / "25" / "run01_20260825"
        slave_dir = run_dir / "sync_nodes" / "slave_b"
        slave_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "mode": "link",
            "start": 1,
            "stop": 3,
            "step": 1,
            "averages": 2,
            "randomize": True,
            "link_formulas": ["P0 * 2"],
            "sequence_name": "master.mot",
        }), encoding="utf-8")
        if include_master_sequence:
            (run_dir / "sequence.mot").write_bytes(b"master-sequence")
        (slave_dir / "config.json").write_text(
            json.dumps({"sequence_name": "slave-original.mot"}), encoding="utf-8"
        )
        if include_slave_sequence:
            (slave_dir / "sequence.mot").write_bytes(b"\xb5slave-sequence")
        (run_dir / "sync_manifest.json").write_text(json.dumps({
            "runtime": {
                "sync_run_id": "sync_test",
                "master_delay_ms": 37.5,
                "slaves": [{"node_id": "slave_b", "name": "Node B", "base_url": "http://10.0.0.21:8000/"}],
            },
            "archive_nodes": {
                "master": {"path": ".", "local": True},
                "slave_b": {"path": "sync_nodes/slave_b", "role": "slave"},
            },
        }), encoding="utf-8")
        return run_dir

    def test_sync_preset_restores_master_and_binary_slave_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_run(root)
            template_target = root / "active" / "seq0.mot"
            manager = self._manager(["slave_b"])
            with patch("app.core.experiment_manager.config.DATA_BASE_DIR", root), patch(
                "app.core.experiment_manager.config.SEQUENCE_TEMPLATE_PATH_LINUX", str(template_target)
            ):
                payload = manager.load_run_preset(
                    "2026", "08", "25", "run01_20260825", include_sync=True
                )

            self.assertEqual(template_target.read_bytes(), b"master-sequence")
            self.assertEqual(payload["config"]["mode"], "link")
            self.assertEqual(payload["sync_preset"]["master_delay_ms"], 37.5)
            self.assertEqual(payload["sync_preset"]["slaves"][0]["node_id"], "slave_b")
            self.assertEqual(payload["sync_preset"]["slaves"][0]["sequence_name"], "slave-original.mot")
            self.assertEqual(
                base64.b64decode(payload["sync_preset"]["slaves"][0]["sequence_content_base64"]),
                b"\xb5slave-sequence",
            )

    def test_sync_preset_rejects_slave_id_mismatch_before_loading_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_run(root)
            template_target = root / "active" / "seq0.mot"
            manager = self._manager(["different_slave"])
            with patch("app.core.experiment_manager.config.DATA_BASE_DIR", root), patch(
                "app.core.experiment_manager.config.SEQUENCE_TEMPLATE_PATH_LINUX", str(template_target)
            ):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    manager.load_run_preset(
                        "2026", "08", "25", "run01_20260825", include_sync=True
                    )
            self.assertFalse(template_target.exists())

    def test_sync_preset_maps_changed_internal_id_by_controller_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_run(root)
            template_target = root / "active" / "seq0.mot"
            manager = self._manager(["slave_new_timestamp_id"])
            manager.settings["sync_slaves"][0].update({
                "name": "MIGA21",
                "base_url": "10.0.0.21:8000",
            })
            with patch("app.core.experiment_manager.config.DATA_BASE_DIR", root), patch(
                "app.core.experiment_manager.config.SEQUENCE_TEMPLATE_PATH_LINUX", str(template_target)
            ):
                payload = manager.load_run_preset(
                    "2026", "08", "25", "run01_20260825", include_sync=True
                )

            restored = payload["sync_preset"]["slaves"][0]
            self.assertEqual(restored["node_id"], "slave_new_timestamp_id")
            self.assertEqual(restored["archive_node_id"], "slave_b")
            self.assertEqual(restored["match_method"], "url")
            self.assertEqual(base64.b64decode(restored["sequence_content_base64"]), b"\xb5slave-sequence")

    def test_sync_preset_requires_historical_replication_for_missing_slave_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_run(root, include_slave_sequence=False)
            template_target = root / "active" / "seq0.mot"
            manager = self._manager(["slave_b"])
            with patch("app.core.experiment_manager.config.DATA_BASE_DIR", root), patch(
                "app.core.experiment_manager.config.SEQUENCE_TEMPLATE_PATH_LINUX", str(template_target)
            ):
                with self.assertRaisesRegex(ValueError, "Sync Historical Archive"):
                    manager.load_run_preset(
                        "2026", "08", "25", "run01_20260825", include_sync=True
                    )
            self.assertFalse(template_target.exists())

    def test_sync_preset_rejects_missing_master_sequence_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_run(root, include_master_sequence=False)
            template_target = root / "active" / "seq0.mot"
            manager = self._manager(["slave_b"])
            with patch("app.core.experiment_manager.config.DATA_BASE_DIR", root), patch(
                "app.core.experiment_manager.config.SEQUENCE_TEMPLATE_PATH_LINUX", str(template_target)
            ):
                with self.assertRaisesRegex(ValueError, "Master sequence"):
                    manager.load_run_preset(
                        "2026", "08", "25", "run01_20260825", include_sync=True
                    )
            self.assertFalse(template_target.exists())


if __name__ == "__main__":
    unittest.main()
