import tempfile
import threading
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from app.core.data_loader import DataLoader
from app.core.data_manager import DataManager
from app.core.experiment_manager import ExperimentManager
from app.core.structures import ExperimentStatus, ScanResult
from app.drivers.power_meter import PowerMeterClient, PowerMeterError
from app.models.schemas import ScanConfig, SystemSettings


class PowerMeterMonitorTests(unittest.TestCase):
    def test_scan_config_defaults_to_monitor_disabled(self):
        config = ScanConfig()
        self.assertFalse(config.transfer_power_monitor_enabled)
        self.assertEqual(config.transfer_power_threshold_percent, 0)

    def test_settings_accept_power_meter_connection(self):
        fields = SystemSettings.model_fields
        self.assertIn("power_meter_url", fields)
        self.assertIn("power_meter_password", fields)
        self.assertIn("power_meter_timeout_s", fields)

    def test_client_rejects_missing_configuration(self):
        with self.assertRaisesRegex(PowerMeterError, "URL"):
            PowerMeterClient("", "secret").read()

    def test_manual_pause_and_resume_are_shot_boundary_events(self):
        manager = object.__new__(ExperimentManager)
        manager.status = ExperimentStatus(is_running=True)
        manager.pause_event = threading.Event()
        manager.pause_event.set()
        manager._pause_lock = threading.Lock()
        manager._active_mode = "scan"
        manager._power_meter_reference_w = 1.0
        manager.get_active_mode = Mock(return_value="scan")

        self.assertEqual(manager.pause_scan()["status"], "success")
        self.assertTrue(manager.status.is_paused)
        self.assertFalse(manager.pause_event.is_set())
        self.assertEqual(manager.resume_scan()["status"], "success")
        self.assertFalse(manager.status.is_paused)
        self.assertTrue(manager.pause_event.is_set())

    def test_threshold_pause_cannot_be_bypassed_by_plain_resume(self):
        manager = object.__new__(ExperimentManager)
        manager.status = ExperimentStatus(is_running=True, is_paused=True, pause_reason="power_threshold")
        manager.pause_event = threading.Event()
        manager._pause_lock = threading.Lock()
        manager._active_mode = "scan"
        manager.get_active_mode = Mock(return_value="scan")
        result = manager.resume_scan()
        self.assertEqual(result["status"], "warning")
        self.assertFalse(manager.pause_event.is_set())

    def test_archive_excludes_invalid_frequency_attempt_from_transfer_summary(self):
        loader = DataLoader()
        points = [
            {"transfer_zero_phase_baseline": False, "power_meter_invalid_reason": "threshold_exceeded"},
            {"transfer_zero_phase_baseline": False, "power_meter_invalid_reason": ""},
            {"transfer_zero_phase_baseline": True, "power_meter_invalid_reason": ""},
        ]
        self.assertEqual(loader._transfer_response_points(points), [points[1]])

    def test_invalidation_updates_csv_and_npz_for_complete_attempt(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DataManager()
            manager.current_run_dir = Path(root)
            manager.waveforms_dir = Path(root) / "waveforms"
            manager.waveforms_dir.mkdir()
            manager.csv_file = Path(root) / "results.csv"
            manager._init_csv(manager.csv_file)
            result = ScanResult(
                parameter=1000.0, timestamp=1.0, all_parameters=[1000.0],
                transfer_frequency_hz=1000.0, transfer_phase_deg=0.0,
                transfer_repeat=1, transfer_frequency_attempt=1,
                power_meter_power_w=1e-6, power_meter_valid=True,
                raw_data_up=[], raw_data_dw=[],
            )
            manager.save_point(result, 1)
            manager.invalidate_transfer_attempt(1000.0, 1, "threshold_exceeded")
            manager.close_run()
            point = DataLoader()._read_results_csv(Path(root), max_points=None)[0]
            self.assertEqual(point["power_meter_invalid_reason"], "threshold_exceeded")
            import numpy as np
            with np.load(Path(root) / "waveforms" / "step_0001.npz") as archive:
                self.assertEqual(str(archive["power_meter_invalid_reason"]), "threshold_exceeded")

    def test_previous_bragg_run_can_supply_recovery_mot_and_parameters(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "2026" / "09" / "23" / "run07_20260923"
            run_dir.mkdir(parents=True)
            (run_dir / "sequence.mot").write_text("<PARAMETER0>", encoding="utf-8")
            (run_dir / "config.json").write_text(json.dumps({
                "mode": "bragg_fringe_calibration", "sequence_name": "calibration.mot",
                "start": 10, "stop": 20, "step": 2,
                "link_formulas": ["P0", "100-P0"],
                "bragg_calibration_target_fringe": 3,
                "bragg_calibration_coarse_repeats": 4,
                "bragg_calibration_fine_points": 9,
                "bragg_calibration_fine_repeats": 6,
            }), encoding="utf-8")
            (run_dir / "bragg_fringe_calibration.json").write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )
            manager = object.__new__(ExperimentManager)
            with patch("app.core.experiment_manager.config.DATA_BASE_DIR", Path(root)):
                catalog = manager.list_bragg_recovery_runs()
                loaded = manager.load_bragg_recovery_run("2026", "09", "23", run_dir.name)
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog[0]["target_fringe_number"], 3)
            self.assertEqual(loaded["calibration_config"]["bragg_calibration_fine_points"], 9)
            self.assertEqual(loaded["source_run"]["run_id"], run_dir.name)
            self.assertTrue(loaded["sequence_content_base64"])

    def test_recovery_catalog_filters_non_calibration_runs(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "2026" / "09" / "23" / "run08_20260923"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(json.dumps({"mode": "transfer_function"}), encoding="utf-8")
            manager = object.__new__(ExperimentManager)
            with patch("app.core.experiment_manager.config.DATA_BASE_DIR", Path(root)):
                self.assertEqual(manager.list_bragg_recovery_runs(), [])


if __name__ == "__main__":
    unittest.main()
