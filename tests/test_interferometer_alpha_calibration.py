import json
import tempfile
import unittest
from pathlib import Path

from app.analysis.interferometer_alpha import (
    alpha_from_probability_percent,
    interpolate_alpha,
    summarize_block,
)
from app.core.experiment_manager import ExperimentManager
from app.core.data_loader import DataLoader
from app.models.schemas import ScanConfig


class InterferometerAlphaAnalysisTests(unittest.TestCase):
    def test_archive_science_points_exclude_calibration_shots(self):
        points = [
            {"step": 1, "parameter": 0.0, "intf_alpha_calibration_block_id": 1},
            {"step": 11, "parameter": 100.0, "intf_alpha_calibration_block_id": None},
        ]
        self.assertEqual(DataLoader._science_points(points), [points[1]])

    def test_probability_percent_is_converted_to_unit_alpha(self):
        self.assertAlmostEqual(alpha_from_probability_percent(30.0, 0.25), 0.4)

    def test_block_summary_uses_fit_probability_statistics(self):
        event = summarize_block([29.0, 30.0, 31.0], [10.0, 11.0, 12.0], 0.25, "alpha-0001", 3)
        self.assertTrue(event["accepted"])
        self.assertAlmostEqual(event["representative_time"], 11.0)
        self.assertAlmostEqual(event["intf_alpha"], 0.4)
        self.assertGreater(event["intf_alpha_sem"], 0)

    def test_block_outside_configured_alpha_range_is_rejected(self):
        event = summarize_block(
            [29.0, 30.0, 31.0], [10.0, 11.0, 12.0], 0.25,
            "alpha-0001", 3, 0.45, 0.55,
        )
        self.assertFalse(event["accepted"])
        self.assertAlmostEqual(event["intf_alpha"], 0.4)
        self.assertIn("outside accepted range [0.45, 0.55]", event["rejection_reason"])
        self.assertEqual(event["accepted_min"], 0.45)
        self.assertEqual(event["accepted_max"], 0.55)

    def test_linear_interpolation_and_edge_extrapolation(self):
        events = [
            {"accepted": True, "representative_time": 10.0, "intf_alpha": 0.3, "calibration_id": "a"},
            {"accepted": True, "representative_time": 20.0, "intf_alpha": 0.5, "calibration_id": "b"},
        ]
        middle = interpolate_alpha(15.0, events)
        self.assertAlmostEqual(middle["value"], 0.4)
        self.assertFalse(middle["extrapolated"])
        before = interpolate_alpha(5.0, events)
        self.assertAlmostEqual(before["value"], 0.3)
        self.assertTrue(before["extrapolated"])

    def test_archive_selection_changes_interpolated_alpha_without_mutating_events(self):
        loader = DataLoader()
        points = [{
            "timestamp": 15.0, "atom_number_dw": 70.0, "atom_number_up": 30.0,
            "atom_number_dw_nofit": 70.0, "atom_number_up_nofit": 30.0,
        }]
        events = [
            {"accepted": True, "representative_time": 10.0, "intf_alpha": 0.2, "calibration_id": "a"},
            {"accepted": False, "representative_time": 15.0, "intf_alpha": 0.9, "calibration_id": "outlier"},
            {"accepted": True, "representative_time": 20.0, "intf_alpha": 0.6, "calibration_id": "b"},
        ]
        result = loader._apply_intf_alpha_history(
            [dict(points[0])], events, {"_system_settings_snapshot": {}}, None
        )
        self.assertAlmostEqual(result[0]["intf_alpha_applied"], 0.4)
        self.assertFalse(events[1]["accepted"])

    def test_archive_analysis_copy_round_trip(self):
        loader = DataLoader()
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root)
            saved = loader.save_intf_alpha_analysis_copy(run_dir, "exclude drift spike", ["b", "a", "a"])
            copies = loader.load_intf_alpha_analysis_copies(run_dir)
            self.assertEqual(copies, [saved])
            self.assertEqual(saved["accepted_calibration_ids"], ["a", "b"])
            self.assertEqual(saved["interpolation"], "linear")
            self.assertEqual(json.loads((run_dir / "intf_alpha_analysis_copies.json").read_text()), copies)

    def test_archive_analysis_copy_requires_name(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "name is required"):
                DataLoader().save_intf_alpha_analysis_copy(Path(root), "  ", ["a"])


class InterferometerAlphaPlanTests(unittest.TestCase):
    def setUp(self):
        ExperimentManager._instance = None
        self.manager = ExperimentManager()

    def tearDown(self):
        ExperimentManager._instance = None

    def test_schema_defaults_to_disabled_ten_shots(self):
        config = ScanConfig()
        self.assertFalse(config.intf_alpha_calibration_enabled)
        self.assertEqual(config.intf_alpha_calibration_shots, 10)

    def test_local_settings_own_the_calibration_mot(self):
        self.manager.settings["intf_alpha_calibration_sequence_name"] = "slave-alpha.mot"
        self.manager.settings["intf_alpha_calibration_sequence_content_base64"] = "c2xhdmU="
        self.assertEqual(self.manager.settings["intf_alpha_calibration_sequence_name"], "slave-alpha.mot")

    def test_transfer_plan_adds_start_phase_boundaries_and_end(self):
        config = {
            "mode": "transfer_function", "scan_dimensions": 1, "parameter_source": "classic",
            "randomize": False, "transfer_frequency_start_hz": 100,
            "transfer_frequency_stop_hz": 200, "transfer_frequency_step_hz": 100,
            "transfer_repeats": 2, "transfer_phase_degrees": [0, 90],
            "intf_alpha_calibration_enabled": True,
        }
        self.manager.settings["intf_alpha_calibration_sequence_content_base64"] = "YQ=="
        plan = self.manager._build_transfer_function_execution(config)
        boundaries = [item["metadata"].get("intf_alpha_calibration_boundary") for item in plan if item["metadata"].get("intf_alpha_calibration_boundary")]
        self.assertEqual(boundaries, ["start", "periodic", "periodic", "periodic", "end"])

    def test_bragg_coarse_plan_checks_interval_after_each_p0_block(self):
        config = {
            "mode": "bragg_fringe_calibration", "scan_dimensions": 1,
            "parameter_source": "classic", "dim1_type": "list",
            "custom_list": "10,20,30,40", "param_type": "float",
            "bragg_calibration_coarse_repeats": 2,
            "bragg_calibration_fine_points": 7,
            "bragg_calibration_fine_repeats": 2,
            "intf_alpha_calibration_enabled": True,
        }
        plan = self.manager._build_bragg_calibration_execution(config)
        boundaries = [
            item for item in plan
            if (item.get("metadata") or {}).get("intf_alpha_calibration_boundary") == "periodic"
        ]
        coarse = [item for item in plan if (item.get("metadata") or {}).get("bragg_calibration_stage") == "coarse"]
        self.assertEqual(len(coarse), 8)
        self.assertEqual(len(boundaries), 3)
        self.assertEqual(config["_bragg_calibration_coarse_shots"], 8)


if __name__ == "__main__":
    unittest.main()
