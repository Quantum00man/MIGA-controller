import unittest

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
        self.assertEqual(boundaries, ["start", "periodic", "periodic", "end"])


if __name__ == "__main__":
    unittest.main()
