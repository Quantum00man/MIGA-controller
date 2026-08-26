import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.analysis.interferometer_phase import (
    calculate_phase,
    monotonic_slope,
    phase_deviation_limits,
    reference_t2_us2,
    source_field,
)
from app.core.data_loader import DataLoader


def calibration(**updates):
    base = {
        "id": "phase-1",
        "name": "Reference fringe",
        "metric_tab": "intf",
        "channel": "up",
        "source_mode": "fit",
        "parameter_values": {"A": 20.0, "C": 50.0, "phi0": 0.2},
        "bragg": {"angular_frequency_rad_per_us2": 0.01},
        "reference_t2_us2": 100.0,
    }
    base.update(updates)
    return base


class InterferometerPhaseTests(unittest.TestCase):
    def test_reference_units_are_converted_to_us_squared(self):
        self.assertEqual(reference_t2_us2({"reference_input_mode": "t", "reference_t_unit": "ms", "reference_value": 2}), 4_000_000)
        self.assertEqual(reference_t2_us2({"reference_input_mode": "t2", "reference_t_unit": "ms", "reference_value": 4}), 4_000_000)

    def test_default_source_is_interferometer_up_fit(self):
        self.assertEqual(source_field({}), "intf_p1")
        self.assertEqual(source_field({"metric_tab": "prob", "channel": "dw", "source_mode": "raw"}), "transition_probability_dw_nofit")

    def test_negative_slope_uses_monotonic_acos_branch(self):
        cal = calibration()
        reference = 0.01 * 100.0 + 0.2
        expected_delta = 0.04
        signal = 50.0 + 20.0 * math.cos(reference + expected_delta)
        result = calculate_phase({"intf_p1": signal}, cal)
        self.assertTrue(result["interferometer_phase_valid"])
        self.assertAlmostEqual(result["interferometer_phase"], expected_delta, places=10)

    def test_positive_slope_uses_mirrored_monotonic_branch(self):
        reference = 4.5
        cal = calibration(
            monotonic_slope="positive",
            parameter_values={"A": 20.0, "C": 50.0, "phi0": 0.0},
            reference_t2_us2=reference / 0.01,
        )
        expected_delta = 0.04
        signal = 50.0 + 20.0 * math.cos(reference + expected_delta)
        result = calculate_phase({"intf_p1": signal}, cal)
        self.assertTrue(result["interferometer_phase_valid"])
        self.assertAlmostEqual(result["interferometer_phase"], expected_delta, places=10)

    def test_legacy_calibration_infers_reference_slope(self):
        self.assertEqual(monotonic_slope(calibration()), "negative")
        positive = calibration(reference_t2_us2=450.0, parameter_values={"A": 20.0, "C": 50.0, "phi0": 0.0})
        self.assertEqual(monotonic_slope(positive), "positive")

    def test_mid_fringe_has_pi_wide_reference_centered_limits(self):
        cal = calibration(
            monotonic_slope="negative",
            parameter_values={"A": 20.0, "C": 50.0, "phi0": 0.0},
            reference_t2_us2=(math.pi / 2) / 0.01,
        )
        lower, upper = phase_deviation_limits(cal)
        self.assertAlmostEqual(lower, -math.pi / 2)
        self.assertAlmostEqual(upper, math.pi / 2)
        self.assertAlmostEqual(upper - lower, math.pi)

    def test_out_of_envelope_is_invalid_not_clipped(self):
        result = calculate_phase({"intf_p1": 71.0}, calibration())
        self.assertFalse(result["interferometer_phase_valid"])
        self.assertIsNone(result["interferometer_phase"])

    def test_phase_is_in_stats_and_allan_as_single_channel(self):
        loader = DataLoader()
        points = [
            {"parameter": 1.0, "all_parameters": [1.0], "interferometer_phase": value}
            for value in (0.01, 0.03, -0.02, 0.00)
        ]
        stats = loader._build_stats_array(points)
        self.assertAlmostEqual(stats[0]["phase_up"], 0.005)
        self.assertIsNone(stats[0]["phase_dw"])
        allan = loader._build_allan_payload(points, 2)
        self.assertIn("phase", allan["metrics"])
        self.assertGreater(len(allan["metrics"]["phase"]["fit"]["up"]["y"]), 0)

    def test_archive_with_legacy_snapshot_recalculates_monotonic_phase(self):
        cal = calibration()
        reference = 0.01 * 100.0 + 0.2
        expected_delta = 0.08
        signal = 50.0 + 20.0 * math.cos(reference + expected_delta)
        point = {
            "parameter": 1.0,
            "all_parameters": [1.0],
            "intf_p1": signal,
            "interferometer_phase": -2.5,
            "interferometer_phase_valid": True,
            "interferometer_phase_calibration_id": "phase-1",
        }
        config = {"scan_dimensions": 1, "_interferometer_phase_calibration_snapshot": cal}
        loader = DataLoader()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run01_20260826"
            run_dir.mkdir()
            with patch.object(loader, "_get_run_dir", return_value=run_dir), \
                    patch.object(loader, "_load_config_data", return_value=config), \
                    patch.object(loader, "_read_results_csv", return_value=[point]):
                payload = loader.load_run("2026", "08", "26", run_dir.name)
        self.assertEqual(payload["interferometer_phase_calibration_provenance"], "run_snapshot_migrated_monotonic")
        self.assertAlmostEqual(payload["data"][0]["interferometer_phase"], expected_delta, places=10)


if __name__ == "__main__":
    unittest.main()
