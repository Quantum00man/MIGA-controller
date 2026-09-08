import math
from pathlib import Path
import unittest

from app.analysis.phase_noise import (
    build_phase_noise_summary,
    calibration_at_mid_fringe,
    validate_mid_fringe_values,
)
from app.core.experiment_manager import ExperimentManager
from app.models.schemas import ScanConfig, SystemSettings


class PhaseNoiseAnalyzeTests(unittest.TestCase):
    def calibration(self):
        return {
            "id": "phase-cal",
            "name": "Interferometer fringe",
            "metric_tab": "intf",
            "channel": "up",
            "source_mode": "fit",
            "parameter_values": {"A": 20.0, "C": 50.0, "phi0": 0.0},
            "bragg": {
                "angular_frequency_rad_per_us2": 0.1,
                "mid_fringe_x": [5.0, 15.0],
            },
        }

    def test_defaults_are_exposed_by_validated_settings(self):
        fields = SystemSettings.model_fields
        self.assertEqual(fields["std_p_interferometer"].default, 1.1)
        self.assertEqual(fields["laser_frequency_phase_noise_mrad"].default, 100.0)
        scan = ScanConfig(mode="phase_noise")
        self.assertEqual(scan.phase_noise_repeats, 10)

    def test_mid_fringe_reference_selects_its_own_slope(self):
        first = calibration_at_mid_fringe(self.calibration(), 5.0)
        second = calibration_at_mid_fringe(self.calibration(), 15.0)
        self.assertEqual(first["reference_t2_us2"], 5.0)
        self.assertEqual(second["reference_t2_us2"], 15.0)
        self.assertIn(first["monotonic_slope"], {"positive", "negative"})
        self.assertIn(second["monotonic_slope"], {"positive", "negative"})

    def test_scan_plan_keeps_repeats_contiguous_and_evaluates_link_formulas(self):
        manager = ExperimentManager.__new__(ExperimentManager)
        config = {
            "mode": "phase_noise",
            "scan_dimensions": 1,
            "phase_noise_mid_fringe_values": [15.0, 5.0],
            "phase_noise_repeats": 3,
            "link_formulas": ["P0 * 2", "P1 + 1"],
            "interferometer_phase_calibration_override": self.calibration(),
        }
        plan = manager._build_phase_noise_execution(config)
        self.assertEqual([item["metadata"]["phase_noise_t2_us2"] for item in plan], [5.0] * 3 + [15.0] * 3)
        self.assertEqual(plan[0]["sequence_parameters"], [5.0, 10.0, 11.0])
        self.assertEqual(plan[3]["sequence_parameters"], [15.0, 30.0, 31.0])
        self.assertFalse(config["randomize"])
        self.assertEqual(config["averages"], 1)

    def test_selected_values_must_come_from_calibration(self):
        self.assertEqual(validate_mid_fringe_values(self.calibration(), [15, 5, 15]), [5.0, 15.0])
        with self.assertRaises(ValueError):
            validate_mid_fringe_values(self.calibration(), [10])

    def test_summary_uses_sample_std_and_quadrature_baseline(self):
        points = [
            {"parameter": 5.0, "interferometer_phase_reference_t2_us2": 5.0,
             "interferometer_phase": value, "interferometer_phase_valid": True}
            for value in (0.0, 0.1, 0.2)
        ] + [
            {"parameter": 15.0, "interferometer_phase_reference_t2_us2": 15.0,
             "interferometer_phase": 0.3, "interferometer_phase_valid": True},
            {"parameter": 15.0, "interferometer_phase_reference_t2_us2": 15.0,
             "interferometer_phase": None, "interferometer_phase_valid": False},
        ]
        rows = build_phase_noise_summary(points, self.calibration(), 1.1, 100.0)
        self.assertAlmostEqual(rows[0]["measured_phase_noise_rad"], 0.1)
        self.assertAlmostEqual(rows[0]["detection_phase_noise_rad"], 1.1 / 20.0)
        self.assertAlmostEqual(rows[0]["laser_phase_noise_rad"], 0.1)
        self.assertAlmostEqual(rows[0]["expected_total_phase_noise_rad"], math.hypot(1.1 / 20.0, 0.1))
        self.assertIsNone(rows[1]["measured_phase_noise_rad"])
        self.assertEqual(rows[1]["valid_phase_count"], 1)
        self.assertEqual(rows[1]["shot_count"], 2)

    def test_pages_expose_new_settings_mode_and_archive_plot(self):
        root = Path(__file__).resolve().parents[1]
        settings = (root / "static" / "settings.html").read_text(encoding="utf-8")
        index = (root / "static" / "index.html").read_text(encoding="utf-8")
        archive = (root / "static" / "archive.html").read_text(encoding="utf-8")
        self.assertIn("Phase Noise Analyze", settings)
        self.assertIn("std_p_interferometer", settings)
        self.assertIn('value="phase_noise"', index)
        self.assertIn("phaseNoiseSelectionPlot", index)
        self.assertIn("renderPhaseNoiseArchivePlot", archive)
        self.assertIn("expected_total_phase_noise_rad", archive)


if __name__ == "__main__":
    unittest.main()
