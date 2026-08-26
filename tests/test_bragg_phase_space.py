import math
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.analysis.phase_space import convert_bragg_points_to_phase_space
from app.core.experiment_manager import ExperimentManager


class BraggPhaseSpaceTests(unittest.TestCase):
    def test_recovers_nearest_phase_branch_across_cycles(self):
        amplitude, offset, omega, phi0 = 0.35, 0.52, 0.8, 0.2
        deviations = [0.08, -0.06, 0.11, -0.09]
        p0_values = [2.0, 9.0, 17.0, 25.0]
        points = []
        for p0, deviation in zip(p0_values, deviations):
            predicted = omega * p0 + phi0
            points.append({"p0": p0, "value": offset + amplitude * math.cos(predicted + deviation)})

        result = convert_bragg_points_to_phase_space(
            points,
            amplitude=amplitude,
            offset=offset,
            angular_frequency_rad_per_us2=omega,
            phase_offset_rad=phi0,
            mid_fringe_fraction=1.0,
        )
        for actual, expected in zip(
            [point["phase_deviation_rad"] for point in result["points"]], deviations
        ):
            self.assertAlmostEqual(actual, expected, places=10)

    def test_mid_fringe_uses_predicted_phase_and_excludes_clipped_points(self):
        result = convert_bragg_points_to_phase_space(
            [
                {"p0": math.pi / 2, "value": 1.4, "shot": 3},
                {"p0": 0.0, "value": 0.0, "shot": 4},
            ],
            amplitude=0.4,
            offset=0.5,
            angular_frequency_rad_per_us2=1.0,
            phase_offset_rad=0.0,
            mid_fringe_fraction=0.5,
        )
        first, second = result["points"]
        self.assertTrue(first["is_mid_fringe"])
        self.assertTrue(first["is_clipped"])
        self.assertFalse(first["is_high_quality"])
        self.assertFalse(second["is_mid_fringe"])
        self.assertEqual(result["statistics"]["mid_fringe"]["count"], 0)

    def test_mid_fringe_statistics_use_sample_standard_deviation(self):
        deviations = [-0.1, 0.0, 0.1]
        points = []
        for index, deviation in enumerate(deviations):
            predicted = math.pi / 2 + index * 2 * math.pi
            points.append({
                "p0": predicted,
                "value": 0.5 + 0.4 * math.cos(predicted + deviation),
            })
        result = convert_bragg_points_to_phase_space(
            points,
            amplitude=0.4,
            offset=0.5,
            angular_frequency_rad_per_us2=1.0,
            phase_offset_rad=0.0,
            mid_fringe_fraction=0.5,
        )
        stats = result["statistics"]["mid_fringe"]
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean_rad"], 0.0, places=10)
        self.assertAlmostEqual(stats["rms_rad"], math.sqrt(0.02 / 3), places=10)
        self.assertAlmostEqual(stats["std_rad"], 0.1, places=10)

    def test_reference_points_correct_offset_before_phase_conversion(self):
        points = [
            {"p0": math.pi / 2, "value": 0.57},
            {"p0": 5 * math.pi / 2, "value": 0.57},
        ]
        result = convert_bragg_points_to_phase_space(
            points,
            amplitude=0.4,
            offset=0.5,
            angular_frequency_rad_per_us2=1.0,
            phase_offset_rad=0.0,
            mid_fringe_fraction=0.5,
            offset_correction_mode="reference_first",
            reference_point_count=2,
        )
        self.assertAlmostEqual(result["offset_correction"], 0.07, places=10)
        self.assertAlmostEqual(result["applied_offset"], 0.57, places=10)
        for point in result["points"]:
            self.assertAlmostEqual(point["phase_deviation_rad"], 0.0, places=10)

    def test_fixed_mid_fringe_ignores_target_p0_for_phase_branch(self):
        reference_phase = 5 * math.pi / 2
        deviations = [-0.08, 0.04, 0.11]
        points = [
            {"p0": scan_index, "value": 0.5 + 0.4 * math.cos(reference_phase + deviation)}
            for scan_index, deviation in enumerate(deviations, start=101)
        ]
        result = convert_bragg_points_to_phase_space(
            points,
            amplitude=0.4,
            offset=0.5,
            angular_frequency_rad_per_us2=0.7,
            phase_offset_rad=0.2,
            mid_fringe_fraction=0.5,
            phase_reference_mode="fixed_mid_fringe",
            reference_phase_rad=reference_phase,
        )
        self.assertEqual(result["phase_reference_mode"], "fixed_mid_fringe")
        for point, expected in zip(result["points"], deviations):
            self.assertAlmostEqual(point["predicted_phase_rad"], reference_phase, places=10)
            self.assertAlmostEqual(point["phase_deviation_rad"], expected, places=10)
            self.assertTrue(point["is_mid_fringe"])

    def test_shot_phase_noise_is_grouped_sample_standard_deviation(self):
        reference_phase = math.pi / 2
        deviations = [-0.1, 0.0, 0.1]
        result = convert_bragg_points_to_phase_space(
            [
                {
                    "p0": 7,
                    "key": "scan-7",
                    "shot": index,
                    "value": 0.5 + 0.4 * math.cos(reference_phase + deviation),
                }
                for index, deviation in enumerate(deviations)
            ],
            amplitude=0.4,
            offset=0.5,
            angular_frequency_rad_per_us2=1.0,
            phase_offset_rad=0.0,
            phase_reference_mode="fixed_mid_fringe",
            reference_phase_rad=reference_phase,
        )
        noise = result["noise_groups"][0]
        self.assertEqual(noise["method"], "shots")
        self.assertEqual(noise["sample_count"], 3)
        self.assertAlmostEqual(noise["phase_std_rad"], 0.1, places=10)
        self.assertAlmostEqual(noise["phase_centered_rms_rad"], math.sqrt(0.02 / 3), places=10)
        self.assertAlmostEqual(result["noise_statistics"]["pooled_std_rad"], 0.1, places=10)

    def test_mean_signal_deviation_propagates_to_phase_noise(self):
        result = convert_bragg_points_to_phase_space(
            [{"p0": 1, "value": 0.5, "value_std": 0.0394, "sample_count": 20}],
            amplitude=0.1,
            offset=0.5,
            angular_frequency_rad_per_us2=1.0,
            phase_offset_rad=0.0,
            phase_reference_mode="fixed_mid_fringe",
            reference_phase_rad=math.pi / 2,
        )
        noise = result["noise_groups"][0]
        self.assertEqual(noise["method"], "propagated")
        self.assertEqual(noise["sample_count"], 20)
        self.assertAlmostEqual(noise["phase_std_rad"], 0.394, places=10)

    def test_saved_calibration_round_trip(self):
        fit_result = {
            "fit_min": 1.0,
            "fit_max": 3.0,
            "fit_x": [1.0, 2.0, 3.0],
            "fit_y": [0.2, 0.5, 0.8],
            "metric_tab": "prob",
            "metric_label": "Transition Prob.",
            "source_key": "fit",
            "source_label": "Fit",
            "channel": "up",
            "channel_label": "Prob F2",
            "parameter_values": {"A": 0.3, "C": 0.5, "phi0": 0.2, "a": 0.01},
            "bragg": {
                "angular_frequency_rad_per_us2": 0.7,
                "wavelength_nm": 780.0,
                "order": 1,
                "mid_fringe_x": [1.5, 2.5],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "settings.json"
            target.write_text("{}", encoding="utf-8")
            manager = ExperimentManager.__new__(ExperimentManager)
            with patch("app.core.experiment_manager.config.USER_JSON_PATH", target):
                saved = manager.save_bragg_phase_calibration(
                    "Morning fringe", fit_result, {"run_id": "run12"}
                )
                loaded = manager.get_bragg_phase_calibrations()
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["id"], saved["id"])
                self.assertEqual(loaded[0]["source"]["run_id"], "run12")
                self.assertEqual(loaded[0]["bragg"]["mid_fringe_x"], [1.5, 2.5])
                self.assertEqual(loaded[0]["monotonic_slope"], "negative")
                self.assertEqual(loaded[0]["phase_conversion_mode"], "monotonic_half_fringe")
                self.assertTrue(manager.delete_bragg_phase_calibration(saved["id"]))
                self.assertEqual(manager.get_bragg_phase_calibrations(), [])
                self.assertEqual(json.loads(target.read_text())["bragg_phase_calibrations"], [])


if __name__ == "__main__":
    unittest.main()
