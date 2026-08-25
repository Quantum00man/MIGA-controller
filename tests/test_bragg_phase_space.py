import math
import unittest

from app.analysis.phase_space import convert_bragg_points_to_phase_space


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


if __name__ == "__main__":
    unittest.main()
