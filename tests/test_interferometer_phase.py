import math
import unittest

from app.analysis.interferometer_phase import calculate_phase, reference_t2_us2, source_field
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

    def test_phase_is_branch_resolved_against_reference(self):
        cal = calibration()
        reference = 0.01 * 100.0 + 0.2
        expected_delta = 0.04
        signal = 50.0 + 20.0 * math.cos(reference + expected_delta)
        result = calculate_phase({"intf_p1": signal}, cal)
        self.assertTrue(result["interferometer_phase_valid"])
        self.assertAlmostEqual(result["interferometer_phase"], expected_delta, places=10)

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


if __name__ == "__main__":
    unittest.main()
