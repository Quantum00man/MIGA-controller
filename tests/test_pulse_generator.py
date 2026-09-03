import unittest

from app.core.pulse_generator import (
    BRAGG_OFF_VOLTAGE,
    generate_bragg_pulse,
    normalized_power_to_voltage,
)
from app.models.schemas import BraggPowerCalibration


CALIBRATION = {
    "lower_asymptote": 0.0,
    "upper_asymptote": 1.0,
    "growth_rate": 2.0,
    "midpoint": 0.0,
    "shape": 1.0,
    "linear_voltage_min": -0.5,
    "linear_voltage_max": 0.5,
    "off_threshold": 0.001,
}


class BraggCalibrationTests(unittest.TestCase):
    def test_selected_linear_interval_is_normalized_to_zero_and_one(self):
        self.assertEqual(normalized_power_to_voltage(0.001, CALIBRATION), BRAGG_OFF_VOLTAGE)
        self.assertAlmostEqual(normalized_power_to_voltage(0.5, CALIBRATION), 0.0, places=12)
        self.assertAlmostEqual(normalized_power_to_voltage(1.0, CALIBRATION), 0.5, places=12)
        self.assertGreater(normalized_power_to_voltage(0.0011, CALIBRATION), -0.5)

    def test_schema_rejects_voltage_interval_outside_hardware_range(self):
        calibration = BraggPowerCalibration(linear_voltage_min=-15.0, linear_voltage_max=15.0)
        self.assertEqual(calibration.linear_voltage_min, -15.0)
        self.assertEqual(calibration.linear_voltage_max, 15.0)
        with self.assertRaises(ValueError):
            BraggPowerCalibration(linear_voltage_min=-15.1)
        with self.assertRaises(ValueError):
            BraggPowerCalibration(linear_voltage_max=15.1)
        with self.assertRaises(ValueError):
            BraggPowerCalibration(linear_voltage_min=1.0, linear_voltage_max=0.0)

    def test_mapping_rejects_invalid_runtime_interval(self):
        invalid = {**CALIBRATION, "linear_voltage_max": 15.1}
        with self.assertRaisesRegex(ValueError, "-15 <= min < max <= 15"):
            normalized_power_to_voltage(0.5, invalid)


class BraggPulseGenerationTests(unittest.TestCase):
    def test_blackman_uses_channel_32_three_decimal_voltages_and_off_edges(self):
        pulse_code, compensation = generate_bragg_pulse(
            fwhm=4.0,
            shape="blackman",
            base_timing=100,
            calibration=CALIBRATION,
        )
        lines = pulse_code.splitlines()
        self.assertTrue(lines[0].startswith("+500.0us Blackman_pulse = -3.000"))
        self.assertTrue(lines[-1].startswith("+0.2us Blackman_pulse = -3.000"))
        self.assertTrue(all(line.endswith("(32)") for line in lines))
        voltage_tokens = [line.split(" = ", 1)[1].split()[0] for line in lines]
        self.assertTrue(all(len(token.split(".")[1]) == 3 for token in voltage_tokens))
        self.assertTrue(any(float(token) > -0.5 for token in voltage_tokens))
        self.assertEqual(compensation, "89.8")

    def test_gaussian_tail_below_threshold_is_off(self):
        pulse_code, _ = generate_bragg_pulse(
            fwhm=2.0,
            shape="gaussian",
            base_timing=100,
            calibration=CALIBRATION,
        )
        lines = pulse_code.splitlines()
        self.assertIn(" = -3.000", lines[1])
        self.assertIn(" = -3.000", lines[-2])

    def test_pulse_cannot_exceed_base_timing(self):
        with self.assertRaisesRegex(ValueError, "exceeds base timing"):
            generate_bragg_pulse(
                fwhm=20.0,
                shape="blackman",
                base_timing=1,
                calibration=CALIBRATION,
            )


if __name__ == "__main__":
    unittest.main()
