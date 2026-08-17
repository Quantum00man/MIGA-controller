import math
import unittest

from app.analysis import physics


class AtomConversionFactorTests(unittest.TestCase):
    def test_default_detection_parameters(self):
        value = physics.calculate_atom_conversion_factor(
            detection_velocity=3.58,
            wavelength=780e-9,
            light_sheet_height=1e-2,
            transimpedance_gain=10e6,
            collection_efficiency=0.02,
            photodiode_responsivity=0.5,
            saturation_ratio=1.5,
            detuning_angular=0.0,
            natural_linewidth_angular=2 * math.pi * 6.02e6,
        )

        self.assertAlmostEqual(value, 1238805950.07661, places=5)

    def test_detuning_uses_angular_frequency_ratio(self):
        gamma = 2 * math.pi * 6.02e6
        resonant = physics.calculate_atom_conversion_factor(
            3.58, 780e-9, 1e-2, 10e6, 0.02, 0.5, 1.5, 0.0, gamma
        )
        detuned = physics.calculate_atom_conversion_factor(
            3.58,
            780e-9,
            1e-2,
            10e6,
            0.02,
            0.5,
            1.5,
            2 * math.pi * 3.01e6,
            gamma,
        )

        expected_ratio = (1 + 1.5 + 1.0) / (1 + 1.5)
        self.assertAlmostEqual(detuned / resonant, expected_ratio)

    def test_rejects_non_positive_denominator_parameter(self):
        with self.assertRaises(ValueError):
            physics.calculate_atom_conversion_factor(
                detection_velocity=3.58,
                wavelength=780e-9,
                light_sheet_height=1e-2,
                transimpedance_gain=10e6,
                collection_efficiency=0.0,
                photodiode_responsivity=0.5,
                saturation_ratio=1.5,
                detuning_angular=0.0,
                natural_linewidth_angular=2 * math.pi * 6.02e6,
            )


