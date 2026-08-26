import asyncio
import math
from pathlib import Path
import unittest

import numpy as np

from app.analysis import fitting
from app.api.routes import fit_archive_scan
from app.models.schemas import ArchiveScanFitRequest, FitModelDefinition


class ArchiveScanFittingTests(unittest.TestCase):
    def test_saved_phase_calibration_is_not_overlaid_on_regular_archive_plot(self):
        archive_html = (Path(__file__).resolve().parents[1] / "static" / "archive.html").read_text(encoding="utf-8")
        self.assertNotIn("buildBraggCalibrationReferenceTrace", archive_html)
        self.assertNotIn("const savedBraggReference = this.showScanFitCurve", archive_html)
        self.assertIn("const scanFitOverlay = this.getCurrentScanFitOverlay()", archive_html)

    def test_bragg_acceleration_and_angle_display_nine_decimal_places(self):
        archive_html = (Path(__file__).resolve().parents[1] / "static" / "archive.html").read_text(encoding="utf-8")
        self.assertIn("formatBraggPrecision(scanFitResult.parameter_values?.a)", archive_html)
        self.assertIn("formatBraggPrecision(scanFitResult.parameter_values?.alpha)", archive_html)
        self.assertIn("numeric.toFixed(9)", archive_html)

    def test_default_scan_models_include_bragg_fringe_fit(self):
        models = fitting.get_default_scan_fit_models()
        bragg = next(model for model in models if model["key"] == "bragg_fringes")

        self.assertEqual(bragg["label"], "Bragg Fringes")
        self.assertIn("phi0", bragg["formula"])

    def test_bragg_fringe_fit_recovers_acceleration_angle_and_phase(self):
        wavelength_nm = 780.0
        bragg_order = 1
        gravity = 9.80665
        expected_alpha_mrad = 0.5
        expected_acceleration = gravity * math.sin(expected_alpha_mrad * 1e-3)
        effective_wavevector = 4.0 * math.pi * bragg_order / (wavelength_nm * 1e-9)
        omega = effective_wavevector * expected_acceleration * 1e-12
        expected_amplitude = 0.23
        expected_offset = 0.51
        expected_phase = 0.4
        x_values = np.linspace(0.0, 5.0e9, 320)
        y_values = expected_offset + expected_amplitude * np.cos(omega * x_values + expected_phase)

        result = fitting.perform_bragg_fringe_fit(
            x_values,
            y_values,
            wavelength_nm=wavelength_nm,
            bragg_order=bragg_order,
            eval_x=x_values,
        )

        self.assertIsNotNone(result)
        params = result.parameter_values
        self.assertAlmostEqual(params["a"], expected_acceleration, delta=expected_acceleration * 1e-4)
        self.assertAlmostEqual(params["alpha"], expected_alpha_mrad, delta=1e-4)
        self.assertAlmostEqual(params["phi0"], expected_phase, delta=1e-3)
        self.assertAlmostEqual(params["A"], expected_amplitude, delta=1e-5)
        self.assertAlmostEqual(params["C"], expected_offset, delta=1e-5)
        self.assertGreater(len(result.mid_fringe_x), 1)
        self.assertTrue(all(x_values[0] <= value <= x_values[-1] for value in result.mid_fringe_x))
        self.assertAlmostEqual(result.mid_fringe_spacing_us2, math.pi / omega, delta=(math.pi / omega) * 1e-4)
        for value in result.mid_fringe_x:
            fitted_value = params["C"] + params["A"] * math.cos(
                result.angular_frequency_rad_per_us2 * value + params["phi0"]
            )
            self.assertAlmostEqual(fitted_value, params["C"], delta=1e-9)

    def test_archive_bragg_fringe_response_includes_physical_metadata(self):
        wavelength_nm = 780.0
        gravity = 9.80665
        acceleration = gravity * math.sin(0.8e-3)
        effective_wavevector = 4.0 * math.pi / (wavelength_nm * 1e-9)
        omega = effective_wavevector * acceleration * 1e-12
        x_values = np.linspace(0.0, 4.0e9, 280)
        y_values = 0.48 + 0.2 * np.cos(omega * x_values - 0.25)
        request = ArchiveScanFitRequest(
            x_values=x_values.tolist(),
            y_values=y_values.tolist(),
            bragg_wavelength_nm=wavelength_nm,
            bragg_order=1,
            model=FitModelDefinition(
                key="bragg_fringes",
                label="Bragg Fringes",
                formula="C + A * cos(omega * x + phi0)",
                parameters=[
                    {"name": "A", "guess": "y_range / 2"},
                    {"name": "omega", "guess": "2 * pi / x_span"},
                    {"name": "phi0", "guess": "0"},
                    {"name": "C", "guess": "y_mean"},
                ],
            ),
        )

        response = asyncio.run(fit_archive_scan(request))

        self.assertEqual(response["model_key"], "bragg_fringes")
        self.assertEqual(response["bragg"]["order"], 1)
        self.assertAlmostEqual(response["bragg"]["wavelength_nm"], wavelength_nm)
        self.assertAlmostEqual(response["parameter_values"]["alpha"], 0.8, delta=1e-3)
        self.assertIn("symbolic_formula", response["bragg"])
        self.assertGreater(len(response["bragg"]["mid_fringe_x"]), 1)
        self.assertGreater(response["bragg"]["mid_fringe_spacing_us2"], 0)

    def test_scan_fit_returns_baseline_removed_curve_area(self):
        request = ArchiveScanFitRequest(
            x_values=[0.0, 1.0, 2.0],
            y_values=[3.0, 3.0, 3.0],
            fit_min=0.0,
            fit_max=2.0,
            model=FitModelDefinition(
                key="fixed_line",
                label="Fixed line",
                formula="signal + offset",
                parameters=[
                    {"name": "signal", "guess": "2.0", "fixed": True},
                    {"name": "offset", "guess": "1.0", "fixed": True},
                ],
                roles={"amplitude": "signal", "offset": "offset"},
                area_mode="window_integral",
            ),
        )

        response = asyncio.run(fit_archive_scan(request))

        self.assertIn("area", response)
        self.assertTrue(math.isfinite(response["area"]))
        self.assertAlmostEqual(response["area"], 4.0)


if __name__ == "__main__":
    unittest.main()
