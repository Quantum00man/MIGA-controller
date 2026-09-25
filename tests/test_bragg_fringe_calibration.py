import math
from pathlib import Path
import unittest

import numpy as np

from app.analysis.bragg_fringe_calibration import (
    build_fine_p0_values,
    coarse_fit,
    fine_fit,
    fine_phase_offsets,
)
from app.core.data_loader import DataLoader


class BraggFringeCalibrationTests(unittest.TestCase):
    def test_fine_offsets_are_center_interleaved(self):
        offsets = fine_phase_offsets(0.6, 7)
        self.assertAlmostEqual(offsets[0], 0.0)
        self.assertGreater(offsets[1], 0.0)
        self.assertLess(offsets[2], 0.0)
        self.assertEqual(sorted(round(abs(value), 12) for value in offsets), [0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6])

    def test_coarse_then_fine_recovers_local_offset_and_midpoint(self):
        amplitude = 20.0
        coarse_offset = 50.0
        local_offset = 51.25
        omega = 0.0001
        phi0 = 0.2
        coarse_points = []
        for p0 in np.linspace(0.0, 100000.0, 81):
            for repeat in range(2):
                coarse_points.append({
                    "bragg_calibration_stage": "coarse",
                    "parameter": p0,
                    "intf_p1": coarse_offset + amplitude * math.cos(omega * p0 + phi0),
                })
        coarse = coarse_fit(coarse_points, 1)
        predicted = coarse["selected_mid_fringe_t2_us2"]
        true_midpoint = predicted + 0.22 / omega
        fine_points = []
        for p0 in build_fine_p0_values(predicted, omega, 0.6, 7):
            for repeat in range(4):
                fine_points.append({
                    "bragg_calibration_stage": "fine",
                    "parameter": p0,
                    "intf_p1": local_offset + amplitude * math.cos(omega * (p0 - true_midpoint) + math.pi / 2),
                })
        result = fine_fit(fine_points, coarse, half_range_rad=0.6)
        self.assertTrue(result["quality_passed"])
        self.assertAlmostEqual(result["mid_fringe_t2_us2"], true_midpoint, places=7)
        self.assertAlmostEqual(result["local_parameter_values"]["C"], local_offset, places=7)
        self.assertAlmostEqual(result["calibration_fit"]["parameter_values"]["C"], local_offset, places=7)

    def test_fine_fit_rejects_correction_outside_requested_range(self):
        omega = 0.1
        coarse = {
            "selected_mid_fringe_t2_us2": 10.0,
            "angular_frequency_rad_per_us2": omega,
            "parameter_values": {"A": 10.0, "C": 5.0, "phi0": math.pi / 2 - 1.0},
            "fit_x": [0.0, 20.0],
        }
        points = [
            {"bragg_calibration_stage": "fine", "parameter": p0,
             "intf_p1": 5.0 + 10.0 * math.cos(omega * (p0 - 10.0) + math.pi / 2 + 0.8)}
            for p0 in build_fine_p0_values(10.0, omega, 0.6, 7)
        ]
        result = fine_fit(points, coarse, half_range_rad=0.6)
        self.assertFalse(result["quality_checks"]["inside_fine_range"])
        self.assertFalse(result["quality_passed"])

    def test_archive_reanalysis_recovers_legacy_rows_without_stage_column(self):
        amplitude, offset, omega, phi0 = 20.0, 50.0, 0.0001, 0.2
        coarse_points = [
            {"parameter": p0, "intf_p1": offset + amplitude * math.cos(omega * p0 + phi0)}
            for p0 in np.linspace(0.0, 100000.0, 81)
        ]
        staged_coarse = [{**point, "bragg_calibration_stage": "coarse"} for point in coarse_points]
        coarse = coarse_fit(staged_coarse, 1)
        center = coarse["selected_mid_fringe_t2_us2"]
        fine_points = [
            {"parameter": p0, "intf_p1": offset + amplitude * math.cos(omega * p0 + phi0)}
            for p0 in build_fine_p0_values(center, omega, 0.6, 7)
        ]
        result = DataLoader._reanalyze_archived_bragg_calibration(
            coarse_points + fine_points,
            {
                "mode": "bragg_fringe_calibration",
                "bragg_calibration_target_fringe": 1,
                "_bragg_calibration_coarse_shots": len(coarse_points),
                "bragg_calibration_fine_phase_half_range_rad": 0.6,
            },
        )
        self.assertTrue(result["archive_reanalysis"])
        self.assertIn(result["status"], {"passed", "failed"})
        self.assertIn("fine", result)

    def test_archive_reanalysis_reports_incomplete_fine_scan(self):
        result = DataLoader._reanalyze_archived_bragg_calibration(
            [{"parameter": value, "intf_p1": value} for value in range(8)],
            {"mode": "bragg_fringe_calibration", "_bragg_calibration_coarse_shots": 8},
        )
        self.assertEqual(result["status"], "reanalysis_failed")
        self.assertIn("completed fine scan", result["error"])

    def test_archive_recovered_fringe_can_be_saved_and_activated(self):
        html = (Path(__file__).parents[1] / "static" / "archive.html").read_text(encoding="utf-8")
        self.assertIn("saveAndActivateArchivedBraggCalibration", html)
        self.assertIn("Save &amp; Activate", html)
        self.assertIn("/archive/bragg-phase-calibrations", html)
        self.assertIn("/settings/interferometer-phase-calibrations/active", html)


if __name__ == "__main__":
    unittest.main()
