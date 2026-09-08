import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from app.api.routes import update_interferometer_beta
from app.core.data_loader import DataLoader
from app.models.schemas import InterferometerBetaApplyRequest


class ArchiveInterferometerBetaOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()
        self.settings = {"intf_alpha": 0.35, "intf_beta": 0.08, "intf_gamma": 0.25}

    def test_fit_up_mean_is_zeroed_inside_unit_interval(self):
        # For P1, the zero is beta=(1+gamma)*n_f2/(n_f1+n_f2).
        points = [
            {"atom_number_dw": 8.4, "atom_number_up": 1.6},
            {"atom_number_dw": 16.8, "atom_number_up": 3.2},
        ]

        result = self.loader._optimize_interferometer_beta_from_points(points, self.settings)

        self.assertAlmostEqual(result["optimized_beta"], 0.2, places=9)
        self.assertAlmostEqual(result["achieved_mean"], 0.0, places=9)
        self.assertTrue(result["exact"])
        self.assertEqual(result["source"], "fit")
        self.assertEqual(result["channel"], "up")
        self.assertEqual(result["sample_count"], 2)

    def test_raw_source_uses_nofit_populations(self):
        points = [{
            "atom_number_dw": 8.4, "atom_number_up": 1.6,
            "atom_number_dw_nofit": 5.2, "atom_number_up_nofit": 4.8,
        }]

        result = self.loader._optimize_interferometer_beta_from_points(
            points, self.settings, source="raw", channel="up"
        )

        self.assertAlmostEqual(result["optimized_beta"], 0.6, places=9)
        self.assertTrue(result["exact"])

    def test_unreachable_down_target_returns_best_boundary_with_warning(self):
        points = [{"atom_number_dw": 8.4, "atom_number_up": 1.6}]

        result = self.loader._optimize_interferometer_beta_from_points(
            points, self.settings, channel="dw", target_mean=0.0
        )

        self.assertFalse(result["exact"])
        self.assertTrue(result["at_boundary"])
        self.assertAlmostEqual(result["optimized_beta"], 1.0)

    def test_settings_endpoint_updates_only_interferometer_beta(self):
        with patch("app.api.routes.manager.update_interferometer_beta", return_value=0.2) as update:
            response = asyncio.run(update_interferometer_beta(InterferometerBetaApplyRequest(beta=0.2)))

        update.assert_called_once_with(0.2)
        self.assertEqual(response.data, {"intf_beta": 0.2})

    def test_archive_ui_exposes_preview_and_settings_actions(self):
        archive_html = (Path(__file__).resolve().parents[1] / "static" / "archive.html").read_text(encoding="utf-8")

        self.assertIn("ALLAN MEAN β OPTIMIZATION", archive_html)
        self.assertIn("/archive/interferometer-beta/optimize", archive_html)
        self.assertIn("optimizeInterferometerBeta", archive_html)
        self.assertIn("Apply β to Settings", archive_html)
        self.assertIn("/settings/interferometer-beta", archive_html)


if __name__ == "__main__":
    unittest.main()
