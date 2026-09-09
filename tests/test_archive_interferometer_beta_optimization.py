import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.analysis import physics
from app.api.routes import update_interferometer_beta, update_optimized_analysis_parameter
from app.core.data_loader import DataLoader
from app.core.experiment_manager import ExperimentManager
from app.models.schemas import AnalysisParameterApplyRequest, InterferometerBetaApplyRequest


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

    def test_alpha_can_zero_fit_atom_number_up_mean(self):
        settings = {**self.settings, "alpha": 0.1, "beta": 0.05, "R": 1.1, "K": 10.0}
        points = []
        for area_up, area_dw in [(0.2, 1.0), (0.4, 2.0)]:
            n_f2, n_f1 = physics.calculate_atom_numbers(
                area_up, area_dw, 1.0, 1.0,
                settings["alpha"], settings["beta"], settings["R"], settings["K"], 0.0,
            )
            points.append({"atom_number_up": n_f2, "atom_number_dw": n_f1})

        result = self.loader._optimize_analysis_parameter_from_points(
            points, settings, parameter="alpha", metric="atoms", channel="up"
        )

        self.assertAlmostEqual(result["optimized_value"], 0.2, places=9)
        self.assertAlmostEqual(result["achieved_mean"], 0.0, places=9)
        self.assertTrue(result["exact"])

    def test_beta_can_zero_raw_atom_number_down_mean(self):
        settings = {**self.settings, "alpha": 0.1, "beta": 0.05, "R": 1.1, "K": 10.0}
        n_f2, n_f1 = physics.calculate_atom_numbers(
            1.0, 0.3, 1.0, 1.0,
            settings["alpha"], settings["beta"], settings["R"], settings["K"], 0.0,
        )
        points = [{"atom_number_up_nofit": n_f2, "atom_number_dw_nofit": n_f1}]

        result = self.loader._optimize_analysis_parameter_from_points(
            points, settings, parameter="beta", metric="atoms", source="raw", channel="dw"
        )

        self.assertAlmostEqual(result["optimized_value"], 0.3, places=9)
        self.assertTrue(result["exact"])

    def test_alpha_can_minimize_sample_standard_deviation(self):
        settings = {**self.settings, "alpha": 0.1, "beta": 0.05, "R": 1.1, "K": 10.0}
        points = []
        for area_up, area_dw in [(1.0, 1.0), (2.0685, 6.0)]:
            n_f2, n_f1 = physics.calculate_atom_numbers(
                area_up, area_dw, 1.0, 1.0,
                settings["alpha"], settings["beta"], settings["R"], settings["K"], 0.0,
            )
            points.append({"atom_number_up": n_f2, "atom_number_dw": n_f1})

        result = self.loader._optimize_analysis_parameter_from_points(
            points,
            settings,
            parameter="alpha",
            metric="atoms",
            statistic="std",
            channel="up",
        )

        self.assertAlmostEqual(result["optimized_value"], 0.2137, places=7)
        self.assertAlmostEqual(result["achieved_statistic"], 0.0, places=7)
        self.assertEqual(result["statistic"], "std")
        self.assertTrue(result["exact"])

    def test_interferometer_beta_rejects_unaffected_metric(self):
        with self.assertRaisesRegex(ValueError, "does not affect"):
            self.loader._optimize_analysis_parameter_from_points(
                [{"atom_number_up": 1.0, "atom_number_dw": 2.0}],
                self.settings,
                parameter="intf_beta",
                metric="atoms",
            )

    def test_settings_endpoint_updates_only_interferometer_beta(self):
        with patch("app.api.routes.manager.update_interferometer_beta", return_value=0.2) as update:
            response = asyncio.run(update_interferometer_beta(InterferometerBetaApplyRequest(beta=0.2)))

        update.assert_called_once_with(0.2)
        self.assertEqual(response.data, {"intf_beta": 0.2})

    def test_generic_settings_endpoint_updates_selected_parameter(self):
        with patch("app.api.routes.manager.update_optimized_analysis_parameter", return_value=0.2) as update:
            response = asyncio.run(update_optimized_analysis_parameter(
                AnalysisParameterApplyRequest(parameter="alpha", value=0.2)
            ))

        update.assert_called_once_with("alpha", 0.2)
        self.assertEqual(response.data, {"parameter": "alpha", "value": 0.2})

    def test_generic_settings_update_preserves_every_other_setting(self):
        manager = SimpleNamespace(
            settings={"alpha": 0.1, "beta": 0.05, "sentinel": "unchanged"},
            _apply_runtime_settings=lambda: None,
            _save_settings_to_disk=lambda: None,
        )

        result = ExperimentManager.update_optimized_analysis_parameter(manager, "alpha", 0.2)

        self.assertEqual(result, 0.2)
        self.assertEqual(manager.settings, {"alpha": 0.2, "beta": 0.05, "sentinel": "unchanged"})

    def test_archive_ui_exposes_preview_and_settings_actions(self):
        archive_html = (Path(__file__).resolve().parents[1] / "static" / "archive.html").read_text(encoding="utf-8")

        self.assertIn("ALLAN PHYSICAL STATISTIC OPTIMIZATION", archive_html)
        self.assertIn("/archive/mean-parameter/optimize", archive_html)
        self.assertIn("optimizeInterferometerBeta", archive_html)
        self.assertIn("Apply Parameter to Settings", archive_html)
        self.assertIn("/settings/analysis-parameter", archive_html)
        self.assertIn('<option value="alpha">ALPHA</option>', archive_html)
        self.assertIn('<option value="beta">BETA</option>', archive_html)
        self.assertIn('<option value="std">Standard Deviation</option>', archive_html)


if __name__ == "__main__":
    unittest.main()
