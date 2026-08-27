import asyncio
import json
import math
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from app.analysis.fitting import perform_sync_differential_ellipse_fit
from app.api.routes import fit_archive_sync_differential
from app.core.data_loader import DataLoader
from app.models.schemas import ArchiveSyncDifferentialFitRequest


class SyncDifferentialEllipseFitTests(unittest.TestCase):
    def test_archive_exposes_saved_differential_fit_controls_and_overlay(self):
        archive_html = (
            Path(__file__).resolve().parents[1] / "static" / "archive.html"
        ).read_text(encoding="utf-8")
        self.assertIn("syncManifest && syncArchiveViewMode === 'differential'", archive_html)
        self.assertIn("Fit Ellipse & Save", archive_html)
        self.assertIn("Delete Saved Fit", archive_html)
        self.assertIn("activeSyncDifferentialFit()", archive_html)
        self.assertIn("Ellipse fit · Δφ=", archive_html)

    def test_physical_ellipse_fit_recovers_phase_and_global_parameters(self):
        random = np.random.default_rng(17)
        theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        expected_phase = 1.13
        x_values = 0.42 + 0.31 * np.cos(theta) + random.normal(0.0, 0.002, len(theta))
        y_values = -0.18 + 0.47 * np.cos(theta + expected_phase) + random.normal(0.0, 0.003, len(theta))

        result = perform_sync_differential_ellipse_fit(x_values, y_values)

        self.assertIsNotNone(result)
        params = result.parameter_values
        self.assertAlmostEqual(params["center_x"], 0.42, delta=0.01)
        self.assertAlmostEqual(params["center_y"], -0.18, delta=0.01)
        self.assertAlmostEqual(params["amplitude_x"], 0.31, delta=0.01)
        self.assertAlmostEqual(params["amplitude_y"], 0.47, delta=0.01)
        self.assertAlmostEqual(params["phase_difference_rad"], expected_phase, delta=0.02)
        self.assertGreater(result.r_squared, 0.99)
        self.assertGreater(result.phase_coverage_rad, 1.8 * math.pi)
        self.assertGreater(result.parameter_uncertainties["phase_difference_rad"], 0.0)
        self.assertEqual(len(result.fit_x), 361)
        self.assertEqual(len(result.fit_y), 361)

    def test_fit_rejects_too_few_or_constant_points(self):
        self.assertIsNone(perform_sync_differential_ellipse_fit(np.arange(7), np.arange(7)))
        self.assertIsNone(perform_sync_differential_ellipse_fit(np.arange(12), np.ones(12)))

    def test_archive_endpoint_saves_source_and_fit_quality(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        request = ArchiveSyncDifferentialFitRequest(
            year="2026",
            month="08",
            day="27",
            run_id="run01",
            x_values=(2.0 + np.cos(theta)).tolist(),
            y_values=(3.0 + 1.5 * np.cos(theta + 0.7)).tolist(),
            source={"slave_id": "slave_b", "aggregation": "shots"},
        )

        def save_fit(_year, _month, _day, _run_id, payload):
            return {**payload, "id": "saved-fit", "created_at": "2026-08-27T12:00:00+00:00"}

        with patch("app.api.routes.data_loader.save_sync_differential_fit", side_effect=save_fit):
            response = asyncio.run(fit_archive_sync_differential(request))

        self.assertEqual(response["id"], "saved-fit")
        self.assertEqual(response["source"]["slave_id"], "slave_b")
        self.assertAlmostEqual(response["parameter_values"]["phase_difference_rad"], 0.7, delta=1e-3)
        self.assertGreater(response["r_squared"], 0.999)


class SyncDifferentialFitPersistenceTests(unittest.TestCase):
    def test_saved_fits_are_loaded_and_deleted_with_the_sync_archive(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "2026" / "08" / "27" / "run01"
            run_dir.mkdir(parents=True)
            (run_dir / "sync_manifest.json").write_text(
                json.dumps({"runtime": {"status": "done"}, "pairs": []}),
                encoding="utf-8",
            )
            loader = DataLoader()
            loader.base_dir = root

            saved = loader.save_sync_differential_fit(
                "2026", "08", "27", "run01",
                {"model_key": "sync_differential_ellipse", "parameter_values": {"phase_difference_rad": 1.2}},
            )
            loaded = loader.load_sync_differential_fits("2026", "08", "27", "run01")

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["id"], saved["id"])
            self.assertIn("created_at", loaded[0])
            self.assertTrue(loader.delete_sync_differential_fit("2026", "08", "27", "run01", saved["id"]))
            self.assertEqual(loader.load_sync_differential_fits("2026", "08", "27", "run01"), [])


if __name__ == "__main__":
    unittest.main()
