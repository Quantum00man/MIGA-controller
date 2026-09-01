import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.analysis.phase_calibration_optimization import optimize_sync_phase_calibrations
from app.api.routes import optimize_archive_sync_phase_calibrations
from app.core.data_loader import DataLoader
from app.models.schemas import ArchiveSyncPhaseCalibrationOptimizeRequest


def calibration(amplitude, offset, calibration_id="cal"):
    x_values = np.linspace(0.0, 100.0, 121)
    omega = 0.02
    phi0 = 0.4
    return {
        "id": calibration_id,
        "name": calibration_id,
        "parameter_values": {"A": amplitude, "C": offset, "phi0": phi0},
        "bragg": {"angular_frequency_rad_per_us2": omega},
        "reference_t2_us2": 50.0,
        "monotonic_slope": "negative",
        "source_field": "intf_p1",
        "fit_x": x_values.tolist(),
        "fit_y": (offset + amplitude * np.cos(omega * x_values + phi0)).tolist(),
    }


class SyncPhaseCalibrationOptimizationTests(unittest.TestCase):
    def test_archive_exposes_joint_ac_optimization_preview(self):
        archive_html = (Path(__file__).resolve().parents[1] / "static" / "archive.html").read_text(encoding="utf-8")
        self.assertIn("Joint A/C calibration optimization", archive_html)
        self.assertIn("runSyncPhaseCalibrationOptimization", archive_html)
        self.assertIn("/archive/sync-phase-calibration-optimize", archive_html)
        self.assertIn("syncPhaseCalibrationOptimizationPlot", archive_html)
        self.assertIn("Preview only; saved calibrations are not modified", archive_html)
        self.assertIn("Optimization preview is applied to Phase Series, Allan and Sequence Statistics", archive_html)
        self.assertIn("syncPhaseOptimizationPreviewRecord", archive_html)
        self.assertIn("resetSyncPhaseCalibrationOptimizationPreview", archive_html)
        self.assertIn("saveSyncPhaseCalibrationOptimization", archive_html)
        self.assertIn("applySavedSyncPhaseCalibrationOptimization", archive_html)
        self.assertIn("deleteSavedSyncPhaseCalibrationOptimization", archive_html)
        self.assertIn("Saved separately from original archive data", archive_html)

    def test_joint_ac_optimization_reduces_sync_allan_and_std(self):
        random = np.random.default_rng(44)
        common_phase = random.normal(0.0, 0.14, 240)
        reference_signal = 0.10 + np.cos(1.45 + common_phase) + random.normal(0.0, 0.002, 240)
        target_signal = 0.20 + 1.20 * np.cos(1.48 + common_phase) + random.normal(0.0, 0.002, 240)
        pairs = [
            {"shot": index, "p0": 50.0, "reference_signal": reference_signal[index], "target_signal": target_signal[index]}
            for index in range(len(common_phase))
        ]

        result = optimize_sync_phase_calibrations(
            pairs,
            calibration(1.08, 0.07, "master-cal"),
            calibration(1.10, 0.24, "slave-cal"),
            objective="combined",
            combined_allan_weight=0.5,
            parameter_bound_fraction=0.2,
            fringe_weight=0.05,
            prior_weight=0.01,
        )

        self.assertLess(result["metrics_after"]["allan_n1_rad"], result["metrics_before"]["allan_n1_rad"])
        self.assertLess(result["metrics_after"]["std_rad"], result["metrics_before"]["std_rad"])
        self.assertEqual(result["pair_count"], 240)
        self.assertEqual(result["optimized_reference_calibration"]["id"], "master-cal")
        self.assertTrue(result["optimized_reference_calibration"]["sync_optimized_preview"])
        self.assertIn("saved_fit_curve", result["fringe_constraint"]["reference"]["source"])
        self.assertEqual(len(result["series"]), 240)
        self.assertIsNotNone(result["series"][0]["reference_after_rad"])
        self.assertIsNotNone(result["series"][0]["target_after_rad"])
        self.assertIsNotNone(result["series"][0]["after_rad"])

    def test_invalid_objective_and_too_few_pairs_are_rejected(self):
        pairs = [{"shot": index, "p0": 1.0, "reference_signal": 0.2, "target_signal": 0.3} for index in range(7)]
        with self.assertRaisesRegex(ValueError, "Objective"):
            optimize_sync_phase_calibrations(pairs, calibration(1, 0), calibration(1, 0), objective="rms")
        with self.assertRaisesRegex(ValueError, "At least 8"):
            optimize_sync_phase_calibrations(pairs, calibration(1, 0), calibration(1, 0))

    def test_archive_endpoint_uses_effective_node_calibrations_and_raw_signals(self):
        master_cal = calibration(1.0, 0.1, "master-cal")
        slave_cal = calibration(1.2, 0.2, "slave-cal")
        rows_master = []
        rows_slave = []
        for shot in range(12):
            phase = 1.4 + 0.03 * math.sin(shot)
            rows_master.append({"sync_shot_index": shot, "sync_p0": 50.0, "intf_p1": 0.1 + math.cos(phase)})
            rows_slave.append({"sync_shot_index": shot, "sync_p0": 50.0, "intf_p1": 0.2 + 1.2 * math.cos(phase + 0.05)})
        loaded = {
            "sync_manifest": {"node_results": {"master": rows_master, "slaves": {"slave-a": rows_slave}}},
            "archive_phase_reference_contexts": {
                "master": {"effective_calibration": master_cal},
                "slave-a": {"effective_calibration": slave_cal},
            },
        }
        request = ArchiveSyncPhaseCalibrationOptimizeRequest(
            year="2026", month="09", day="01", run_id="sync01",
            reference_node_id="master", target_node_id="slave-a",
        )

        with patch("app.api.routes.data_loader.load_run", return_value=loaded):
            response = asyncio.run(optimize_archive_sync_phase_calibrations(request))

        self.assertEqual(response["reference_node_id"], "master")
        self.assertEqual(response["target_node_id"], "slave-a")
        self.assertEqual(response["pair_count"], 12)
        self.assertEqual(response["source_fields"], {"reference": "intf_p1", "target": "intf_p1"})
        self.assertEqual(response["settings"]["objective"], "allan")
        self.assertEqual(response["settings"]["parameter_bound_fraction"], 0.1)

    def test_saved_optimization_is_a_separate_sidecar_and_is_deletable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "2026" / "09" / "01" / "sync01"
            run_dir.mkdir(parents=True)
            manifest_text = json.dumps({"runtime": {"status": "done"}, "pairs": []})
            config_text = json.dumps({"mode": "sync", "sentinel": "original"})
            results_text = "Step,Interferometer_Phase_Rad\n0,0.1\n"
            (run_dir / "sync_manifest.json").write_text(manifest_text, encoding="utf-8")
            (run_dir / "config.json").write_text(config_text, encoding="utf-8")
            (run_dir / "results.csv").write_text(results_text, encoding="utf-8")
            loader = DataLoader()
            loader.base_dir = root
            payload = {
                "reference_node_id": "master",
                "target_node_id": "slave-a",
                "parameters_before": {"reference": {"A": 1.0, "C": 0.1}},
                "parameters_after": {"reference": {"A": 1.01, "C": 0.11}},
                "metrics_before": {"allan_n1_rad": 0.02},
                "metrics_after": {"allan_n1_rad": 0.01},
                "series": [
                    {"shot": 0, "p0": 50.0, "after_rad": 0.1},
                    {"shot": 1, "p0": 50.0, "after_rad": 0.11},
                ],
            }

            saved = loader.save_sync_phase_calibration_optimization(
                "2026", "09", "01", "sync01", payload, "quiet pair"
            )
            records = loader.load_sync_phase_calibration_optimizations(
                "2026", "09", "01", "sync01"
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["id"], saved["id"])
            self.assertEqual(records[0]["name"], "quiet pair")
            self.assertEqual(records[0]["storage_mode"], "archive_sidecar_preview")
            self.assertEqual((run_dir / "config.json").read_text(encoding="utf-8"), config_text)
            self.assertEqual((run_dir / "results.csv").read_text(encoding="utf-8"), results_text)
            self.assertEqual((run_dir / "sync_manifest.json").read_text(encoding="utf-8"), manifest_text)
            self.assertTrue(loader.delete_sync_phase_calibration_optimization(
                "2026", "09", "01", "sync01", saved["id"]
            ))
            self.assertEqual(loader.load_sync_phase_calibration_optimizations(
                "2026", "09", "01", "sync01"
            ), [])


if __name__ == "__main__":
    unittest.main()
