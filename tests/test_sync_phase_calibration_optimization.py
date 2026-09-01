import asyncio
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.analysis.phase_calibration_optimization import optimize_sync_phase_calibrations
from app.api.routes import optimize_archive_sync_phase_calibrations
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


if __name__ == "__main__":
    unittest.main()
