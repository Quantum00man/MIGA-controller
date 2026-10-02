import math
import json
import tempfile
from copy import deepcopy
from unittest.mock import patch
import unittest
from pathlib import Path

from app.core.data_loader import DataLoader


class AllanStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()

    def test_sequence_statistics_are_computed_from_finite_values(self):
        points = [
            {"signal": 1.0},
            {"signal": 2.0},
            {"signal": float("nan")},
            {"signal": 3.0},
            {"signal": 4.0},
        ]

        channel = self.loader._build_allan_channel(points, ("signal",), [1, 2])
        statistics = channel["sequence_statistics"]

        self.assertEqual(statistics["sample_count"], 4)
        self.assertAlmostEqual(statistics["mean"], 2.5)
        self.assertAlmostEqual(statistics["rms"], math.sqrt(7.5))
        self.assertAlmostEqual(statistics["standard_deviation"], math.sqrt(5.0 / 3.0))
        self.assertEqual(len(channel["edf_white"]), 2)
        self.assertGreater(channel["error_plus"][0], channel["error_minus"][0])
        self.assertLess(channel["ci_lower"][0], channel["y"][0])
        self.assertGreater(channel["ci_upper"][0], channel["y"][0])

    def test_total_channel_statistics_use_the_combined_value(self):
        points = [
            {"up": 2.0, "down": 3.0},
            {"up": 4.0, "down": 5.0},
        ]

        channel = self.loader._build_allan_channel(points, ("up", "down"), [1])
        statistics = channel["sequence_statistics"]

        self.assertEqual(statistics["sample_count"], 2)
        self.assertAlmostEqual(statistics["mean"], 7.0)
        self.assertAlmostEqual(statistics["rms"], math.sqrt(53.0))
        self.assertAlmostEqual(statistics["standard_deviation"], math.sqrt(8.0))

    def test_empty_channel_returns_empty_statistics(self):
        channel = self.loader._build_allan_channel([], ("signal",), [1])

        self.assertEqual(channel["sequence_statistics"]["sample_count"], 0)
        self.assertIsNone(channel["sequence_statistics"]["mean"])
        self.assertIsNone(channel["sequence_statistics"]["rms"])
        self.assertIsNone(channel["sequence_statistics"]["standard_deviation"])

    def test_alpha_copy_allan_reuses_full_node_results_without_fitting(self):
        points = [
            {"step": i, "parameter": i, "timestamp": float(i),
             "atom_number_up": 10 + i * i, "atom_number_dw": 30 - i,
             "atom_number_up_nofit": 10 + i * i, "atom_number_dw_nofit": 30 - i}
            for i in range(8)
        ]
        events = [
            {"calibration_id": "a", "representative_time": 0.0, "intf_alpha": 0.2, "accepted": True},
            {"calibration_id": "b", "representative_time": 7.0, "intf_alpha": 0.7, "accepted": True},
        ]
        config = {"mode": "standard", "scan_dimensions": 1, "randomize": False,
                  "intf_alpha_calibration_enabled": True}
        with tempfile.TemporaryDirectory() as root:
            node = Path(root) / "master"
            node.mkdir()
            (node / "intf_alpha_calibrations.json").write_text(json.dumps(events))
            with patch.object(self.loader, "_get_run_dir", return_value=Path(root)), \
                 patch.object(self.loader, "_resolve_archive_node_dir", return_value=node) as resolve, \
                 patch.object(self.loader, "_load_config_data", return_value=config), \
                 patch.object(self.loader, "_read_results_csv", side_effect=lambda *a, **k: deepcopy(points)) as read, \
                 patch.object(self.loader, "_archive_phase_reference_context", return_value=("master", {}, {})), \
                 patch.object(self.loader, "_load_waveform_arrays", side_effect=AssertionError("Waveforms must not be loaded")):
                results = []
                for method in ("linear", "nearest"):
                    result = self.loader.calculate_allan_run(
                        "2026", "10", "02", "run", 2, "recalculated", node_id="master",
                        metric="intf", source="fit",
                        intf_alpha_selection={"accepted_calibration_ids": ["a", "b"], "interpolation_method": method},
                    )
                    expected = self.loader._apply_intf_alpha_history(deepcopy(points), events, config, None, method)
                    self.assertEqual(result["metrics"], self.loader._build_allan_payload(expected, 2, "intf", "fit")["metrics"])
                    self.assertEqual(result["sequence_length"], 8)
                    self.assertEqual(set(result["metrics"]), {"intf"})
                    self.assertEqual(set(result["metrics"]["intf"]), {"fit"})
                    results.append(result["metrics"]["intf"]["fit"]["up"]["y"])
                self.assertNotEqual(*results)
                resolve.assert_called_with(Path(root), "master")
                self.assertTrue(all(call.kwargs.get("max_points") is None for call in read.call_args_list))

    def test_archive_allan_uses_confidence_bands_and_y_axis_modes(self):
        archive = (Path(__file__).resolve().parents[1] / "static" / "archive.html").read_text(encoding="utf-8")
        self.assertIn("allanConfidenceBandTrace", archive)
        self.assertIn("allanYAxisMode", archive)
        self.assertIn("syncPhaseAllanYAxis", archive)
        self.assertNotIn("array: this.getAllanDisplayErrors(series, 'error_plus')", archive)


if __name__ == "__main__":
    unittest.main()
