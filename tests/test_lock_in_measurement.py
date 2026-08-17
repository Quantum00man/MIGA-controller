import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from app.analysis.lock_in import build_lock_in_analysis

try:
    from app.core.experiment_manager import ExperimentManager
except ModuleNotFoundError as exc:
    if exc.name != "lxml":
        raise
    ExperimentManager = None


@unittest.skipIf(ExperimentManager is None, "lxml is not installed in this Python environment")
class LockInExecutionTests(unittest.TestCase):
    def test_builds_typed_abba_blocks_with_reference_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sequence = Path(temp_dir) / "sequence.mot"
            sequence.write_text("DDS1 [<PARAMETER0>]\n", encoding="utf-8")
            manager = object.__new__(ExperimentManager)
            manager.settings = {}
            scan_config = {
                "mode": "lock_in",
                "scan_dimensions": 1,
                "param_type": "int",
                "lock_in_a_value": 11.2,
                "lock_in_b_value": 5.6,
                "averages": 2,
                "randomize": False,
            }

            with (
                mock.patch.object(config, "USE_SIMULATION", True),
                mock.patch.object(config, "SEQUENCE_TEMPLATE_PATH_WIN", str(sequence)),
            ):
                points = manager._build_lock_in_execution(scan_config)

            self.assertEqual([point["sequence_parameters"] for point in points], [[11], [6], [6], [11]] * 2)
            self.assertEqual([point["metadata"]["lock_in_reference"] for point in points[:4]], [1, -1, -1, 1])
            self.assertEqual([point["metadata"]["lock_in_state"] for point in points[:4]], ["a", "b", "b", "a"])
            self.assertEqual([point["metadata"]["lock_in_position"] for point in points[:4]], [1, 2, 3, 4])
            self.assertEqual(points[0]["metadata"]["lock_in_block_index"], 1)
            self.assertEqual(points[4]["metadata"]["lock_in_block_index"], 2)
            self.assertFalse(scan_config["randomize"])

    def test_requires_parameter0_in_sequence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sequence = Path(temp_dir) / "sequence.mot"
            sequence.write_text("DDS1 [123]\n", encoding="utf-8")
            manager = object.__new__(ExperimentManager)
            manager.settings = {}
            with (
                mock.patch.object(config, "USE_SIMULATION", True),
                mock.patch.object(config, "SEQUENCE_TEMPLATE_PATH_WIN", str(sequence)),
            ):
                with self.assertRaisesRegex(ValueError, "PARAMETER0"):
                    manager._build_lock_in_execution({
                        "scan_dimensions": 1,
                        "param_type": "float",
                        "lock_in_a_value": 1,
                        "lock_in_b_value": 0.5,
                        "averages": 1,
                    })


class LockInAnalysisTests(unittest.TestCase):
    def make_block(self, block_index, values):
        points = []
        for position, (state, reference, value) in enumerate(zip("abba", (1, -1, -1, 1), values), start=1):
            points.append({
                "step": (block_index - 1) * 4 + position,
                "parameter": 1 if state == "a" else 0.5,
                "lock_in_block_index": block_index,
                "lock_in_position": position,
                "lock_in_state": state,
                "lock_in_reference": reference,
                "atom_number_up": value,
            })
        return points

    def test_computes_block_values_mean_sem_and_incomplete_count(self):
        points = self.make_block(1, (12, 4, 6, 10)) + self.make_block(2, (14, 6, 6, 14))
        points.pop()  # The second block is incomplete and must not be paired across blocks.
        analysis = build_lock_in_analysis(points, expected_blocks=2)

        self.assertEqual(analysis["complete_blocks"], 1)
        self.assertEqual(analysis["incomplete_blocks"], 1)
        row = analysis["blocks"][0]
        self.assertEqual([row[f"atom_number_up_s{i}"] for i in range(1, 5)], [12.0, 4.0, 6.0, 10.0])
        self.assertAlmostEqual(row["atom_number_up_s_a"], 11.0)
        self.assertAlmostEqual(row["atom_number_up_s_b"], 5.0)
        self.assertAlmostEqual(row["atom_number_up_x"], 3.0)
        self.assertAlmostEqual(row["atom_number_up_r"], 0.375)
        self.assertAlmostEqual(row["atom_number_up_e"], 1.0)
        summary = analysis["metrics"]["atom_number_up"]
        self.assertEqual(summary["x_count"], 1)
        self.assertAlmostEqual(summary["x_mean"], 3.0)
        self.assertAlmostEqual(summary["x_sem"], 0.0)

    def test_sem_is_standard_deviation_over_square_root_n(self):
        points = self.make_block(1, (12, 4, 6, 10)) + self.make_block(2, (14, 6, 6, 14))
        summary = build_lock_in_analysis(points, expected_blocks=2)["metrics"]["atom_number_up"]
        self.assertAlmostEqual(summary["x_mean"], 3.5)
        self.assertAlmostEqual(summary["x_sem"], 0.5)


if __name__ == "__main__":
    unittest.main()
