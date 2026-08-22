import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.core.data_loader import DataLoader
from app.core.data_manager import RESULTS_CSV_HEADER


class MarkerOptimizationArchiveTests(unittest.TestCase):
    def make_run(self, base: Path, report: dict, rows: list[dict]) -> tuple[DataLoader, Path]:
        run_dir = base / "2026" / "08" / "22" / "run01_20260822"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps({"sequence_name": "source_marked.mot", "run_label": "Raman optimization"}),
            encoding="utf-8",
        )
        (run_dir / "marker_optimization_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_HEADER)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        loader = DataLoader()
        loader.base_dir = base
        return loader, run_dir

    @staticmethod
    def row(value: float, storage_step: int, workflow_step: int | None = None) -> dict:
        row = {
            "Step": storage_step,
            "Timestamp": float(storage_step),
            "Parameter_P0": value,
            "All_Parameters": str(value),
            "Atom_UP": value * 10,
            "Atom_DW": value * 20,
            "Temp_UP": value * 0.1,
            "Temp_DW": value * 0.2,
            "Sigma_UP": value * 0.01,
            "Sigma_DW": value * 0.02,
            "Center_UP": value * 0.001,
            "Center_DW": value * 0.002,
            "Amp_UP": value * 0.3,
            "Amp_DW": value * 0.4,
            "Prob_UP_F2": value * 2,
            "Prob_DW_F1": value * 3,
            "NF_Atom_UP": value * 11,
            "NF_Atom_DW": value * 21,
            "NF_Temp_UP": value * 0.11,
            "NF_Temp_DW": value * 0.21,
            "NF_Sigma_UP": value * 0.011,
            "NF_Sigma_DW": value * 0.021,
            "NF_Center_UP": value * 0.0011,
            "NF_Center_DW": value * 0.0021,
            "NF_Amp_UP": value * 0.31,
            "NF_Amp_DW": value * 0.41,
            "NF_Prob_UP": value * 2.1,
            "NF_Prob_DW": value * 3.1,
            "TailMean_UP": value * 0.001,
            "TailMean_DW": value * 0.002,
        }
        if workflow_step is not None:
            row.update({
                "Workflow_Step": workflow_step,
                "Workflow_Marker": f"MARKER_{workflow_step}",
                "Workflow_Point": storage_step,
                "Workflow_Repeat": 1,
                "Workflow_Shot": storage_step,
                "Workflow_Randomized": 1,
            })
        return row

    @staticmethod
    def report() -> dict:
        return {
            "workflow_name": "Sequential Raman",
            "run_label": "Raman optimization",
            "phase": "completed",
            "completed_steps": 2,
            "total_steps": 2,
            "applied_values": {"MARKER_1": 2, "MARKER_2": 11},
            "steps": [
                {
                    "index": 1,
                    "marker_id": "MARKER_1",
                    "marker_name": "Labeling detuning",
                    "status": "completed",
                    "objective": "spectral_center",
                    "metric_key": "transition_probability_up",
                    "metric_label": "Transition Probability UP",
                    "metric_source": "fit",
                    "average_count": 1,
                    "randomize": True,
                    "start": 1,
                    "stop": 2,
                    "step": 1,
                    "applied_value": 2,
                    "points": [
                        {"value": 1, "repeats": [2]},
                        {"value": 2, "repeats": [4]},
                    ],
                    "analysis": {
                        "model": "gaussian",
                        "selected_value": 2,
                        "fit_x_dense": [1, 1.5, 2],
                        "fit_curve_dense": [2, 3.5, 4],
                    },
                },
                {
                    "index": 2,
                    "marker_id": "MARKER_2",
                    "marker_name": "Pulse duration",
                    "status": "completed",
                    "objective": "rabi_pi",
                    "metric_key": "transition_probability_dw",
                    "metric_label": "Transition Probability DOWN",
                    "metric_source": "fit",
                    "average_count": 1,
                    "randomize": False,
                    "start": 10,
                    "stop": 11,
                    "step": 1,
                    "applied_value": 11,
                    "points": [
                        {"value": 10, "repeats": [30]},
                        {"value": 11, "repeats": [33]},
                    ],
                    "analysis": {"model": "damped_rabi", "selected_value": 11},
                },
            ],
        }

    def test_new_archive_groups_full_metrics_by_persisted_workflow_step(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            loader, run_dir = self.make_run(
                base,
                self.report(),
                [self.row(1, 1, 1), self.row(2, 2, 1), self.row(10, 3, 2), self.row(11, 4, 2)],
            )
            for filename in (
                "source_marked_original.mot",
                "source_marked_optimized.mot",
                "marker_optimization_report.pdf",
                "workflow_preset.json",
                "source_marked_marker_optimization_report.zip",
            ):
                (run_dir / filename).write_bytes(b"artifact")

            payload = loader.load_run("2026", "08", "22", "run01_20260822")
            optimization = payload["marker_optimization"]

            self.assertEqual(payload["config"]["mode"], "marker_optimization")
            self.assertTrue(payload["run_entry"]["has_marker_optimization"])
            self.assertEqual(optimization["run_label"], "Raman optimization")
            self.assertEqual(len(optimization["steps"]), 2)
            self.assertEqual([point["parameter"] for point in optimization["steps"][0]["data"]], [1.0, 2.0])
            self.assertEqual([point["parameter"] for point in optimization["steps"][1]["data"]], [10.0, 11.0])
            first_stats = optimization["steps"][0]["stats"][0]
            self.assertAlmostEqual(first_stats["atoms_up"], 10.0)
            self.assertAlmostEqual(first_stats["temp_up"], 0.1)
            self.assertAlmostEqual(first_stats["prob_up"], 2.0)
            self.assertIn("fit_x_dense", optimization["steps"][0]["analysis"])
            self.assertEqual(
                {artifact["kind"] for artifact in optimization["artifacts"]},
                {"original_sequence", "optimized_sequence", "report_pdf", "report_json", "workflow_preset", "report_bundle"},
            )
            artifact, name = loader.get_marker_optimization_artifact(
                "2026", "08", "22", "run01_20260822", "original_sequence"
            )
            self.assertEqual(artifact, run_dir / "source_marked_original.mot")
            self.assertEqual(name, "source_marked_original.mot")
            with self.assertRaises(FileNotFoundError):
                loader.get_marker_optimization_artifact(
                    "2026", "08", "22", "run01_20260822", "../../config"
                )

    def test_legacy_archive_recovers_contiguous_steps_from_report_repeat_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            loader, _ = self.make_run(
                Path(directory),
                self.report(),
                [self.row(1, 1), self.row(2, 2), self.row(10, 3), self.row(11, 4)],
            )
            payload = loader.load_run("2026", "08", "22", "run01_20260822")
            steps = payload["marker_optimization"]["steps"]
            self.assertEqual([point["parameter"] for point in steps[0]["data"]], [1.0, 2.0])
            self.assertEqual([point["parameter"] for point in steps[1]["data"]], [10.0, 11.0])


if __name__ == "__main__":
    unittest.main()
