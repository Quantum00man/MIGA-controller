import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from app.core.marker_optimization_manager import (
    MarkerOptimizationManager,
    _damped_rabi,
    _gaussian,
    _nearest_scanned_point,
    analyze_marker_scan,
)


class MarkerOptimizationAnalysisTests(unittest.TestCase):
    @staticmethod
    def points(x, y):
        return [
            {"value": float(xv), "metric_mean": float(yv), "metric_std": 0.01, "metric_sem": 0.005, "repeats": [float(yv)]}
            for xv, yv in zip(x, y)
        ]

    def test_nearest_scanned_point_tie_uses_lower_value(self):
        self.assertEqual(_nearest_scanned_point([140, 142], 141), 140)

    def test_spectral_center_applies_unscanned_integer_optimum(self):
        x = np.arange(130, 153, 2, dtype=float)
        y = _gaussian(x, 0.08, 0.82, 141.2, 4.0)
        result = analyze_marker_scan(self.points(x, y), "spectral_center", minimum_r_squared=0.99, marker_kind="dds_element", marker_decimals=0)
        self.assertAlmostEqual(result["continuous_optimum"], 141.2, places=4)
        self.assertEqual(result["selected_value"], 141)
        self.assertNotIn(result["selected_value"], x.tolist())
        self.assertFalse(result["selected_was_sampled"])
        self.assertGreaterEqual(len(result["fit_x_dense"]), 500)
        self.assertEqual(len(result["fit_x_dense"]), len(result["fit_curve_dense"]))
        self.assertGreater(result["r_squared"], 0.999)

    def test_spectral_center_respects_dac_decimal_resolution(self):
        x = np.arange(0.0, 1.01, 0.1, dtype=float)
        y = _gaussian(x, 0.03, 0.9, 0.3334, 0.16)
        result = analyze_marker_scan(
            self.points(x, y), "spectral_center", minimum_r_squared=0.99,
            marker_kind="dac_value", marker_decimals=3,
        )
        self.assertAlmostEqual(result["continuous_optimum"], 0.3334, places=4)
        self.assertEqual(result["selected_value"], 0.333)
        self.assertFalse(result["selected_was_sampled"])

    def test_damped_rabi_applies_unscanned_integer_pi_time(self):
        x = np.arange(10, 131, 10, dtype=float)
        y = _damped_rabi(x, 0.04, 0.91, 53.0, 450.0)
        result = analyze_marker_scan(self.points(x, y), "rabi_pi", minimum_r_squared=0.99, marker_kind="duration", marker_decimals=0)
        self.assertAlmostEqual(result["continuous_optimum"], 53.0, places=3)
        self.assertEqual(result["selected_value"], 53)
        self.assertIsInstance(result["selected_value"], int)
        self.assertNotIn(result["selected_value"], x.tolist())
        self.assertFalse(result["selected_was_sampled"])

    def test_boundary_optimum_stops_raw_objective(self):
        with self.assertRaisesRegex(ValueError, "scan boundary"):
            analyze_marker_scan(self.points([1, 2, 3], [1, 2, 3]), "maximize")

    def test_low_fit_quality_stops_step(self):
        x = np.arange(10, 131, 10, dtype=float)
        y = np.asarray([0.7, 0.1, 0.8, 0.2, 0.4, 0.9, 0.3, 0.65, 0.22, 0.79, 0.11, 0.51, 0.35])
        with self.assertRaisesRegex(ValueError, "Fit quality"):
            analyze_marker_scan(self.points(x, y), "rabi_pi", minimum_r_squared=0.9999)

    def test_point_count_scan_supports_integer_marker_grid(self):
        definition = {"kind": "dds_element", "hard_min": 0, "hard_max": 20}
        values = MarkerOptimizationManager._scan_values(
            {"start": 0, "stop": 20, "step": 5, "scan_method": "n_points"}, definition
        )
        self.assertEqual(values, [0.0, 5.0, 10.0, 15.0, 20.0])
        with self.assertRaisesRegex(ValueError, "integer grid"):
            MarkerOptimizationManager._scan_values(
                {"start": 0, "stop": 10, "step": 4, "scan_method": "n_points"}, definition
            )

    def test_integer_scan_grid_validation(self):
        definition = {"kind": "duration", "hard_min": 1, "hard_max": 100}
        step = {"start": 10, "stop": 30, "step": 5}
        self.assertEqual(MarkerOptimizationManager._scan_values(step, definition), [10.0, 15.0, 20.0, 25.0, 30.0])
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            MarkerOptimizationManager._scan_values({"start": 10, "stop": 20, "step": 0.5}, definition)


class MarkerOptimizationReportTests(unittest.TestCase):
    def test_report_bundle_contains_scientific_outputs_and_sequences(self):
        manager = MarkerOptimizationManager(mock.Mock())
        points = [
            {"value": 1.0, "metric_mean": 0.2, "metric_std": 0.0, "metric_sem": 0.0, "repeats": [0.2]},
            {"value": 2.0, "metric_mean": 0.8, "metric_std": 0.0, "metric_sem": 0.0, "repeats": [0.8]},
            {"value": 3.0, "metric_mean": 0.3, "metric_std": 0.0, "metric_sem": 0.0, "repeats": [0.3]},
        ]
        manager._status = {
            **manager._idle_status(),
            "phase": "completed",
            "run_id": "run01_20260821",
            "stop_reason": "all_steps_completed",
            "total_steps": 1,
            "applied_values": {"TEST_DAC": 2.0},
            "steps": [{
                "index": 1,
                "marker_id": "TEST_DAC",
                "marker_name": "Test DAC",
                "marker_kind": "dac_value",
                "objective": "maximize",
                "randomize": True,
                "metric_label": "Transition Probability UP",
                "status": "completed",
                "points": points,
                "analysis": {
                    "model": "none",
                    "selected_value": 2.0,
                    "selected_was_sampled": True,
                    "continuous_optimum": 2.0,
                    "r_squared": None,
                    "fit_curve": [None, None, None],
                    "residuals": [None, None, None],
                },
                "digital_state_choices": {"ENABLE": "on"},
                "digital_states": {"ENABLE": "ON"},
                "digital_conditions": [{
                    "id": "ENABLE", "display_name": "Raman enable",
                    "current_state": "OFF", "selection": "on", "effective_state": "ON",
                }],
            }],
        }
        payload = {"workflow_name": "Test workflow", "sequence_name": "source_marked.mot", "steps": []}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "results.csv").write_text("Step,Value\n1,0.2\n", encoding="utf-8")
            (run_dir / "step_01_execution_conditions.mot").write_text("STATE ON\n", encoding="utf-8")
            artifacts = manager._finalize_artifacts(run_dir, payload, "ORIGINAL\n", "OPTIMIZED\n", "utf-8")
            self.assertTrue(artifacts["report_pdf"].is_file())
            self.assertGreater(artifacts["report_pdf"].stat().st_size, 1000)
            report = json.loads(artifacts["report_json"].read_text(encoding="utf-8"))
            self.assertEqual(report["phase"], "completed")
            self.assertEqual(report["report_version"], 3)
            self.assertEqual(report["steps"][0]["digital_states"], {"ENABLE": "ON"})
            self.assertTrue(report["steps"][0]["randomize"])
            preset = json.loads(artifacts["workflow_preset"].read_text(encoding="utf-8"))
            self.assertEqual(preset["steps"][0]["digital_states"], {"ENABLE": "on"})
            self.assertTrue(preset["steps"][0]["randomize"])
            scan_csv = next(run_dir.glob("*_scan.csv")).read_text(encoding="utf-8")
            self.assertIn("digital_states_json", scan_csv)
            self.assertIn('ENABLE', scan_csv)
            self.assertIn("randomized", scan_csv)
            fit_csv = next(run_dir.glob("*_fit_residuals.csv")).read_text(encoding="utf-8")
            self.assertIn("continuous_optimum", fit_csv)
            self.assertIn("applied_was_sampled", fit_csv)
            with zipfile.ZipFile(artifacts["report_bundle"]) as archive:
                names = set(archive.namelist())
            self.assertIn("source_marked_original.mot", names)
            self.assertIn("source_marked_optimized.mot", names)
            self.assertIn("marker_optimization_report.pdf", names)
            self.assertIn("marker_optimization_report.json", names)
            self.assertIn("results.csv", names)
            self.assertIn("workflow_preset.json", names)
            self.assertIn("step_01_execution_conditions.mot", names)
            self.assertTrue(any(name.endswith("_scan.csv") for name in names))
            self.assertTrue(any(name.endswith("_fit_residuals.csv") for name in names))



class MarkerOptimizationWorkflowTests(unittest.TestCase):
    def test_validation_uses_embedded_definition_without_settings_profile(self):
        from types import SimpleNamespace

        from app.core.sequence_markers import embed_marker_definitions

        scan_definition = {
            "id": "FREQ", "display_name": "Frequency", "kind": "dds_element",
            "decimals": 0, "hard_min": 1, "hard_max": 9,
            "default_start": 1, "default_stop": 9, "default_step": 1,
            "default_method": "step_size", "expected_command": "DDS1",
            "expected_channel": "2", "has_compensation": False,
        }
        state_definition = {
            "id": "ENABLE", "display_name": "Raman enable", "kind": "digital_state",
            "expected_command": "TTL_AOM_Raman1", "expected_channel": "49",
        }
        content = embed_marker_definitions(
            "###SCAN:FREQ###\n+1us DDS1 [5] (2)\n"
            "###STATE:ENABLE###\n+1us TTL_AOM_Raman1 = OFF (49)\n",
            [scan_definition, state_definition],
        )
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "renamed.mot"
            template.write_text(content, encoding="utf-8")
            experiment = SimpleNamespace(settings={"template_path": str(template)})
            manager = MarkerOptimizationManager(experiment)

            _, _, _, definitions, steps = manager._validate_payload({
                "sequence_name": "renamed.mot",
                "steps": [{
                    "marker_id": "FREQ", "objective": "maximize",
                    "metric_key": "transition_probability_up", "metric_source": "fit",
                    "average_count": 1, "randomize": True, "start": 1, "stop": 3, "step": 1,
                    "scan_method": "step_size", "digital_states": {"ENABLE": "on"},
                }],
            })

        self.assertEqual({item["id"] for item in definitions}, {"FREQ", "ENABLE"})
        self.assertEqual(steps[0]["values"], [1.0, 2.0, 3.0])
        self.assertTrue(steps[0]["randomize"])
        self.assertEqual(steps[0]["digital_state_choices"], {"ENABLE": "on"})
        self.assertEqual(steps[0]["digital_states"], {"ENABLE": "ON"})
        self.assertEqual(steps[0]["digital_conditions"][0]["current_state"], "OFF")

    def test_randomize_shuffles_all_point_average_shots_and_keeps_sorted_aggregates(self):
        from types import SimpleNamespace

        experiment = mock.Mock()
        experiment.on_data_ready = None
        execution_order = []

        def execute(params, execution, **kwargs):
            execution_order.append(float(params[0]))
            return {"params": params, "metadata": kwargs.get("metadata", {})}

        def process(job, fit_config, **kwargs):
            value = float(job["params"][0])
            return SimpleNamespace(transition_probability_up=value), {
                "stream_type": "marker_optimization_shot",
                "time_axis": [0.0, 0.001],
                "raw_data_up": [0.0, 1.0],
                "raw_data_dw": [0.0, 0.5],
                "fit_data_up": [0.0, 1.0],
                "fit_data_dw": [0.0, 0.5],
            }

        experiment.execute_single_measurement.side_effect = execute
        experiment.process_measurement_job.side_effect = process
        manager = MarkerOptimizationManager(experiment)
        step = {
            "index": 1, "marker_id": "FREQ", "marker_name": "Frequency",
            "metric_key": "transition_probability_up", "metric_source": "fit",
            "values": [1.0, 2.0, 3.0], "average_count": 2, "randomize": True,
            "digital_states": {},
        }
        manager._status = {
            **manager._idle_status(), "is_running": True, "steps": [step],
        }
        with mock.patch("app.core.marker_optimization_manager.random.shuffle", side_effect=lambda items: items.reverse()) as shuffle:
            points, next_shot = manager._evaluate_step(
                step, {}, {}, Path("sequence.mot"), mock.Mock(), 0,
            )

        shuffle.assert_called_once()
        self.assertEqual(execution_order, [3.0, 2.0, 1.0, 3.0, 2.0, 1.0])
        self.assertEqual([point["value"] for point in points], [1.0, 2.0, 3.0])
        self.assertTrue(all(len(point["repeats"]) == 2 for point in points))
        self.assertEqual(next_shot, 6)
        self.assertEqual(manager.get_status()["total_points"], 6)

    def test_failure_in_later_step_keeps_prior_applied_value_and_finalizes_report(self):

        from types import SimpleNamespace
        content = (
            "###SCAN:FREQ###\n"
            "+1us DDS1 [1] (2)\n"
            "###SCAN:POWER###\n"
            "+1us AOM_Raman =10.000 (23)\n"
            "###STATE:ENABLE###\n"
            "+1us TTL_AOM_Raman1 = OFF (49)\n"
        )
        definitions = [
            {"id":"FREQ","display_name":"Frequency","kind":"dds_element","decimals":0,"hard_min":1,"hard_max":3,"default_start":1,"default_stop":3,"default_step":1,"default_method":"step_size","expected_command":"DDS1","expected_channel":"2","has_compensation":False},
            {"id":"POWER","display_name":"Power","kind":"dac_value","decimals":3,"hard_min":10,"hard_max":12,"default_start":10,"default_stop":12,"default_step":1,"default_method":"step_size","expected_command":"AOM_Raman","expected_channel":"23","has_compensation":False},
            {"id":"ENABLE","display_name":"Raman enable","kind":"digital_state","expected_command":"TTL_AOM_Raman1","expected_channel":"49"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            template = base / "source.mot"
            template.write_text(content, encoding="utf-8")
            run_dir = base / "run"
            run_dir.mkdir()

            experiment = mock.Mock()
            experiment.settings = {"template_path": str(template), "sequence_marker_profiles": {"source": definitions}}
            experiment.on_data_ready = None
            experiment.execute_single_measurement.side_effect = lambda params, execution, **kwargs: {"params":params,"metadata":kwargs.get("metadata",{})}
            def process(job, fit_config, **kwargs):
                marker = job["metadata"]["workflow_marker"]
                value = float(job["params"][0])
                metric = (1.0 - abs(value - 2.0)) if marker == "FREQ" else value
                return SimpleNamespace(transition_probability_up=metric), {"stream_type":"marker_optimization_shot"}
            experiment.process_measurement_job.side_effect = process

            class FakeDataManager:
                current_run_dir = run_dir
                current_run_id_str = "run_test"
                def init_run(self, payload): pass
                def close_run(self): pass

            manager = MarkerOptimizationManager(experiment)
            steps = [
                {"index":1,"marker_id":"FREQ","marker_name":"Frequency","marker_kind":"dds_element","objective":"maximize","metric_key":"transition_probability_up","metric_label":"Transition Probability UP","metric_source":"fit","average_count":1,"minimum_r_squared":.75,"start":1,"stop":3,"step":1,"values":[1.0,2.0,3.0],"status":"pending","points":[],"analysis":None,"error":None,"digital_states":{"ENABLE":"ON"},"digital_state_choices":{"ENABLE":"on"},"digital_conditions":[]},
                {"index":2,"marker_id":"POWER","marker_name":"Power","marker_kind":"dac_value","objective":"maximize","metric_key":"transition_probability_up","metric_label":"Transition Probability UP","metric_source":"fit","average_count":1,"minimum_r_squared":.75,"start":10,"stop":12,"step":1,"values":[10.0,11.0,12.0],"status":"pending","points":[],"analysis":None,"error":None,"digital_states":{"ENABLE":"OFF"},"digital_state_choices":{"ENABLE":"off"},"digital_conditions":[]},
            ]
            manager._status = {**manager._idle_status(),"is_running":True,"phase":"running","total_steps":2,"steps":steps,"applied_values":{}}
            captured = {}
            def finalize(path, payload, original, working, encoding):
                captured["working"] = working
                captured["step_1_execution"] = (path / "step_01_execution_conditions.mot").read_text(encoding="utf-8")
                captured["step_2_execution"] = (path / "step_02_execution_conditions.mot").read_text(encoding="utf-8")
                artifact = path / "report.zip"
                artifact.write_bytes(b"report")
                return {"report_bundle": artifact}
            manager._finalize_artifacts = finalize
            with mock.patch("app.core.marker_optimization_manager.DataManager", FakeDataManager):
                manager._run(
                    {"sequence_name":"source.mot"},
                    {},
                    (template, content, "utf-8", definitions, steps),
                )
            status = manager.get_status()
            self.assertEqual(status["phase"], "failed")
            self.assertEqual(status["steps"][0]["status"], "completed")
            self.assertEqual(status["steps"][1]["status"], "failed")
            self.assertEqual(status["applied_values"], {"FREQ": 2})
            self.assertIn("DDS1 [2]", captured["working"])
            self.assertIn("TTL_AOM_Raman1 = OFF", captured["working"])
            self.assertIn("TTL_AOM_Raman1 = ON", captured["step_1_execution"])
            self.assertIn("TTL_AOM_Raman1 = OFF", captured["step_2_execution"])
            self.assertIn("/marker-optimization/download/report_bundle", status["export_urls"]["report_bundle"])

if __name__ == "__main__":
    unittest.main()
