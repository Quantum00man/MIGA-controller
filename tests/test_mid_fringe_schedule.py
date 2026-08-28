import asyncio
import json
import math
from pathlib import Path
import tempfile
import unittest

from app.core.mid_fringe_schedule import (
    build_mid_fringe_task,
    build_virtual_scan_config,
    calibration_at_reference,
    load_prepared_queue,
    save_prepared_queue,
    validate_selected_mid_fringes,
    virtual_shot_count,
)
from app.core.experiment_manager import ExperimentManager
from app.core.link_export import render_link_mot
from app.models.schemas import ArchiveMidFringeScheduleRequest
from app.api import routes


class MidFringeScheduleTests(unittest.TestCase):
    def calibration(self, phi0=0.0):
        return {
            "id": "calibration-a",
            "name": "Node A fringe",
            "parameter_values": {"A": 2.0, "C": 5.0, "phi0": phi0},
            "bragg": {"angular_frequency_rad_per_us2": 0.2, "mid_fringe_x": [1.0, 2.0]},
            "fit_x": [0.0, 1.0, 2.0],
            "fit_y": [7.0, 6.96, 6.84],
            "fit_min": 0.0,
            "fit_max": 2.0,
            "channel": "up",
            "source_field": "interferometer_p_up",
        }

    def test_reference_only_changes_the_temporary_origin_and_uses_own_curve(self):
        first = calibration_at_reference(self.calibration(phi0=0.0), 1.0)
        second = calibration_at_reference(self.calibration(phi0=math.pi), 1.0)

        self.assertEqual(first["reference_t2_us2"], 1.0)
        self.assertEqual(first["reference_value"], 1.0)
        self.assertEqual(first["reference_input_mode"], "t2")
        self.assertNotEqual(first["monotonic_slope"], second["monotonic_slope"])
        self.assertEqual(first["name"], "Node A fringe")

    def test_virtual_scan_is_a_standard_one_dimensional_shot_counter(self):
        source = {
            "mode": "link",
            "start": 10,
            "stop": 20,
            "step": 2,
            "averages": 8,
            "randomize": True,
            "link_formulas": ["P0 * 2"],
        }
        calibration = calibration_at_reference(self.calibration(), 1.0)
        result = build_virtual_scan_config(source, 1.0, calibration, 1, 100, 1, "master.mot")

        self.assertEqual(result["mode"], "standard")
        self.assertEqual((result["start"], result["stop"], result["step"]), (1.0, 100.0, 1.0))
        self.assertEqual(result["averages"], 1)
        self.assertFalse(result["randomize"])
        self.assertEqual(result["interferometer_phase_calibration_override"]["reference_t2_us2"], 1.0)

    def test_selected_master_mid_fringe_drives_all_link_parameters(self):
        manager = ExperimentManager.__new__(ExperimentManager)
        source = {
            "mode": "link",
            "scan_dimensions": 1,
            "dim1_type": "range",
            "dim1_method": "step_size",
            "param_type": "float",
            "start": 1,
            "stop": 2,
            "step": 1,
            "link_formulas": ["P0 * 2", "P1 + 3"],
        }
        parameters = manager.build_link_export_parameter_sets(source, p0=12.5)[0]
        rendered = render_link_mot("<PARAMETER0>|<PARAMETER1>|<PARAMETER2>", parameters)
        self.assertEqual(parameters, [12.5, 25.0, 28.0])
        self.assertEqual(rendered, "12.500000|25.000000|28.000000")

    def test_mid_fringe_selection_and_shot_count_are_validated(self):
        fit = {"bragg": {"mid_fringe_x": [10.0, 20.0]}}
        self.assertEqual(validate_selected_mid_fringes(fit, [20, 10, 20]), [10.0, 20.0])
        self.assertEqual(virtual_shot_count(1, 100, 1), 100)
        with self.assertRaises(ValueError):
            validate_selected_mid_fringes(fit, [15])
        with self.assertRaises(ValueError):
            virtual_shot_count(10, 1, 1)

    def test_task_note_and_prepared_queue_persist(self):
        task = build_mid_fringe_task(
            batch_id="midfringe_test",
            index=0,
            reference=12.5,
            sequence_name="master.mot",
            sequence_snapshot="fixed master sequence",
            config={"start": 1, "stop": 100, "step": 1},
            shot_count=100,
            execution_mode="scan",
            sync=None,
        )
        self.assertIn("P0=12.5", task["note"])
        with tempfile.TemporaryDirectory() as temporary:
            payload = {"batch_id": "midfringe_test", "tasks": [task]}
            save_prepared_queue(Path(temporary), payload)
            loaded = load_prepared_queue(Path(temporary), "midfringe_test")
            self.assertEqual(loaded, json.loads(json.dumps(payload)))

    def test_archive_and_index_expose_the_prepared_queue_workflow(self):
        root = Path(__file__).resolve().parents[1]
        archive_html = (root / "static" / "archive.html").read_text(encoding="utf-8")
        index_html = (root / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("Generate phase-noise schedule", archive_html)
        self.assertIn("Add to Scheduled Queue", archive_html)
        self.assertIn("openPreparedMidFringeQueue", archive_html)
        self.assertIn("importPreparedQueueFromUrl", index_html)
        self.assertIn("scheduledTaskQueueCard", index_html)

    def test_regular_archive_route_builds_fixed_sequences_without_changing_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "2026" / "08" / "28" / "run01_20260828"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(json.dumps({
                "mode": "link",
                "scan_dimensions": 1,
                "dim1_type": "range",
                "dim1_method": "step_size",
                "param_type": "float",
                "start": 10,
                "stop": 20,
                "step": 1,
                "link_formulas": ["P0 * 2", "P1 + 3"],
                "sequence_name": "master.mot",
            }), encoding="utf-8")
            (run_dir / "sequence.mot").write_text(
                "<PARAMETER0>|<PARAMETER1>|<PARAMETER2>", encoding="utf-8"
            )
            fit_result = {
                "model_key": "bragg_fringes",
                "fit_min": 0.0,
                "fit_max": 30.0,
                "fit_x": [0.0, 15.0, 30.0],
                "fit_y": [7.0, 3.0, 7.0],
                "parameter_values": {"A": 2.0, "C": 5.0, "phi0": 0.0},
                "bragg": {
                    "angular_frequency_rad_per_us2": 0.2,
                    "mid_fringe_x": [12.5, 20.0],
                },
                "metric_tab": "intf",
                "channel": "up",
                "source_key": "intf_p",
            }
            request = ArchiveMidFringeScheduleRequest(
                year="2026", month="08", day="28", run_id="run01_20260828",
                fit_result=fit_result, mid_fringe_values=[12.5],
                shot_start=1, shot_stop=3, shot_step=1, gap_seconds=10,
            )
            original_base = routes.data_loader.base_dir
            routes.data_loader.base_dir = root
            try:
                response = asyncio.run(routes.prepare_archive_mid_fringe_schedule(request))
                saved = load_prepared_queue(run_dir, response["batch_id"])
            finally:
                routes.data_loader.base_dir = original_base

            task = saved["tasks"][0]
            self.assertEqual(task["sequence_snapshot"], "12.500000|25.000000|28.000000")
            self.assertTrue(task["temporary_sequence"])
            self.assertEqual(task["config"]["mode"], "standard")
            self.assertEqual(task["config"]["averages"], 1)
            self.assertEqual(
                task["config"]["interferometer_phase_calibration_override"]["reference_t2_us2"],
                12.5,
            )


if __name__ == "__main__":
    unittest.main()
