import math
import unittest
import csv
import json
import queue
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from app.analysis.transfer_function import (
    bragg_phase_modulation_rad,
    build_differential_transfer_function_summary,
    build_transfer_function_summary,
)
from app.core.experiment_manager import ExperimentManager, transfer_recovery_wait_seconds
from app.core.data_manager import DataManager, RESULTS_CSV_HEADER
from app.core.data_loader import DataLoader
from app.drivers.tti_generator import (
    set_tti_test_frequency,
    set_tti_test_output,
    set_tti_test_phase,
    TtiConnectionSettings,
    TtiGeneratorClient,
    TtiGeneratorError,
)


class FakeSocket:
    def __init__(self, replies):
        self.replies = bytearray("".join(f"{reply}\n" for reply in replies).encode("ascii"))
        self.sent = []

    def settimeout(self, _timeout):
        pass

    def sendall(self, payload):
        self.sent.append(payload.decode("ascii"))

    def recv(self, size):
        block = bytes(self.replies[:size])
        del self.replies[:size]
        return block

    def close(self):
        pass


class TtiGeneratorClientTests(unittest.TestCase):
    def test_connect_and_frequency_command_use_tg5012a_protocol(self):
        fake = FakeSocket(["TTi,TG5012A,1234,1.00", "1"])
        with patch("app.drivers.tti_generator.socket.create_connection", return_value=fake):
            with TtiGeneratorClient(TtiConnectionSettings("192.168.1.8")) as client:
                client.set_ch1_frequency(1_250_000.5)
        self.assertEqual(fake.sent, [
            "*IDN?\n",
            "CHN 1;FREQ 1250000.5;*OPC?\n",
        ])

    def test_wrong_model_is_rejected(self):
        fake = FakeSocket(["TTi,TGF3162,1234,1.00"])
        with patch("app.drivers.tti_generator.socket.create_connection", return_value=fake):
            with self.assertRaisesRegex(TtiGeneratorError, "Expected TG5012A"):
                TtiGeneratorClient(TtiConnectionSettings("192.168.1.8")).connect()

    def test_settings_frequency_action_connects_and_sets_only_ch1_frequency(self):
        fake = FakeSocket(["TTi,TG5012A,1234,1.00", "1"])
        with patch("app.drivers.tti_generator.socket.create_connection", return_value=fake):
            identity = set_tti_test_frequency(
                TtiConnectionSettings("192.168.1.8"),
                9876.54321,
            )
        self.assertIn("TG5012A", identity)
        self.assertEqual(fake.sent, [
            "*IDN?\n",
            "CHN 1;FREQ 9876.54321;*OPC?\n",
        ])

    def test_tg5012a_can_set_channel_two(self):
        fake = FakeSocket(["TTi,TG5012A,1234,1.00", "1"])
        settings = TtiConnectionSettings("192.168.1.8", model="TG5012A", channel=2)
        with patch("app.drivers.tti_generator.socket.create_connection", return_value=fake):
            with TtiGeneratorClient(settings) as client:
                client.set_frequency(1234.5)
        self.assertEqual(fake.sent, ["*IDN?\n", "CHN 2;FREQ 1234.5;*OPC?\n"])

    def test_tgf3162_uses_fire_and_forget_frequency_command(self):
        fake = FakeSocket(["THURLBY THANDAR,TGF3162,1234,1.03"])
        settings = TtiConnectionSettings("192.168.1.9", model="TGF3162", channel=2)
        with patch("app.drivers.tti_generator.socket.create_connection", return_value=fake):
            with TtiGeneratorClient(settings) as client:
                client.set_frequency(2_000_000)
        self.assertEqual(fake.sent, ["*IDN?\n", "CHN 2;FREQ 2000000\n"])

    def test_tg5012a_phase_command_is_confirmed(self):
        fake = FakeSocket(["TTi,TG5012A,1234,1.00", "1"])
        with patch("app.drivers.tti_generator.socket.create_connection", return_value=fake):
            with TtiGeneratorClient(TtiConnectionSettings("192.168.1.8", channel=2)) as client:
                client.set_phase(90)
        self.assertEqual(fake.sent, ["*IDN?\n", "CHN 2;PHASE 90;*OPC?\n"])

    def test_settings_phase_action_connects_and_sets_only_selected_phase(self):
        fake = FakeSocket(["TTi,TG5012A,1234,1.00", "1"])
        settings = TtiConnectionSettings("192.168.1.8", model="TG5012A", channel=2)
        with patch("app.drivers.tti_generator.socket.create_connection", return_value=fake):
            identity = set_tti_test_phase(settings, 90)
        self.assertIn("TG5012A", identity)
        self.assertEqual(fake.sent, ["*IDN?\n", "CHN 2;PHASE 90;*OPC?\n"])

    def test_tgf3162_phase_command_is_fire_and_forget(self):
        fake = FakeSocket(["THURLBY THANDAR,TGF3162,1234,1.03"])
        settings = TtiConnectionSettings("192.168.1.9", model="TGF3162", channel=1)
        with patch("app.drivers.tti_generator.socket.create_connection", return_value=fake):
            with TtiGeneratorClient(settings) as client:
                client.set_phase(0)
        self.assertEqual(fake.sent, ["*IDN?\n", "CHN 1;PHASE 0\n"])

    def test_settings_output_action_confirms_tg5012a_selected_channel(self):
        fake = FakeSocket(["TTi,TG5012A,1234,1.00", "1"])
        settings = TtiConnectionSettings("192.168.1.8", model="TG5012A", channel=2)
        with patch("app.drivers.tti_generator.socket.create_connection", return_value=fake):
            identity = set_tti_test_output(settings, True)
        self.assertIn("TG5012A", identity)
        self.assertEqual(fake.sent, ["*IDN?\n", "CHN 2;OUTPUT ON;*OPC?\n"])

    def test_tgf3162_output_command_is_fire_and_forget(self):
        fake = FakeSocket(["THURLBY THANDAR,TGF3162,1234,1.03"])
        settings = TtiConnectionSettings("192.168.1.9", model="TGF3162", channel=1)
        with patch("app.drivers.tti_generator.socket.create_connection", return_value=fake):
            set_tti_test_output(settings, False)
        self.assertEqual(fake.sent, ["*IDN?\n", "CHN 1;OUTPUT OFF\n"])


class TransferFunctionPlanTests(unittest.TestCase):
    def test_low_frequency_recovery_wait_rounds_extra_time_up(self):
        self.assertEqual(transfer_recovery_wait_seconds(0.2, 2.6), 3)
        self.assertEqual(transfer_recovery_wait_seconds(0.38, 2.6), 1)
        self.assertEqual(transfer_recovery_wait_seconds(1.0, 2.6), 0)
        self.assertEqual(transfer_recovery_wait_seconds(0, 2.6), 0)

    def test_frontend_does_not_treat_null_frequency_as_transfer_function_point(self):
        index_html = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "const isTransferFunctionPoint = this.hasFiniteNumericValue(data.transfer_frequency_hz);",
            index_html,
        )
        self.assertIn(
            "value !== null && value !== undefined && value !== ''",
            index_html,
        )

    def test_frontend_allows_scheduled_transfer_function_and_estimates_its_shots(self):
        index_html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('<option value="transfer_function">Transfer Function</option>', index_html)
        self.assertIn("isTransferFunctionMode(mode = this.config.mode)", index_html)
        self.assertIn("transfer_function: 'Transfer Function'", index_html)
        self.assertIn('aria-label="SYNC Transfer Function statistic"', index_html)
        self.assertIn("getTransferRecoveryEstimateSeconds", index_html)
        self.assertIn("low-frequency recovery", index_html)

    def test_settings_offer_rigol_transfer_generator_and_reuse_its_connection(self):
        settings_html = (Path(__file__).resolve().parents[1] / "static" / "settings.html").read_text(encoding="utf-8")
        self.assertIn('<option value="DG4162">RIGOL DG4162</option>', settings_html)
        self.assertIn("transferGeneratorConnectionPayload()", settings_html)
        self.assertIn("useRigol ? this.s.rigol_host : this.s.tti_host", settings_html)

    def test_phase_transfer_plot_keeps_its_single_panel_x_axis_visible(self):
        index_html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("const showXAxis = !isUp || this.currentTab === 'phase';", index_html)
        self.assertIn("showticklabels: showXAxis, automargin: true", index_html)

    def test_burst_time_plan_uses_timing_parameters_at_fixed_tti_frequency(self):
        config = {
            "scan_dimensions": 1,
            "parameter_source": "classic",
            "mode": "transfer_burst_time_scan",
            "randomize": False,
            "start": 10,
            "stop": 20,
            "step": 10,
            "dim1_type": "range",
            "dim1_method": "step_size",
            "param_type": "float",
            "mode_param": 100,
            "transfer_burst_time_frequency_hz": 2500,
            "transfer_repeats": 2,
            "transfer_phase_degrees": [0],
        }
        plan = self.manager._build_transfer_function_execution(config)
        self.assertEqual([point["sequence_parameters"] for point in plan], [[10.0, 90.0], [10.0, 90.0], [20.0, 80.0], [20.0, 80.0]])
        self.assertEqual([point["metadata"]["transfer_frequency_hz"] for point in plan], [10.0, 10.0, 20.0, 20.0])
        self.assertTrue(all(point["metadata"]["transfer_generator_frequency_hz"] == 2500.0 for point in plan))
        self.assertTrue(all(point["metadata"]["transfer_response_axis"] == "p0" for point in plan))

    def test_archive_exposes_sync_differential_transfer_function(self):
        archive_html = (Path(__file__).resolve().parents[1] / "static" / "archive.html").read_text(encoding="utf-8")
        self.assertIn("Differential Transfer Function", archive_html)
        self.assertIn("syncArchiveTransferDifferentialRows()", archive_html)
        self.assertIn("syncArchiveTransferPhaseDifferencePlot", archive_html)
        self.assertIn("syncArchiveTransferPathNormalizedPlot", archive_html)
        self.assertIn("phase_difference_0deg_mean_rad", archive_html)
        self.assertIn("differential_path_s2", archive_html)
        self.assertIn("differential_path_noise_s2", archive_html)
        self.assertIn("renderSyncArchiveTransferFunctionPlot(element)", archive_html)
        self.assertIn("downloadSyncTransferFunctionDifferentialCSV", archive_html)
        self.assertIn("row.differential_s2", archive_html)
        self.assertIn("Averaged by frequency and generator phase", archive_html)
        self.assertIn("SYNC Transfer Function Phase · Per-frequency mean", archive_html)

    def setUp(self):
        self.manager = ExperimentManager.__new__(ExperimentManager)
        self.manager.settings = {"tti_model": "TG5012A", "tti_channel": 1}

    def test_plan_runs_fixed_sequence_repeated_at_each_frequency(self):
        config = {
            "scan_dimensions": 1,
            "parameter_source": "classic",
            "mode": "transfer_function",
            "randomize": False,
            "transfer_frequency_start_hz": 100,
            "transfer_frequency_stop_hz": 300,
            "transfer_frequency_step_hz": 100,
            "transfer_burst_time_frequency_hz": None,
            "transfer_repeats": 3,
            "transfer_settling_time_s": 0,
        }
        plan = self.manager._build_transfer_function_execution(config)
        self.assertEqual(len(plan), 18)
        self.assertTrue(all(point["sequence_parameters"] == [] for point in plan))
        self.assertEqual(
            [point["metadata"]["transfer_frequency_hz"] for point in plan],
            ([100.0] * 3 + [200.0] * 3 + [300.0] * 3) * 2,
        )
        self.assertEqual(
            [point["metadata"]["transfer_phase_deg"] for point in plan],
            [0.0] * 9 + [90.0] * 9,
        )
        self.assertEqual(config["transfer_frequency_values_hz"], [100.0, 200.0, 300.0])
        self.assertEqual(config["transfer_burst_time_frequency_hz"], 1000.0)
        self.assertEqual(config["averages"], 1)
        self.assertEqual(config["transfer_settling_time_s"], 5.0)
        self.assertEqual(config["transfer_generator_model"], "TG5012A")
        self.assertEqual(config["transfer_generator_channel"], 1)
        self.assertEqual(config["transfer_frequency_modulation_mhz"], 1.0)
        self.assertEqual(config["transfer_atom_mirror_distance_m"], 2.23)
        self.assertEqual(config["transfer_phase_noise_sigma_mrad"], 100.0)
        self.assertEqual(config["transfer_phase_degrees"], [0.0, 90.0])
        self.assertEqual(config["transfer_phase_scan_mode"], "phase_blocks")
        self.assertFalse(config["transfer_control_output"])
        self.assertTrue(all(
            point["metadata"]["transfer_frequency_modulation_mhz"] == 1.0
            for point in plan
        ))
        self.assertTrue(all(
            point["metadata"]["transfer_atom_mirror_distance_m"] == 2.23
            for point in plan
        ))
        self.assertTrue(all(
            point["metadata"]["transfer_phase_noise_sigma_mrad"] == 100.0
            for point in plan
        ))

    def test_plan_supports_descending_frequency(self):
        config = {
            "scan_dimensions": 1,
            "parameter_source": "classic",
            "randomize": False,
            "transfer_frequency_start_hz": 3,
            "transfer_frequency_stop_hz": 1,
            "transfer_frequency_step_hz": 1,
            "transfer_repeats": 2,
        }
        plan = self.manager._build_transfer_function_execution(config)
        self.assertEqual(
            [point["metadata"]["transfer_frequency_hz"] for point in plan],
            [3.0, 3.0, 2.0, 2.0, 1.0, 1.0] * 2,
        )

    def test_plan_can_run_only_one_selected_phase(self):
        config = {
            "scan_dimensions": 1,
            "parameter_source": "classic",
            "randomize": False,
            "transfer_frequency_start_hz": 100,
            "transfer_frequency_stop_hz": 200,
            "transfer_frequency_step_hz": 100,
            "transfer_repeats": 2,
            "transfer_phase_degrees": [90],
        }
        plan = self.manager._build_transfer_function_execution(config)
        self.assertEqual(len(plan), 4)
        self.assertTrue(all(point["metadata"]["transfer_phase_deg"] == 90.0 for point in plan))

    def test_plan_can_interleave_phases_at_each_frequency(self):
        config = {
            "scan_dimensions": 1,
            "parameter_source": "classic",
            "randomize": False,
            "transfer_frequency_start_hz": 100,
            "transfer_frequency_stop_hz": 200,
            "transfer_frequency_step_hz": 100,
            "transfer_repeats": 2,
            "transfer_phase_degrees": [0, 90],
            "transfer_phase_scan_mode": "frequency_interleaved",
            "transfer_control_output": True,
        }
        plan = self.manager._build_transfer_function_execution(config)
        self.assertEqual(
            [point["metadata"]["transfer_frequency_hz"] for point in plan],
            [100.0] * 4 + [200.0] * 4,
        )
        self.assertEqual(
            [point["metadata"]["transfer_phase_deg"] for point in plan],
            [0.0, 0.0, 90.0, 90.0] * 2,
        )
        self.assertTrue(all(point["metadata"]["transfer_control_output"] for point in plan))

    def test_plan_can_converge_from_frequency_endpoints(self):
        config = {
            "scan_dimensions": 1, "parameter_source": "classic", "mode": "transfer_function",
            "randomize": False, "transfer_frequency_start_hz": 100, "transfer_frequency_stop_hz": 500,
            "transfer_frequency_step_hz": 100, "transfer_repeats": 2,
            "transfer_phase_degrees": [0], "transfer_frequency_order": "symmetric_converging",
        }
        plan = self.manager._build_transfer_function_execution(config)
        self.assertEqual(
            [point["metadata"]["transfer_frequency_hz"] for point in plan if point["metadata"]["transfer_repeat"] == 1],
            [100.0, 500.0, 200.0, 400.0, 300.0],
        )
        self.assertEqual(config["transfer_frequency_values_hz"], [100.0, 500.0, 200.0, 400.0, 300.0])

    def test_plan_can_randomize_frequency_order_once_for_all_phase_blocks(self):
        config = {
            "scan_dimensions": 1, "parameter_source": "classic", "mode": "transfer_function",
            "randomize": False, "transfer_frequency_start_hz": 100, "transfer_frequency_stop_hz": 300,
            "transfer_frequency_step_hz": 100, "transfer_repeats": 2,
            "transfer_phase_degrees": [0, 90], "transfer_frequency_order": "random",
        }

        def reverse_in_place(values):
            values.reverse()

        with patch("app.core.experiment_manager.random.shuffle", side_effect=reverse_in_place) as shuffle:
            plan = self.manager._build_transfer_function_execution(config)

        self.assertEqual(shuffle.call_count, 1)
        self.assertEqual(config["transfer_frequency_values_hz"], [300.0, 200.0, 100.0])
        self.assertEqual(
            [point["metadata"]["transfer_frequency_hz"] for point in plan],
            ([300.0] * 2 + [200.0] * 2 + [100.0] * 2) * 2,
        )

    def test_plan_prepends_output_off_zero_phase_calibration(self):
        config = {
            "scan_dimensions": 1, "parameter_source": "classic", "mode": "transfer_function",
            "randomize": False, "transfer_frequency_start_hz": 100, "transfer_frequency_stop_hz": 100,
            "transfer_frequency_step_hz": 100, "transfer_repeats": 2,
            "transfer_control_output": True, "transfer_calibrate_zero_phase": True,
            "transfer_zero_phase_repeats": 3,
        }
        plan = self.manager._build_transfer_function_execution(config)
        baseline = plan[:3]
        self.assertTrue(all(item["metadata"]["transfer_zero_phase_baseline"] for item in baseline))
        self.assertEqual([item["metadata"]["transfer_zero_phase_repeat"] for item in baseline], [1, 2, 3])
        self.assertEqual(len(plan), 7)

    def test_plan_inserts_periodic_zero_phase_after_complete_frequency_points(self):
        config = {
            "scan_dimensions": 1, "parameter_source": "classic", "mode": "transfer_function",
            "randomize": False, "transfer_frequency_start_hz": 100, "transfer_frequency_stop_hz": 300,
            "transfer_frequency_step_hz": 100, "transfer_repeats": 2,
            "transfer_phase_degrees": [0, 90], "transfer_phase_scan_mode": "frequency_interleaved",
            "transfer_control_output": True, "transfer_calibrate_zero_phase": True,
            "transfer_zero_phase_repeats": 3, "transfer_periodic_zero_phase": True,
            "transfer_zero_phase_frequency_interval": 2,
        }
        plan = self.manager._build_transfer_function_execution(config)
        baseline_indices = [index for index, item in enumerate(plan) if item["metadata"].get("transfer_zero_phase_baseline")]
        self.assertEqual(baseline_indices, [0, 1, 2, 11, 12, 13])
        self.assertEqual([plan[index]["metadata"]["transfer_zero_phase_block_id"] for index in baseline_indices], [0, 0, 0, 1, 1, 1])
        self.assertEqual(len(plan), 18)

    def test_invalid_phase_scan_mode_is_rejected(self):
        config = {
            "scan_dimensions": 1,
            "parameter_source": "classic",
            "randomize": False,
            "transfer_frequency_start_hz": 100,
            "transfer_frequency_stop_hz": 100,
            "transfer_frequency_step_hz": 100,
            "transfer_repeats": 2,
            "transfer_phase_scan_mode": "unknown",
        }
        with self.assertRaisesRegex(ValueError, "phase scan mode"):
            self.manager._build_transfer_function_execution(config)

    def test_acquisition_controls_output_once_and_cleans_up_after_error(self):
        manager = ExperimentManager.__new__(ExperimentManager)
        manager.settings = {"tti_host": "192.168.1.8", "tti_port": 9221, "tti_timeout_s": 3}
        manager.stop_flag = False
        manager.status = SimpleNamespace(message="")
        manager.data_queue = queue.Queue()
        manager._scan_finalize_error = None
        manager._restore_ac_stark_dds = lambda _context: None
        manager.execute_single_measurement = Mock(side_effect=RuntimeError("measurement failed"))
        client = Mock()
        client.connect.return_value = "TTi,TG5012A,1234,1.00"
        parameters = [{
            "sequence_parameters": [],
            "metadata": {"transfer_frequency_hz": 100.0, "transfer_phase_deg": 0.0},
        }]
        scan_config = {
            "mode": "transfer_function",
            "scan_dimensions": 1,
            "transfer_generator_model": "TG5012A",
            "transfer_generator_channel": 1,
            "transfer_control_output": True,
            "transfer_settling_time_s": 0,
        }
        with patch("app.core.experiment_manager.config.USE_SIMULATION", False), \
                patch("app.core.experiment_manager.TtiGeneratorClient", return_value=client):
            manager._acquisition_loop(parameters, scan_config)

        self.assertEqual(client.set_output.call_args_list, [call(True), call(False)])
        client.close.assert_called_once_with()
        self.assertIn("measurement failed", manager._scan_finalize_error)

    def test_acquisition_can_control_rigol_frequency_phase_and_output(self):
        manager = ExperimentManager.__new__(ExperimentManager)
        manager.settings = {"rigol_host": "192.168.1.40", "rigol_port": 5555, "rigol_timeout_s": 3}
        manager.stop_flag = False
        manager.status = SimpleNamespace(message="")
        manager.data_queue = queue.Queue()
        manager._scan_finalize_error = None
        manager._restore_ac_stark_dds = lambda _context: None
        manager.execute_single_measurement = Mock(side_effect=RuntimeError("measurement failed"))
        client = Mock()
        client.connect.return_value = "RIGOL TECHNOLOGIES,DG4162,123,1.0"
        parameters = [{
            "sequence_parameters": [],
            "metadata": {"transfer_frequency_hz": 100.0, "transfer_phase_deg": 90.0},
        }]
        scan_config = {
            "mode": "transfer_function", "scan_dimensions": 1,
            "transfer_generator_model": "DG4162", "transfer_generator_channel": 2,
            "transfer_control_output": True, "transfer_settling_time_s": 0,
        }
        with patch("app.core.experiment_manager.config.USE_SIMULATION", False), patch(
            "app.core.experiment_manager.RigolGeneratorClient", return_value=client
        ):
            manager._acquisition_loop(parameters, scan_config)

        client.set_frequency.assert_called_once_with(2, 100.0)
        client.set_phase.assert_called_once_with(2, 90.0)
        self.assertEqual(client.set_output.call_args_list, [call(2, True), call(2, False)])
        client.close.assert_called_once_with()
        self.assertIn("measurement failed", manager._scan_finalize_error)

    def test_acquisition_checks_recovery_only_between_transfer_shots(self):
        manager = ExperimentManager.__new__(ExperimentManager)
        manager.settings = {"tti_host": "192.168.1.8", "tti_port": 9221, "tti_timeout_s": 3}
        manager.stop_flag = False
        manager.status = SimpleNamespace(message="")
        manager.data_queue = queue.Queue()
        manager._scan_finalize_error = None
        manager._restore_ac_stark_dds = lambda _context: None
        manager.execute_single_measurement = Mock(return_value={})
        client = Mock()
        client.connect.return_value = "TTi,TG5012A,1234,1.00"
        parameters = [{
            "sequence_parameters": [],
            "metadata": {
                "transfer_frequency_hz": 0.2,
                "transfer_phase_deg": 0.0,
                "transfer_repeat": repeat,
            },
        } for repeat in (1, 2)]
        scan_config = {
            "mode": "transfer_function", "scan_dimensions": 1,
            "transfer_generator_model": "TG5012A", "transfer_generator_channel": 1,
            "transfer_control_output": False, "transfer_settling_time_s": 0,
        }
        with patch("app.core.experiment_manager.config.USE_SIMULATION", False), patch(
            "app.core.experiment_manager.TtiGeneratorClient", return_value=client
        ), patch(
            "app.core.experiment_manager.transfer_recovery_wait_seconds", return_value=0
        ) as recovery:
            manager._acquisition_loop(parameters, scan_config)

        recovery.assert_called_once()
        self.assertEqual(recovery.call_args.args[0], 0.2)
        self.assertEqual(manager.execute_single_measurement.call_count, 2)


class TransferFunctionStatisticsTests(unittest.TestCase):
    def test_differential_summary_subtracts_signed_quadratures_before_squaring(self):
        def record(phase, modulation, distance):
            return {
                "interferometer_phase": phase,
                "interferometer_phase_valid": True,
                "transfer_frequency_modulation_mhz": modulation,
                "transfer_atom_mirror_distance_m": distance,
                "transfer_phase_noise_sigma_mrad": 100.0,
            }

        master_amplitude = bragg_phase_modulation_rad(1.0, 2.0)
        slave_amplitude = bragg_phase_modulation_rad(2.0, 3.0)
        pairs = []
        for phase_deg, master_s, slave_s in ((0.0, 3.0, 1.0), (90.0, 5.0, 2.0)):
            pairs.append({
                "slave_node_id": "slave_b",
                "master": {
                    **record(master_s * master_amplitude, 1.0, 2.0),
                    "transfer_frequency_hz": 100.0,
                    "transfer_phase_deg": phase_deg,
                },
                "slave": {
                    **record(slave_s * slave_amplitude, 2.0, 3.0),
                    "transfer_frequency_hz": 100.0,
                    "transfer_phase_deg": phase_deg,
                },
            })

        row = build_differential_transfer_function_summary(pairs)[0]
        self.assertAlmostEqual(row["delta_s_0deg_mean"], 2.0)
        self.assertAlmostEqual(row["delta_s_90deg_mean"], 3.0)
        self.assertAlmostEqual(row["delta_s_0deg_s2"], 4.0)
        self.assertAlmostEqual(row["delta_s_90deg_s2"], 9.0)
        self.assertAlmostEqual(row["differential_s2"], 13.0)
        expected_component_noise = (0.1 / master_amplitude) ** 2 + (0.1 / slave_amplitude) ** 2
        self.assertAlmostEqual(row["delta_s_0deg_noise_s2"], expected_component_noise)
        self.assertAlmostEqual(row["delta_s_90deg_noise_s2"], expected_component_noise)
        self.assertAlmostEqual(row["differential_noise_s2"], 2 * expected_component_noise)
        self.assertAlmostEqual(row["differential_magnitude"], 13.0 ** 0.5)
        self.assertAlmostEqual(
            row["phase_difference_0deg_mean_rad"],
            3.0 * master_amplitude - slave_amplitude,
        )
        self.assertAlmostEqual(
            row["phase_difference_90deg_mean_rad"],
            5.0 * master_amplitude - 2.0 * slave_amplitude,
        )
        self.assertAlmostEqual(
            row["phase_difference_magnitude_rad"],
            math.hypot(
                3.0 * master_amplitude - slave_amplitude,
                5.0 * master_amplitude - 2.0 * slave_amplitude,
            ),
        )
        path_denominator = 4.0 * math.pi * 1_000_000.0 * (2.0 - 3.0) / 299_792_458.0
        expected_path_s2 = (
            (3.0 * master_amplitude - slave_amplitude) ** 2
            + (5.0 * master_amplitude - 2.0 * slave_amplitude) ** 2
        ) / path_denominator ** 2
        expected_path_noise_floor = 2.0 * (0.1 ** 2 + 0.1 ** 2) / path_denominator ** 2
        self.assertAlmostEqual(row["differential_path_s2"], expected_path_s2)
        self.assertAlmostEqual(row["differential_path_noise_s2"], expected_path_noise_floor)
        self.assertTrue(row["quadrature_complete"])

        zero_path_pairs = [
            {**pair, "slave": {**pair["slave"], "transfer_atom_mirror_distance_m": 2.0}}
            for pair in pairs
        ]
        zero_path_row = build_differential_transfer_function_summary(zero_path_pairs)[0]
        self.assertIsNone(zero_path_row["differential_path_s2"])
        self.assertIsNone(zero_path_row["differential_path_noise_s2"])

    def test_summary_uses_sample_standard_deviation_and_total_per_shot(self):
        rows = [
            {
                "transfer_frequency_hz": 100.0,
                "atom_number_up": up,
                "atom_number_dw": down,
                "atom_number_up_nofit": up + 1,
                "atom_number_dw_nofit": down + 1,
                "intf_p1": up / 10,
                "intf_p2": down / 10,
                "intf_p1_nofit": up / 20,
                "intf_p2_nofit": down / 20,
                "interferometer_phase": phase,
                "interferometer_phase_valid": True,
            }
            for up, down, phase in [(1, 10, 0.1), (2, 8, 0.2), (3, 6, 0.3)]
        ]
        summary = build_transfer_function_summary(rows)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["atom_number_up_fit_std"], 1.0)
        self.assertAlmostEqual(summary[0]["atom_number_total_fit_std"], 1.0)
        self.assertAlmostEqual(summary[0]["interferometer_phase_std"], 0.1)
        self.assertAlmostEqual(summary[0]["atom_number_up_fit_mean"], 2.0)
        self.assertAlmostEqual(summary[0]["atom_number_total_fit_mean"], 10.0)
        self.assertAlmostEqual(summary[0]["interferometer_phase_mean"], 0.2)
        self.assertEqual(summary[0]["interferometer_phase_count"], 3)

    def test_frontends_can_switch_between_std_mean_s2_and_phase2(self):
        archive_html = (
            Path(__file__).resolve().parents[1] / "static" / "archive.html"
        ).read_text(encoding="utf-8")
        index_html = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("transferFunctionStatistic: 'std'", archive_html)
        self.assertIn("transferFunctionYAxisScale: 'linear'", archive_html)
        self.assertIn("setTransferFunctionStatistic('mean')", archive_html)
        self.assertIn("setTransferFunctionStatistic('s2')", archive_html)
        self.assertIn("setTransferFunctionStatistic('phase2')", archive_html)
        self.assertIn("setTransferFunctionYAxisScale('log')", archive_html)
        self.assertIn("`interferometer_phase_${statistic}`", archive_html)
        self.assertIn("`${statisticLabel} quadrature sum`", archive_html)
        self.assertIn("Allan phase-noise floor", archive_html)
        self.assertIn("row.interferometer_phase_noise_phase2_rad2", archive_html)
        self.assertIn("row.interferometer_phase_noise_s2", archive_html)
        self.assertIn("`${fieldBase}_${phaseLabel}_${statistic}`", archive_html)
        self.assertIn("`${labels[index]} ${statisticLabel} at ${phaseDeg}°`", archive_html)
        self.assertIn("isSquareSummaryArchive()", archive_html)
        self.assertIn("return this.isPhaseNoiseArchive() || this.isTransferFunctionArchive()", archive_html)
        self.assertIn('v-show="!isSquareSummaryArchive()"', archive_html)
        self.assertIn("transferFunctionStatistic: 'std'", index_html)
        self.assertIn("transferFunctionYAxisScale: 'linear'", index_html)
        self.assertIn("setTransferFunctionStatistic('s2')", index_html)
        self.assertIn("setTransferFunctionStatistic('phase2')", index_html)
        self.assertIn("setTransferFunctionYAxisScale('log')", index_html)
        self.assertIn("transferBraggPhaseAmplitudeRad()", index_html)
        self.assertIn("`${statisticLabel} quadrature sum`", index_html)
        self.assertIn("Allan phase-noise floor", index_html)
        self.assertIn("this.transferPhaseNoiseSigmaMrad", index_html)
        self.assertNotIn("S² phase-noise floor", index_html)
        self.assertIn("`${channel} ${statisticLabel} at ${phaseDeg}°`", index_html)
        self.assertIn('v-model="config.transfer_phase_degrees"', index_html)
        self.assertIn('id="transferPhase0"', index_html)
        self.assertIn('id="transferPhase90"', index_html)
        self.assertIn('v-model="config.transfer_phase_scan_mode"', index_html)
        self.assertIn('v-model="config.transfer_frequency_order"', index_html)
        self.assertIn('symmetric_converging', index_html)
        self.assertIn('<option value="random">Random</option>', index_html)
        self.assertIn('v-model="config.transfer_control_output"', index_html)

        settings_html = (
            Path(__file__).resolve().parents[1] / "static" / "settings.html"
        ).read_text(encoding="utf-8")
        self.assertIn("setTtiTestOutput(true)", settings_html)
        self.assertIn("setTtiTestOutput(false)", settings_html)
        self.assertIn("/settings/tti/test-output", settings_html)

    def test_archive_csv_uses_transfer_function_frequency_summary(self):
        archive_html = (
            Path(__file__).resolve().parents[1] / "static" / "archive.html"
        ).read_text(encoding="utf-8")

        self.assertIn("await this.exportTransferFunctionCSV()", archive_html)
        self.assertIn("['Atom_Number_Total_Fit_Mean', 'atom_number_total_fit_mean']", archive_html)
        self.assertIn("['Interferometer_P1_Fit_Std', 'intf_p1_fit_std']", archive_html)
        self.assertIn("['Interferometer_Phase_Mean_Rad', 'interferometer_phase_mean']", archive_html)
        self.assertIn("['S2_0deg', 'interferometer_phase_0deg_s2']", archive_html)
        self.assertIn("['S2_90deg', 'interferometer_phase_90deg_s2']", archive_html)
        self.assertIn("['S2_Quadrature_Sum', 'interferometer_phase_s2']", archive_html)
        self.assertIn("['Delta_Phi2_0deg_Rad2', 'interferometer_phase_0deg_phase2_rad2']", archive_html)
        self.assertIn("['Delta_Phi2_90deg_Rad2', 'interferometer_phase_90deg_phase2_rad2']", archive_html)
        self.assertIn("['Delta_Phi2_Quadrature_Sum_Rad2', 'interferometer_phase_phase2_rad2']", archive_html)
        self.assertIn("['Delta_Phi2_Allan_Noise_Floor_Rad2', 'interferometer_phase_noise_phase2_rad2']", archive_html)
        self.assertNotIn("S2_Phase_Noise_Floor", archive_html)
        self.assertIn("v-model.number=\"analysis.transfer_frequency_modulation_mhz\"", archive_html)
        self.assertIn("v-model.number=\"analysis.transfer_atom_mirror_distance_m\"", archive_html)
        self.assertIn("['Atom_Mirror_Distance_M', 'atom_mirror_distance_m']", archive_html)

        settings_html = (
            Path(__file__).resolve().parents[1] / "static" / "settings.html"
        ).read_text(encoding="utf-8")
        self.assertIn("v-model.number=\"s.transfer_atom_mirror_distance_m\"", settings_html)
        self.assertIn("v-model.number=\"s.transfer_phase_noise_sigma_mrad\"", settings_html)

    def test_summary_calculates_s2_from_780_nm_frequency_modulation(self):
        rows = [
            {
                "transfer_frequency_hz": 100.0,
                "interferometer_phase": phase,
                "interferometer_phase_valid": True,
            }
            for phase in (0.05, 0.15)
        ]
        phase_amplitude = bragg_phase_modulation_rad(1.0)
        summary = build_transfer_function_summary(rows, frequency_modulation_mhz=1.0)

        self.assertIsNotNone(phase_amplitude)
        self.assertAlmostEqual(summary[0]["bragg_phase_modulation_rad"], phase_amplitude)
        self.assertAlmostEqual(summary[0]["interferometer_phase_s2"], (0.1 / phase_amplitude) ** 2)
        self.assertEqual(summary[0]["frequency_modulation_mhz"], 1.0)
        self.assertEqual(summary[0]["atom_mirror_distance_m"], 2.23)

    def test_phase_normalization_uses_user_defined_atom_mirror_distance(self):
        default_phase = bragg_phase_modulation_rad(1.0, 2.23)
        doubled_phase = bragg_phase_modulation_rad(1.0, 4.46)
        summary = build_transfer_function_summary([
            {
                "transfer_frequency_hz": 100.0,
                "interferometer_phase": 0.1,
                "interferometer_phase_valid": True,
            }
        ], 1.0, None, 4.46)

        self.assertAlmostEqual(doubled_phase, 2 * default_phase)
        self.assertEqual(summary[0]["atom_mirror_distance_m"], 4.46)
        self.assertAlmostEqual(summary[0]["interferometer_phase_s2"], (0.1 / doubled_phase) ** 2)

    def test_archive_recalculation_rebuilds_summary_with_normalization_overrides(self):
        loader = DataLoader()
        recalculated_point = {
            "step": 0,
            "parameter": 100.0,
            "all_parameters": [100.0],
            "transfer_frequency_hz": 100.0,
            "transfer_phase_deg": 0.0,
            "interferometer_phase": 0.2,
            "interferometer_phase_valid": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "config.json").write_text(json.dumps({
                "mode": "transfer_function",
                "scan_dimensions": 1,
                "transfer_phase_degrees": [0.0],
            }), encoding="utf-8")
            with patch.object(loader, "_get_run_dir", return_value=run_dir), \
                    patch.object(loader, "_read_results_csv", return_value=[{"step": 0}]), \
                    patch.object(loader, "_load_waveform_arrays", return_value={}), \
                    patch.object(loader, "_recalculate_point", return_value=recalculated_point):
                payload = loader.recalculate_run("2026", "09", "08", "run01", {
                    "transfer_frequency_modulation_mhz": 2.0,
                    "transfer_atom_mirror_distance_m": 3.0,
                    "transfer_phase_noise_sigma_mrad": 250.0,
                })

        row = payload["transfer_function_summary"][0]
        expected_phi = bragg_phase_modulation_rad(2.0, 3.0)
        self.assertEqual(row["frequency_modulation_mhz"], 2.0)
        self.assertEqual(row["atom_mirror_distance_m"], 3.0)
        self.assertAlmostEqual(row["interferometer_phase_0deg_s2"], (0.2 / expected_phi) ** 2)
        self.assertEqual(row["transfer_phase_noise_sigma_mrad"], 250.0)
        self.assertAlmostEqual(row["interferometer_phase_noise_phase2_rad2"], 0.25 ** 2)
        self.assertAlmostEqual(row["interferometer_phase_noise_s2"], (0.25 / expected_phi) ** 2)

    def test_s2_is_unavailable_without_archived_modulation_amplitude(self):
        summary = build_transfer_function_summary([
            {
                "transfer_frequency_hz": 100.0,
                "interferometer_phase": 0.1,
                "interferometer_phase_valid": True,
            }
        ])
        self.assertIsNone(summary[0]["interferometer_phase_s2"])

    def test_summary_contains_two_phase_components_and_quadrature_sum(self):
        rows = [
            {
                "transfer_frequency_hz": 100.0,
                "transfer_phase_deg": phase_deg,
                "interferometer_phase": value,
                "interferometer_phase_valid": True,
            }
            for phase_deg, value in [(0.0, 0.1), (0.0, 0.2), (90.0, 0.3), (90.0, 0.5)]
        ]
        phase_amplitude = bragg_phase_modulation_rad(1.0)
        summary = build_transfer_function_summary(rows, 1.0, [0.0, 90.0])
        components = summary[0]["interferometer_phase_s2_components"]

        self.assertEqual([component["phase_deg"] for component in components], [0.0, 90.0])
        self.assertAlmostEqual(components[0]["s2"], (0.15 / phase_amplitude) ** 2)
        self.assertAlmostEqual(components[1]["s2"], (0.4 / phase_amplitude) ** 2)
        self.assertAlmostEqual(summary[0]["interferometer_phase_s2"], components[0]["s2"] + components[1]["s2"])
        self.assertAlmostEqual(summary[0]["interferometer_phase_0deg_s2"], components[0]["s2"])
        self.assertAlmostEqual(summary[0]["interferometer_phase_90deg_s2"], components[1]["s2"])
        self.assertAlmostEqual(components[0]["phase2_rad2"], 0.15 ** 2)
        self.assertAlmostEqual(components[1]["phase2_rad2"], 0.4 ** 2)
        self.assertAlmostEqual(summary[0]["interferometer_phase_0deg_phase2_rad2"], 0.15 ** 2)
        self.assertAlmostEqual(summary[0]["interferometer_phase_90deg_phase2_rad2"], 0.4 ** 2)
        self.assertAlmostEqual(summary[0]["interferometer_phase_phase2_rad2"], 0.15 ** 2 + 0.4 ** 2)
        self.assertAlmostEqual(summary[0]["interferometer_phase_noise_phase2_rad2"], 2 * 0.1 ** 2)
        self.assertAlmostEqual(summary[0]["interferometer_phase_noise_s2"], 2 * (0.1 / phase_amplitude) ** 2)
        self.assertEqual(summary[0]["transfer_phase_noise_sigma_mrad"], 100.0)

    def test_summary_contains_per_phase_statistics_for_each_plotted_metric(self):
        rows = [
            {
                "transfer_frequency_hz": 100.0,
                "transfer_phase_deg": phase_deg,
                "atom_number_up": value,
                "atom_number_dw": value * 2,
                "intf_p1": value / 10,
                "intf_p2": value / 20,
                "interferometer_phase": value / 100,
                "interferometer_phase_valid": True,
            }
            for phase_deg, value in ((0.0, 10.0), (0.0, 12.0), (90.0, 20.0), (90.0, 24.0))
        ]
        summary = build_transfer_function_summary(rows, 1.0, [0.0, 90.0])[0]

        self.assertEqual(summary["atom_number_up_fit_0deg_count"], 2)
        self.assertAlmostEqual(summary["atom_number_up_fit_0deg_mean"], 11.0)
        self.assertAlmostEqual(summary["atom_number_up_fit_0deg_std"], 2 ** 0.5)
        self.assertEqual(summary["atom_number_up_fit_90deg_count"], 2)
        self.assertAlmostEqual(summary["atom_number_up_fit_90deg_mean"], 22.0)
        self.assertAlmostEqual(summary["atom_number_up_fit_90deg_std"], 2 * (2 ** 0.5))
        self.assertAlmostEqual(summary["intf_p1_fit_0deg_mean"], 1.1)
        self.assertAlmostEqual(summary["intf_p1_fit_90deg_mean"], 2.2)
        self.assertAlmostEqual(summary["interferometer_phase_0deg_mean_rad"], 0.11)
        self.assertAlmostEqual(summary["interferometer_phase_90deg_mean_rad"], 0.22)

    def test_summary_infers_phases_from_archived_points_when_config_is_missing(self):
        summary = build_transfer_function_summary([
            {
                "transfer_frequency_hz": 100.0,
                "transfer_phase_deg": phase_deg,
                "atom_number_up": value,
                "interferometer_phase": value / 100,
                "interferometer_phase_valid": True,
            }
            for phase_deg, value in ((0.0, 10.0), (0.0, 12.0), (90.0, 20.0), (90.0, 24.0))
        ], 1.0)

        self.assertEqual(
            [component["phase_deg"] for component in summary[0]["interferometer_phase_s2_components"]],
            [0.0, 90.0],
        )
        self.assertAlmostEqual(summary[0]["atom_number_up_fit_0deg_mean"], 11.0)
        self.assertAlmostEqual(summary[0]["atom_number_up_fit_90deg_mean"], 22.0)

    def test_single_phase_summary_does_not_report_quadrature_sum(self):
        summary = build_transfer_function_summary([
            {
                "transfer_frequency_hz": 100.0,
                "transfer_phase_deg": 0.0,
                "interferometer_phase": 0.1,
                "interferometer_phase_valid": True,
            }
        ], 1.0, [0.0])
        self.assertIsNotNone(summary[0]["interferometer_phase_0deg_s2"])
        self.assertIsNone(summary[0]["interferometer_phase_s2"])
        self.assertAlmostEqual(summary[0]["interferometer_phase_0deg_phase2_rad2"], 0.1 ** 2)
        self.assertIsNone(summary[0]["interferometer_phase_phase2_rad2"])
        self.assertAlmostEqual(summary[0]["interferometer_phase_noise_phase2_rad2"], 0.1 ** 2)
        self.assertAlmostEqual(summary[0]["interferometer_phase_noise_s2"], (0.1 / bragg_phase_modulation_rad(1.0)) ** 2)

    def test_transfer_csv_exports_phase_and_flat_s2_columns(self):
        self.assertIn("TTI_Phase_Deg", RESULTS_CSV_HEADER)
        self.assertIn("Transfer_Zero_Phase_Baseline", RESULTS_CSV_HEADER)
        self.assertIn("Interferometer_Phase_Raw_Rad", RESULTS_CSV_HEADER)
        summary = build_transfer_function_summary([
            {
                "transfer_frequency_hz": 100.0,
                "transfer_phase_deg": phase,
                "interferometer_phase": 0.1,
                "interferometer_phase_valid": True,
            }
            for phase in (0.0, 90.0)
        ], 1.0, [0.0, 90.0])
        with tempfile.TemporaryDirectory() as directory:
            manager = DataManager()
            manager.current_run_dir = Path(directory)
            manager.save_transfer_function_summary(summary)
            with open(Path(directory) / "transfer_function_summary.csv", newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        self.assertIn("interferometer_phase_0deg_s2", row)
        self.assertIn("interferometer_phase_90deg_s2", row)
        self.assertIn("interferometer_phase_s2", row)
        self.assertIn("interferometer_phase_0deg_phase2_rad2", row)
        self.assertIn("interferometer_phase_90deg_phase2_rad2", row)
        self.assertIn("interferometer_phase_phase2_rad2", row)
        self.assertIn("interferometer_phase_noise_phase2_rad2", row)
        self.assertIn("interferometer_phase_noise_s2", row)
        self.assertNotIn("interferometer_phase_s2_components", row)

    def test_archive_overwrite_persists_transfer_normalization_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "2026" / "09" / "08" / "run01"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(json.dumps({
                "mode": "transfer_function",
                "transfer_frequency_modulation_mhz": 1.0,
            }), encoding="utf-8")
            (run_dir / "results.csv").write_text(",".join(RESULTS_CSV_HEADER) + "\n", encoding="utf-8")
            summary = [{
                "frequency_hz": 100.0,
                "frequency_modulation_mhz": 2.0,
                "atom_mirror_distance_m": 3.0,
                "interferometer_phase_s2_components": [],
                "interferometer_phase_s2": 4.0,
            }]
            with patch("app.core.data_manager.config.DATA_BASE_DIR", directory):
                DataManager().overwrite_run(
                    "2026", "09", "08", "run01",
                    {
                        "transfer_frequency_modulation_mhz": 2.0,
                        "transfer_atom_mirror_distance_m": 3.0,
                        "transfer_phase_noise_sigma_mrad": 80.0,
                    },
                    [],
                    summary,
                )

            saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            saved_summary = json.loads((run_dir / "transfer_function_summary.json").read_text(encoding="utf-8"))
            with open(run_dir / "transfer_function_summary.csv", newline="", encoding="utf-8") as handle:
                csv_row = next(csv.DictReader(handle))
        self.assertEqual(saved_config["transfer_frequency_modulation_mhz"], 2.0)
        self.assertEqual(saved_config["transfer_atom_mirror_distance_m"], 3.0)
        self.assertEqual(saved_config["transfer_phase_noise_sigma_mrad"], 80.0)
        self.assertEqual(saved_summary[0]["interferometer_phase_s2"], 4.0)
        self.assertEqual(csv_row["atom_mirror_distance_m"], "3.0")
        self.assertNotIn("interferometer_phase_s2_components", csv_row)

    def test_summary_excludes_invalid_calibrated_phase(self):
        rows = [
            {"transfer_frequency_hz": 100.0, "interferometer_phase": 0.1, "interferometer_phase_valid": True},
            {"transfer_frequency_hz": 100.0, "interferometer_phase": 99.0, "interferometer_phase_valid": False},
            {"transfer_frequency_hz": 100.0, "interferometer_phase": 0.3, "interferometer_phase_valid": True},
        ]
        summary = build_transfer_function_summary(rows)
        self.assertEqual(summary[0]["interferometer_phase_count"], 2)
        self.assertAlmostEqual(summary[0]["interferometer_phase_std"], 2 ** 0.5 / 10)

    def test_archive_filters_baselines_and_can_rebase_from_another_block(self):
        loader = DataLoader()
        points = [
            {"transfer_zero_phase_baseline": True, "transfer_zero_phase_block_id": 0,
             "interferometer_phase_raw": 1.0, "interferometer_phase": 1.0},
            {"transfer_zero_phase_baseline": True, "transfer_zero_phase_block_id": 0,
             "interferometer_phase_raw": 1.2, "interferometer_phase": 1.2},
            {"transfer_zero_phase_baseline": True, "transfer_zero_phase_block_id": 1,
             "interferometer_phase_raw": 2.0, "interferometer_phase": 2.0},
            {"transfer_frequency_hz": 100.0, "transfer_phase_deg": 0.0,
             "transfer_zero_phase_block_id": 0, "interferometer_phase_raw": 1.5,
             "interferometer_phase": 0.4, "interferometer_phase_valid": True},
        ]
        self.assertEqual(len(loader._transfer_response_points(points)), 1)
        recorded = loader._apply_transfer_zero_phase_reference(points, {"transfer_zero_phase_reference_mode": "recorded"})
        self.assertAlmostEqual(recorded[-1]["interferometer_phase"], 0.4)
        alternate = loader._apply_transfer_zero_phase_reference(points, {
            "transfer_zero_phase_reference_mode": "block",
            "transfer_zero_phase_reference_block_id": 1,
        })
        self.assertAlmostEqual(alternate[-1]["interferometer_phase"], -0.5)


if __name__ == "__main__":
    unittest.main()
