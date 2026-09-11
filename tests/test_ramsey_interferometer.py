import queue
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.core.experiment_manager import ExperimentManager
from app.drivers.rigol_generator import (
    RigolConnectionSettings,
    RigolGeneratorClient,
    RigolGeneratorError,
    interpolate_power_dbm,
)
from app.models.schemas import ScanConfig, SystemSettings


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


class RigolGeneratorTests(unittest.TestCase):
    def test_connect_and_set_both_frequencies_with_readback(self):
        fake = FakeSocket([
            "RIGOL TECHNOLOGIES,DG4162,DG4E000000000,00.01",
            "109000000",
            "111000000",
        ])
        with patch("app.drivers.rigol_generator.socket.create_connection", return_value=fake) as connect:
            with RigolGeneratorClient(RigolConnectionSettings("192.168.1.40")) as client:
                client.set_frequency_pair(109_000_000, 111_000_000)
        connect.assert_called_once_with(("192.168.1.40", 5555), timeout=3.0)
        self.assertEqual(fake.sent, [
            "*IDN?\n",
            ":SOURce1:FREQuency:FIXed 109000000\n",
            ":SOURce1:FREQuency:FIXed?\n",
            ":SOURce2:FREQuency:FIXed 111000000\n",
            ":SOURce2:FREQuency:FIXed?\n",
        ])

    def test_wrong_instrument_is_rejected(self):
        fake = FakeSocket(["RIGOL TECHNOLOGIES,DG1022,123,1.0"])
        with patch("app.drivers.rigol_generator.socket.create_connection", return_value=fake):
            with self.assertRaisesRegex(RigolGeneratorError, "Expected RIGOL DG4162"):
                RigolGeneratorClient(RigolConnectionSettings("192.168.1.40")).connect()

    def test_manual_output_control_is_channel_specific(self):
        fake = FakeSocket([
            "RIGOL TECHNOLOGIES,DG4162,DG4E000000000,00.01",
            "ON",
        ])
        with patch("app.drivers.rigol_generator.socket.create_connection", return_value=fake):
            with RigolGeneratorClient(RigolConnectionSettings("192.168.1.40")) as client:
                client.set_output(2, True)
        self.assertEqual(fake.sent, [
            "*IDN?\n",
            ":OUTPut2:STATe ON\n",
            ":OUTPut2:STATe?\n",
        ])

    def test_power_lookup_interpolates_and_clamps_endpoints(self):
        table = [
            {"frequency_mhz": 100, "power_dbm": -4},
            {"frequency_mhz": 110, "power_dbm": -6},
        ]
        self.assertEqual(interpolate_power_dbm(95, table), -4)
        self.assertEqual(interpolate_power_dbm(115, table), -6)
        self.assertAlmostEqual(interpolate_power_dbm(105, table), -5)

    def test_frequency_and_dbm_power_are_set_and_verified(self):
        fake = FakeSocket([
            "RIGOL TECHNOLOGIES,DG4162,DG4E000000000,00.01",
            "110000000", "SIN", "DBM", "-5.11",
        ])
        with patch("app.drivers.rigol_generator.socket.create_connection", return_value=fake):
            with RigolGeneratorClient(RigolConnectionSettings("192.168.1.40")) as client:
                client.set_frequency_and_power(1, 110_000_000, -5.11)
        self.assertIn(":SOURce1:FUNCtion SINusoid\n", fake.sent)
        self.assertIn(":SOURce1:VOLTage:UNIT DBM\n", fake.sent)
        self.assertIn(":SOURce1:VOLTage -5.11\n", fake.sent)


class RamseyPlanTests(unittest.TestCase):
    def setUp(self):
        self.manager = ExperimentManager.__new__(ExperimentManager)
        self.manager.settings = {"ramsey_center_frequency_mhz": 110.0}

    def test_plan_scans_delta_f_and_repeats_fixed_sequence(self):
        config = {
            "mode": "ramsey_interferometer",
            "scan_dimensions": 1,
            "parameter_source": "classic",
            "randomize": False,
            "ramsey_delta_start_mhz": 0.0,
            "ramsey_delta_stop_mhz": 0.2,
            "ramsey_delta_step_mhz": 0.1,
            "ramsey_repeats": 2,
            "ramsey_settling_time_s": 3.5,
        }
        plan = self.manager._build_ramsey_interferometer_execution(config)
        self.assertEqual(len(plan), 6)
        self.assertTrue(all(point["sequence_parameters"] == [] for point in plan))
        self.assertEqual(
            [point["metadata"]["ramsey_delta_f_mhz"] for point in plan],
            [0.0, 0.0, 0.1, 0.1, 0.2, 0.2],
        )
        self.assertEqual(plan[-1]["metadata"]["ramsey_ch1_frequency_mhz"], 109.8)
        self.assertEqual(plan[-1]["metadata"]["ramsey_ch2_frequency_mhz"], 110.2)
        self.assertAlmostEqual(plan[-1]["metadata"]["ramsey_ch1_power_dbm"], -5.122)
        self.assertAlmostEqual(plan[-1]["metadata"]["ramsey_ch2_power_dbm"], -5.082)
        self.assertEqual(config["ramsey_settling_time_s"], 3.5)
        self.assertEqual(config["averages"], 1)

    def test_negative_delta_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.manager._build_ramsey_interferometer_execution({
                "scan_dimensions": 1,
                "parameter_source": "classic",
                "randomize": False,
                "ramsey_delta_start_mhz": -0.1,
                "ramsey_delta_stop_mhz": 0.1,
                "ramsey_delta_step_mhz": 0.1,
            })

    def test_pair_outside_generator_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "0 to 160 MHz"):
            self.manager._build_ramsey_interferometer_execution({
                "scan_dimensions": 1,
                "parameter_source": "classic",
                "randomize": False,
                "ramsey_delta_start_mhz": 51,
                "ramsey_delta_stop_mhz": 51,
                "ramsey_delta_step_mhz": 1,
            })

    def test_acquisition_sets_pair_once_for_all_repeats(self):
        manager = ExperimentManager.__new__(ExperimentManager)
        manager.settings = {"rigol_host": "192.168.1.40", "rigol_port": 5555, "rigol_timeout_s": 3}
        manager.stop_flag = False
        manager.status = SimpleNamespace(message="")
        manager.data_queue = queue.Queue()
        manager._scan_finalize_error = None
        manager._restore_ac_stark_dds = lambda _context: None
        manager.execute_single_measurement = Mock(return_value={"shot_duration_sec": 0.1})
        client = Mock()
        client.connect.return_value = "RIGOL TECHNOLOGIES,DG4162,123,1.0"
        parameters = [{
            "sequence_parameters": [],
            "metadata": {
                "ramsey_delta_f_mhz": 0.1,
                "ramsey_ch1_frequency_mhz": 109.9,
                "ramsey_ch2_frequency_mhz": 110.1,
                "ramsey_repeat": repeat,
            },
        } for repeat in (1, 2)]
        config = {"mode": "ramsey_interferometer", "scan_dimensions": 1, "ramsey_settling_time_s": 0}
        with patch("app.core.experiment_manager.config.USE_SIMULATION", False), patch(
            "app.core.experiment_manager.RigolGeneratorClient", return_value=client
        ):
            manager._acquisition_loop(parameters, config)
        client.set_frequency_power_pair.assert_called_once()
        args = client.set_frequency_power_pair.call_args.args
        self.assertEqual(args[0], 109_900_000.0)
        self.assertAlmostEqual(args[1], -5.116)
        self.assertEqual(args[2], 110_100_000.0)
        self.assertAlmostEqual(args[3], -5.096)
        client.close.assert_called_once_with()
        self.assertEqual(manager.execute_single_measurement.call_count, 2)


class RamseyFrontendContractTests(unittest.TestCase):
    def test_settings_and_scan_controls_are_present(self):
        root = Path(__file__).resolve().parents[1]
        settings_html = (root / "static" / "settings.html").read_text(encoding="utf-8")
        index_html = (root / "static" / "index.html").read_text(encoding="utf-8")
        archive_html = (root / "static" / "archive.html").read_text(encoding="utf-8")
        self.assertIn("tab==='ramsey'", settings_html)
        self.assertIn("/settings/rigol/test-pair", settings_html)
        self.assertIn('value="ramsey_interferometer"', index_html)
        self.assertIn("'Δf (MHz)'", index_html)
        self.assertIn("isRamseyInterferometerArchive", archive_html)

    def test_schema_defaults_match_requested_behavior(self):
        config = ScanConfig()
        self.assertEqual(config.ramsey_settling_time_s, 5.0)
        self.assertEqual(config.ramsey_delta_start_mhz, 0.0)
        self.assertEqual(SystemSettings.model_fields["rigol_port"].default, 5555)
        self.assertEqual(len(SystemSettings.model_fields["ramsey_frequency_power_table"].default_factory()), 20)
