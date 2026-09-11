import queue
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.core.experiment_manager import ExperimentManager
from app.drivers.rigol_generator import (
    RigolConnectionSettings,
    RigolGeneratorClient,
    RigolGeneratorError,
)
from app.models.schemas import ScanConfig


class FakeVisaInstrument:
    def __init__(self, identity="RIGOL TECHNOLOGIES,DG4162,DG4E000000000,00.01"):
        self.identity = identity
        self.timeout = None
        self.read_termination = None
        self.write_termination = None
        self.writes = []
        self.frequencies = {1: 110_000_000.0, 2: 110_000_000.0}
        self.outputs = {1: False, 2: False}

    def write(self, command):
        self.writes.append(command)
        if command.startswith(":SOURce"):
            channel = int(command[len(":SOURce")])
            self.frequencies[channel] = float(command.split()[-1])
        elif command.startswith(":OUTPut"):
            channel = int(command[len(":OUTPut")])
            self.outputs[channel] = command.endswith("ON")

    def query(self, command):
        if command == "*IDN?":
            return self.identity
        if command.startswith(":SOURce"):
            return str(self.frequencies[int(command[len(":SOURce")])])
        if command.startswith(":OUTPut"):
            return "ON" if self.outputs[int(command[len(":OUTPut")])] else "OFF"
        raise AssertionError(command)

    def close(self):
        pass


class FakeVisaResourceManager:
    def __init__(self, instrument):
        self.instrument = instrument
        self.opened_resource = None

    def open_resource(self, resource):
        self.opened_resource = resource
        return self.instrument

    def close(self):
        pass


class RigolGeneratorTests(unittest.TestCase):
    def test_connect_and_set_both_frequencies_with_readback(self):
        instrument = FakeVisaInstrument()
        manager = FakeVisaResourceManager(instrument)
        fake_pyvisa = SimpleNamespace(ResourceManager=lambda backend: manager)
        with patch.dict(sys.modules, {"pyvisa": fake_pyvisa}):
            with RigolGeneratorClient(RigolConnectionSettings("192.168.1.40")) as client:
                client.set_frequency_pair(109_000_000, 111_000_000)
        self.assertEqual(manager.opened_resource, "TCPIP0::192.168.1.40::INSTR")
        self.assertEqual(instrument.writes, [
            ":SOURce1:FREQuency:FIXed 109000000",
            ":SOURce2:FREQuency:FIXed 111000000",
        ])

    def test_wrong_instrument_is_rejected(self):
        instrument = FakeVisaInstrument("RIGOL TECHNOLOGIES,DG1022,123,1.0")
        manager = FakeVisaResourceManager(instrument)
        with patch.dict(sys.modules, {"pyvisa": SimpleNamespace(ResourceManager=lambda backend: manager)}):
            with self.assertRaisesRegex(RigolGeneratorError, "Expected RIGOL DG4162"):
                RigolGeneratorClient(RigolConnectionSettings("192.168.1.40")).connect()

    def test_manual_output_control_is_channel_specific(self):
        instrument = FakeVisaInstrument()
        manager = FakeVisaResourceManager(instrument)
        with patch.dict(sys.modules, {"pyvisa": SimpleNamespace(ResourceManager=lambda backend: manager)}):
            with RigolGeneratorClient(RigolConnectionSettings("192.168.1.40")) as client:
                client.set_output(2, True)
        self.assertEqual(instrument.writes, [":OUTPut2:STATe ON"])


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
        manager.settings = {"rigol_host": "192.168.1.40", "rigol_timeout_s": 3, "rigol_visa_resource": ""}
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
        client.set_frequency_pair.assert_called_once_with(109_900_000.0, 110_100_000.0)
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
