import unittest
from unittest.mock import patch

from app.analysis.transfer_function import build_transfer_function_summary
from app.core.experiment_manager import ExperimentManager
from app.drivers.tti_generator import (
    set_tti_test_frequency,
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


class TransferFunctionPlanTests(unittest.TestCase):
    def setUp(self):
        self.manager = ExperimentManager.__new__(ExperimentManager)

    def test_plan_runs_fixed_sequence_repeated_at_each_frequency(self):
        config = {
            "scan_dimensions": 1,
            "parameter_source": "classic",
            "mode": "transfer_function",
            "randomize": False,
            "transfer_frequency_start_hz": 100,
            "transfer_frequency_stop_hz": 300,
            "transfer_frequency_step_hz": 100,
            "transfer_repeats": 3,
            "transfer_settling_time_s": 0,
        }
        plan = self.manager._build_transfer_function_execution(config)
        self.assertEqual(len(plan), 9)
        self.assertTrue(all(point["sequence_parameters"] == [] for point in plan))
        self.assertEqual(
            [point["metadata"]["transfer_frequency_hz"] for point in plan],
            [100.0] * 3 + [200.0] * 3 + [300.0] * 3,
        )
        self.assertEqual(config["transfer_frequency_values_hz"], [100.0, 200.0, 300.0])
        self.assertEqual(config["averages"], 1)
        self.assertEqual(config["transfer_settling_time_s"], 5.0)

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
            [3.0, 3.0, 2.0, 2.0, 1.0, 1.0],
        )


class TransferFunctionStatisticsTests(unittest.TestCase):
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
        self.assertEqual(summary[0]["interferometer_phase_count"], 3)

    def test_summary_excludes_invalid_calibrated_phase(self):
        rows = [
            {"transfer_frequency_hz": 100.0, "interferometer_phase": 0.1, "interferometer_phase_valid": True},
            {"transfer_frequency_hz": 100.0, "interferometer_phase": 99.0, "interferometer_phase_valid": False},
            {"transfer_frequency_hz": 100.0, "interferometer_phase": 0.3, "interferometer_phase_valid": True},
        ]
        summary = build_transfer_function_summary(rows)
        self.assertEqual(summary[0]["interferometer_phase_count"], 2)
        self.assertAlmostEqual(summary[0]["interferometer_phase_std"], 2 ** 0.5 / 10)


if __name__ == "__main__":
    unittest.main()
