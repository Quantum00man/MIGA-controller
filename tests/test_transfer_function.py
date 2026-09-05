import unittest
from pathlib import Path
from unittest.mock import patch

from app.analysis.transfer_function import bragg_phase_modulation_rad, build_transfer_function_summary
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


class TransferFunctionPlanTests(unittest.TestCase):
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
        self.assertEqual(config["transfer_generator_model"], "TG5012A")
        self.assertEqual(config["transfer_generator_channel"], 1)
        self.assertEqual(config["transfer_frequency_modulation_mhz"], 1.0)
        self.assertTrue(all(
            point["metadata"]["transfer_frequency_modulation_mhz"] == 1.0
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
        self.assertAlmostEqual(summary[0]["atom_number_up_fit_mean"], 2.0)
        self.assertAlmostEqual(summary[0]["atom_number_total_fit_mean"], 10.0)
        self.assertAlmostEqual(summary[0]["interferometer_phase_mean"], 0.2)
        self.assertEqual(summary[0]["interferometer_phase_count"], 3)

    def test_frontends_can_switch_between_std_mean_and_s2(self):
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
        self.assertIn("setTransferFunctionYAxisScale('log')", archive_html)
        self.assertIn("`interferometer_phase_${statistic}`", archive_html)
        self.assertIn("transferFunctionStatistic: 'std'", index_html)
        self.assertIn("transferFunctionYAxisScale: 'linear'", index_html)
        self.assertIn("setTransferFunctionStatistic('s2')", index_html)
        self.assertIn("setTransferFunctionYAxisScale('log')", index_html)
        self.assertIn("transferBraggPhaseAmplitudeRad()", index_html)

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

    def test_s2_is_unavailable_without_archived_modulation_amplitude(self):
        summary = build_transfer_function_summary([
            {
                "transfer_frequency_hz": 100.0,
                "interferometer_phase": 0.1,
                "interferometer_phase_valid": True,
            }
        ])
        self.assertIsNone(summary[0]["interferometer_phase_s2"])

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
