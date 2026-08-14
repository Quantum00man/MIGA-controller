from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from lxml import etree

import config
from app.drivers.dds_table import (
    DdsCommandError,
    DdsTableError,
    amplitude_from_power,
    build_ac_stark_table,
    generate_ratio_values,
    power_from_amplitude,
    validate_dds_table,
    write_and_verify_dds_table,
)


CALIBRATION = dict(config.DEFAULT_RAMAN_POWER_CALIBRATION)


def write_table(path: Path, element_numbers):
    root = etree.Element("ad9958")
    for number in element_numbers:
        element = etree.SubElement(root, "elem", n=str(number))
        for channel_name in ("ch0", "ch1"):
            channel = etree.SubElement(element, channel_name)
            etree.SubElement(channel, "mode").text = "sf"
            etree.SubElement(channel, "fr").text = "80000000"
            etree.SubElement(channel, "am").text = "100"
    etree.ElementTree(root).write(
        str(path),
        xml_declaration=True,
        encoding="ISO-8859-1",
        pretty_print=True,
    )


class DdsRatioAxisTests(unittest.TestCase):
    def test_ratio_axis_is_inclusive_and_decimal_stable(self):
        self.assertEqual(generate_ratio_values(0.5, 0.7, 0.1), [0.5, 0.6, 0.7])
        self.assertEqual(generate_ratio_values(0.7, 0.5, 0.1), [0.7, 0.6, 0.5])

    def test_ratio_axis_rejects_nonpositive_values(self):
        with self.assertRaises(DdsTableError):
            generate_ratio_values(0, 1, 0.1)
        with self.assertRaises(DdsTableError):
            generate_ratio_values(0.5, 1, 0)


class RamanCalibrationTests(unittest.TestCase):
    def test_curve_inverse_returns_safe_integer_amplitude(self):
        amplitude = amplitude_from_power(50.0, CALIBRATION)
        self.assertIsInstance(amplitude, int)
        self.assertGreaterEqual(amplitude, 0)
        self.assertLessEqual(amplitude, 1023)
        self.assertAlmostEqual(power_from_amplitude(amplitude, CALIBRATION), 50.0, delta=0.5)

    def test_curve_rejects_unreachable_power(self):
        with self.assertRaises(DdsTableError):
            amplitude_from_power(CALIBRATION["upper_asymptote"], CALIBRATION)

    def test_curve_evaluation_is_stable_for_large_exponents(self):
        steep_curve = {**CALIBRATION, "growth_rate": 1_000_000.0, "midpoint": 500.0}
        low_power = power_from_amplitude(0, steep_curve)
        high_power = power_from_amplitude(1023, steep_curve)
        self.assertEqual(low_power, steep_curve["lower_asymptote"])
        self.assertAlmostEqual(high_power, steep_curve["upper_asymptote"])


class DdsTableGenerationTests(unittest.TestCase):
    def test_appends_fixed_elements_after_largest_existing_n(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.xml"
            output = Path(temp_dir) / "generated.xml"
            write_table(source, [0, 3, 10])

            plans = build_ac_stark_table(
                source,
                output,
                [0.5, 1.0, 2.0],
                100.0,
                CALIBRATION,
                CALIBRATION,
            )

            self.assertEqual([plan.element for plan in plans], [11, 12, 13])
            self.assertEqual(validate_dds_table(source), {"element_count": 3, "max_element": 10})
            self.assertEqual(validate_dds_table(output), {"element_count": 6, "max_element": 13})

            tree = etree.parse(str(output))
            generated = tree.findall("./elem")[-3:]
            for element, plan in zip(generated, plans):
                self.assertEqual(element.findtext("./ch0/mode"), "sf")
                self.assertEqual(element.findtext("./ch1/mode"), "sf")
                self.assertEqual(element.findtext("./ch0/fr"), "80000000")
                self.assertEqual(element.findtext("./ch1/fr"), "80000000")
                self.assertEqual(int(element.findtext("./ch0/am")), plan.amplitude_r1)
                self.assertEqual(int(element.findtext("./ch1/am")), plan.amplitude_r2)
                self.assertLessEqual(plan.amplitude_r1, 1023)
                self.assertLessEqual(plan.amplitude_r2, 1023)
                self.assertAlmostEqual(
                    plan.requested_power_r1 + plan.requested_power_r2,
                    100.0,
                    places=10,
                )

    def test_rejects_generation_past_element_500(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.xml"
            output = Path(temp_dir) / "generated.xml"
            write_table(source, [500])
            with self.assertRaises(DdsTableError):
                build_ac_stark_table(
                    source,
                    output,
                    [1.0],
                    100.0,
                    CALIBRATION,
                    CALIBRATION,
                )


class DdsWriterTests(unittest.TestCase):
    @mock.patch("app.drivers.dds_table.subprocess.run")
    def test_write_then_verify_uses_exact_cli_order(self, run_mock):
        run_mock.return_value = SimpleNamespace(returncode=0, stdout="Verify Ok", stderr="")
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = Path(temp_dir) / "writetable.py"
            table = Path(temp_dir) / "table.xml"
            writer.touch()
            write_table(table, [0])

            write_and_verify_dds_table(writer, table)

            self.assertEqual(run_mock.call_count, 2)
            first_command = run_mock.call_args_list[0].args[0]
            second_command = run_mock.call_args_list[1].args[0]
            self.assertEqual(first_command, ["python3", str(writer.resolve()), "-w", str(table.resolve())])
            self.assertEqual(second_command, ["python3", str(writer.resolve()), "-v", str(table.resolve())])
            self.assertEqual(run_mock.call_args_list[0].kwargs["cwd"], str(writer.parent.resolve()))

    @mock.patch("app.drivers.dds_table.subprocess.run")
    def test_writer_nonzero_exit_is_reported(self, run_mock):
        run_mock.return_value = SimpleNamespace(returncode=1, stdout="", stderr="Device not found")
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = Path(temp_dir) / "writetable.py"
            table = Path(temp_dir) / "table.xml"
            writer.touch()
            write_table(table, [0])
            with self.assertRaises(DdsCommandError):
                write_and_verify_dds_table(writer, table)


if __name__ == "__main__":
    unittest.main()
