from pathlib import Path
import tempfile
import unittest
from unittest import mock

from lxml import etree

import config
from app.core.experiment_manager import ExperimentManager
from app.core.structures import ScanResult


CALIBRATION = dict(config.DEFAULT_RAMAN_POWER_CALIBRATION)


def write_dds_table(path: Path):
    root = etree.Element("ad9958")
    element = etree.SubElement(root, "elem", n="4")
    for name in ("ch0", "ch1"):
        channel = etree.SubElement(element, name)
        etree.SubElement(channel, "mode").text = "sf"
        etree.SubElement(channel, "fr").text = "80000000"
        etree.SubElement(channel, "am").text = "100"
    etree.ElementTree(root).write(str(path), encoding="ISO-8859-1", xml_declaration=True)


class AcStarkExecutionPlanTests(unittest.TestCase):
    def test_plan_uses_selected_group_and_existing_average_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            source = base_dir / "base.xml"
            sequence = base_dir / "sequence.mot"
            write_dds_table(source)
            sequence.write_text(
                "DDS1 [<PARAMETER0>]\nDDS5 [<PARAMETER1>]\n",
                encoding="utf-8",
            )

            invalid_up_curve = {**CALIBRATION, "upper_asymptote": 10.0}
            manager = object.__new__(ExperimentManager)
            manager.settings = {
                "dds_writetable_path": "",
                "raman_up_r1_calibration": invalid_up_curve,
                "raman_up_r2_calibration": invalid_up_curve,
                "raman_down_r1_calibration": dict(CALIBRATION),
                "raman_down_r2_calibration": dict(CALIBRATION),
            }
            scan_config = {
                "mode": "ac_stark",
                "scan_dimensions": 1,
                "randomize": False,
                "ac_stark_raman_group": "down",
                "ac_stark_left_p0": 11,
                "ac_stark_right_p0": 22,
                "ac_stark_ratio_start": 1.0,
                "ac_stark_ratio_stop": 2.0,
                "ac_stark_ratio_step": 1.0,
                "ac_stark_total_power": 100.0,
                "averages": 2,
            }

            with (
                mock.patch.object(config, "USE_SIMULATION", True),
                mock.patch.object(config, "DDS_TABLE_UPLOAD_PATH", source),
                mock.patch.object(config, "BASE_DIR", base_dir),
                mock.patch.object(config, "SEQUENCE_TEMPLATE_PATH_WIN", str(sequence)),
            ):
                points, context = manager._build_ac_stark_execution(scan_config)

            self.assertEqual(len(points), 8)
            first_pass = [point["sequence_parameters"] for point in points[:4]]
            second_pass = [point["sequence_parameters"] for point in points[4:]]
            self.assertEqual(first_pass, [[11, 5], [22, 5], [11, 6], [22, 6]])
            self.assertEqual(second_pass, first_pass)
            self.assertEqual(
                [point["metadata"]["ac_stark_side"] for point in points[:4]],
                ["left", "right", "left", "right"],
            )
            self.assertTrue(all(point["metadata"]["ac_stark_raman_group"] == "down" for point in points))
            self.assertEqual(points[0]["metadata"]["ac_stark_repeat_index"], 1)
            self.assertEqual(points[4]["metadata"]["ac_stark_repeat_index"], 2)
            self.assertEqual(len(context["ratio_plan"]), 2)
            self.assertTrue(Path(context["generated_xml"]).is_file())

    def test_summary_reports_right_minus_left_and_sem(self):
        manager = object.__new__(ExperimentManager)
        results = [
            ScanResult(
                parameter=11,
                timestamp=1,
                ac_stark_ratio=1.0,
                ac_stark_side="left",
                atom_number_up=value,
            )
            for value in (8.0, 12.0)
        ] + [
            ScanResult(
                parameter=22,
                timestamp=1,
                ac_stark_ratio=1.0,
                ac_stark_side="right",
                atom_number_up=value,
            )
            for value in (13.0, 17.0)
        ]

        summary = manager._build_ac_stark_summary(results)

        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["atom_number_up_left_mean"], 10.0)
        self.assertAlmostEqual(summary[0]["atom_number_up_right_mean"], 15.0)
        self.assertAlmostEqual(summary[0]["atom_number_up_difference"], 5.0)
        self.assertAlmostEqual(summary[0]["atom_number_up_difference_sem"], 8.0 ** 0.5)



class AcStarkArchiveSummaryTests(unittest.TestCase):
    def test_archive_parser_groups_ratio_sides_and_preserves_metadata(self):
        from app.core.data_loader import DataLoader

        loader = DataLoader()
        rows = []
        for side, parameter, values in (
            ('left', 11, (8.0, 12.0)),
            ('right', 22, (13.0, 17.0)),
        ):
            for index, value in enumerate(values):
                rows.append(loader._row_to_point({
                    'Step': str(len(rows)),
                    'Parameter_P0': str(parameter),
                    'All_Parameters': str(parameter),
                    'Atom_UP': str(value),
                    'NF_Atom_UP': str(value + 1.0),
                    'Prob_UP_F2': str(value / 10.0),
                    'NF_Prob_UP': str(value / 20.0),
                    'AC_Stark_Ratio': '1.25',
                    'AC_Stark_Side': side.upper(),
                    'AC_Stark_DDS_Element': '11',
                    'AC_Stark_Power_R1': '60',
                    'AC_Stark_Power_R2': '48',
                    'AC_Stark_Amplitude_R1': '321',
                    'AC_Stark_Amplitude_R2': '287',
                    'AC_Stark_Actual_Power_R1': '59.8',
                    'AC_Stark_Actual_Power_R2': '48.2',
                }))

        summary = loader._build_ac_stark_summary(rows)

        self.assertEqual(len(summary), 1)
        item = summary[0]
        self.assertEqual(item['ratio'], 1.25)
        self.assertEqual(item['left_count'], 2)
        self.assertEqual(item['right_count'], 2)
        self.assertEqual(item['left_key'], '11.000000')
        self.assertEqual(item['right_key'], '22.000000')
        self.assertEqual(item['amplitude_r1'], 321)
        self.assertAlmostEqual(item['atom_number_up_left_mean'], 10.0)
        self.assertAlmostEqual(item['atom_number_up_right_mean'], 15.0)
        self.assertAlmostEqual(item['atom_number_up_difference'], 5.0)
        self.assertAlmostEqual(item['atom_number_up_difference_sem'], 8.0 ** 0.5)
        self.assertAlmostEqual(item['atom_number_up_nofit_difference'], 5.0)
        self.assertAlmostEqual(item['transition_probability_up_difference'], 0.5)

if __name__ == "__main__":
    unittest.main()
