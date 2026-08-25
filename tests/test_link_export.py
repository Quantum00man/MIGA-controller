import io
import unittest
import zipfile

from app.core.experiment_manager import ExperimentManager
from app.core.link_export import build_link_zip_export, build_single_link_export


TEMPLATE = "Header\nP0=<PARAMETER0>\nP1=<PARAMETER1>\nP2=<PARAMETER2>\n"


class LinkExportTests(unittest.TestCase):
    def test_single_export_replaces_all_linked_parameters(self):
        payload, filename = build_single_link_export(
            TEMPLATE,
            "utf-8",
            "linked_sequence.mot",
            [10.0, 20.0, 25.0],
        )

        content = payload.decode("utf-8")
        self.assertEqual(filename, "linked_sequence_P0_10.mot")
        self.assertNotIn("<PARAMETER", content)
        self.assertIn("P0=10.000000", content)
        self.assertIn("P1=20.000000", content)
        self.assertIn("P2=25.000000", content)

    def test_single_export_preserves_latin1_encoding(self):
        payload, _ = build_single_link_export(
            "Séquence\nP0=<PARAMETER0>\n",
            "latin-1",
            "sequence.mot",
            [4],
        )

        self.assertEqual(payload.decode("latin-1"), "Séquence\nP0=4\n")

    def test_zip_uses_p0_filenames_and_scan_order(self):
        payload, filename = build_link_zip_export(
            TEMPLATE,
            "utf-8",
            "linked.mot",
            [[3.0, 6.0, 7.0], [1.0, 2.0, 3.0], [2.0, 4.0, 5.0]],
        )

        self.assertEqual(filename, "linked_link_P0_3-2.zip")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertEqual(
                archive.namelist(),
                ["linked_P0_3.mot", "linked_P0_1.mot", "linked_P0_2.mot"],
            )
            self.assertIn("P2=3.000000", archive.read("linked_P0_1.mot").decode("utf-8"))

    def test_export_rejects_unresolved_template_parameters(self):
        with self.assertRaisesRegex(ValueError, "PARAMETER2"):
            build_single_link_export(
                TEMPLATE,
                "utf-8",
                "linked.mot",
                [1.0, 2.0],
            )


class LinkExportParameterTests(unittest.TestCase):
    def setUp(self):
        self.manager = object.__new__(ExperimentManager)

    def test_single_p0_evaluates_formula_chain(self):
        parameter_sets = self.manager.build_link_export_parameter_sets(
            {
                "mode": "link",
                "scan_dimensions": 1,
                "param_type": "float",
                "link_formulas": ["P0 * 2", "P1 + 5"],
            },
            p0=10,
        )

        self.assertEqual(parameter_sets, [[10.0, 20.0, 25.0]])

    def test_scan_export_ignores_averages_and_randomization(self):
        parameter_sets = self.manager.build_link_export_parameter_sets({
            "mode": "link",
            "scan_dimensions": 1,
            "dim1_type": "list",
            "custom_list": "3, 1, 2",
            "param_type": "float",
            "link_formulas": ["P0 * 10"],
            "averages": 5,
            "randomize": True,
        })

        self.assertEqual(parameter_sets, [[3.0, 30.0], [1.0, 10.0], [2.0, 20.0]])

    def test_integer_mode_matches_runtime_rounding(self):
        parameter_sets = self.manager.build_link_export_parameter_sets(
            {
                "mode": "link",
                "scan_dimensions": 1,
                "param_type": "int",
                "link_formulas": ["P0 / 2"],
            },
            p0=5.6,
        )

        self.assertEqual(parameter_sets, [[6, 3]])

    def test_scan_export_rejects_more_than_200_points(self):
        with self.assertRaisesRegex(ValueError, "Link ZIP export is limited to 200"):
            self.manager.build_link_export_parameter_sets({
                "mode": "link",
                "scan_dimensions": 1,
                "dim1_type": "range",
                "dim1_method": "n_points",
                "start": 1,
                "stop": 201,
                "step": 201,
                "link_formulas": ["P0 * 2"],
            })


if __name__ == "__main__":
    unittest.main()
