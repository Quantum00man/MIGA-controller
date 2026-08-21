import io
import unittest
from unittest.mock import patch
import zipfile

from app.core.bragg_export import (
    build_bragg_zip_export,
    build_single_bragg_export,
    sequence_filename_stem,
)
from app.core.experiment_manager import ExperimentManager


CALIBRATION = {
    "lower_asymptote": 0.0,
    "upper_asymptote": 1.0,
    "growth_rate": 2.0,
    "midpoint": 0.0,
    "shape": 1.0,
    "linear_voltage_min": -0.5,
    "linear_voltage_max": 0.5,
    "off_threshold": 0.001,
}
TEMPLATE = "Header\n<PARAMETER0>\n+<PARAMETER1>us Tail = 0\t\t(1)\n"


class BraggExportTests(unittest.TestCase):
    def test_single_export_uses_uploaded_stem_and_fwhm(self):
        payload, filename = build_single_bragg_export(
            TEMPLATE, "upload.mot", 50, "blackman", 1000, CALIBRATION
        )
        content = payload.decode("utf-8")
        self.assertEqual(filename, "upload_50us.mot")
        self.assertNotIn("<PARAMETER", content)
        self.assertIn("Blackman_pulse", content)
        self.assertIn("-3.000", content)
        self.assertIn("(32)", content)

    def test_decimal_fwhm_is_preserved_without_trailing_zeroes(self):
        _, filename = build_single_bragg_export(
            TEMPLATE, "experiment.mot", 50.5, "gaussian", 1000, CALIBRATION
        )
        self.assertEqual(filename, "experiment_50.5us.mot")

    def test_default_display_name_resolves_to_seq0(self):
        self.assertEqual(sequence_filename_stem("Default (seq0.mot)"), "seq0")

    def test_zip_deduplicates_fwhm_filenames_in_scan_order(self):
        payload, filename = build_bragg_zip_export(
            TEMPLATE, "upload.mot", [10, 20, 20.0, 30], "blackman", 1000, CALIBRATION
        )
        self.assertEqual(filename, "upload_blackman_10-30us.zip")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertEqual(
                archive.namelist(),
                ["upload_10us.mot", "upload_20us.mot", "upload_30us.mot"],
            )
            self.assertNotIn("<PARAMETER", archive.read("upload_20us.mot").decode("utf-8"))

    def test_template_requires_bragg_pulse_placeholder(self):
        with self.assertRaisesRegex(ValueError, "PARAMETER0"):
            build_single_bragg_export(
                "<PARAMETER1>\n", "upload.mot", 10, "blackman", 1000, CALIBRATION
            )

    def test_compensation_placeholder_is_optional_and_comment_header_is_safe(self):
        payload, _ = build_single_bragg_export(
            "# Other <PARAMETER0>\n# Bragg pulse <PARAMETER0>\n# Another <PARAMETER0>\nTail\n",
            "upload.mot",
            10,
            "blackman",
            1000,
            CALIBRATION,
        )
        content = payload.decode("utf-8")
        self.assertIn("# Bragg pulse \n+500.0us Blackman_pulse = -3.000", content)
        self.assertEqual(content.count("+500.0us Blackman_pulse = -3.000"), 1)
        self.assertNotIn("<PARAMETER0>", content)

    def test_zip_rejects_more_than_200_unique_files(self):
        with self.assertRaisesRegex(ValueError, "limited to 200"):
            build_bragg_zip_export(
                TEMPLATE, "upload.mot", range(1, 202), "blackman", 1000, CALIBRATION
            )

    def test_single_export_enforces_content_size_limit(self):
        with patch("app.core.bragg_export.MAX_BRAGG_EXPORT_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "100 MB"):
                build_single_bragg_export(
                    TEMPLATE, "upload.mot", 10, "blackman", 1000, CALIBRATION
                )


class BraggScanValuesTests(unittest.TestCase):
    def setUp(self):
        self.manager = object.__new__(ExperimentManager)

    def test_export_ignores_averages_and_randomization(self):
        values = self.manager.build_bragg_export_fwhm_values({
            "mode": "bragg_rabi",
            "scan_dimensions": 1,
            "dim1_type": "list",
            "custom_list": "30, 10, 20",
            "param_type": "float",
            "averages": 5,
            "randomize": True,
        })
        self.assertEqual(values, [30.0, 10.0, 20.0])

    def test_export_supports_point_count_range(self):
        values = self.manager.build_bragg_export_fwhm_values({
            "mode": "bragg_rabi",
            "scan_dimensions": 1,
            "dim1_type": "range",
            "dim1_method": "n_points",
            "start": 10,
            "stop": 40,
            "step": 4,
        })
        self.assertEqual(values, [10.0, 20.0, 30.0, 40.0])

    def test_export_rejects_more_than_200_requested_points_before_generation(self):
        with self.assertRaisesRegex(ValueError, "limited to 200"):
            self.manager.build_bragg_export_fwhm_values({
                "mode": "bragg_rabi",
                "scan_dimensions": 1,
                "dim1_type": "range",
                "dim1_method": "n_points",
                "start": 1,
                "stop": 201,
                "step": 201,
            })

    def test_export_rejects_nonpositive_fwhm(self):
        with self.assertRaisesRegex(ValueError, "positive finite"):
            self.manager.build_bragg_export_fwhm_values({
                "mode": "bragg_rabi",
                "scan_dimensions": 1,
                "dim1_type": "list",
                "custom_list": "10, 0, 20",
            })


if __name__ == "__main__":
    unittest.main()
