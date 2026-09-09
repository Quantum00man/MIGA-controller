import base64
import lzma
import math
import struct
import unittest
from xml.etree import ElementTree as ET

from app.core.archive_labplot import METRICS, build_archive_project
from app.core.labplot_export import Curve, Plot, Worksheet, build_project


def stats(offset=0.0):
    rows = []
    for index, x in enumerate((10.0, 20.0, 30.0)):
        row = {"x": x}
        for metric in METRICS.values():
            for prefix in {metric["fit"], metric["raw"]}:
                row[f"{prefix}_up"] = offset + index + 1.0
                row[f"{prefix}_dw"] = offset + index + 2.0
        row["phase_up"] = offset + index / 10
        rows.append(row)
    return rows


class FakeLoader:
    def __init__(self, sync=False):
        self.sync = sync
        self.manifest = None
        if sync:
            pairs = []
            for index in range(6):
                p0 = float(index // 2)
                pairs.append({
                    "slave_node_id": "slave_1",
                    "sync_shot_index": index,
                    "sync_p0": p0,
                    "sync_parameters": [p0],
                    "master": {"intf_p1": 20 + index, "intf_p1_nofit": 120 + index},
                    "slave": {"intf_p1": 40 + index * 0.5, "intf_p1_nofit": 140 + index * 0.5},
                })
            self.manifest = {
                "runtime": {"master_node_id": "MIGA22", "slaves": [{"node_id": "slave_1", "name": "MIGA21"}]},
                "archive_nodes": {
                    "master": {"path": ".", "name": "MIGA22"},
                    "slave_1": {"path": "sync_nodes/slave_1", "name": "MIGA21"},
                },
                "pairs": pairs,
            }

    def load_run(self, year, month, day, run_id, node_id=None, current_phase_calibration=None):
        node = node_id or "master"
        calibration = {
            "channel": "up", "fit_x": [10, 15, 20], "fit_y": [30, 45, 30]
        }
        payload = {
            "stats": stats(10 if node == "slave_1" else 0),
            "sync_manifest": self.manifest,
            "interferometer_phase_calibration": calibration,
            "sync_differential_fits": [],
        }
        if self.sync and node_id is None:
            payload["sync_differential_fits"] = [{
                "source": {
                    "aggregation": "shots", "slave_id": "slave_1", "data_mode": "fit",
                    "x_field": "intf_p1", "y_field": "intf_p1",
                },
                "fit_x": [19, 24, 26, 19], "fit_y": [41, 43, 41, 41],
            }]
        return payload


class TransferFunctionLoader:
    def __init__(self, summary=None):
        self.summary = summary or []

    def load_run(self, year, month, day, run_id, node_id=None, current_phase_calibration=None):
        return {
            "config": {"mode": "transfer_function"},
            "transfer_function_summary": self.summary,
        }


class PhaseNoiseLoader:
    def load_run(self, year, month, day, run_id, node_id=None, current_phase_calibration=None):
        return {
            "config": {"mode": "phase_noise"},
            "phase_noise_summary": [],
        }


def transfer_summary():
    rows = []
    for index, frequency in enumerate((10.0, 20.0, 30.0), start=1):
        row = {
            "frequency_hz": frequency,
            "interferometer_phase_s2_components": [],
            "interferometer_phase_s2": index * 10.0,
            "interferometer_phase_phase2_rad2": 5 * (index / 10) ** 2,
            "interferometer_phase_noise_phase2_rad2": 0.02,
            "transfer_phase_noise_sigma_mrad": 100.0,
        }
        for phase_deg, factor in ((0.0, 1.0), (90.0, 2.0)):
            phase_label = f"{int(phase_deg)}deg"
            component = {
                "phase_deg": phase_deg,
                "mean_rad": factor * index / 10,
                "std_rad": factor * index / 100,
                "s2": factor * index,
                "phase2_rad2": (factor * index / 10) ** 2,
            }
            row["interferometer_phase_s2_components"].append(component)
            for source, source_factor in (("fit", 1.0), ("nofit", 100.0)):
                for channel, channel_factor in (("up", 1.0), ("dw", 2.0), ("total", 3.0)):
                    base = f"atom_number_{channel}_{source}_{phase_label}"
                    row[f"{base}_mean"] = source_factor * channel_factor * factor * index
                    row[f"{base}_std"] = source_factor * channel_factor * factor * index / 10
                for channel, channel_factor in (("p1", 1.0), ("p2", 2.0)):
                    base = f"intf_{channel}_{source}_{phase_label}"
                    row[f"{base}_mean"] = source_factor * channel_factor * factor * index
                    row[f"{base}_std"] = source_factor * channel_factor * factor * index / 10
        rows.append(row)
    return rows


class LabPlotExportTests(unittest.TestCase):
    def parse(self, payload):
        return ET.fromstring(lzma.decompress(payload))

    def test_minimal_project_embeds_double_columns(self):
        payload = build_project("Test", [Worksheet("Metric", [
            Plot("UP", "P0", "Value", [Curve("Data", [1, 2], [3, 4])])
        ])])
        root = self.parse(payload)
        self.assertEqual(root.attrib["version"], "2.12.1")
        columns = list(root.iter("column"))
        self.assertEqual(len(columns), 2)
        encoded = (columns[0].find("output_filter").tail or "").strip()
        values = struct.unpack("=2d", base64.b64decode(encoded))
        self.assertEqual(values, (1.0, 2.0))

    def test_standard_archive_has_one_worksheet_per_metric_and_stacked_channels(self):
        payload = build_archive_project(
            FakeLoader(), "2026", "08", "31", "run01", ["atoms", "intf", "phase"],
            current_fit={
                "model_key": "bragg_fringes", "channel": "dw",
                "fit_x": [10, 15, 20], "fit_y": [31, 44, 31],
            },
        )
        root = self.parse(payload)
        worksheets = {item.attrib["name"]: item for item in root.iter("worksheet")}
        self.assertEqual(set(worksheets), {"Atom Number", "Interferometer P (%)", "Interferometer Phase (rad)"})
        self.assertEqual(len(list(worksheets["Atom Number"].iter("cartesianPlot"))), 2)
        intf_curves = [item.attrib["name"] for item in worksheets["Interferometer P (%)"].iter("xyCurve")]
        self.assertIn("Archive fringe fit", intf_curves)
        self.assertIn("Current Bragg fringe fit", intf_curves)

    def test_archive_uses_accessible_scientific_publication_theme(self):
        payload = build_archive_project(
            FakeLoader(), "2026", "08", "31", "run01", ["intf"],
        )
        root = self.parse(payload)
        plot = next(root.iter("cartesianPlot"))
        axes = {axis.attrib["name"]: axis for axis in plot.findall("axis")}

        for text_format in root.iter("format"):
            self.assertEqual(text_format.attrib["fontFamily"], "Arial")
        for labels in root.iter("labels"):
            self.assertEqual(labels.attrib["fontFamily"], "Arial")
            self.assertAlmostEqual(float(labels.attrib["fontPointSize"]), 38.805556, places=5)

        self.assertEqual(axes["x"].find("textLabel/format").attrib["fontPointSize"], "12")
        self.assertEqual(axes["y"].find("textLabel/format").attrib["fontPointSize"], "12")
        self.assertEqual(axes["x"].find("textLabel/geometry").attrib["rotationAngle"], "0")
        self.assertEqual(axes["y"].find("textLabel/geometry").attrib["rotationAngle"], "90")
        self.assertEqual(axes["x"].find("majorGrid").attrib["style"], "0")
        self.assertEqual(axes["y"].find("majorGrid").attrib["style"], "1")
        self.assertTrue(all(grid.attrib["style"] == "0" for grid in root.iter("minorGrid")))

        plot_title = next(label for label in plot.findall("textLabel") if label.attrib["name"].endswith("Title"))
        self.assertEqual(plot_title.find("geometry").attrib["visible"], "0")
        legend = plot.find("cartesianPlotLegend/general")
        self.assertIsNotNone(legend)
        self.assertEqual(legend.attrib["fontFamily"], "Arial")
        self.assertAlmostEqual(float(legend.attrib["fontPointSize"]), 35.277778, places=5)
        legend_geometry = plot.find("cartesianPlotLegend/geometry")
        self.assertEqual(legend_geometry.attrib["horizontalPosition"], "2")
        self.assertEqual(legend_geometry.attrib["verticalPosition"], "0")

        data_curve = next(curve for curve in plot.findall("xyCurve") if "FIT" in curve.attrib["name"])
        self.assertEqual(data_curve.find("lines").attrib["width"], "4.2")
        self.assertEqual(data_curve.find("symbols").attrib["size"], "18")
        self.assertEqual(
            tuple(data_curve.find("lines").attrib[f"color_{channel}"] for channel in ("r", "g", "b")),
            ("0", "114", "178"),
        )
        fit_curve = next(
            curve for curve in plot.findall("xyCurve")
            if curve.attrib["name"].lower().endswith("fringe fit")
        )
        self.assertEqual(fit_curve.find("lines").attrib["style"], "2")
        self.assertEqual(
            tuple(fit_curve.find("lines").attrib[f"color_{channel}"] for channel in ("r", "g", "b")),
            ("35", "35", "35"),
        )

    def test_sync_archive_exports_all_hosts_and_two_differential_worksheets(self):
        payload = build_archive_project(
            FakeLoader(sync=True), "2026", "08", "31", "run_sync", ["intf"], include_differential=True
        )
        root = self.parse(payload)
        worksheets = {item.attrib["name"]: item for item in root.iter("worksheet")}
        self.assertIn("Interferometer P (%)", worksheets)
        self.assertIn("Differential - Every Shot", worksheets)
        self.assertIn("Differential - Average", worksheets)
        intf_names = [item.attrib["name"] for item in worksheets["Interferometer P (%)"].iter("xyCurve")]
        self.assertTrue(any("MIGA22" in name for name in intf_names))
        self.assertTrue(any("MIGA21" in name for name in intf_names))
        shot_names = [item.attrib["name"] for item in worksheets["Differential - Every Shot"].iter("xyCurve")]
        self.assertEqual(shot_names, ["Measured pairs", "Ellipse fit"])

    def test_sync_differential_respects_nofit_source(self):
        payload = build_archive_project(
            FakeLoader(sync=True), "2026", "08", "31", "run_sync", ["intf"],
            source="nofit", include_differential=True,
        )
        root = self.parse(payload)
        worksheet = next(
            item for item in root.iter("worksheet") if item.attrib["name"] == "Differential - Every Shot"
        )
        curve_names = [item.attrib["name"] for item in worksheet.iter("xyCurve")]
        self.assertEqual(curve_names, ["Measured pairs"])
        spreadsheet = next(
            item for item in root.iter("spreadsheet")
            if item.attrib["name"] == "Data - Differential - Every Shot"
        )
        first_column = next(spreadsheet.iter("column"))
        values = struct.unpack("=6d", base64.b64decode(first_column.find("output_filter").tail))
        self.assertEqual(values[0], 120.0)

    def test_transfer_function_uses_frequency_axis_and_exports_all_phase_statistics(self):
        payload = build_archive_project(
            TransferFunctionLoader(transfer_summary()),
            "2026", "09", "08", "run_tf", ["atoms", "intf", "phase"],
        )
        root = self.parse(payload)
        worksheets = {item.attrib["name"]: item for item in root.iter("worksheet")}
        self.assertEqual(set(worksheets), {
            "Atom Number - Mean",
            "Atom Number - Standard Deviation",
            "Interferometer P - Mean",
            "Interferometer P - Standard Deviation",
            "Interferometer Phase - Mean",
            "Interferometer Phase - Standard Deviation",
            "Transfer Function S2",
            "Interferometer Phase Squared",
        })
        s2_curves = [item.attrib["name"] for item in worksheets["Transfer Function S2"].iter("xyCurve")]
        self.assertEqual(s2_curves, ["0 deg", "90 deg", "Quadrature sum"])
        phase2_curves = [
            item.attrib["name"]
            for item in worksheets["Interferometer Phase Squared"].iter("xyCurve")
        ]
        self.assertEqual(
            phase2_curves,
            ["0 deg", "90 deg", "Quadrature sum", "Allan phase-noise floor"],
        )
        noise_curve = next(
            item for item in worksheets["Interferometer Phase Squared"].iter("xyCurve")
            if item.attrib["name"] == "Allan phase-noise floor"
        )
        self.assertEqual(noise_curve.find("lines").attrib["style"], "2")
        self.assertEqual(noise_curve.find("symbols").attrib["symbolsStyle"], "0")
        spreadsheet = next(
            item for item in root.iter("spreadsheet")
            if item.attrib["name"] == "Data - Transfer Function S2"
        )
        first_column = next(spreadsheet.iter("column"))
        frequencies = struct.unpack("=3d", base64.b64decode(first_column.find("output_filter").tail))
        self.assertEqual(frequencies, (10.0, 20.0, 30.0))

    def test_transfer_function_uses_current_recalculated_summary_and_nofit_source(self):
        saved = transfer_summary()
        current = transfer_summary()
        current[0]["frequency_hz"] = 1234.0
        payload = build_archive_project(
            TransferFunctionLoader(saved),
            "2026", "09", "08", "run_tf", ["atoms"], source="nofit",
            transfer_function_summary=current,
        )
        root = self.parse(payload)
        spreadsheet = next(
            item for item in root.iter("spreadsheet")
            if item.attrib["name"] == "Data - Atom Number - Mean"
        )
        columns = list(spreadsheet.iter("column"))
        x_values = struct.unpack("=3d", base64.b64decode(columns[0].find("output_filter").tail))
        y_values = struct.unpack("=3d", base64.b64decode(columns[1].find("output_filter").tail))
        self.assertEqual(x_values[0], 20.0)
        self.assertEqual(x_values[-1], 1234.0)
        self.assertEqual(y_values, (200.0, 300.0, 100.0))

    def test_phase_noise_exports_standard_and_selected_allan_orders(self):
        rows = []
        for index, t2 in enumerate((1_000_000.0, 4_000_000.0), start=1):
            rows.append({
                "t2_us2": t2,
                "t_ms": math.sqrt(t2) / 1000.0,
                "measured_phase_noise_rad": 0.1 * index,
                "expected_total_phase_noise_rad": 0.08 * index,
                "detection_phase_noise_rad": 0.03 * index,
                "laser_phase_noise_rad": 0.05 * index,
                "allan_deviations": [
                    {
                        "order": order,
                        "measured_phase_noise_rad": 0.1 * index / math.sqrt(order),
                        "expected_total_phase_noise_rad": 0.08 * index / math.sqrt(order),
                        "detection_phase_noise_rad": 0.03 * index / math.sqrt(order),
                        "laser_phase_noise_rad": 0.05 * index / math.sqrt(order),
                    }
                    for order in (1, 2)
                ],
            })
        payload = build_archive_project(
            PhaseNoiseLoader(), "2026", "09", "09", "run_pn", ["phase"],
            phase_noise_summary=rows, phase_noise_x_axis="t",
        )
        root = self.parse(payload)
        worksheets = {item.attrib["name"]: item for item in root.iter("worksheet")}
        self.assertEqual(set(worksheets), {
            "Phase Noise - Standard Deviation", "Phase Noise - Allan Deviation",
        })
        allan_plots = list(worksheets["Phase Noise - Allan Deviation"].iter("cartesianPlot"))
        self.assertEqual(len(allan_plots), 2)
        self.assertEqual([item.attrib["name"] for item in allan_plots], [
            "Phase Noise Allan n=1", "Phase Noise Allan n=2",
        ])
        spreadsheet = next(
            item for item in root.iter("spreadsheet")
            if item.attrib["name"] == "Data - Phase Noise - Standard Deviation"
        )
        first_column = next(spreadsheet.iter("column"))
        x_values = struct.unpack("=2d", base64.b64decode(first_column.find("output_filter").tail))
        self.assertEqual(x_values, (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
