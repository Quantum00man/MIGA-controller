import base64
import lzma
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


if __name__ == "__main__":
    unittest.main()
