import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.core.archive_collection_store import ArchiveCollectionStore
from app.core.data_loader import DataLoader
from app.core.data_manager import RESULTS_CSV_HEADER


class ArchiveCollectionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.run_dir = self.base / "2026" / "08" / "22" / "run01"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "config.json").write_text(
            json.dumps({"run_label": "Original label", "sequence_name": "scan.mot"}),
            encoding="utf-8",
        )
        self.store = ArchiveCollectionStore(self.base, self.base / "collections.sqlite3")
        self.reference = {"year": "2026", "month": "08", "day": "22", "run_id": "run01"}
        self.metadata = {
            "source_type": "scan",
            "original_label": "Original label",
            "sequence_name": "scan.mot",
            "summary": "1D standard",
        }
        self.preview = {"metric": "prob", "x": [1, 2], "up": [0.2, 0.4], "down": [0.8, 0.6]}

    def tearDown(self):
        self.temp.cleanup()

    def test_nested_folders_reject_duplicates_cycles_and_nonempty_delete(self):
        paper = self.store.create_folder("Paper")
        figure = self.store.create_folder("Figure 1", paper["id"])
        with self.assertRaises(ValueError):
            self.store.create_folder("paper")
        with self.assertRaises(ValueError):
            self.store.update_folder(paper["id"], parent_id=figure["id"])

        self.store.create_favorite(
            figure["id"], self.reference, self.metadata, self.preview, alias="Panel a"
        )
        with self.assertRaises(ValueError):
            self.store.delete_folder(figure["id"])
        with self.assertRaises(ValueError):
            self.store.delete_folder(paper["id"])

    def test_same_run_can_appear_in_multiple_folders_but_not_twice_in_one(self):
        first = self.store.create_folder("First")
        second = self.store.create_folder("Second")
        favorite = self.store.create_favorite(
            first["id"], self.reference, self.metadata, self.preview, alias="Renamed"
        )
        with self.assertRaises(ValueError):
            self.store.create_favorite(first["id"], self.reference, self.metadata, self.preview)
        duplicate = self.store.create_favorite(second["id"], self.reference, self.metadata, self.preview)
        self.assertNotEqual(favorite["id"], duplicate["id"])
        config = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["run_label"], "Original label")

    def test_batch_actions_and_integrity_tracking(self):
        source = self.store.create_folder("Source")
        target = self.store.create_folder("Target")
        favorite = self.store.create_favorite(source["id"], self.reference, self.metadata, self.preview)
        copied = self.store.batch_favorites("copy", [favorite["id"]], target["id"])
        self.assertEqual(copied, {"changed": 1, "skipped": 0})
        snapshot = self.store.snapshot()
        self.assertEqual(len(snapshot["favorites"]), 2)
        self.assertTrue(all(item["integrity"] == "ok" for item in snapshot["favorites"]))

        (self.run_dir / "config.json").write_text('{"changed": true}', encoding="utf-8")
        self.assertTrue(all(item["integrity"] == "modified" for item in self.store.snapshot()["favorites"]))
        ids = [item["id"] for item in self.store.snapshot()["favorites"]]
        removed = self.store.batch_favorites("remove", ids)
        self.assertEqual(removed["changed"], 2)
        self.assertFalse(self.store.snapshot()["favorites"])

    def test_archive_ui_shares_loaded_timeline_or_collection_run_by_url(self):
        archive_html = (
            Path(__file__).resolve().parents[1] / "static" / "archive.html"
        ).read_text(encoding="utf-8")
        self.assertIn("Share Run Link", archive_html)
        self.assertIn("Share loaded run", archive_html)
        self.assertIn("loadSharedRunFromUrl", archive_html)
        self.assertIn("url.searchParams.set('run', String(reference.run_id))", archive_html)
        self.assertIn("this.fetchArchiveTree().then(() => this.loadSharedRunFromUrl())", archive_html)
        self.assertIn("document.execCommand('copy')", archive_html)


class ArchiveCollectionPreviewTests(unittest.TestCase):
    def test_standard_scan_preview_uses_selected_metric(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            run_dir = base / "2026" / "08" / "22" / "run01"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps({"scan_dimensions": 1, "run_label": "Preview"}), encoding="utf-8"
            )
            with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_HEADER)
                writer.writeheader()
                for index, value in enumerate((1.0, 2.0, 3.0)):
                    writer.writerow({
                        "Step": index,
                        "Parameter_P0": value,
                        "All_Parameters": str(value),
                        "Prob_UP_F2": value / 10,
                        "Prob_DW_F1": 1 - value / 10,
                    })
            loader = DataLoader()
            loader.base_dir = base
            preview = loader.build_collection_preview("2026", "08", "22", "run01", "prob")
            self.assertEqual(preview["metric"], "prob")
            self.assertEqual(preview["x"], [1.0, 2.0, 3.0])
            self.assertEqual(preview["up"], [0.1, 0.2, 0.3])
            self.assertEqual(preview["down"], [0.9, 0.8, 0.7])


if __name__ == "__main__":
    unittest.main()
