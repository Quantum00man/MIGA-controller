from pathlib import Path
import tempfile
import unittest

from app.api.routes import _writetable_folder_listing


class WritetableFolderBrowserTests(unittest.TestCase):
    def test_lists_directories_and_accepts_folder_containing_writer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            writer_folder = root / "PREPARE FOR THE AC STARK"
            writer_folder.mkdir()
            writer = writer_folder / "writetable.py"
            writer.write_text("# test writer\n", encoding="utf-8")
            (root / "empty").mkdir()
            (root / ".hidden").mkdir()

            root_listing = _writetable_folder_listing("", root_path=root)
            self.assertEqual(
                [item["name"] for item in root_listing["directories"]],
                ["empty", "PREPARE FOR THE AC STARK"],
            )

            selected = _writetable_folder_listing(str(writer), root_path=root)
            self.assertTrue(selected["contains_writetable"])
            self.assertEqual(selected["current"], str(writer_folder.resolve()))
            self.assertEqual(selected["writetable_path"], str(writer.resolve()))

    def test_rejects_navigation_outside_configured_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "Documents"
            root.mkdir()
            with self.assertRaises(ValueError):
                _writetable_folder_listing(str(root.parent), root_path=root)


if __name__ == "__main__":
    unittest.main()
