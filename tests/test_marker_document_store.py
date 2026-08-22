import tempfile
import unittest
from pathlib import Path

from app.core.marker_document_store import SequenceMarkerDocumentStore


class SequenceMarkerDocumentStoreTests(unittest.TestCase):
    def test_saved_document_survives_new_store_instance_and_downloads_marked_name(self):
        content = "###SCAN:FREQ###\n+1us DDS1 [141] (2)\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "documents"
            first = SequenceMarkerDocumentStore(root)
            record = first.save("raman_sequence.mot", content, "utf-8")
            self.assertEqual(record["marked_filename"], "raman_sequence_marked.mot")
            self.assertEqual(record["marker_count"], 1)

            restarted = SequenceMarkerDocumentStore(root)
            listing = restarted.list()
            self.assertEqual(listing["last_profile"], "raman_sequence")
            self.assertEqual(len(listing["documents"]), 1)
            loaded_record, loaded_content = restarted.load(profile_key="raman_sequence")
            self.assertEqual(loaded_record["filename"], "raman_sequence.mot")
            self.assertEqual(loaded_content, content)
            payload, download_name = restarted.download(sequence_name="raman_sequence_marked.mot")
            self.assertEqual(download_name, "raman_sequence_marked.mot")
            self.assertEqual(payload.decode("utf-8"), content)

    def test_same_profile_updates_document_without_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceMarkerDocumentStore(Path(directory))
            store.save("scan.mot", "###SCAN:A###\n+1us DDS1 [1] (2)\n", "utf-8")
            store.save("scan_marked.mot", "###SCAN:A###\n+1us DDS1 [2] (2)\n", "utf-8")
            listing = store.list()
            self.assertEqual(len(listing["documents"]), 1)
            _, content = store.load(sequence_name="scan.mot")
            self.assertIn("DDS1 [2]", content)

    def test_different_sequence_profiles_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceMarkerDocumentStore(Path(directory))
            store.save("first.mot", "###SCAN:SAME###\n+1us DDS1 [1] (2)\n", "utf-8")
            store.save("second.mot", "###SCAN:SAME###\n+1us DDS1 [2] (2)\n", "utf-8")
            listing = store.list()
            self.assertEqual({item["profile_key"] for item in listing["documents"]}, {"first", "second"})
            self.assertEqual(listing["last_profile"], "second")
            _, first_content = store.load(profile_key="first")
            _, second_content = store.load(profile_key="second")
            self.assertIn("[1]", first_content)
            self.assertIn("[2]", second_content)

    def test_filename_is_sanitized_and_latin1_encoding_is_preserved(self):
        content = "###SCAN:POWER###\n+1us AOM =1.000 (23) # café\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SequenceMarkerDocumentStore(root)
            record = store.save("../../unsafe.mot", content, "latin-1")
            self.assertEqual(record["filename"], "unsafe.mot")
            self.assertNotIn("unsafe", record["storage_name"])
            payload, _ = store.download(profile_key="unsafe")
            self.assertEqual(payload.decode("latin-1"), content)
            self.assertTrue((root / record["storage_name"]).is_file())

    def test_missing_document_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceMarkerDocumentStore(Path(directory))
            with self.assertRaises(FileNotFoundError):
                store.load(profile_key="missing")


if __name__ == "__main__":
    unittest.main()
