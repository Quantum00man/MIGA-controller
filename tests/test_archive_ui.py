import json
from pathlib import Path
import tempfile
import unittest

from app.archive.ui_context import clone_metadata
from app.archive.ui_settings import ServerSettings


class ArchiveUiStorageTests(unittest.TestCase):
    def test_metadata_versions_share_only_read_only_waveforms(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / 'raw'
            (raw / 'waveforms').mkdir(parents=True)
            (raw / 'waveforms/step_0000.npz').write_bytes(b'unchanged-waveform')
            (raw / 'config.json').write_text('{"original":true}')
            (raw / 'results.csv').write_text('original-results')
            first = base / 'versions/one/run'
            clone_metadata(raw, first, raw)
            self.assertTrue((first / 'waveforms').is_symlink())
            self.assertFalse((first / 'config.json').is_symlink())
            (first / 'config.json').write_text('{"analysis":1}')
            second = base / 'versions/two/run'
            clone_metadata(first, second, raw)
            (second / 'config.json').write_text('{"analysis":2}')
            self.assertEqual(json.loads((raw / 'config.json').read_text()), {'original': True})
            self.assertEqual(json.loads((first / 'config.json').read_text()), {'analysis': 1})
            self.assertEqual((second / 'waveforms/step_0000.npz').read_bytes(), b'unchanged-waveform')
            self.assertEqual((raw / 'results.csv').read_text(), 'original-results')

    def test_unexpected_metadata_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / 'raw'
            raw.mkdir()
            external = base / 'outside.json'
            external.write_text('{}')
            (raw / 'config.json').symlink_to(external)
            with self.assertRaises(ValueError):
                clone_metadata(raw, base / 'version', raw)

    def test_scientific_preferences_are_append_only_and_device_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            master = ServerSettings(base / 'master')
            slave = ServerSettings(base / 'slave')
            master._save_user_json_payload({'scan_fit_models': []})
            first = json.loads((base / 'master/latest.json').read_text())['id']
            master._save_user_json_payload({'scan_fit_models': [], 'active_bragg_phase_calibration_id': ''})
            self.assertEqual(json.loads((base / 'master' / (first + '.json')).read_text()), {'scan_fit_models': []})
            self.assertEqual(slave._load_user_json_payload(), {})
            self.assertEqual(len(list((base / 'master').glob('*.json'))), 3)


if __name__ == '__main__':
    unittest.main()
