import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.api.routes import _default_index_ui_state, _save_index_ui_state_record
from app.core.experiment_manager import ExperimentManager


class AtomicSettingsPersistenceTests(unittest.TestCase):
    def test_system_settings_replace_a_read_only_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "user_settings.json"
            target.write_text('{"old": true}', encoding="utf-8")
            target.chmod(0o444)
            manager = ExperimentManager.__new__(ExperimentManager)
            manager.settings = {"sync_role": "master", "sync_slaves": []}

            with patch("app.core.experiment_manager.config.SETTINGS_FILE_PATH", target):
                manager._save_settings_to_disk()

            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["sync_role"], "master")

    def test_index_state_replaces_a_read_only_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "index_ui_state.json"
            target.write_text('{}', encoding="utf-8")
            target.chmod(0o444)
            state = _default_index_ui_state()
            state["runMode"] = "sync"

            with patch("app.api.routes.config.INDEX_UI_STATE_PATH", target):
                record = _save_index_ui_state_record(state, "test-client")

            saved = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(record["state"]["runMode"], "sync")
            self.assertEqual(saved["state"]["runMode"], "sync")


if __name__ == "__main__":
    unittest.main()
