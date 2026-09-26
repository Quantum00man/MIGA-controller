import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.api.routes import _default_index_ui_state, _save_index_ui_state_record
from app.core.experiment_manager import ExperimentManager
from app.models.schemas import SystemSettings


class AtomicSettingsPersistenceTests(unittest.TestCase):
    def test_system_settings_preserve_local_intf_alpha_calibration_mot(self):
        manager = ExperimentManager()
        payload = SystemSettings(**{
            key: value for key, value in manager.get_settings().items()
            if key in SystemSettings.model_fields
        }).model_dump()
        payload["intf_alpha_calibration_sequence_name"] = "local-alpha.mot"
        payload["intf_alpha_calibration_sequence_content_base64"] = "YWxwaGE="
        payload["intf_alpha_calibration_accepted_min"] = 0.25
        payload["intf_alpha_calibration_accepted_max"] = 0.65
        payload["intf_alpha_calibration_fit_center_up"] = 12.5
        payload["intf_alpha_calibration_fit_width_up"] = 3.0
        payload["intf_alpha_calibration_fit_center_dw"] = 14.5
        payload["intf_alpha_calibration_fit_width_dw"] = 4.0

        validated = SystemSettings(**payload).model_dump()

        self.assertEqual(validated["intf_alpha_calibration_sequence_name"], "local-alpha.mot")
        self.assertEqual(validated["intf_alpha_calibration_sequence_content_base64"], "YWxwaGE=")
        self.assertEqual(validated["intf_alpha_calibration_accepted_min"], 0.25)
        self.assertEqual(validated["intf_alpha_calibration_accepted_max"], 0.65)
        self.assertEqual(validated["intf_alpha_calibration_fit_center_up"], 12.5)
        self.assertEqual(validated["intf_alpha_calibration_fit_width_up"], 3.0)
        self.assertEqual(validated["intf_alpha_calibration_fit_center_dw"], 14.5)
        self.assertEqual(validated["intf_alpha_calibration_fit_width_dw"], 4.0)

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
