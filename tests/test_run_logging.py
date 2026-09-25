import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.api.routes import router
from app.core.data_loader import DataLoader
from app.core.data_manager import DataManager


class RunLoggingTests(unittest.TestCase):
    def test_data_manager_creates_and_closes_run_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("config.DATA_BASE_DIR", temporary), patch("config.USE_SIMULATION", True), patch(
                "config.SEQUENCE_TEMPLATE_PATH_WIN", str(Path(temporary) / "missing.mot")
            ):
                manager = DataManager()
                manager.init_run({"mode": "standard", "run_label": "logged"})
                manager.log_event("shot.started", parameters=[1.25])
                manager.close_run(status="completed", message="Done")
                records = [json.loads(line) for line in (manager.current_run_dir / "run.log").read_text().splitlines()]
        self.assertEqual([record["event"] for record in records], ["run.initialized", "shot.started", "run.closed"])
        self.assertEqual(records[1]["parameters"], [1.25])
        self.assertEqual(records[-1]["status"], "completed")

    def test_loader_builds_time_sorted_merged_sync_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "2026" / "09" / "25" / "run00_20260925"
            slave = root / "sync_nodes" / "slave-a"
            slave.mkdir(parents=True)
            (root / "run.log").write_text(json.dumps({"timestamp_unix_ms": 20, "event": "master"}) + "\n")
            (slave / "run.log").write_text(json.dumps({"timestamp_unix_ms": 10, "event": "slave"}) + "\n")
            with patch("config.DATA_BASE_DIR", Path(temporary)):
                loader = DataLoader()
                payload, filename = loader.build_merged_sync_run_log("2026", "09", "25", "run00_20260925")
            records = [json.loads(line) for line in payload.decode().splitlines()]
        self.assertEqual(filename, "run00_20260925_merged_run.log")
        self.assertEqual([record["event"] for record in records], ["slave", "master"])
        self.assertEqual([record["source_node"] for record in records], ["slave-a", "master"])

    def test_run_log_download_routes_and_archive_controls_exist(self):
        paths = {route.path for route in router.routes}
        self.assertIn("/archive/run-log/{year}/{month}/{day}/{run_id}", paths)
        self.assertIn("/archive/run-log/{year}/{month}/{day}/{run_id}/merged", paths)
        archive_html = Path("static/archive.html").read_text(encoding="utf-8")
        self.assertIn("Download Run Log", archive_html)
        self.assertIn("Merged run log", archive_html)


if __name__ == "__main__":
    unittest.main()
