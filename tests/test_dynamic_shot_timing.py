import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.core.experiment_manager import ExperimentManager


class DynamicShotTimingTests(unittest.TestCase):
    def test_acquisition_measures_each_launch_cycle(self):
        manager = ExperimentManager.__new__(ExperimentManager)
        manager.stop_flag = False
        manager._scan_finalize_error = None
        manager.data_queue = Mock()
        manager.settings = {}
        manager.execute_single_measurement = Mock(return_value={"idx": 0, "total": 1})
        manager._resolve_scan_dimensions = Mock(return_value=1)
        manager._restore_ac_stark_dds = Mock(return_value=None)

        with patch("app.core.experiment_manager.time.monotonic", side_effect=[10.0, 11.75]):
            manager._acquisition_loop([[1]], {"mode": "standard"})

        measured_job = manager.data_queue.put.call_args_list[0].args[0]
        self.assertEqual(measured_job["shot_duration_sec"], 1.75)

    def test_index_accumulates_and_persists_completed_scan_average(self):
        index_html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("recordShotDuration(data = {})", index_html)
        self.assertIn("this.shotTiming.totalDurationSec / this.shotTiming.sampleCount", index_html)
        self.assertIn("this.scheduleSettings.singlePointDurationSec = Number(learned.toFixed(6))", index_html)
        self.assertIn("'shot_duration_sec': job.get('shot_duration_sec')", Path(
            Path(__file__).resolve().parents[1] / "app" / "core" / "experiment_manager.py"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
