import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.core.data_loader import DataLoader


def make_run(root: Path, year: str, month: str, day: str, run_id: str, label: str = "") -> None:
    run_dir = root / year / month / day / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"run_label": label, "mode": "standard"}), encoding="utf-8")


def test_archive_navigation_excludes_non_date_directories_and_finds_latest():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        make_run(root, "2025", "12", "31", "run09_20251231")
        make_run(root, "2026", "01", "02", "run01_20260102", "latest")
        make_run(root, "2026", "01", "02", "run00_20260102")
        log_dir = root / "schedule_logs"
        log_dir.mkdir()
        (log_dir / "2026-01-02.jsonl").write_text("{}\n", encoding="utf-8")

        loader = DataLoader()
        loader.base_dir = root

        assert loader.list_archive_years() == ["2025", "2026"]
        assert loader.list_archive_months("2026") == ["01"]
        assert loader.list_archive_days("2026", "01") == ["02"]
        assert [item["id"] for item in loader.list_archive_runs("2026", "01", "02")] == [
            "run00_20260102", "run01_20260102"
        ]
        assert "schedule_logs" not in loader.get_archive_tree()
        original_builder = loader._build_run_entry
        with patch.object(loader, "_build_run_entry", wraps=original_builder) as build_entry:
            latest = loader.get_latest_archive_run()
        assert build_entry.call_count == 1
        assert latest["run_id"] == "run01_20260102"
        assert latest["run"]["run_label"] == "latest"
