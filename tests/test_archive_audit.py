import csv
import json
import tempfile
from pathlib import Path

from app.core.archive_audit import ArchiveAuditService
from app.core.archive_collection_store import ArchiveCollectionStore


def _make_run(root: Path, run_id: str, *, sync_status: str = "") -> Path:
    run = root / "2026" / "09" / "22" / run_id
    (run / "waveforms").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"mode": "standard"}), encoding="utf-8")
    with (run / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["Step", "Parameter_P0"])
    (run / "sequence.mot").write_text("WAIT\n", encoding="utf-8")
    (run / "waveforms" / "step_0000.npz").write_bytes(b"fixture")
    if sync_status:
        (run / "sync_manifest.json").write_text(
            json.dumps({"archive_replication": {"status": sync_status}}), encoding="utf-8"
        )
    return run


def test_archive_audit_is_read_only_for_runs_and_saves_reports_outside_timeline():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = _make_run(root, "run00_20260922", sync_status="complete")
        before = {path.relative_to(run): path.stat().st_mtime_ns for path in run.rglob("*") if path.is_file()}
        store = ArchiveCollectionStore(root)
        service = ArchiveAuditService(root, store)

        report = service.scan()
        report_id = service.save(report)

        after = {path.relative_to(run): path.stat().st_mtime_ns for path in run.rglob("*") if path.is_file()}
        assert before == after
        assert report["read_only_scan"] is True
        assert report["summary"]["run_count"] == 1
        assert report["summary"]["sync_runs"] == 1
        assert report["summary"]["complete_sync_runs"] == 1
        assert service.report_path(report_id, "json").is_file()
        assert service.report_path(report_id, "html").is_file()
        assert len(service.list_reports()) == 1


def test_archive_audit_reports_legacy_missing_files_and_collection_references():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = root / "2025" / "01" / "02" / "run00_20250102"
        run.mkdir(parents=True)
        (run / "config.json").write_text("not-json", encoding="utf-8")
        store = ArchiveCollectionStore(root)
        folder = store.create_folder("Important")
        store.create_favorite(
            folder["id"],
            {"year": "2025", "month": "01", "day": "02", "run_id": run.name},
            {"source_type": "scan"},
            {},
        )
        # Removing the run tests a stale Collection reference without modifying during the scan.
        for path in run.iterdir():
            path.unlink()
        run.rmdir()

        report = ArchiveAuditService(root, store).scan()
        codes = {item["code"] for item in report["issues"]}
        assert report["summary"]["run_count"] == 0
        assert report["summary"]["missing_collection_references"] == 1
        assert "missing_collection_reference" in codes


def test_archive_audit_detects_incomplete_sync_and_invalid_legacy_run():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = _make_run(root, "run02_20260922", sync_status="incomplete")
        (run / "config.json").write_text("[]", encoding="utf-8")
        (run / "sequence.mot").unlink()

        report = ArchiveAuditService(root).scan()
        codes = {item["code"] for item in report["issues"]}
        assert {"invalid_config", "missing_sequence", "incomplete_sync_replication"} <= codes
