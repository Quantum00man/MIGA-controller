"""Read-only archive integrity audits with reports stored outside run directories."""

from __future__ import annotations

import csv
import html
import json
import os
import socket
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPORT_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _directory_size(path: Path) -> tuple[int, int]:
    size = 0
    files = 0
    for root, _, names in os.walk(path):
        for name in names:
            candidate = Path(root) / name
            try:
                stat = candidate.stat()
            except OSError:
                continue
            size += stat.st_size
            files += 1
    return size, files


class ArchiveAuditService:
    """Runs bounded, non-mutating scans and persists JSON/HTML reports."""

    def __init__(self, base_dir: Path, collection_store: Any = None, reports_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.collection_store = collection_store
        self.reports_dir = Path(reports_dir or (self.base_dir / "audit_reports"))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="archive-audit")

    @staticmethod
    def _run_directories(base_dir: Path) -> Iterable[tuple[str, str, str, Path]]:
        if not base_dir.is_dir():
            return
        for year in sorted(base_dir.iterdir()):
            if not year.is_dir() or len(year.name) != 4 or not year.name.isdigit():
                continue
            for month in sorted(year.iterdir()):
                if not month.is_dir() or len(month.name) != 2 or not month.name.isdigit():
                    continue
                for day in sorted(month.iterdir()):
                    if not day.is_dir() or len(day.name) != 2 or not day.name.isdigit():
                        continue
                    for run in sorted(day.iterdir()):
                        if run.is_dir() and run.name.startswith("run"):
                            yield year.name, month.name, day.name, run

    def start(self) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        state = {
            "job_id": job_id,
            "status": "queued",
            "started_at": None,
            "finished_at": None,
            "progress": {"scanned_runs": 0, "total_runs": 0},
            "report_id": None,
            "error": "",
        }
        with self._lock:
            self._jobs[job_id] = state
        self._executor.submit(self._run_job, job_id)
        return dict(state)

    def status(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            state = self._jobs.get(str(job_id))
            if not state:
                raise FileNotFoundError("Archive audit job was not found")
            return json.loads(json.dumps(state))

    def _set_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(updates)

    def _run_job(self, job_id: str) -> None:
        try:
            self._set_job(job_id, status="running", started_at=_utc_now())
            report = self.scan(job_id=job_id)
            report_id = self.save(report)
            self._set_job(
                job_id, status="complete", finished_at=_utc_now(), report_id=report_id,
                progress={"scanned_runs": report["summary"]["run_count"], "total_runs": report["summary"]["run_count"]},
            )
        except Exception as exc:
            self._set_job(job_id, status="failed", finished_at=_utc_now(), error=str(exc))

    def scan(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        started = _utc_now()
        run_dirs = list(self._run_directories(self.base_dir))
        if job_id:
            self._set_job(job_id, progress={"scanned_runs": 0, "total_runs": len(run_dirs)})

        issues: List[Dict[str, Any]] = []
        runs: List[Dict[str, Any]] = []
        totals = {"bytes": 0, "files": 0, "sync_runs": 0, "complete_sync_runs": 0, "waveform_files": 0}
        by_year: Dict[str, Dict[str, int]] = {}

        for index, (year, month, day, run_dir) in enumerate(run_dirs, start=1):
            relative = run_dir.relative_to(self.base_dir).as_posix()
            size_bytes, file_count = _directory_size(run_dir)
            totals["bytes"] += size_bytes
            totals["files"] += file_count
            waveform_count = len(list((run_dir / "waveforms").glob("*.npz"))) if (run_dir / "waveforms").is_dir() else 0
            totals["waveform_files"] += waveform_count
            run_issues: List[str] = []

            config_path = run_dir / "config.json"
            results_path = run_dir / "results.csv"
            if not config_path.is_file():
                run_issues.append("missing_config")
            else:
                try:
                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        run_issues.append("invalid_config")
                except (OSError, ValueError, UnicodeError):
                    run_issues.append("invalid_config")

            if not results_path.is_file():
                run_issues.append("missing_results")
            else:
                try:
                    with results_path.open("r", newline="", encoding="utf-8-sig") as handle:
                        header = next(csv.reader(handle), [])
                    if not header:
                        run_issues.append("empty_results")
                except (OSError, csv.Error, UnicodeError):
                    run_issues.append("invalid_results")

            if not (run_dir / "sequence.mot").is_file():
                run_issues.append("missing_sequence")
            if not waveform_count:
                run_issues.append("missing_waveforms")

            sync_status = "not_sync"
            manifest_path = run_dir / "sync_manifest.json"
            if manifest_path.is_file():
                totals["sync_runs"] += 1
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    replication = manifest.get("archive_replication") or {}
                    sync_status = str(replication.get("status") or "unknown")
                    if sync_status == "complete":
                        totals["complete_sync_runs"] += 1
                    else:
                        run_issues.append("incomplete_sync_replication")
                except (OSError, ValueError, UnicodeError):
                    sync_status = "invalid"
                    run_issues.append("invalid_sync_manifest")

            for issue in run_issues:
                issues.append({"severity": "warning", "code": issue, "run": relative})
            runs.append({
                "year": year, "month": month, "day": day, "run_id": run_dir.name,
                "path": relative, "size_bytes": size_bytes, "file_count": file_count,
                "waveform_count": waveform_count, "sync_status": sync_status, "issues": run_issues,
            })
            year_row = by_year.setdefault(year, {"runs": 0, "bytes": 0})
            year_row["runs"] += 1
            year_row["bytes"] += size_bytes
            if job_id:
                self._set_job(job_id, progress={"scanned_runs": index, "total_runs": len(run_dirs)})

        collection = self._audit_collections()
        issues.extend(collection.pop("issues"))
        return {
            "report_version": REPORT_VERSION,
            "report_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8],
            "started_at": started,
            "completed_at": _utc_now(),
            "host": socket.gethostname(),
            "archive_root": str(self.base_dir.resolve()),
            "read_only_scan": True,
            "summary": {
                "run_count": len(runs), "total_bytes": totals["bytes"], "file_count": totals["files"],
                "waveform_files": totals["waveform_files"], "sync_runs": totals["sync_runs"],
                "complete_sync_runs": totals["complete_sync_runs"],
                "incomplete_sync_runs": totals["sync_runs"] - totals["complete_sync_runs"],
                "issue_count": len(issues), **collection,
            },
            "by_year": by_year,
            "issues": issues,
            "runs": runs,
        }

    def _audit_collections(self) -> Dict[str, Any]:
        if self.collection_store is None:
            return {"collection_folders": 0, "collection_favorites": 0, "missing_collection_references": 0, "issues": []}
        snapshot = self.collection_store.snapshot()
        favorites = snapshot.get("favorites") or []
        missing = [item for item in favorites if item.get("integrity") == "missing"]
        return {
            "collection_folders": len(snapshot.get("folders") or []),
            "collection_favorites": len(favorites),
            "missing_collection_references": len(missing),
            "issues": [{"severity": "warning", "code": "missing_collection_reference", "run": item.get("source_ref", "")} for item in missing],
        }

    def save(self, report: Dict[str, Any]) -> str:
        report_id = str(report["report_id"])
        json_path = self.reports_dir / f"audit_{report_id}.json"
        html_path = self.reports_dir / f"audit_{report_id}.html"
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(json_path)
        html_temporary = html_path.with_suffix(".html.tmp")
        html_temporary.write_text(self._render_html(report), encoding="utf-8")
        html_temporary.replace(html_path)
        return report_id

    def list_reports(self) -> List[Dict[str, Any]]:
        reports = []
        for path in sorted(self.reports_dir.glob("audit_*.json"), reverse=True):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                reports.append({
                    "report_id": report.get("report_id"), "completed_at": report.get("completed_at"),
                    "host": report.get("host"), "archive_root": report.get("archive_root"),
                    "summary": report.get("summary") or {},
                })
            except (OSError, ValueError):
                continue
        return reports

    def report_path(self, report_id: str, format_name: str) -> Path:
        safe_id = str(report_id or "")
        if not safe_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in safe_id):
            raise ValueError("Invalid archive audit report identifier")
        suffix = ".html" if format_name == "html" else ".json"
        path = self.reports_dir / f"audit_{safe_id}{suffix}"
        if not path.is_file():
            raise FileNotFoundError("Archive audit report was not found")
        return path

    @staticmethod
    def _render_html(report: Dict[str, Any]) -> str:
        summary = report.get("summary") or {}
        issue_rows = "".join(
            f"<tr><td>{html.escape(str(item.get('severity', '')))}</td><td>{html.escape(str(item.get('code', '')))}</td><td>{html.escape(str(item.get('run', '')))}</td></tr>"
            for item in report.get("issues") or []
        ) or '<tr><td colspan="3">No issues found.</td></tr>'
        return f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>MIGA Archive Audit</title>
<style>body{{font:15px system-ui;margin:32px;color:#263244}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dbe3ec;text-align:left}}code{{background:#f4f6f9;padding:2px 5px}}</style></head><body>
<h1>MIGA Archive Audit</h1><p>Read-only scan completed {html.escape(str(report.get('completed_at', '')))}</p>
<p><strong>Archive root:</strong> <code>{html.escape(str(report.get('archive_root', '')))}</code></p>
<ul><li>Runs: {int(summary.get('run_count', 0))}</li><li>Total bytes: {int(summary.get('total_bytes', 0))}</li>
<li>SYNC runs: {int(summary.get('sync_runs', 0))}</li><li>Issues: {int(summary.get('issue_count', 0))}</li>
<li>Collection favorites: {int(summary.get('collection_favorites', 0))}</li></ul>
<h2>Issues</h2><table><thead><tr><th>Severity</th><th>Code</th><th>Run</th></tr></thead><tbody>{issue_rows}</tbody></table>
</body></html>"""
