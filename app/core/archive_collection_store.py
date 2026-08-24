import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ArchiveCollectionStore:
    """SQLite index of virtual archive folders and favorite run references."""

    def __init__(self, base_dir: Path, database_path: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = Path(database_path or (self.base_dir / "archive_collections.sqlite3"))
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS collection_folders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id INTEGER NOT NULL DEFAULT 0,
                    name TEXT NOT NULL COLLATE NOCASE,
                    position INTEGER NOT NULL DEFAULT 0,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    UNIQUE(parent_id, name)
                );
                CREATE TABLE IF NOT EXISTS archive_favorites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    folder_id INTEGER NOT NULL,
                    year TEXT NOT NULL,
                    month TEXT NOT NULL,
                    day TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    alias TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    preview_metric TEXT NOT NULL DEFAULT 'prob',
                    preview_step INTEGER,
                    source_type TEXT NOT NULL DEFAULT 'scan',
                    original_label TEXT NOT NULL DEFAULT '',
                    sequence_name TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    preview_json TEXT NOT NULL DEFAULT '{}',
                    fingerprint_json TEXT NOT NULL DEFAULT '{}',
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    FOREIGN KEY(folder_id) REFERENCES collection_folders(id),
                    UNIQUE(folder_id, year, month, day, run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_archive_favorites_folder
                    ON archive_favorites(folder_id, updated_at_ms DESC);
                """
            )

    @staticmethod
    def _clean_name(value: Any, label: str = "Name") -> str:
        name = str(value or "").strip()
        if not name:
            raise ValueError(f"{label} cannot be empty")
        if len(name) > 160:
            raise ValueError(f"{label} is too long")
        return name

    def _require_folder(self, connection: sqlite3.Connection, folder_id: int) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM collection_folders WHERE id = ?", (folder_id,)).fetchone()
        if row is None:
            raise FileNotFoundError("Collection folder not found")
        return row

    def create_folder(self, name: str, parent_id: int = 0) -> Dict[str, Any]:
        clean_name = self._clean_name(name, "Folder name")
        parent_id = int(parent_id or 0)
        now = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            if parent_id:
                self._require_folder(connection, parent_id)
            try:
                cursor = connection.execute(
                    "INSERT INTO collection_folders(parent_id,name,created_at_ms,updated_at_ms) VALUES(?,?,?,?)",
                    (parent_id, clean_name, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("A folder with this name already exists here") from exc
            return self._folder_payload(connection, int(cursor.lastrowid))

    def _folder_payload(self, connection: sqlite3.Connection, folder_id: int) -> Dict[str, Any]:
        row = self._require_folder(connection, folder_id)
        count = connection.execute(
            "SELECT COUNT(*) FROM archive_favorites WHERE folder_id = ?", (folder_id,)
        ).fetchone()[0]
        return {**dict(row), "item_count": int(count)}

    def update_folder(self, folder_id: int, name: Optional[str] = None, parent_id: Optional[int] = None) -> Dict[str, Any]:
        folder_id = int(folder_id)
        with self._lock, self._connect() as connection:
            current = self._require_folder(connection, folder_id)
            next_name = self._clean_name(name, "Folder name") if name is not None else current["name"]
            next_parent = int(parent_id) if parent_id is not None else int(current["parent_id"])
            if next_parent == folder_id:
                raise ValueError("A folder cannot contain itself")
            if next_parent:
                self._require_folder(connection, next_parent)
                cursor = next_parent
                seen = set()
                while cursor:
                    if cursor == folder_id:
                        raise ValueError("Cannot move a folder into one of its descendants")
                    if cursor in seen:
                        break
                    seen.add(cursor)
                    parent = connection.execute(
                        "SELECT parent_id FROM collection_folders WHERE id = ?", (cursor,)
                    ).fetchone()
                    cursor = int(parent[0]) if parent else 0
            try:
                connection.execute(
                    "UPDATE collection_folders SET name=?, parent_id=?, updated_at_ms=? WHERE id=?",
                    (next_name, next_parent, int(time.time() * 1000), folder_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("A folder with this name already exists here") from exc
            return self._folder_payload(connection, folder_id)

    def delete_folder(self, folder_id: int) -> None:
        folder_id = int(folder_id)
        with self._lock, self._connect() as connection:
            self._require_folder(connection, folder_id)
            has_children = connection.execute(
                "SELECT 1 FROM collection_folders WHERE parent_id=? LIMIT 1", (folder_id,)
            ).fetchone()
            has_items = connection.execute(
                "SELECT 1 FROM archive_favorites WHERE folder_id=? LIMIT 1", (folder_id,)
            ).fetchone()
            if has_children or has_items:
                raise ValueError("Only empty folders can be deleted")
            connection.execute("DELETE FROM collection_folders WHERE id=?", (folder_id,))

    def _run_dir(self, year: str, month: str, day: str, run_id: str) -> Path:
        parts = [str(year), str(month), str(day), str(run_id)]
        if any(not part or part in {".", ".."} or "/" in part or "\\" in part for part in parts):
            raise ValueError("Invalid archive run reference")
        path = self.base_dir.joinpath(*parts)
        try:
            path.resolve().relative_to(self.base_dir.resolve())
        except ValueError as exc:
            raise ValueError("Invalid archive run reference") from exc
        return path

    def source_fingerprint(self, year: str, month: str, day: str, run_id: str) -> Dict[str, Any]:
        run_dir = self._run_dir(year, month, day, run_id)
        files = {}
        for name in ("config.json", "results.csv", "marker_optimization_report.json"):
            path = run_dir / name
            if path.is_file():
                stat = path.stat()
                files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return {"exists": run_dir.is_dir(), "files": files}

    def _favorite_payload(self, row: sqlite3.Row) -> Dict[str, Any]:
        payload = dict(row)
        payload["preview"] = json.loads(payload.pop("preview_json") or "{}")
        stored_fingerprint = json.loads(payload.pop("fingerprint_json") or "{}")
        current = self.source_fingerprint(payload["year"], payload["month"], payload["day"], payload["run_id"])
        payload["integrity"] = "missing" if not current.get("exists") else ("ok" if current == stored_fingerprint else "modified")
        payload["display_name"] = payload["alias"] or payload["original_label"] or payload["run_id"]
        payload["source_ref"] = f'{payload["year"]}/{payload["month"]}/{payload["day"]}/{payload["run_id"]}'
        return payload

    def create_favorite(self, folder_id: int, reference: Dict[str, Any], metadata: Dict[str, Any], preview: Dict[str, Any], alias: str = "", note: str = "", preview_metric: str = "prob", preview_step: Optional[int] = None) -> Dict[str, Any]:
        folder_id = int(folder_id)
        now = int(time.time() * 1000)
        values = tuple(str(reference[key]) for key in ("year", "month", "day", "run_id"))
        fingerprint = self.source_fingerprint(*values)
        if not fingerprint.get("exists"):
            raise FileNotFoundError("Archive run not found")
        with self._lock, self._connect() as connection:
            self._require_folder(connection, folder_id)
            try:
                cursor = connection.execute(
                    """INSERT INTO archive_favorites(
                    folder_id,year,month,day,run_id,alias,note,preview_metric,preview_step,
                    source_type,original_label,sequence_name,summary,preview_json,fingerprint_json,
                    created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (folder_id, *values, str(alias or "").strip(), str(note or "").strip(), preview_metric,
                     preview_step, metadata.get("source_type", "scan"), metadata.get("original_label", ""),
                     metadata.get("sequence_name", ""), metadata.get("summary", ""),
                     json.dumps(preview, separators=(",", ":")), json.dumps(fingerprint, separators=(",", ":")), now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("This run is already saved in the selected folder") from exc
            row = connection.execute("SELECT * FROM archive_favorites WHERE id=?", (cursor.lastrowid,)).fetchone()
            return self._favorite_payload(row)

    def update_favorite(self, favorite_id: int, **changes: Any) -> Dict[str, Any]:
        allowed = {"folder_id", "alias", "note", "preview_metric", "preview_step", "preview", "fingerprint"}
        updates = {key: value for key, value in changes.items() if key in allowed and value is not None}
        with self._lock, self._connect() as connection:
            current = connection.execute("SELECT * FROM archive_favorites WHERE id=?", (favorite_id,)).fetchone()
            if current is None:
                raise FileNotFoundError("Favorite not found")
            if "folder_id" in updates:
                updates["folder_id"] = int(updates["folder_id"])
                self._require_folder(connection, updates["folder_id"])
            if "preview" in updates:
                updates["preview_json"] = json.dumps(updates.pop("preview"), separators=(",", ":"))
            if "fingerprint" in updates:
                updates["fingerprint_json"] = json.dumps(updates.pop("fingerprint"), separators=(",", ":"))
            updates["updated_at_ms"] = int(time.time() * 1000)
            assignments = ",".join(f"{key}=?" for key in updates)
            try:
                connection.execute(f"UPDATE archive_favorites SET {assignments} WHERE id=?", (*updates.values(), favorite_id))
            except sqlite3.IntegrityError as exc:
                raise ValueError("This run is already saved in the selected folder") from exc
            row = connection.execute("SELECT * FROM archive_favorites WHERE id=?", (favorite_id,)).fetchone()
            return self._favorite_payload(row)

    def delete_favorite(self, favorite_id: int) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM archive_favorites WHERE id=?", (int(favorite_id),))
            if cursor.rowcount == 0:
                raise FileNotFoundError("Favorite not found")

    def batch_favorites(self, action: str, favorite_ids: List[int], folder_id: Optional[int] = None) -> Dict[str, int]:
        action = str(action or "").lower()
        ids = list(dict.fromkeys(int(value) for value in favorite_ids))
        if not ids:
            raise ValueError("Select at least one favorite")
        if action == "remove":
            placeholders = ",".join("?" for _ in ids)
            with self._lock, self._connect() as connection:
                existing = int(connection.execute(
                    f"SELECT COUNT(*) FROM archive_favorites WHERE id IN ({placeholders})", ids
                ).fetchone()[0])
                connection.execute(
                    f"DELETE FROM archive_favorites WHERE id IN ({placeholders})", ids
                )
            return {"changed": existing, "skipped": len(ids) - existing}
        if action not in {"copy", "move"} or folder_id is None:
            raise ValueError("Invalid batch action")
        changed = skipped = 0
        with self._lock, self._connect() as connection:
            self._require_folder(connection, int(folder_id))
            for favorite_id in ids:
                row = connection.execute("SELECT * FROM archive_favorites WHERE id=?", (favorite_id,)).fetchone()
                if row is None:
                    skipped += 1
                    continue
                if action == "move":
                    try:
                        connection.execute("UPDATE archive_favorites SET folder_id=?,updated_at_ms=? WHERE id=?", (folder_id, int(time.time() * 1000), favorite_id))
                        changed += 1
                    except sqlite3.IntegrityError:
                        skipped += 1
                else:
                    columns = [key for key in row.keys() if key != "id"]
                    values = [folder_id if key == "folder_id" else row[key] for key in columns]
                    try:
                        connection.execute(f"INSERT INTO archive_favorites({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", values)
                        changed += 1
                    except sqlite3.IntegrityError:
                        skipped += 1
        return {"changed": changed, "skipped": skipped}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            folders = [self._folder_payload(connection, int(row["id"])) for row in connection.execute("SELECT * FROM collection_folders ORDER BY position,name")]
            favorites = [self._favorite_payload(row) for row in connection.execute("SELECT * FROM archive_favorites ORDER BY updated_at_ms DESC")]
        return {"folders": folders, "favorites": favorites}
