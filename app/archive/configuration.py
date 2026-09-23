from __future__ import annotations

import json
import os
import shutil
import socket
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path(os.environ.get("MIGA_ARCHIVE_CONFIG", "~/.config/miga-archive/config.json")).expanduser()


class ArchiveServerConfiguration:
    def __init__(self, path: Path = DEFAULT_CONFIG_PATH):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            return {"configuration_error": "Archive Server configuration is unreadable"}
        return payload if isinstance(payload, dict) else {"configuration_error": "Archive Server configuration is invalid"}

    def _save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def register_device(self, device: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.load()
        archive_root = payload.get("archive_root")
        if not archive_root:
            raise ValueError("Initialize the archive root before registering devices")
        device_id = str(device.get("device_id") or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", device_id):
            raise ValueError("Device ID must use lowercase letters, numbers, dots, underscores or hyphens")
        role = str(device.get("role") or "standalone").strip().lower()
        if role not in {"master", "slave", "standalone"}:
            raise ValueError("Device role must be master, slave or standalone")
        transport = str(device.get("transport") or "ssh").strip().lower()
        if transport not in {"ssh", "local"}:
            raise ValueError("Device transport must be ssh or local")
        host = str(device.get("host") or "").strip()
        ssh_user = str(device.get("ssh_user") or "").strip()
        source_path = str(device.get("source_path") or "").strip()
        if not source_path or not Path(source_path).is_absolute():
            raise ValueError("Device source path must be absolute")
        if transport == "ssh" and (not host or not ssh_user):
            raise ValueError("SSH devices require a host and SSH user")
        record = {
            "device_id": device_id,
            "name": str(device.get("name") or device_id).strip() or device_id,
            "role": role, "transport": transport, "host": host, "ssh_user": ssh_user,
            "source_path": source_path.rstrip("/"),
            "collection_path": str(device.get("collection_path") or (Path(source_path) / "archive_collections.sqlite3")),
            "enabled": bool(device.get("enabled", True)),
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        devices = dict(payload.get("devices") or {})
        previous = devices.get(device_id) or {}
        record["registered_at"] = previous.get("registered_at") or record["registered_at"]
        devices[device_id] = record
        payload["devices"] = devices
        self._save(payload)
        device_root = Path(archive_root) / "devices" / device_id
        for name in ("runs", "collection-imports", "state"):
            (device_root / name).mkdir(parents=True, exist_ok=True)
        marker_path = device_root / "device.json"
        marker = {key: record[key] for key in ("device_id", "name", "role", "registered_at")}
        temporary = marker_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(marker, indent=2), encoding="utf-8")
        temporary.replace(marker_path)
        return record

    @staticmethod
    def inspect_root(value: str) -> Dict[str, Any]:
        path = Path(str(value or "").strip()).expanduser()
        result: Dict[str, Any] = {
            "path": str(path), "exists": path.is_dir(), "readable": False, "writable": False,
            "is_mount": False, "filesystem": "", "free_bytes": None, "marker": None, "warnings": [],
        }
        if not path.is_dir():
            result["warnings"].append("The selected directory does not exist")
            return result
        result["readable"] = os.access(path, os.R_OK | os.X_OK)
        result["writable"] = os.access(path, os.W_OK | os.X_OK)
        result["is_mount"] = os.path.ismount(path)
        try:
            result["free_bytes"] = shutil.disk_usage(path).free
        except OSError:
            result["warnings"].append("Free space could not be determined")
        try:
            resolved = path.resolve()
            best_match = (Path("/"), "")
            for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 3:
                    continue
                mountpoint = Path(fields[1].replace("\\040", " "))
                if resolved == mountpoint or mountpoint in resolved.parents:
                    if len(str(mountpoint)) >= len(str(best_match[0])):
                        best_match = (mountpoint, fields[2])
            result["mountpoint"], result["filesystem"] = str(best_match[0]), best_match[1]
        except OSError:
            result["warnings"].append("Linux mount information could not be read")
        if result["filesystem"].startswith("fuse") or "afp" in result["filesystem"].lower():
            result["warnings"].append("Desktop/AFP mounts are suitable for testing only and may disappear after logout")
        if not result["is_mount"] and result.get("mountpoint") == "/":
            result["warnings"].append("The directory is on the local root filesystem, not a dedicated NAS mount")
        marker_path = path / "archive-root.json"
        if marker_path.is_file():
            try:
                result["marker"] = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result["warnings"].append("archive-root.json exists but is invalid")
        return result

    def initialize(self, archive_root: str) -> Dict[str, Any]:
        inspection = self.inspect_root(archive_root)
        if not inspection["exists"] or not inspection["readable"] or not inspection["writable"]:
            raise ValueError("Archive root must exist and be readable and writable")
        root = Path(inspection["path"]).resolve()
        marker_path = root / "archive-root.json"
        marker = inspection.get("marker")
        if marker and marker.get("type") != "miga-archive-root":
            raise ValueError("The existing archive-root.json does not identify a MIGA archive")
        if not marker:
            marker = {
                "type": "miga-archive-root", "format_version": 1,
                "archive_uuid": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
                "created_by": socket.gethostname(),
            }
            temporary = marker_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(marker, indent=2), encoding="utf-8")
            temporary.replace(marker_path)
        for name in ("devices", "incoming", "quarantine", "derived", "reports"):
            (root / name).mkdir(exist_ok=True)
        payload = {
            "version": 1, "archive_root": str(root), "archive_uuid": marker["archive_uuid"],
            "initialized_at": datetime.now(timezone.utc).isoformat(),
        }
        payload["devices"] = dict(self.load().get("devices") or {})
        self._save(payload)
        return payload
