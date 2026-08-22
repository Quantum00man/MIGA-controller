from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.sequence_markers import (
    decode_mot_bytes,
    encode_mot_text,
    inspect_sequence_markers,
    marked_filename,
    sequence_marker_profile_key,
)


class SequenceMarkerDocumentStore:
    """Persistent, profile-scoped storage for marked MOT editor documents."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self._lock = threading.Lock()

    @staticmethod
    def _storage_name(profile_key: str) -> str:
        digest = hashlib.sha256(profile_key.encode("utf-8")).hexdigest()[:24]
        return f"{digest}.mot"

    def _empty_index(self) -> Dict[str, Any]:
        return {"version": 1, "last_profile": None, "documents": {}}

    def _read_index(self) -> Dict[str, Any]:
        if not self.index_path.is_file():
            return self._empty_index()
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return self._empty_index()
        if not isinstance(payload, dict) or not isinstance(payload.get("documents"), dict):
            return self._empty_index()
        payload.setdefault("version", 1)
        payload.setdefault("last_profile", None)
        return payload

    def _write_index(self, payload: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / ".index.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.index_path)

    def save(self, filename: str, content: str, encoding: str) -> Dict[str, Any]:
        original_name = Path(str(filename or "sequence.mot")).name or "sequence.mot"
        profile_key = sequence_marker_profile_key(original_name)
        payload = encode_mot_text(content, encoding)
        inspection = inspect_sequence_markers(content)
        updated_at_ms = int(time.time() * 1000)
        storage_name = self._storage_name(profile_key)
        record = {
            "profile_key": profile_key,
            "filename": original_name,
            "marked_filename": marked_filename(original_name),
            "encoding": str(encoding or "utf-8"),
            "storage_name": storage_name,
            "size_bytes": len(payload),
            "line_count": int(inspection.get("line_count", 0)),
            "marker_count": int(inspection.get("marker_count", 0)),
            "updated_at_ms": updated_at_ms,
        }
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.root / f".{storage_name}.tmp"
            temporary.write_bytes(payload)
            temporary.replace(self.root / storage_name)
            index = self._read_index()
            index["documents"][profile_key] = record
            index["last_profile"] = profile_key
            self._write_index(index)
        return dict(record)

    def list(self) -> Dict[str, Any]:
        with self._lock:
            index = self._read_index()
            documents = []
            dirty = False
            for profile_key, raw in index["documents"].items():
                if not isinstance(raw, dict):
                    dirty = True
                    continue
                path = self.root / str(raw.get("storage_name") or self._storage_name(profile_key))
                if not path.is_file():
                    dirty = True
                    continue
                record = dict(raw)
                record["profile_key"] = profile_key
                record["size_bytes"] = path.stat().st_size
                documents.append(record)
            documents.sort(key=lambda item: int(item.get("updated_at_ms", 0)), reverse=True)
            valid_profiles = {item["profile_key"] for item in documents}
            last_profile = index.get("last_profile") if index.get("last_profile") in valid_profiles else None
            if dirty:
                index["documents"] = {item["profile_key"]: item for item in documents}
                index["last_profile"] = last_profile
                self._write_index(index)
            return {"documents": documents, "last_profile": last_profile}

    def load(self, *, sequence_name: str = "", profile_key: str = "") -> Tuple[Dict[str, Any], str]:
        requested_profile = str(profile_key or "").strip()
        if not requested_profile and sequence_name:
            requested_profile = sequence_marker_profile_key(sequence_name)
        with self._lock:
            index = self._read_index()
            if not requested_profile:
                requested_profile = str(index.get("last_profile") or "")
            record = index.get("documents", {}).get(requested_profile)
            if not isinstance(record, dict):
                raise FileNotFoundError("Saved marked MOT document was not found")
            path = self.root / str(record.get("storage_name") or self._storage_name(requested_profile))
            if not path.is_file():
                raise FileNotFoundError("Saved marked MOT file is missing")
            payload = path.read_bytes()
        content, detected_encoding = decode_mot_bytes(payload)
        loaded = dict(record)
        loaded["profile_key"] = requested_profile
        loaded["encoding"] = str(record.get("encoding") or detected_encoding)
        return loaded, content

    def download(self, *, sequence_name: str = "", profile_key: str = "") -> Tuple[bytes, str]:
        record, content = self.load(sequence_name=sequence_name, profile_key=profile_key)
        return encode_mot_text(content, record.get("encoding", "utf-8")), str(record["marked_filename"])
