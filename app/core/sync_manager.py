from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime
import base64
import binascii
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import tempfile
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import requests
import zipfile

import config
from app.core.experiment_manager import ExperimentManager


SYNC_RESULT_FIELDS = (
    "atom_number_up", "atom_number_dw", "amplitude_up", "amplitude_dw",
    "tail_mean_up_raw", "tail_mean_dw_raw",
    "sigma_up", "sigma_dw", "temperature_up", "temperature_dw",
    "arrival_time_up", "arrival_time_dw",
    "transition_probability_up", "transition_probability_dw",
    "intf_n1", "intf_n2", "intf_p1", "intf_p2",
    "atom_number_up_nofit", "atom_number_dw_nofit",
    "amplitude_up_nofit", "amplitude_dw_nofit",
    "sigma_up_nofit", "sigma_dw_nofit",
    "temperature_up_nofit", "temperature_dw_nofit",
    "arrival_time_up_nofit", "arrival_time_dw_nofit",
    "transition_probability_up_nofit", "transition_probability_dw_nofit",
    "intf_n1_nofit", "intf_n2_nofit", "intf_p1_nofit", "intf_p2_nofit",
    "interferometer_phase", "interferometer_phase_valid",
    "interferometer_phase_source_value",
    "interferometer_phase_calibration_id", "interferometer_phase_calibration_name",
    "interferometer_phase_reference_t2_us2",
)


class SyncManager:
    """Coordinates one local master and one or more HTTP-controlled slave nodes."""

    def __init__(self, experiment_manager: ExperimentManager):
        self.manager = experiment_manager
        self._lock = threading.RLock()
        self._prepared: Dict[str, Dict[str, Any]] = {}
        self._node_results: Deque[Tuple[int, Dict[str, Any]]] = deque(maxlen=256)
        self._node_result_sequence = 0
        self._node_active_sync_run_id = ""
        self._master_results: Dict[int, Dict[str, Any]] = {}
        self._slave_results: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._emitted_pairs: set[Tuple[str, int]] = set()
        self._monitor_thread: Optional[threading.Thread] = None
        self._runtime = self._empty_runtime()
        self.manager.add_data_listener(self._capture_local_result)

    @staticmethod
    def _empty_runtime() -> Dict[str, Any]:
        return {
            "active": False,
            "sync_run_id": "",
            "status": "idle",
            "message": "IDLE",
            "master_delay_ms": 0.0,
            "expected_shots": 0,
            "master_step": 0,
            "slaves": [],
            "paired_count": 0,
            "started_at_ms": None,
            "finished_at_ms": None,
            "stop_requested": False,
            "archive_replication": {"status": "idle", "nodes": {}},
        }

    @staticmethod
    def _safe_node_id(value: Any) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._")
        if not normalized:
            raise ValueError("Sync archive node id is empty")
        return normalized[:96]

    def _find_sync_run_dir(self, sync_run_id: str) -> Path:
        with self._lock:
            prepared = self._prepared.get(sync_run_id) or {}
            prepared_dir = prepared.get("run_dir")
            runtime_dir = self._runtime.get("master_run_dir") if self._runtime.get("sync_run_id") == sync_run_id else None
        for candidate in (prepared_dir, runtime_dir):
            if candidate and Path(candidate).is_dir():
                return Path(candidate).resolve()

        base_dir = Path(config.DATA_BASE_DIR).resolve()
        for config_path in base_dir.glob("*/*/*/run*/config.json"):
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str(payload.get("sync_run_id") or "") == str(sync_run_id):
                return config_path.parent.resolve()
        raise FileNotFoundError(f"Archive for Sync run {sync_run_id} was not found")

    def get_sync_run_dir(self, sync_run_id: str) -> Path:
        return self._find_sync_run_dir(sync_run_id)

    @staticmethod
    def _archive_checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def build_archive_bundle(self, sync_run_id: str) -> Tuple[Path, Dict[str, Any]]:
        run_dir = self._find_sync_run_dir(sync_run_id)
        temp = tempfile.NamedTemporaryFile(prefix="miga_sync_archive_", suffix=".zip", delete=False)
        temp_path = Path(temp.name)
        temp.close()
        try:
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for source in sorted(run_dir.rglob("*")):
                    relative = source.relative_to(run_dir)
                    if not source.is_file() or "sync_nodes" in relative.parts:
                        continue
                    if source.name.endswith((".tmp", ".part")):
                        continue
                    archive.write(source, relative.as_posix())
            checksum = self._archive_checksum(temp_path)
            return temp_path, {
                "sync_run_id": sync_run_id,
                "run_id": run_dir.name,
                "sha256": checksum,
                "size_bytes": temp_path.stat().st_size,
            }
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _extract_archive_safely(archive_path: Path, target: Path) -> None:
        target_root = target.resolve()
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                destination = (target / member.filename).resolve()
                if destination != target_root and target_root not in destination.parents:
                    raise ValueError("Sync archive contains an unsafe path")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

    def _install_archive_bundle(
        self,
        run_dir: Path,
        source_node_id: str,
        archive_path: Path,
        expected_sha256: str = "",
    ) -> Dict[str, Any]:
        source_id = self._safe_node_id(source_node_id)
        checksum = self._archive_checksum(archive_path)
        if expected_sha256 and checksum.lower() != str(expected_sha256).strip().lower():
            raise ValueError("Sync archive checksum mismatch")
        sync_root = run_dir / "sync_nodes"
        sync_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{source_id}_", dir=sync_root))
        target = sync_root / source_id
        backup = sync_root / f".{source_id}.previous"
        try:
            self._extract_archive_safely(archive_path, staging)
            if not (staging / "results.csv").is_file() or not (staging / "config.json").is_file():
                raise ValueError("Sync archive is missing results.csv or config.json")
            if backup.exists():
                shutil.rmtree(backup)
            if target.exists():
                target.replace(backup)
            staging.replace(target)
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        return {
            "node_id": source_id,
            "path": f"sync_nodes/{source_id}",
            "sha256": checksum,
            "size_bytes": archive_path.stat().st_size,
            "received_at_ms": int(time.time() * 1000),
        }

    def _merge_newer_replica_phase_metadata(
        self, master_run_dir: Path, replica_relative_path: str
    ) -> bool:
        master_path = Path(master_run_dir) / "sync_phase_analysis.json"
        replica_path = Path(master_run_dir) / replica_relative_path / "sync_phase_analysis.json"
        if not replica_path.is_file():
            return False
        try:
            replica = json.loads(replica_path.read_text(encoding="utf-8"))
            master = json.loads(master_path.read_text(encoding="utf-8")) if master_path.is_file() else {}
        except (OSError, ValueError):
            return False
        if int(replica.get("version") or 0) != 2:
            return False
        if int(replica.get("revision") or 0) <= int(master.get("revision") or 0):
            return False
        if master_path.is_file():
            shutil.copy2(master_path, Path(master_run_dir) / "sync_phase_analysis.previous.json")
        temporary = master_path.with_suffix(master_path.suffix + ".tmp")
        temporary.write_text(json.dumps(replica, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(master_path)
        return True

    def install_node_archive(
        self,
        sync_run_id: str,
        source_node_id: str,
        archive_path: Path,
        expected_sha256: str = "",
    ) -> Dict[str, Any]:
        run_dir = self._find_sync_run_dir(sync_run_id)
        installed = self._install_archive_bundle(run_dir, source_node_id, archive_path, expected_sha256)
        if self._safe_node_id(source_node_id) == "master":
            source_manifest_path = run_dir / installed["path"] / "sync_manifest.json"
            try:
                manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {"runtime": {"sync_run_id": sync_run_id}, "pairs": [], "node_results": {}}
            with self._lock:
                prepared = self._prepared.get(sync_run_id) or {}
            local_id = self._safe_node_id(prepared.get("slave_node_id") or self.manager.get_settings().get("sync_node_name") or "slave")
            source_replication = manifest.get("archive_replication") or {}
            master_copy_of_local = (source_replication.get("nodes") or {}).get(local_id) or {}
            bidirectional_complete = master_copy_of_local.get("pull_status") == "complete"
            manifest["archive_nodes"] = {
                "master": {**installed, "role": "master", "local": False},
                local_id: {"node_id": local_id, "role": "slave", "local": True, "path": ".", "run_id": run_dir.name},
            }
            manifest["archive_replication"] = {
                "status": "complete" if bidirectional_complete else "incomplete",
                "updated_at_ms": int(time.time() * 1000),
                "nodes": {
                    "master": {"status": "complete", "push_status": "complete", **installed},
                    local_id: {
                        "status": "complete" if bidirectional_complete else "incomplete",
                        "pull_status": master_copy_of_local.get("pull_status") or "unknown",
                    },
                },
            }
            self._atomic_write_json(run_dir / "sync_manifest.json", manifest)
        return installed

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def settings_snapshot(self) -> Dict[str, Any]:
        settings = self.manager.get_settings()
        return {
            "role": str(settings.get("sync_role") or "standalone"),
            "node_name": str(settings.get("sync_node_name") or "Local controller"),
            "slaves": deepcopy(settings.get("sync_slaves") or []),
            "token_configured": bool(str(settings.get("sync_shared_token") or "")),
        }

    def authorize(self, token: str, client_ip: str = "") -> None:
        settings = self.manager.get_settings()
        expected = str(settings.get("sync_shared_token") or "")
        if expected and str(token or "") != expected:
            raise PermissionError("Invalid Sync token")
        allowed_ip = str(settings.get("sync_allowed_master_ip") or "").strip()
        if allowed_ip and client_ip and client_ip != allowed_ip:
            raise PermissionError(f"Sync control is not allowed from {client_ip}")

    @staticmethod
    def _normalize_url(value: str) -> str:
        url = str(value or "").strip().rstrip("/")
        if not url:
            raise ValueError("Slave URL is required")
        if "://" not in url:
            url = "http://" + url
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid slave URL: {value}")
        return url

    def _headers(self) -> Dict[str, str]:
        token = str(self.manager.get_settings().get("sync_shared_token") or "")
        return {"X-MIGA-Sync-Token": token} if token else {}

    def health(self) -> Dict[str, Any]:
        return {
            **self.settings_snapshot(),
            "controller_busy": bool(self.manager.get_active_mode()),
            "experiment": {
                "is_running": bool(self.manager.status.is_running),
                "current_step": int(self.manager.status.current_step or 0),
                "total_steps": int(self.manager.status.total_steps or 0),
                "message": self.manager.status.message,
            },
        }

    def test_node(self, base_url: str) -> Dict[str, Any]:
        url = self._normalize_url(base_url)
        response = requests.get(f"{url}/sync/node/health", headers=self._headers(), timeout=3.0)
        response.raise_for_status()
        return response.json()

    def prepare_node(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if str(self.manager.get_settings().get("sync_role") or "standalone") != "slave":
            raise ValueError("This controller is not configured as a Sync slave")
        if self.manager.get_active_mode():
            raise ValueError(f"Slave is busy with {self.manager.get_active_mode()}")
        sync_run_id = str(payload.get("sync_run_id") or "").strip()
        if not sync_run_id:
            raise ValueError("sync_run_id is required")
        encoded_content = str(payload.get("sequence_content_base64") or "").strip()
        content = str(payload.get("sequence_content") or "")
        try:
            content_bytes = base64.b64decode(encoded_content, validate=True) if encoded_content else content.encode("utf-8")
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Slave sequence encoding is invalid") from exc
        if not content_bytes:
            raise ValueError("Slave sequence content is empty")
        if len(content_bytes) > 10 * 1024 * 1024:
            raise ValueError("Slave sequence exceeds 10 MB")
        plan = payload.get("shot_plan") or []
        if not isinstance(plan, list) or not plan:
            raise ValueError("Sync shot plan is empty")

        run_dir = Path(config.BASE_DIR) / "temp" / "sync" / sync_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        sequence_path = run_dir / "slave_sequence.mot"
        sequence_path.write_bytes(content_bytes)
        record = {
            **deepcopy(payload),
            "sequence_path": str(sequence_path),
            "prepared_at_ms": int(time.time() * 1000),
        }
        with self._lock:
            if self._prepared and not self.manager.status.is_running:
                self._prepared.clear()
            self._prepared[sync_run_id] = record
            self._node_results.clear()
            self._node_result_sequence = 0
            self._node_active_sync_run_id = ""
        return {"ready": True, "sync_run_id": sync_run_id, "shot_count": len(plan)}

    def start_node(self, sync_run_id: str) -> Dict[str, Any]:
        with self._lock:
            prepared = deepcopy(self._prepared.get(sync_run_id))
        if not prepared:
            raise ValueError("Sync run was not prepared on this slave")
        plan = prepared.get("shot_plan") or []
        scan_config = dict(prepared.get("scan_config") or {})
        node_name = str(self.manager.get_settings().get("sync_node_name") or "Slave")
        parameters = []
        for index, raw_parameters in enumerate(plan):
            shot_parameters = list(raw_parameters) if isinstance(raw_parameters, list) else [raw_parameters]
            parameters.append({
                "sequence_parameters": [],
                "metadata": {
                    "sync_run_id": sync_run_id,
                    "sync_role": "slave",
                    "sync_node_id": node_name,
                    "sync_shot_index": index,
                    "sync_p0": shot_parameters[0] if shot_parameters else None,
                    # Slave analysis is tagged with the shared P0 only. The
                    # remaining Master parameters are retained for audit but
                    # are never written into the Slave sequence.
                    "sync_parameters": shot_parameters[:1],
                    "sync_master_parameters": shot_parameters,
                },
            })
        scan_config.update({
            "mode": "standard",
            "parameter_source": "classic",
            "marker_axes": [],
            "averages": 1,
            "randomize": False,
            "sequence_name": prepared.get("sequence_name") or "slave.mot",
            "_template_path_override": prepared["sequence_path"],
            "_sync_slave": True,
            "sync_run_id": sync_run_id,
            "sync_role": "slave",
            "sync_master_node_id": prepared.get("master_node_id"),
            "sync_shot_plan": plan,
        })
        result = self.manager.start_scan(scan_config, parameters_override=parameters)
        if result.get("status") == "error":
            raise ValueError(result.get("message") or "Slave scan failed to start")
        with self._lock:
            self._node_active_sync_run_id = sync_run_id
            if sync_run_id in self._prepared:
                self._prepared[sync_run_id]["run_id"] = self.manager.data_manager.current_run_id_str
                self._prepared[sync_run_id]["run_dir"] = str(self.manager.data_manager.current_run_dir)
        return {
            **result,
            "sync_run_id": sync_run_id,
            "shot_count": len(parameters),
            "run_id": self.manager.data_manager.current_run_id_str,
        }

    def stop_node(self, sync_run_id: str) -> Dict[str, Any]:
        with self._lock:
            if sync_run_id != self._node_active_sync_run_id:
                raise ValueError("This Sync run is not active on the slave")
        result = self.manager.stop_scan()
        return {**result, "sync_run_id": sync_run_id}

    def node_status(self, sync_run_id: str, after: int = 0) -> Dict[str, Any]:
        with self._lock:
            if sync_run_id != self._node_active_sync_run_id:
                raise ValueError("This Sync run is not active on the slave")
            results = [
                {"sequence": sequence, "payload": deepcopy(payload)}
                for sequence, payload in self._node_results
                if sequence > int(after or 0)
            ]
            latest_sequence = self._node_result_sequence
        return {
            "sync_run_id": sync_run_id,
            "is_running": bool(self.manager.status.is_running),
            "current_step": int(self.manager.status.current_step or 0),
            "total_steps": int(self.manager.status.total_steps or 0),
            "message": self.manager.status.message,
            "run_id": self.manager.data_manager.current_run_id_str,
            "latest_sequence": latest_sequence,
            "results": results,
        }

    @staticmethod
    def _plan_values(item: Any) -> List[Any]:
        if isinstance(item, dict) and "sequence_parameters" in item:
            raw = item.get("sequence_parameters")
        else:
            raw = item
        if isinstance(raw, (list, tuple)):
            return list(raw)
        return [raw]

    def _master_plan(self, parameters: List[Any], sync_run_id: str, node_name: str) -> List[Any]:
        decorated = []
        for index, item in enumerate(parameters):
            values = self._plan_values(item)
            existing_metadata = item.get("metadata") if isinstance(item, dict) else {}
            decorated.append({
                "sequence_parameters": values,
                "metadata": {
                    **(existing_metadata or {}),
                    "sync_run_id": sync_run_id,
                    "sync_role": "master",
                    "sync_node_id": node_name,
                    "sync_shot_index": index,
                    "sync_p0": values[0] if values else None,
                    "sync_parameters": values,
                },
            })
        return decorated

    def start_master(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        settings = self.manager.get_settings()
        if str(settings.get("sync_role") or "standalone") != "master":
            raise ValueError("This controller is not configured as a Sync master")
        if self.manager.get_active_mode():
            raise ValueError(f"Master is busy with {self.manager.get_active_mode()}")
        with self._lock:
            if self._runtime.get("active"):
                raise ValueError("A Sync run is already active")

        slaves = [item for item in (payload.get("slaves") or []) if item.get("enabled", True)]
        if not slaves:
            raise ValueError("At least one enabled Sync slave is required")
        scan_config = dict(payload.get("scan_config") or {})
        parameters = self.manager.build_scan_parameter_plan(scan_config)
        if not parameters:
            raise ValueError("Sync scan contains no shots")
        shot_plan = [self._plan_values(item) for item in parameters]
        sync_run_id = f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        node_name = str(settings.get("sync_node_name") or "Master")
        delay_ms = max(0.0, float(payload.get("master_delay_ms") or 0.0))
        slave_states = []
        seen_node_ids = set()
        seen_urls = set()

        for raw_slave in slaves:
            node_id = self._safe_node_id(raw_slave.get("node_id") or raw_slave.get("name") or "slave")
            base_url = self._normalize_url(raw_slave.get("base_url"))
            if node_id in seen_node_ids:
                raise ValueError(f"Duplicate Slave node id: {node_id}")
            if base_url in seen_urls:
                raise ValueError(f"Duplicate Slave URL: {base_url}")
            seen_node_ids.add(node_id)
            seen_urls.add(base_url)
            prepare_payload = {
                "sync_run_id": sync_run_id,
                "master_node_id": node_name,
                "slave_node_id": node_id,
                "scan_config": scan_config,
                "shot_plan": shot_plan,
                "sequence_name": raw_slave.get("sequence_name") or "slave.mot",
                "sequence_content": raw_slave.get("sequence_content") or "",
                "sequence_content_base64": raw_slave.get("sequence_content_base64") or "",
            }
            response = requests.post(
                f"{base_url}/sync/node/prepare",
                json=prepare_payload,
                headers=self._headers(),
                timeout=8.0,
            )
            response.raise_for_status()
            if not response.json().get("ready"):
                raise ValueError(f"Slave {raw_slave.get('name') or node_id} did not become ready")
            slave_states.append({
                "node_id": node_id,
                "name": raw_slave.get("name") or node_id,
                "base_url": base_url,
                "status": "ready",
                "current_step": 0,
                "cursor": 0,
            })

        started_slaves = []
        try:
            for slave in slave_states:
                response = requests.post(
                    f"{slave['base_url']}/sync/node/start",
                    json={"sync_run_id": sync_run_id},
                    headers=self._headers(),
                    timeout=8.0,
                )
                response.raise_for_status()
                slave["status"] = "running"
                started_slaves.append(slave)
            if delay_ms:
                time.sleep(delay_ms / 1000.0)

            master_parameters = self._master_plan(parameters, sync_run_id, node_name)
            scan_config.update({
                "sync_run_id": sync_run_id,
                "sync_role": "master",
                "sync_master_delay_ms": delay_ms,
                "sync_shot_plan": shot_plan,
            })
            # Install the runtime before start_scan launches its worker. A fast
            # first shot can otherwise arrive before the listener knows this run.
            with self._lock:
                self._master_results = {}
                self._slave_results = {slave["node_id"]: {} for slave in slave_states}
                self._emitted_pairs = set()
                self._runtime = {
                    "active": True,
                    "sync_run_id": sync_run_id,
                    "status": "starting",
                    "message": "STARTING MASTER",
                    "master_delay_ms": delay_ms,
                    "expected_shots": len(shot_plan),
                    "master_step": 0,
                    "slaves": deepcopy(slave_states),
                    "paired_count": 0,
                    "started_at_ms": int(time.time() * 1000),
                    "finished_at_ms": None,
                    "stop_requested": False,
                    "master_run_id": "",
                    "master_run_dir": "",
                    "archive_replication": {"status": "pending", "nodes": {}},
                }
            result = self.manager.start_scan(scan_config, parameters_override=master_parameters)
            if result.get("status") == "error":
                raise ValueError(result.get("message") or "Master scan failed to start")
        except Exception as exc:
            for slave in started_slaves:
                try:
                    requests.post(
                        f"{slave['base_url']}/sync/node/stop",
                        json={"sync_run_id": sync_run_id}, headers=self._headers(), timeout=3.0,
                    )
                except Exception:
                    pass
            with self._lock:
                self._runtime["active"] = False
                self._runtime["status"] = "error"
                self._runtime["message"] = str(exc)
                self._runtime["finished_at_ms"] = int(time.time() * 1000)
            raise

        with self._lock:
            self._runtime["status"] = "running"
            self._runtime["message"] = "SYNC RUNNING"
            self._runtime["slaves"] = slave_states
            self._runtime["master_run_id"] = self.manager.data_manager.current_run_id_str
            self._runtime["master_run_dir"] = str(self.manager.data_manager.current_run_dir)
        self._write_archive_snapshot()
        self._monitor_thread = threading.Thread(target=self._monitor_master, name="miga-sync-monitor", daemon=True)
        self._monitor_thread.start()
        return self.status()

    def stop_master(self, reason: str = "User stop requested") -> Dict[str, Any]:
        with self._lock:
            runtime = deepcopy(self._runtime)
            if not runtime.get("active"):
                raise ValueError("No Sync run is active")
            self._runtime["status"] = "stopping"
            self._runtime["message"] = reason
            self._runtime["stop_requested"] = True
        sync_run_id = runtime.get("sync_run_id") or ""
        for slave in runtime.get("slaves") or []:
            try:
                requests.post(
                    f"{slave['base_url']}/sync/node/stop",
                    json={"sync_run_id": sync_run_id}, headers=self._headers(), timeout=3.0,
                )
                slave["status"] = "stopping"
            except Exception as exc:
                slave["status"] = "unreachable"
                slave["error"] = str(exc)
        delay_ms = float(runtime.get("master_delay_ms") or 0.0)
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        self.manager.stop_scan()
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            state = deepcopy(self._runtime)
        state["master_step"] = int(self.manager.status.current_step or state.get("master_step") or 0)
        state["master_running"] = bool(self.manager.status.is_running)
        return state

    def _capture_local_result(self, payload: Dict[str, Any]) -> None:
        sync_run_id = str(payload.get("sync_run_id") or "")
        role = str(payload.get("sync_role") or "")
        if not sync_run_id or role not in {"master", "slave"}:
            return
        if role == "slave":
            with self._lock:
                self._node_result_sequence += 1
                self._node_results.append((self._node_result_sequence, deepcopy(payload)))
            return
        with self._lock:
            if sync_run_id != self._runtime.get("sync_run_id"):
                return
            shot_index = int(payload.get("sync_shot_index", -1))
            if shot_index >= 0:
                self._master_results[shot_index] = deepcopy(payload)
                self._runtime["master_step"] = shot_index + 1
        self._emit_available_pairs()

    @staticmethod
    def _compact_result(payload: Dict[str, Any]) -> Dict[str, Any]:
        compact = {
            key: payload.get(key)
            for key in SYNC_RESULT_FIELDS
        }
        compact.update({
            "sync_node_id": payload.get("sync_node_id"),
            "sync_role": payload.get("sync_role"),
            "sync_shot_index": payload.get("sync_shot_index"),
            "sync_p0": payload.get("sync_p0"),
            "sync_parameters": payload.get("sync_parameters"),
            "sync_master_parameters": payload.get("sync_master_parameters"),
            "all_parameters": payload.get("all_parameters"),
            "timestamp": payload.get("timestamp"),
            "error": payload.get("error"),
        })
        return compact

    def _emit_available_pairs(self) -> None:
        emitted = []
        with self._lock:
            for slave_id, results in self._slave_results.items():
                for shot_index, slave_payload in results.items():
                    key = (slave_id, shot_index)
                    master_payload = self._master_results.get(shot_index)
                    if key in self._emitted_pairs or master_payload is None:
                        continue
                    if master_payload.get("error") or slave_payload.get("error"):
                        self._emitted_pairs.add(key)
                        continue
                    pair = {
                        "stream_type": "sync_pair",
                        "sync_run_id": self._runtime.get("sync_run_id"),
                        "sync_shot_index": shot_index,
                        "sync_p0": master_payload.get("sync_p0"),
                        "sync_parameters": master_payload.get("sync_parameters")
                            or master_payload.get("all_parameters")
                            or [master_payload.get("sync_p0")],
                        "slave_node_id": slave_id,
                        "master": self._compact_result(master_payload),
                        "slave": self._compact_result(slave_payload),
                    }
                    self._emitted_pairs.add(key)
                    emitted.append(pair)
            self._runtime["paired_count"] = sum(
                1
                for slave_id, shot_index in self._emitted_pairs
                if not self._master_results.get(shot_index, {}).get("error")
                and not self._slave_results.get(slave_id, {}).get(shot_index, {}).get("error")
            )
        for pair in emitted:
            self.manager.publish_data(pair, notify_listeners=False)
        if emitted:
            self._write_archive_snapshot()

    def _monitor_master(self) -> None:
        failed_reason = ""
        while True:
            with self._lock:
                if not self._runtime.get("active"):
                    return
                sync_run_id = self._runtime.get("sync_run_id")
                slaves = deepcopy(self._runtime.get("slaves") or [])
                expected = int(self._runtime.get("expected_shots") or 0)
                stop_requested = bool(self._runtime.get("stop_requested"))
            if stop_requested:
                break
            all_slaves_complete = True
            updated_states = []
            for slave in slaves:
                try:
                    response = requests.get(
                        f"{slave['base_url']}/sync/node/status/{sync_run_id}",
                        params={"after": int(slave.get("cursor") or 0)},
                        headers=self._headers(), timeout=3.0,
                    )
                    response.raise_for_status()
                    node_state = response.json()
                    for item in node_state.get("results") or []:
                        remote_payload = item.get("payload") or {}
                        remote_payload.update({
                            "sync_role": "slave",
                            "sync_node_id": slave["node_id"],
                            "sync_run_id": sync_run_id,
                        })
                        shot_index = int(remote_payload.get("sync_shot_index", -1))
                        if shot_index >= 0:
                            with self._lock:
                                self._slave_results.setdefault(slave["node_id"], {})[shot_index] = deepcopy(remote_payload)
                        self.manager.publish_data(remote_payload, notify_listeners=False)
                    slave["cursor"] = int(node_state.get("latest_sequence") or slave.get("cursor") or 0)
                    slave["current_step"] = int(node_state.get("current_step") or 0)
                    slave["run_id"] = node_state.get("run_id") or slave.get("run_id")
                    slave["status"] = "running" if node_state.get("is_running") else "done"
                    if not node_state.get("is_running") and slave["current_step"] < expected:
                        failed_reason = f"Slave {slave['name']} stopped before completing the shot plan"
                    all_slaves_complete = all_slaves_complete and slave["current_step"] >= expected
                except Exception as exc:
                    slave["status"] = "unreachable"
                    slave["error"] = str(exc)
                    failed_reason = f"Slave {slave['name']} disconnected"
                    all_slaves_complete = False
                updated_states.append(slave)
            with self._lock:
                self._runtime["slaves"] = updated_states
            self._emit_available_pairs()
            with self._lock:
                stop_requested = bool(self._runtime.get("stop_requested"))
            if stop_requested:
                failed_reason = ""
                break
            if failed_reason:
                self.stop_master(failed_reason)
                break
            master_complete = not self.manager.status.is_running and int(self.manager.status.current_step or 0) >= expected
            if master_complete and all_slaves_complete:
                break
            time.sleep(0.25)

        with self._lock:
            was_stopped = bool(self._runtime.get("stop_requested"))
            self._runtime["active"] = False
            self._runtime["status"] = "error" if failed_reason else ("stopped" if was_stopped else "done")
            self._runtime["message"] = failed_reason or ("SYNC STOPPED" if was_stopped else "SYNC DONE")
            self._runtime["finished_at_ms"] = int(time.time() * 1000)
        self._write_archive_snapshot()
        if not failed_reason and not was_stopped:
            self._replicate_archives()

    def _replicate_archives(self, run_dir: Optional[Path] = None, runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            state = deepcopy(runtime or self._runtime)
        sync_run_id = str(state.get("sync_run_id") or "")
        if not sync_run_id:
            raise ValueError("Sync run id is missing")
        master_run_dir = Path(run_dir or state.get("master_run_dir") or self._find_sync_run_dir(sync_run_id)).resolve()
        replication = {
            "status": "running",
            "started_at_ms": int(time.time() * 1000),
            "updated_at_ms": int(time.time() * 1000),
            "nodes": {},
        }
        with self._lock:
            self._runtime["archive_replication"] = deepcopy(replication)
        self._update_replication_manifest(master_run_dir, state, replication)

        for slave in state.get("slaves") or []:
            node_id = self._safe_node_id(slave.get("node_id") or slave.get("name"))
            node_state = {
                "status": "running", "pull_status": "pulling", "push_status": "pending",
                "name": slave.get("name") or node_id,
            }
            replication["nodes"][node_id] = node_state
            temp_path = None
            try:
                response = requests.get(
                    f"{slave['base_url']}/sync/node/archive/{sync_run_id}",
                    headers=self._headers(), timeout=(8.0, 300.0), stream=True,
                )
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(prefix=f"miga_{node_id}_", suffix=".zip", delete=False) as handle:
                    temp_path = Path(handle.name)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                installed = self._install_archive_bundle(
                    master_run_dir,
                    node_id,
                    temp_path,
                    response.headers.get("X-MIGA-Archive-SHA256", ""),
                )
                self._merge_newer_replica_phase_metadata(master_run_dir, installed["path"])
                node_state.update({"pull_status": "complete", "remote_run_id": response.headers.get("X-MIGA-Archive-Run-Id", ""), **installed})
            except Exception as exc:
                node_state.update({"pull_status": "failed", "pull_error": str(exc)})
            finally:
                if temp_path:
                    temp_path.unlink(missing_ok=True)

        replication["updated_at_ms"] = int(time.time() * 1000)
        with self._lock:
            self._runtime["archive_replication"] = deepcopy(replication)
        self._update_replication_manifest(master_run_dir, state, replication)

        master_bundle = None
        try:
            master_bundle, master_meta = self.build_archive_bundle(sync_run_id)
            for slave in state.get("slaves") or []:
                node_id = self._safe_node_id(slave.get("node_id") or slave.get("name"))
                node_state = replication["nodes"].setdefault(node_id, {})
                node_state["push_status"] = "pushing"
                try:
                    with master_bundle.open("rb") as handle:
                        response = requests.post(
                            f"{slave['base_url']}/sync/node/archive/{sync_run_id}/master",
                            headers={**self._headers(), "X-MIGA-Archive-SHA256": master_meta["sha256"]},
                            files={"archive": ("master.zip", handle, "application/zip")},
                            timeout=(8.0, 300.0),
                        )
                    response.raise_for_status()
                    node_state["push_status"] = "complete"
                    node_state["master_push"] = response.json()
                except Exception as exc:
                    node_state.update({"push_status": "failed", "push_error": str(exc)})
        except Exception as exc:
            for node_state in replication["nodes"].values():
                if node_state.get("push_status") != "complete":
                    node_state.update({"push_status": "failed", "push_error": str(exc)})
        finally:
            if master_bundle:
                master_bundle.unlink(missing_ok=True)

        for node_state in replication["nodes"].values():
            node_state["status"] = "complete" if (
                node_state.get("pull_status") == "complete" and node_state.get("push_status") == "complete"
            ) else "incomplete"
        replication["status"] = "complete" if replication["nodes"] and all(
            item.get("status") == "complete" for item in replication["nodes"].values()
        ) else "incomplete"
        replication["updated_at_ms"] = int(time.time() * 1000)
        with self._lock:
            self._runtime["archive_replication"] = deepcopy(replication)
        self._update_replication_manifest(master_run_dir, state, replication)
        return deepcopy(replication)

    def _update_replication_manifest(
        self,
        run_dir: Path,
        runtime: Dict[str, Any],
        replication: Dict[str, Any],
    ) -> None:
        manifest_path = Path(run_dir) / "sync_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {"runtime": deepcopy(runtime), "pairs": [], "node_results": {}}
        saved_runtime = dict(manifest.get("runtime") or runtime)
        saved_runtime["archive_replication"] = deepcopy(replication)
        manifest["runtime"] = saved_runtime
        manifest["archive_replication"] = deepcopy(replication)
        archive_nodes = dict(manifest.get("archive_nodes") or {})
        archive_nodes["master"] = {
            "node_id": "master", "role": "master", "local": True,
            "path": ".", "run_id": Path(run_dir).name,
        }
        for slave in runtime.get("slaves") or []:
            node_id = self._safe_node_id(slave.get("node_id") or slave.get("name"))
            node_state = replication.get("nodes", {}).get(node_id) or {}
            replica_path = Path(run_dir) / "sync_nodes" / node_id
            if replica_path.is_dir():
                archive_nodes[node_id] = {
                    "node_id": node_id, "role": "slave", "local": False,
                    "path": f"sync_nodes/{node_id}",
                    "run_id": node_state.get("remote_run_id") or node_id,
                    "name": slave.get("name") or node_id,
                    "sha256": node_state.get("sha256"),
                }
        manifest["archive_nodes"] = archive_nodes
        self._atomic_write_json(manifest_path, manifest)

    def retry_archive_replication(self, run_dir: Path) -> Dict[str, Any]:
        manifest_path = Path(run_dir) / "sync_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Sync manifest not found")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive_nodes = manifest.get("archive_nodes") or {}
        runtime = dict(manifest.get("runtime") or {})
        is_legacy_master_archive = not archive_nodes and bool(
            runtime.get("sync_run_id") and runtime.get("slaves")
        )
        if not is_legacy_master_archive and archive_nodes.get("master", {}).get("local") is not True:
            raise ValueError("Archive replication can only be retried from the Master controller")
        runtime["master_run_dir"] = str(Path(run_dir).resolve())
        return self._replicate_archives(Path(run_dir), runtime)

    @staticmethod
    def _archive_manifest(run_dir: Path) -> Dict[str, Any]:
        path = Path(run_dir) / "sync_manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def archive_local_node_id(self, run_dir: Path) -> str:
        manifest = self._archive_manifest(run_dir)
        for node_id, entry in (manifest.get("archive_nodes") or {}).items():
            if isinstance(entry, dict) and entry.get("local") is True:
                return str(node_id)
        return "master" if str((manifest.get("runtime") or {}).get("sync_role") or "") == "master" else ""

    def _archive_node_base_url(
        self, run_dir: Path, node_id: str, coordinator_url: str = ""
    ) -> str:
        manifest = self._archive_manifest(run_dir)
        target = str(node_id or "").strip()
        if target == "master":
            return str(coordinator_url or "").rstrip("/")
        for slave in (manifest.get("runtime") or {}).get("slaves") or []:
            candidate = self._safe_node_id(slave.get("node_id") or slave.get("name"))
            if candidate == target:
                return self._normalize_url(slave.get("base_url"))
        return ""

    def get_archive_node_phase_calibrations(
        self,
        run_dir: Path,
        node_id: str,
        coordinator_url: str = "",
    ) -> Dict[str, Any]:
        local_node = self.archive_local_node_id(run_dir)
        target = str(node_id or local_node or "master")
        if target == local_node:
            active = self.manager.get_active_bragg_phase_calibration()
            return {
                "node_id": target,
                "source": "local_settings",
                "calibrations": self.manager.get_bragg_phase_calibrations(),
                "active": active,
            }
        base_url = self._archive_node_base_url(run_dir, target, coordinator_url)
        if not base_url:
            raise ValueError(f"Controller URL for SYNC node {target} is unavailable")
        response = requests.get(
            f"{base_url}/sync/node/phase-calibrations",
            headers=self._headers(),
            timeout=8.0,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "node_id": target,
            "source": "remote_settings",
            "calibrations": payload.get("calibrations") or [],
            "active": payload.get("active"),
        }

    def distribute_phase_analysis_metadata(
        self,
        run_dir: Path,
        metadata: Dict[str, Any],
        coordinator_url: str,
    ) -> Dict[str, Any]:
        manifest = self._archive_manifest(run_dir)
        runtime = manifest.get("runtime") or {}
        sync_run_id = str(runtime.get("sync_run_id") or "")
        if not sync_run_id:
            raise ValueError("SYNC run id is missing from the Archive")
        if self.archive_local_node_id(run_dir) != "master":
            raise ValueError("Only the Master controller can distribute SYNC phase metadata")
        wire = deepcopy(metadata)
        wire.update({
            "version": 2,
            "sync_status": "synced",
            "sync_message": "",
            "coordinator_url": str(coordinator_url or "").rstrip("/"),
        })
        nodes: Dict[str, Any] = {}
        for slave in runtime.get("slaves") or []:
            node_id = self._safe_node_id(slave.get("node_id") or slave.get("name"))
            try:
                response = requests.post(
                    f"{self._normalize_url(slave.get('base_url'))}/sync/node/archive-phase-analysis/{sync_run_id}",
                    json=wire,
                    headers=self._headers(),
                    timeout=8.0,
                )
                response.raise_for_status()
                nodes[node_id] = {"status": "synced", **(response.json() or {})}
            except Exception as exc:
                nodes[node_id] = {"status": "pending", "error": str(exc)}
        complete = bool(nodes) and all(item.get("status") == "synced" for item in nodes.values())
        return {
            "status": "synced" if complete else "pending",
            "message": "" if complete else "One or more SYNC nodes did not receive the phase metadata",
            "nodes": nodes,
            "coordinator_url": wire["coordinator_url"],
        }

    def advertised_phase_coordinator_url(self, run_dir: Path, request_base_url: str) -> str:
        value = str(request_base_url or "").rstrip("/")
        parsed = urlparse(value)
        hostname = str(parsed.hostname or "").lower()
        if hostname not in {"localhost", "127.0.0.1", "::1"}:
            return value
        manifest = self._archive_manifest(run_dir)
        for slave in (manifest.get("runtime") or {}).get("slaves") or []:
            peer = urlparse(self._normalize_url(slave.get("base_url")))
            if not peer.hostname:
                continue
            try:
                family = socket.AF_INET6 if ":" in peer.hostname else socket.AF_INET
                with socket.socket(family, socket.SOCK_DGRAM) as probe:
                    probe.connect((peer.hostname, peer.port or 80))
                    local_ip = str(probe.getsockname()[0])
                host = f"[{local_ip}]" if ":" in local_ip else local_ip
                netloc = f"{host}:{parsed.port}" if parsed.port else host
                return urlunparse((parsed.scheme or "http", netloc, parsed.path, "", "", "")).rstrip("/")
            except OSError:
                continue
        return value

    def forward_phase_analysis_metadata(
        self,
        sync_run_id: str,
        metadata: Dict[str, Any],
        coordinator_url: str,
    ) -> Dict[str, Any]:
        target = str(coordinator_url or "").rstrip("/")
        if not target:
            raise ValueError("Master coordinator URL is unavailable; update from the Master Archive first")
        response = requests.post(
            f"{target}/sync/node/archive-phase-analysis/{sync_run_id}/merge",
            json=metadata,
            headers=self._headers(),
            timeout=12.0,
        )
        response.raise_for_status()
        return response.json()

    def _write_archive_snapshot(self) -> None:
        with self._lock:
            configured_run_dir = self._runtime.get("master_run_dir")
        run_dir = Path(configured_run_dir) if configured_run_dir else self.manager.data_manager.current_run_dir
        if not run_dir:
            return
        with self._lock:
            runtime = deepcopy(self._runtime)
            pairs = []
            for slave_id, shot_index in sorted(self._emitted_pairs, key=lambda item: (item[1], item[0])):
                master = self._master_results.get(shot_index)
                slave = self._slave_results.get(slave_id, {}).get(shot_index)
                if master is None or slave is None:
                    continue
                if master.get("error") or slave.get("error"):
                    continue
                pairs.append({
                    "sync_shot_index": shot_index,
                    "sync_p0": master.get("sync_p0"),
                    "sync_parameters": master.get("sync_parameters")
                        or master.get("all_parameters")
                        or [master.get("sync_p0")],
                    "slave_node_id": slave_id,
                    "master": self._compact_result(master),
                    "slave": self._compact_result(slave),
                })
            node_results = {
                "master": [
                    self._compact_result(payload)
                    for _, payload in sorted(self._master_results.items())
                ],
                "slaves": {
                    slave_id: [
                        self._compact_result(payload)
                        for _, payload in sorted(results.items())
                    ]
                    for slave_id, results in self._slave_results.items()
                },
            }
            archive_nodes = {
                "master": {
                    "node_id": "master", "role": "master", "local": True,
                    "path": ".", "run_id": Path(run_dir).name,
                }
            }
            for slave in runtime.get("slaves") or []:
                node_id = self._safe_node_id(slave.get("node_id") or slave.get("name"))
                replica_path = Path(run_dir) / "sync_nodes" / node_id
                if replica_path.is_dir():
                    archive_nodes[node_id] = {
                        "node_id": node_id, "role": "slave", "local": False,
                        "path": f"sync_nodes/{node_id}", "run_id": replica_path.name,
                        "name": slave.get("name") or node_id,
                    }
        target = Path(run_dir) / "sync_manifest.json"
        self._atomic_write_json(target, {
            "runtime": runtime,
            "pairs": pairs,
            "node_results": node_results,
            "archive_nodes": archive_nodes,
            "archive_replication": runtime.get("archive_replication") or {"status": "idle", "nodes": {}},
        })
