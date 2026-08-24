from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime
import base64
import binascii
import json
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from uuid import uuid4

import requests

import config
from app.core.experiment_manager import ExperimentManager


SYNC_RESULT_FIELDS = (
    "atom_number_up", "atom_number_dw", "amplitude_up", "amplitude_dw",
    "transition_probability_up", "transition_probability_dw",
    "intf_n1", "intf_n2", "intf_p1", "intf_p2",
    "atom_number_up_nofit", "atom_number_dw_nofit",
    "amplitude_up_nofit", "amplitude_dw_nofit",
    "transition_probability_up_nofit", "transition_probability_dw_nofit",
    "intf_n1_nofit", "intf_n2_nofit", "intf_p1_nofit", "intf_p2_nofit",
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
        }

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
            node_id = str(raw_slave.get("node_id") or raw_slave.get("name") or "slave").strip()
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

    def _write_archive_snapshot(self) -> None:
        run_dir = self.manager.data_manager.current_run_dir
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
                    "slave_node_id": slave_id,
                    "master": self._compact_result(master),
                    "slave": self._compact_result(slave),
                })
        target = Path(run_dir) / "sync_manifest.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"runtime": runtime, "pairs": pairs}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)
