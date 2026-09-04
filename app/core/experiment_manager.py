import time
import base64
import threading
import traceback
import json
import random
import math
import queue
import os
import shutil
from copy import deepcopy
import shlex
import subprocess
import numpy as np
from pathlib import Path
from itertools import product
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import asdict

import config
from app.drivers.hardware import SequenceEditor, ExperimentDriver, RedPitayaDriver
from app.drivers import dds_table
from app.drivers.vcd_parser import VCDParser
from app.analysis import fitting, physics, interferometer_phase
from app.analysis.lock_in import build_lock_in_analysis
from app.analysis.transfer_function import build_transfer_function_summary
from app.models.schemas import ScanConfig
from app.core.data_manager import DataManager
from app.core.structures import ExperimentStatus, ScanResult
from app.core.pulse_generator import generate_bragg_pulse
from app.core.sequence_markers import (
    decode_mot_bytes,
    marker_definitions_for_sequence,
    normalize_marker_profiles,
    normalize_marker_definitions,
    validate_auto_marker_scan,
)
from app.drivers.tti_generator import TtiConnectionSettings, TtiGeneratorClient

class ExperimentManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ExperimentManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'): return
        self.initialized = True

        self.seq_editor = SequenceEditor()
        self.driver = ExperimentDriver()

        self.rp_driver_red = RedPitayaDriver(ip_address=config.RP_IP_RED_REAL, timeout=config.NETWORK_TIMEOUT)
        self.rp_driver_green = RedPitayaDriver(ip_address=config.RP_IP_GREEN_REAL, timeout=config.NETWORK_TIMEOUT)

        self.data_manager = DataManager()
        self.settings = self._load_initial_settings()
        self._apply_runtime_settings()

        self.status = ExperimentStatus()
        self.stop_flag = False
        
        self.data_queue = queue.Queue()
        self.acq_thread: Optional[threading.Thread] = None
        self.proc_thread: Optional[threading.Thread] = None
        self.on_data_ready: Optional[Callable[[Dict[str, Any]], None]] = None
        self._data_listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._activity_lock = threading.Lock()
        self._active_mode: Optional[str] = None
        self._scan_finalize_error: Optional[str] = None
        self._active_phase_calibration_for_run: Optional[Dict[str, Any]] = None

    def _default_tmot_args(self) -> str:
        return config.TMOT_EXTRA_ARGS_WIN if config.IS_WINDOWS else config.TMOT_EXTRA_ARGS_LINUX

    def _git_base_command(self) -> List[str]:
        return [
            "git",
            "-c",
            f"safe.directory={config.BASE_DIR}",
            "-C",
            str(config.BASE_DIR),
        ]

    def _run_git_command(self, args: List[str]) -> str:
        completed = subprocess.run(
            self._git_base_command() + args,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode != 0:
            raise ValueError(stderr or stdout or f"Git command failed: {' '.join(args)}")
        return stdout

    def _git_ref_exists(self, ref_name: str) -> bool:
        completed = subprocess.run(
            self._git_base_command() + ["rev-parse", "--verify", "--quiet", ref_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode == 0

    def _get_git_remote_url(self) -> str:
        try:
            return self._run_git_command(["remote", "get-url", "origin"])
        except Exception:
            return ""

    def _get_git_current_branch(self) -> str:
        try:
            return self._run_git_command(["branch", "--show-current"])
        except Exception:
            return ""

    def _list_git_branches(self) -> List[str]:
        refs = self._run_git_command([
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads",
            "refs/remotes/origin",
        ])
        branches: List[str] = []
        seen = set()
        for raw_line in refs.splitlines():
            branch_name = raw_line.strip()
            if not branch_name or branch_name == "origin/HEAD":
                continue
            if branch_name.startswith("origin/"):
                branch_name = branch_name[len("origin/"):]
            if branch_name not in seen:
                seen.add(branch_name)
                branches.append(branch_name)
        return sorted(branches)

    def _get_git_commit(self, ref_name: str) -> str:
        try:
            return self._run_git_command(["rev-parse", ref_name])
        except Exception:
            return ""

    def _get_git_commit_short(self, ref_name: str) -> str:
        try:
            return self._run_git_command(["rev-parse", "--short", ref_name])
        except Exception:
            return ""

    def _get_git_commit_subject(self, ref_name: str) -> str:
        try:
            return self._run_git_command(["log", "-1", "--pretty=%s", ref_name])
        except Exception:
            return ""

    def _get_git_ahead_behind(self, local_ref: str, remote_ref: str) -> Tuple[Optional[int], Optional[int]]:
        if not local_ref or not remote_ref:
            return None, None
        try:
            counts = self._run_git_command(["rev-list", "--left-right", "--count", f"{local_ref}...{remote_ref}"])
            ahead_raw, behind_raw = (counts.split() + ["0", "0"])[:2]
            return int(ahead_raw), int(behind_raw)
        except Exception:
            return None, None

    def _build_update_comparison(self, current_branch: str, current_commit: str, configured_branch: str) -> Dict[str, Any]:
        comparison_branch = str(configured_branch or current_branch or "main").strip() or "main"
        local_ref = "HEAD" if current_branch == comparison_branch and current_commit else ""
        local_exists = bool(local_ref)
        if not local_exists and self._git_ref_exists(f"refs/heads/{comparison_branch}"):
            local_ref = f"refs/heads/{comparison_branch}"
            local_exists = True

        remote_ref = f"refs/remotes/origin/{comparison_branch}"
        remote_exists = self._git_ref_exists(remote_ref)

        local_commit = current_commit if current_branch == comparison_branch and current_commit else self._get_git_commit(local_ref)
        local_commit_short = self._get_git_commit_short(local_ref) if local_ref else ""
        remote_commit = self._get_git_commit(remote_ref) if remote_exists else ""
        remote_commit_short = self._get_git_commit_short(remote_ref) if remote_exists else ""
        remote_subject = self._get_git_commit_subject(remote_ref) if remote_exists else ""
        ahead, behind = self._get_git_ahead_behind(local_ref, remote_ref) if local_exists and remote_exists else (None, None)

        branch_matches_checkout = bool(current_branch) and current_branch == comparison_branch
        is_latest = branch_matches_checkout and remote_exists and ahead == 0 and behind == 0

        if not remote_exists:
            version_status = "unknown"
            version_message = f"Remote branch origin/{comparison_branch} is not available locally yet. Click Refresh Remote to fetch the latest GitHub state."
        elif not branch_matches_checkout:
            version_status = "branch_mismatch"
            version_message = f"Current checkout is on {current_branch or '-'}, while the selected update branch is {comparison_branch}."
        elif ahead and behind:
            version_status = "diverged"
            version_message = f"Local checkout has diverged from origin/{comparison_branch} ({ahead} ahead, {behind} behind)."
        elif behind:
            version_status = "behind"
            version_message = f"Local checkout is behind origin/{comparison_branch} by {behind} commit(s)."
        elif ahead:
            version_status = "ahead"
            version_message = f"Local checkout is ahead of origin/{comparison_branch} by {ahead} commit(s)."
        else:
            version_status = "latest"
            version_message = f"Local checkout is up to date with origin/{comparison_branch}."

        return {
            "comparison_branch": comparison_branch,
            "comparison_branch_matches_checkout": branch_matches_checkout,
            "comparison_local_exists": local_exists,
            "comparison_local_commit": local_commit,
            "comparison_local_commit_short": local_commit_short,
            "comparison_remote_exists": remote_exists,
            "comparison_remote_commit": remote_commit,
            "comparison_remote_commit_short": remote_commit_short,
            "comparison_remote_subject": remote_subject,
            "comparison_ahead": ahead,
            "comparison_behind": behind,
            "is_latest": is_latest,
            "version_status": version_status,
            "version_message": version_message,
        }

    def _is_allowed_update_repo_url(self, repo_url: str) -> bool:
        normalized = str(repo_url or "").strip()
        if not normalized:
            return False
        return normalized.startswith("https://github.com/") or normalized.startswith("git@github.com:")

    def _normalize_update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        remote_url = self._get_git_remote_url()
        current_branch = self._get_git_current_branch()
        repo_url = str(settings.get("update_repo_url") or remote_url or "").strip()
        branch = str(settings.get("update_branch") or current_branch or "main").strip() or "main"

        if repo_url and not self._is_allowed_update_repo_url(repo_url):
            repo_url = remote_url

        settings["update_repo_url"] = repo_url
        settings["update_branch"] = branch
        return settings

    def get_update_status(self) -> Dict[str, Any]:
        remote_url = self._get_git_remote_url()
        current_branch = self._get_git_current_branch()
        current_commit = self._run_git_command(["rev-parse", "HEAD"])
        current_commit_short = self._run_git_command(["rev-parse", "--short", "HEAD"])
        last_commit_subject = self._run_git_command(["log", "-1", "--pretty=%s"])
        dirty_entries = [line for line in self._run_git_command(["status", "--porcelain"]).splitlines() if line.strip()]
        configured_repo_url = str(self.settings.get("update_repo_url") or remote_url or "").strip()
        configured_branch = str(self.settings.get("update_branch") or current_branch or "main").strip() or "main"
        branches = self._list_git_branches()
        comparison = self._build_update_comparison(current_branch, current_commit, configured_branch)
        return {
            "repo_root": str(config.BASE_DIR),
            "origin_url": remote_url,
            "configured_repo_url": configured_repo_url,
            "configured_branch": configured_branch,
            "current_branch": current_branch,
            "current_commit": current_commit,
            "current_commit_short": current_commit_short,
            "last_commit_subject": last_commit_subject,
            "branches": branches,
            "dirty": bool(dirty_entries),
            "dirty_entries": dirty_entries,
            "reload_expected": True,
            **comparison,
        }

    def fetch_update_metadata(self, repo_url: Optional[str] = None, branch: Optional[str] = None) -> Dict[str, Any]:
        target_repo_url = str(repo_url or self.settings.get("update_repo_url") or self._get_git_remote_url() or "").strip()
        target_branch = str(branch or self.settings.get("update_branch") or self._get_git_current_branch() or "main").strip() or "main"
        if not self._is_allowed_update_repo_url(target_repo_url):
            raise ValueError("Update repo URL must be a GitHub HTTPS or SSH URL")

        current_remote = self._get_git_remote_url()
        if current_remote != target_repo_url:
            self._run_git_command(["remote", "set-url", "origin", target_repo_url])

        self._run_git_command(["fetch", "--prune", "origin"])
        self.settings["update_repo_url"] = target_repo_url
        self.settings["update_branch"] = target_branch
        self._save_settings_to_disk()
        return self.get_update_status()

    def apply_system_update(self, repo_url: Optional[str] = None, branch: Optional[str] = None) -> Dict[str, Any]:
        status = self.fetch_update_metadata(repo_url=repo_url, branch=branch)
        target_repo_url = status.get("configured_repo_url") or status.get("origin_url")
        target_branch = str(branch or status.get("configured_branch") or status.get("current_branch") or "main").strip() or "main"

        local_exists = self._git_ref_exists(f"refs/heads/{target_branch}")
        remote_exists = self._git_ref_exists(f"refs/remotes/origin/{target_branch}")
        if not local_exists and not remote_exists:
            raise ValueError(f"Branch '{target_branch}' was not found in origin")

        current_branch = status.get("current_branch") or ""
        if current_branch != target_branch:
            if local_exists:
                self._run_git_command(["checkout", target_branch])
            else:
                self._run_git_command(["checkout", "-b", target_branch, "--track", f"origin/{target_branch}"])

        pull_output = self._run_git_command(["pull", "--ff-only", "origin", target_branch]) if remote_exists else ""
        self.settings["update_repo_url"] = target_repo_url
        self.settings["update_branch"] = target_branch
        self._save_settings_to_disk()

        updated_status = self.get_update_status()
        updated_status["pull_output"] = pull_output
        updated_status["requested_repo_url"] = target_repo_url
        updated_status["requested_branch"] = target_branch
        updated_status["reload_expected"] = True
        return updated_status

    def _build_run_display_name(self, run_id: str, run_label: str) -> str:
        clean_label = str(run_label or "").strip()
        return f"{run_id} | {clean_label}" if clean_label else run_id

    def _load_run_preset_config(self, run_dir: Path) -> Dict[str, Any]:
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found for run: {run_dir.name}")

        with open(config_path, "r") as handle:
            stored_config = json.load(handle)

        defaults = ScanConfig().dict()
        allowed_keys = set(defaults.keys())
        restored = {**defaults}
        restored.update({key: value for key, value in stored_config.items() if key in allowed_keys})
        restored["run_label"] = str(restored.get("run_label") or "").strip()
        restored["sequence_name"] = str(restored.get("sequence_name") or "").strip()
        restored["scan_dimensions"] = self._resolve_scan_dimensions(restored)
        restored["dim2_enabled"] = restored["scan_dimensions"] >= 2
        restored["dim3_enabled"] = restored["scan_dimensions"] >= 3
        return restored

    def _load_sync_run_preset(self, run_dir: Path) -> Dict[str, Any]:
        manifest_path = run_dir / "sync_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Selected run is not a SYNC run")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("Selected SYNC run has an invalid sync_manifest.json") from exc

        if str(self.settings.get("sync_role") or "standalone").strip().lower() != "master":
            raise ValueError("SYNC presets can only be loaded on a controller configured as Master")

        runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
        archive_nodes = manifest.get("archive_nodes") if isinstance(manifest.get("archive_nodes"), dict) else {}
        runtime_slaves = runtime.get("slaves") if isinstance(runtime.get("slaves"), list) else []
        historical_nodes: Dict[str, Dict[str, Any]] = {}
        for item in runtime_slaves:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id") or "").strip()
            if node_id:
                historical_nodes[node_id] = dict(item)
        for node_id, item in archive_nodes.items():
            if node_id == "master" or not isinstance(item, dict):
                continue
            historical_nodes.setdefault(str(node_id), dict(item))
        if not historical_nodes:
            raise ValueError("Selected SYNC run does not contain any archived Slave nodes")

        configured_slaves = {
            str(item.get("id") or "").strip(): dict(item)
            for item in (self.settings.get("sync_slaves") or [])
            if isinstance(item, dict) and item.get("enabled", True) and str(item.get("id") or "").strip()
        }

        def normalized_url(value: Any) -> str:
            result = str(value or "").strip().lower().rstrip("/")
            for prefix in ("http://", "https://"):
                if result.startswith(prefix):
                    result = result[len(prefix):]
                    break
            return result

        def normalized_name(value: Any) -> str:
            return " ".join(str(value or "").strip().casefold().split())

        node_mapping: Dict[str, Tuple[str, str]] = {}
        unused_configured_ids = set(configured_slaves)
        unmatched_historical = []
        for historical_id, historical in historical_nodes.items():
            if historical_id in unused_configured_ids:
                node_mapping[historical_id] = (historical_id, "id")
                unused_configured_ids.remove(historical_id)
                continue
            historical_url = normalized_url(historical.get("base_url"))
            url_matches = [
                node_id for node_id in unused_configured_ids
                if historical_url and normalized_url(configured_slaves[node_id].get("base_url")) == historical_url
            ]
            if len(url_matches) == 1:
                current_id = url_matches[0]
                node_mapping[historical_id] = (current_id, "url")
                unused_configured_ids.remove(current_id)
                continue
            historical_name = normalized_name(historical.get("name"))
            name_matches = [
                node_id for node_id in unused_configured_ids
                if historical_name and normalized_name(configured_slaves[node_id].get("name")) == historical_name
            ]
            if len(name_matches) == 1:
                current_id = name_matches[0]
                node_mapping[historical_id] = (current_id, "name")
                unused_configured_ids.remove(current_id)
                continue
            unmatched_historical.append(historical_id)

        if unmatched_historical or unused_configured_ids:
            details = []
            if unmatched_historical:
                details.append("unmatched archived Slaves: " + ", ".join(sorted(unmatched_historical)))
            if unused_configured_ids:
                details.append("current Slaves not present in the run: " + ", ".join(sorted(unused_configured_ids)))
            raise ValueError(
                "SYNC Slave configuration does not match the archived run ("
                + "; ".join(details)
                + "). Matching uses ID, then controller URL, then node name."
            )

        run_root = run_dir.resolve()
        restored_slaves = []
        missing_sequences = []
        for historical_id in sorted(historical_nodes):
            current_id, match_method = node_mapping[historical_id]
            archive_entry = archive_nodes.get(historical_id) if isinstance(archive_nodes.get(historical_id), dict) else {}
            relative_path = str(archive_entry.get("path") or f"sync_nodes/{historical_id}")
            node_run_dir = (run_dir / relative_path).resolve()
            if node_run_dir != run_root and run_root not in node_run_dir.parents:
                raise ValueError(f"Archived path for Slave {historical_id} is invalid")
            sequence_path = node_run_dir / "sequence.mot"
            if not sequence_path.is_file():
                missing_sequences.append(historical_id)
                continue
            node_config = {}
            try:
                node_config = json.loads((node_run_dir / "config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            configured = configured_slaves[current_id]
            restored_slaves.append({
                "node_id": current_id,
                "archive_node_id": historical_id,
                "match_method": match_method,
                "name": configured.get("name") or historical_nodes[historical_id].get("name") or current_id,
                "sequence_name": str(node_config.get("sequence_name") or sequence_path.name),
                "sequence_content_base64": base64.b64encode(sequence_path.read_bytes()).decode("ascii"),
            })
        if missing_sequences:
            raise ValueError(
                "Archived Slave sequence.mot is missing for: " + ", ".join(missing_sequences)
                + ". Run Sync Historical Archive first."
            )
        return {
            "sync_run_id": str(runtime.get("sync_run_id") or ""),
            "master_delay_ms": max(0.0, float(runtime.get("master_delay_ms") or 0.0)),
            "slaves": restored_slaves,
        }

    def load_run_preset(
        self, year: str, month: str, day: str, run_id: str, include_sync: bool = False
    ) -> Dict[str, Any]:
        if getattr(self.status, "is_running", False):
            raise ValueError("Cannot load a previous run while an experiment is running")

        run_dir = Path(config.DATA_BASE_DIR) / year / month / day / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        restored_config = self._load_run_preset_config(run_dir)
        source_sequence = run_dir / "sequence.mot"
        if include_sync and not source_sequence.is_file():
            raise ValueError("Archived Master sequence.mot is missing")
        sync_preset = self._load_sync_run_preset(run_dir) if include_sync else None
        target_sequence = Path(config.SEQUENCE_TEMPLATE_PATH_WIN if config.IS_WINDOWS else config.SEQUENCE_TEMPLATE_PATH_LINUX)
        sequence_loaded = False
        sequence_name = restored_config.get("sequence_name") or source_sequence.name

        if source_sequence.exists():
            target_sequence.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_sequence, target_sequence)
            sequence_loaded = True

        restored_config["sequence_name"] = sequence_name
        run_label = restored_config.get("run_label", "")
        return {
            "config": restored_config,
            "sequence_loaded": sequence_loaded,
            "sequence_name": sequence_name,
            "run_id": run_id,
            "run_label": run_label,
            "display_name": self._build_run_display_name(run_id, run_label),
            "sync_preset": sync_preset,
        }

    def _normalize_tmot_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        tmot_path = (settings.get("tmot_path") or "").strip()
        tmot_args = settings.get("tmot_args")

        if tmot_args is None:
            tmot_args = self._default_tmot_args()

        # Backward compatibility for legacy values like "/path/to/tmot4 -e".
        if tmot_path and not os.path.exists(tmot_path):
            try:
                parts = shlex.split(tmot_path)
            except ValueError:
                parts = []

            if parts:
                settings["tmot_path"] = parts[0]
                if len(parts) > 1 and not str(tmot_args).strip():
                    tmot_args = " ".join(parts[1:])

        settings["tmot_args"] = tmot_args
        return settings

    def _infer_hardware_platform(self, settings: Dict[str, Any]) -> str:
        platform_name = str(settings.get("hardware_platform") or "").strip().lower()
        if platform_name in {"redpitaya", "daq"}:
            return platform_name

        rp_ip_red = str(settings.get("rp_ip_red") or "").strip().lower()
        if rp_ip_red.startswith("127.0.0.1") or rp_ip_red.startswith("localhost") or ":" in rp_ip_red:
            return "daq"
        return "redpitaya"

    def _normalize_hardware_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        settings["hardware_platform"] = self._infer_hardware_platform(settings)

        try:
            daq_rate = float(settings.get("daq_rate", 0) or 0)
        except (TypeError, ValueError):
            daq_rate = 0.0

        if daq_rate <= 0:
            try:
                decimation = int(settings.get("decimation", config.DEFAULT_ANALYSIS_SETTINGS.get("decimation", 8192)))
            except (TypeError, ValueError):
                decimation = int(config.DEFAULT_ANALYSIS_SETTINGS.get("decimation", 8192))
            settings["daq_rate"] = round(125000000 / decimation, 6) if decimation > 0 else 0.0
        else:
            settings["daq_rate"] = daq_rate

        return settings

    def _apply_runtime_settings(self):
        self.rp_driver_red.real_ip = self.settings.get("rp_ip_red", config.RP_IP_RED_REAL)
        self.rp_driver_red.timeout = int(self.settings.get("network_timeout", config.NETWORK_TIMEOUT))
        self.rp_driver_green.real_ip = self.settings.get("rp_ip_green", config.RP_IP_GREEN_REAL)
        self.rp_driver_green.timeout = int(self.settings.get("network_timeout", config.NETWORK_TIMEOUT))

    def _load_initial_settings(self) -> Dict[str, Any]:
        # 1. 定义所有参数的默认值
        base_settings = {
            "hardware_platform": "redpitaya",
            "voltage_limit": 0.015,
            "rp_ip_red": config.RP_IP_RED_REAL,
            "rp_ip_green": config.RP_IP_GREEN_REAL,
            "daq_rate": round(125000000 / config.DEFAULT_ANALYSIS_SETTINGS.get("decimation", 8192), 6),
            "network_timeout": config.NETWORK_TIMEOUT,
            "g_const": config.G_CONST,
            "link_total_time": config.LINK_TOTAL_TIME,
            "tmot_path": config.TMOT_BINARY_PATH_WIN if config.IS_WINDOWS else config.TMOT_BINARY_PATH_LINUX,
            "tmot_args": config.TMOT_EXTRA_ARGS_WIN if config.IS_WINDOWS else config.TMOT_EXTRA_ARGS_LINUX,
            "cmot_path": config.CMOT_BINARY_PATH_WIN if config.IS_WINDOWS else config.CMOT_BINARY_PATH_LINUX,
            "template_path": config.SEQUENCE_TEMPLATE_PATH_WIN if config.IS_WINDOWS else config.SEQUENCE_TEMPLATE_PATH_LINUX,
            "dds_writetable_path": config.DDS_WRITETABLE_PATH_LINUX,
            "raman_up_r1_calibration": dict(config.DEFAULT_RAMAN_POWER_CALIBRATION),
            "raman_up_r2_calibration": dict(config.DEFAULT_RAMAN_POWER_CALIBRATION),
            "raman_down_r1_calibration": dict(config.DEFAULT_RAMAN_POWER_CALIBRATION),
            "raman_down_r2_calibration": dict(config.DEFAULT_RAMAN_POWER_CALIBRATION),
            "bragg_power_calibration": dict(config.DEFAULT_BRAGG_POWER_CALIBRATION),
            "sequence_marker_definitions": [],
            "sequence_marker_profiles": {},
            "sync_role": "standalone",
            "sync_node_name": "Local controller",
            "sync_shared_token": "",
            "sync_allowed_master_ip": "",
            "sync_slaves": [],
            "tti_host": "",
            "tti_port": 9221,
            "tti_timeout_s": 3.0,
            
            # --- [关键修复] 显式添加这三个参数的默认值 ---
            "intf_alpha": 0.35,
            "intf_beta": 0.07636,
            "intf_gamma": 0.25,
            # ----------------------------------------
            
            # 合并 config.py 中的默认分析参数
            **config.DEFAULT_ANALYSIS_SETTINGS
        }
        
        # 2. 从文件加载用户保存的设置，覆盖默认值
        settings_path = Path(config.SETTINGS_FILE_PATH)
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    saved_settings = json.load(f)
                # 遍历保存的设置，覆盖 base_settings
                for k, v in saved_settings.items():
                    base_settings[k] = v 
                print(f"[Settings] Loaded user settings from {settings_path}")
            except Exception as e:
                print(f"[Settings] Failed to load user settings: {e}")

        base_settings.setdefault("fit_model_key", "gaussian")
        base_settings.setdefault("fit_models", fitting.get_default_fit_models())

        normalized = self._normalize_atom_area_settings(
            self._normalize_fit_settings(
                self._normalize_update_settings(
                    self._normalize_hardware_settings(self._normalize_tmot_settings(base_settings))
                ),
                strict=False
            )
        )

        normalized["sequence_marker_definitions"] = normalize_marker_definitions(
            normalized.get("sequence_marker_definitions"), strict=False
        )
        normalized["sequence_marker_profiles"] = normalize_marker_profiles(
            normalized.get("sequence_marker_profiles"), strict=False
        )
        return self._normalize_k_calibration_settings(normalized)

    def _save_settings_to_disk(self):
        target = Path(config.SETTINGS_FILE_PATH)
        temp_path = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target)
            print(f"[Settings] Saved to {target}")
        except Exception as e:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"[Settings] Save failed: {e}")
            raise

    def _load_user_json_payload(self) -> Dict[str, Any]:
        payload_path = Path(config.USER_JSON_PATH)
        if not payload_path.exists():
            return {}
        try:
            with open(payload_path, 'r') as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            print(f"[User JSON] Load failed: {exc}")
            return {}

    def _save_user_json_payload(self, payload: Dict[str, Any]):
        try:
            with open(config.USER_JSON_PATH, 'w') as f:
                json.dump(payload, f, indent=4)
            print(f"[User JSON] Saved to {config.USER_JSON_PATH}")
        except Exception as exc:
            print(f"[User JSON] Save failed: {exc}")

    def _normalize_custom_scan_fit_models(self, models: List[Dict[str, Any]], strict: bool = False) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for idx, model in enumerate(models or []):
            try:
                norm = fitting.normalize_fit_model_definition(model, fallback_key=f"user_scan_fit_model_{idx + 1}")
                raw_key = str(norm.get('key') or '').strip()
                if raw_key.startswith('user_'):
                    base_key = raw_key
                else:
                    base_key = f"user_{fitting.sanitize_model_key(norm.get('label') or raw_key or f'user_scan_fit_model_{idx + 1}') }"
                key = base_key
                suffix = 1
                while key in seen:
                    suffix += 1
                    key = f"{base_key}_{suffix}"
                norm['key'] = key
                error = fitting.validate_fit_model_definition(norm)
                if error:
                    raise ValueError(error)
                normalized.append(norm)
                seen.add(key)
            except Exception as exc:
                if strict:
                    raise ValueError(f"Invalid custom scan fit model #{idx + 1}: {exc}")
                print(f"[User JSON] Skipping invalid custom scan fit model #{idx + 1}: {exc}")
        return normalized

    def get_custom_scan_fit_models(self) -> List[Dict[str, Any]]:
        payload = self._load_user_json_payload()
        raw_models = payload.get('scan_fit_models')
        if not isinstance(raw_models, list):
            raw_models = []
        return self._normalize_custom_scan_fit_models(raw_models, strict=False)

    def get_scan_fit_models(self) -> List[Dict[str, Any]]:
        return fitting.get_default_scan_fit_models() + self.get_custom_scan_fit_models()

    def save_custom_scan_fit_model(self, model_definition: Dict[str, Any], name: str) -> Dict[str, Any]:
        label = str(name or '').strip()
        if not label:
            raise ValueError('Model name is required')

        source_key = str((model_definition or {}).get('key') or '').strip()
        requested_key = source_key if source_key.startswith('user_') else label
        candidate = fitting.normalize_fit_model_definition(
            {
                **(model_definition or {}),
                'key': requested_key,
                'label': label,
            },
            fallback_key=label,
        )
        if not candidate['key'].startswith('user_'):
            candidate['key'] = f"user_{candidate['key']}"

        validation_error = fitting.validate_fit_model_definition(candidate)
        if validation_error:
            raise ValueError(f"Invalid model: {validation_error}")

        payload = self._load_user_json_payload()
        models = self.get_custom_scan_fit_models()
        replaced = False
        for idx, model in enumerate(models):
            if model.get('key') == candidate['key']:
                models[idx] = candidate
                replaced = True
                break
        if not replaced:
            models.append(candidate)

        payload['scan_fit_models'] = self._normalize_custom_scan_fit_models(models, strict=True)
        self._save_user_json_payload(payload)
        return candidate

    def _normalize_bragg_phase_calibration(self, calibration: Dict[str, Any]) -> Dict[str, Any]:
        source = dict(calibration or {})
        if source.get('model_key') not in {None, '', 'bragg_fringes'}:
            raise ValueError('Only Bragg fringe fits can be saved as phase calibrations')
        parameters = dict(source.get('parameter_values') or {})
        bragg = dict(source.get('bragg') or {})
        required = {
            'A': parameters.get('A'),
            'C': parameters.get('C'),
            'phi0': parameters.get('phi0'),
            'omega': bragg.get('angular_frequency_rad_per_us2'),
        }
        try:
            numeric = {key: float(value) for key, value in required.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError('Bragg phase calibration is missing required parameters') from exc
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError('Bragg phase calibration contains non-finite parameters')
        if numeric['A'] <= 0 or numeric['omega'] <= 0:
            raise ValueError('Bragg phase calibration requires positive A and angular frequency')
        try:
            fit_x = [float(value) for value in source.get('fit_x') or []]
            fit_y = [float(value) for value in source.get('fit_y') or []]
        except (TypeError, ValueError) as exc:
            raise ValueError('Bragg phase calibration curve must be numeric') from exc
        if len(fit_x) != len(fit_y) or len(fit_x) < 2:
            raise ValueError('Bragg phase calibration requires a fitted reference curve')
        if not all(math.isfinite(value) for value in fit_x + fit_y):
            raise ValueError('Bragg phase calibration curve contains non-finite values')
        try:
            fit_min = float(source.get('fit_min', min(fit_x)))
            fit_max = float(source.get('fit_max', max(fit_x)))
        except (TypeError, ValueError) as exc:
            raise ValueError('Bragg phase calibration range must be numeric') from exc
        if fit_min > fit_max:
            fit_min, fit_max = fit_max, fit_min
        source_metadata = dict(source.get('source') or {})
        metric_tab = str(source.get('metric_tab') or source_metadata.get('metric_tab') or 'intf').strip().lower()
        channel = 'dw' if str(source.get('channel') or source_metadata.get('channel') or 'up').strip().lower() == 'dw' else 'up'
        source_key = str(source.get('source_key') or source_metadata.get('source_key') or 'intf_p').strip()
        source_mode = str(source.get('source_mode') or source_metadata.get('source_mode') or '').strip().lower()
        if source_mode not in {'fit', 'raw'}:
            source_mode = 'raw' if source_key.startswith('nf_') or 'nofit' in source_key.lower() else 'fit'
        mid_fringe_x = []
        for value in bragg.get('mid_fringe_x') or []:
            try:
                numeric_mid = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_mid) and numeric_mid >= 0:
                mid_fringe_x.append(numeric_mid)
        default_reference = min(mid_fringe_x, key=lambda value: abs(value - (fit_min + fit_max) / 2.0)) if mid_fringe_x else max(0.0, (fit_min + fit_max) / 2.0)
        reference_t2 = source.get('reference_t2_us2')
        try:
            reference_t2 = float(reference_t2) if reference_t2 is not None else float(default_reference)
        except (TypeError, ValueError):
            reference_t2 = float(default_reference)
        if not math.isfinite(reference_t2) or reference_t2 < 0:
            reference_t2 = float(default_reference)
        try:
            reference_value = float(source.get('reference_value'))
        except (TypeError, ValueError):
            reference_value = reference_t2
        if not math.isfinite(reference_value) or reference_value < 0:
            reference_value = reference_t2
        monotonic_direction = str(source.get('monotonic_slope') or '').strip().lower()
        if monotonic_direction not in {'negative', 'positive'}:
            derivative = -numeric['A'] * numeric['omega'] * math.sin(numeric['omega'] * reference_t2 + numeric['phi0'])
            monotonic_direction = 'positive' if derivative > 0 else 'negative'
        return {
            'id': str(source.get('id') or f"bragg_{time.time_ns()}"),
            'name': str(source.get('name') or '').strip(),
            'created_at': str(source.get('created_at') or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
            'model_key': 'bragg_fringes',
            'model_label': 'Bragg Fringes',
            'source': source_metadata,
            'metric_tab': metric_tab,
            'metric_label': str(source.get('metric_label') or ''),
            'source_key': source_key,
            'source_label': str(source.get('source_label') or ''),
            'source_mode': source_mode,
            'source_field': interferometer_phase.source_field({
                'metric_tab': metric_tab, 'channel': channel,
                'source_key': source_key, 'source_mode': source_mode,
            }),
            'channel': channel,
            'channel_label': str(source.get('channel_label') or ''),
            'fit_min': fit_min,
            'fit_max': fit_max,
            'fit_x': fit_x,
            'fit_y': fit_y,
            'parameter_values': {**parameters, **{key: numeric[key] for key in ('A', 'C', 'phi0')}},
            'bragg': {**bragg, 'angular_frequency_rad_per_us2': numeric['omega'], 'mid_fringe_x': mid_fringe_x},
            'reference_input_mode': 't' if str(source.get('reference_input_mode') or '').lower() == 't' else 't2',
            'reference_t_unit': 'ms' if str(source.get('reference_t_unit') or '').lower() == 'ms' else 'us',
            'reference_value': reference_value,
            'reference_t2_us2': reference_t2,
            'monotonic_slope': monotonic_direction,
            'phase_conversion_mode': 'monotonic_half_fringe',
        }

    def get_bragg_phase_calibrations(self) -> List[Dict[str, Any]]:
        payload = self._load_user_json_payload()
        calibrations = payload.get('bragg_phase_calibrations')
        if not isinstance(calibrations, list):
            return []
        normalized = []
        for calibration in calibrations:
            try:
                normalized.append(self._normalize_bragg_phase_calibration(calibration))
            except (TypeError, ValueError) as exc:
                print(f"[User JSON] Skipping invalid Bragg phase calibration: {exc}")
        return sorted(normalized, key=lambda item: item['created_at'], reverse=True)

    def build_bragg_phase_calibration(self, name: str, fit_result: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        label = str(name or '').strip()
        if not label:
            raise ValueError('Calibration name is required')
        return self._normalize_bragg_phase_calibration({
            **dict(fit_result or {}),
            'name': label,
            'source': dict(source or {}),
        })

    def save_bragg_phase_calibration(self, name: str, fit_result: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        calibration = self.build_bragg_phase_calibration(name, fit_result, source)
        payload = self._load_user_json_payload()
        calibrations = self.get_bragg_phase_calibrations()
        calibrations.append(calibration)
        payload['bragg_phase_calibrations'] = calibrations
        if not str(payload.get('active_bragg_phase_calibration_id') or '').strip():
            payload['active_bragg_phase_calibration_id'] = calibration['id']
        self._save_user_json_payload(payload)
        return calibration

    def get_active_bragg_phase_calibration(self) -> Optional[Dict[str, Any]]:
        payload = self._load_user_json_payload()
        target = str(payload.get('active_bragg_phase_calibration_id') or '').strip()
        calibrations = self.get_bragg_phase_calibrations()
        if target:
            selected = next((item for item in calibrations if item.get('id') == target), None)
            if selected is not None:
                return selected
        return calibrations[0] if calibrations else None

    def update_bragg_phase_calibration(self, calibration_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        target = str(calibration_id or '').strip()
        payload = self._load_user_json_payload()
        calibrations = self.get_bragg_phase_calibrations()
        for index, calibration in enumerate(calibrations):
            if calibration.get('id') != target:
                continue
            merged = {**calibration, **dict(updates or {}), 'id': target}
            calibrations[index] = self._normalize_bragg_phase_calibration(merged)
            payload['bragg_phase_calibrations'] = calibrations
            self._save_user_json_payload(payload)
            return calibrations[index]
        raise ValueError('Interferometer phase calibration not found')

    def set_active_bragg_phase_calibration(self, calibration_id: str) -> Optional[Dict[str, Any]]:
        target = str(calibration_id or '').strip()
        calibration = next((item for item in self.get_bragg_phase_calibrations() if item.get('id') == target), None)
        if target and calibration is None:
            raise ValueError('Interferometer phase calibration not found')
        payload = self._load_user_json_payload()
        payload['active_bragg_phase_calibration_id'] = target
        self._save_user_json_payload(payload)
        return calibration

    def delete_bragg_phase_calibration(self, calibration_id: str) -> bool:
        target = str(calibration_id or '').strip()
        payload = self._load_user_json_payload()
        calibrations = self.get_bragg_phase_calibrations()
        kept = [item for item in calibrations if item.get('id') != target]
        if len(kept) == len(calibrations):
            return False
        payload['bragg_phase_calibrations'] = kept
        if str(payload.get('active_bragg_phase_calibration_id') or '') == target:
            payload['active_bragg_phase_calibration_id'] = kept[0]['id'] if kept else ''
        self._save_user_json_payload(payload)
        return True

    def get_settings(self) -> Dict[str, Any]: return self.settings

    def _normalize_fit_settings(self, settings: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        fit_models = fitting.normalize_fit_model_list(settings.get("fit_models") or fitting.get_default_fit_models())
        validation_errors = []
        for model in fit_models:
            fit_model_error = fitting.validate_fit_model_definition(model)
            if fit_model_error:
                validation_errors.append(f"{model.get('label') or model.get('key')}: {fit_model_error}")

        if validation_errors:
            if strict:
                raise ValueError(f"Fit model invalid: {validation_errors[0]}")
            print(f"[Settings] Invalid fit model configuration detected, falling back to defaults: {validation_errors[0]}")
            fit_models = fitting.get_default_fit_models()
        selected_fit_model = fitting.get_fit_model_by_key(fit_models, settings.get("fit_model_key", "gaussian"))

        settings["fit_models"] = fit_models
        settings["fit_model_key"] = selected_fit_model["key"]
        return settings

    def _normalize_atom_area_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        method = str(settings.get("atom_area_method") or "legacy").strip().lower()
        if method not in {"legacy", "edge_line"}:
            method = "legacy"

        try:
            baseline_points = int(settings.get("atom_area_baseline_points", 2))
        except (TypeError, ValueError):
            baseline_points = 2

        settings["atom_area_method"] = method
        settings["atom_area_baseline_points"] = max(1, baseline_points)
        return settings

    def _normalize_k_calibration_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        defaults = config.DEFAULT_ANALYSIS_SETTINGS
        keys = (
            "k_detection_velocity_m_s",
            "k_wavelength_nm",
            "k_light_sheet_height_cm",
            "k_transimpedance_gain_mohm",
            "k_collection_efficiency",
            "k_photodiode_responsivity_a_w",
            "k_saturation_ratio",
            "k_detuning_mhz",
            "k_natural_linewidth_mhz",
        )
        for key in keys:
            try:
                value = float(settings.get(key, defaults[key]))
            except (TypeError, ValueError):
                value = float(defaults[key])
            settings[key] = value

        settings["K"] = physics.calculate_atom_conversion_factor(
            detection_velocity=settings["k_detection_velocity_m_s"],
            wavelength=settings["k_wavelength_nm"] * 1e-9,
            light_sheet_height=settings["k_light_sheet_height_cm"] * 1e-2,
            transimpedance_gain=settings["k_transimpedance_gain_mohm"] * 1e6,
            collection_efficiency=settings["k_collection_efficiency"],
            photodiode_responsivity=settings["k_photodiode_responsivity_a_w"],
            saturation_ratio=settings["k_saturation_ratio"],
            detuning_angular=2.0 * math.pi * settings["k_detuning_mhz"] * 1e6,
            natural_linewidth_angular=2.0 * math.pi * settings["k_natural_linewidth_mhz"] * 1e6,
        )
        return settings

    def update_settings(self, new_settings: Dict[str, Any]):
        if new_settings.get('tmot_args') is None:
            new_settings['tmot_args'] = self.settings.get('tmot_args', self._default_tmot_args())
        new_settings = self._normalize_k_calibration_settings(self._normalize_atom_area_settings(
            self._normalize_fit_settings(
                self._normalize_update_settings(
                    self._normalize_hardware_settings(self._normalize_tmot_settings(new_settings))
                ),
                strict=True
            )
        ))
        new_settings["sequence_marker_definitions"] = normalize_marker_definitions(
            new_settings.get("sequence_marker_definitions"), strict=True
        )
        new_settings["sequence_marker_profiles"] = normalize_marker_profiles(
            new_settings.get("sequence_marker_profiles"), strict=True
        )
        self.settings.update(new_settings)
        self._apply_runtime_settings()
        self._save_settings_to_disk()
        print(f">>> System Settings Updated: {self.settings}")
    

    def get_analysis_config(self) -> Dict[str, Any]:
        return self.settings

    def add_data_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        if listener not in self._data_listeners:
            self._data_listeners.append(listener)

    def publish_data(self, payload: Dict[str, Any], *, notify_listeners: bool = True) -> None:
        if notify_listeners:
            for listener in list(self._data_listeners):
                try:
                    listener(payload)
                except Exception as exc:
                    print(f"[Data Listener] {exc}")
        if self.on_data_ready:
            self.on_data_ready(payload)

    def update_analysis_config(self, new_config: Dict[str, Any]):
        self.update_settings(new_config)

    def set_simulation_mode(self, enabled: bool):
        config.USE_SIMULATION = enabled
        print(f">>> System Mode Switched: {'SIMULATION' if enabled else 'REAL HARDWARE'}")

    def get_active_mode(self) -> Optional[str]:
        with self._activity_lock:
            return self._active_mode

    def acquire_run_slot(self, mode: str) -> Tuple[bool, str]:
        requested_mode = str(mode or '').strip().lower() or 'unknown'
        with self._activity_lock:
            if self._active_mode:
                if self._active_mode == requested_mode:
                    return False, f"{requested_mode.title()} already running"
                return False, f"System busy with {self._active_mode}"
            self._active_mode = requested_mode
        return True, ''

    def release_run_slot(self, mode: Optional[str] = None):
        requested_mode = str(mode or '').strip().lower() or None
        with self._activity_lock:
            if requested_mode is None or self._active_mode == requested_mode:
                self._active_mode = None

    def refresh_runtime_settings_from_disk(self):
        try:
            print("[Manager] Auto-configuring driver before run...")
            current_settings = self._load_initial_settings()
            self.settings.update(current_settings)
            self._apply_runtime_settings()
            print("[Manager] Runtime driver settings restored from disk.")
        except Exception as exc:
            print(f"[Manager Error] Failed to auto-configure driver: {exc}")

    def load_settings_snapshot_from_disk(self) -> Dict[str, Any]:
        """Read normalized settings without mutating manager or driver state."""
        return self._load_initial_settings()

    def get_fit_model_bundle(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        fit_models = self.settings.get('fit_models') or fitting.get_default_fit_models()
        selected_fit_model = fitting.get_fit_model_by_key(
            fit_models,
            self.settings.get('fit_model_key', 'gaussian'),
        )
        fit_model_error = fitting.validate_fit_model_definition(selected_fit_model)
        if fit_model_error:
            raise ValueError(f"System fit model invalid: {fit_model_error}")
        return fit_models, selected_fit_model

    def build_fit_config(self, config_payload: Dict[str, Any]) -> Dict[str, Any]:
        fit_models, selected_fit_model = self.get_fit_model_bundle()
        return {
            'center_up': float(config_payload.get('fit_center_up', 0) or 0),
            'width_up': float(config_payload.get('fit_width_up', 0) or 0),
            'center_dw': float(config_payload.get('fit_center_dw', 0) or 0),
            'width_dw': float(config_payload.get('fit_width_dw', 0) or 0),
            'model_key': self.settings.get('fit_model_key', selected_fit_model['key']),
            'models': fit_models,
        }

    def _validate_ac_stark_sequence(self) -> None:
        template_path = Path(
            config.SEQUENCE_TEMPLATE_PATH_WIN
            if config.USE_SIMULATION
            else self.settings.get('template_path', config.SEQUENCE_TEMPLATE_PATH_LINUX)
        )
        if not template_path.is_file():
            raise ValueError(f"Sequence template not found: {template_path}")
        content = template_path.read_text(encoding="utf-8", errors="replace")
        active_content = "\n".join(line.split("#", 1)[0] for line in content.splitlines())
        missing = [
            placeholder
            for placeholder in ("<PARAMETER0>", "<PARAMETER1>")
            if placeholder not in active_content
        ]
        if missing:
            raise ValueError(
                "AC Stark sequence must use " + " and ".join(missing) + " in executable lines"
            )

    def _validate_lock_in_sequence(self) -> None:
        template_path = Path(
            config.SEQUENCE_TEMPLATE_PATH_WIN
            if config.USE_SIMULATION
            else self.settings.get('template_path', config.SEQUENCE_TEMPLATE_PATH_LINUX)
        )
        if not template_path.is_file():
            raise ValueError(f"Sequence template not found: {template_path}")
        content = template_path.read_text(encoding="utf-8", errors="replace")
        active_content = "\n".join(line.split("#", 1)[0] for line in content.splitlines())
        if "<PARAMETER0>" not in active_content:
            raise ValueError("Lock-in Measurement sequence must use <PARAMETER0> in an executable line")

    def _build_lock_in_execution(self, scan_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._resolve_scan_dimensions(scan_config) != 1:
            raise ValueError("Lock-in Measurement only supports a 1D scan")
        if scan_config.get('randomize'):
            raise ValueError("Lock-in Measurement does not support Randomize")
        self._validate_lock_in_sequence()

        target_type = str(scan_config.get('param_type') or 'float').strip().lower()
        if target_type not in {'int', 'float'}:
            raise ValueError("Lock-in PARAMETER0 type must be Int or Float")

        def normalize(value: Any) -> Any:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError("Lock-in A and B values must be numeric")
            if not math.isfinite(numeric):
                raise ValueError("Lock-in A and B values must be finite")
            return int(round(numeric)) if target_type == 'int' else round(numeric, 6)

        a_value = normalize(scan_config.get('lock_in_a_value'))
        b_value = normalize(scan_config.get('lock_in_b_value'))
        block_count = max(1, int(scan_config.get('averages', 1)))
        states = (('a', a_value, 1), ('b', b_value, -1), ('b', b_value, -1), ('a', a_value, 1))
        parameters: List[Dict[str, Any]] = []
        for block_index in range(1, block_count + 1):
            for position, (state, value, reference) in enumerate(states, start=1):
                parameters.append({
                    'sequence_parameters': [value],
                    'metadata': {
                        'lock_in_block_index': block_index,
                        'lock_in_position': position,
                        'lock_in_state': state,
                        'lock_in_reference': reference,
                    },
                })

        scan_config['scan_dimensions'] = 1
        scan_config['dim2_enabled'] = False
        scan_config['dim3_enabled'] = False
        scan_config['randomize'] = False
        scan_config['lock_in_a_value'] = a_value
        scan_config['lock_in_b_value'] = b_value
        return parameters

    def _build_transfer_function_execution(self, scan_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._resolve_scan_dimensions(scan_config) != 1:
            raise ValueError("Transfer Function only supports a 1D live scan")
        if scan_config.get("randomize"):
            raise ValueError("Transfer Function does not support Randomize")
        if str(scan_config.get("parameter_source") or "classic").lower() != "classic":
            raise ValueError("Transfer Function requires classic fixed-sequence execution")

        start = float(scan_config.get("transfer_frequency_start_hz", 0))
        stop = float(scan_config.get("transfer_frequency_stop_hz", 0))
        step = float(scan_config.get("transfer_frequency_step_hz", 0))
        if not all(math.isfinite(value) for value in (start, stop, step)):
            raise ValueError("Transfer Function frequencies must be finite")
        if step == 0:
            raise ValueError("Transfer Function frequency step cannot be zero")
        repeats = int(scan_config.get("transfer_repeats", 10))
        if repeats < 2:
            raise ValueError("Transfer Function requires at least 2 repeats per frequency")

        direction = 1.0 if stop >= start else -1.0
        effective_step = abs(step) * direction
        tolerance = abs(effective_step) * 1e-9 + 1e-12
        compare = (lambda value: value <= stop + tolerance) if direction > 0 else (lambda value: value >= stop - tolerance)
        frequencies: List[float] = []
        current = start
        while compare(current):
            frequencies.append(round(current, 6))
            if len(frequencies) > 10000:
                raise ValueError("Transfer Function scan exceeds 10000 frequency points")
            current += effective_step
        if not frequencies:
            frequencies = [round(start, 6)]

        parameters: List[Dict[str, Any]] = []
        for frequency_index, frequency in enumerate(frequencies, start=1):
            for repeat_index in range(1, repeats + 1):
                parameters.append({
                    "sequence_parameters": [],
                    "metadata": {
                        "display_parameters": [frequency],
                        "transfer_frequency_hz": frequency,
                        "transfer_frequency_index": frequency_index,
                        "transfer_frequency_count": len(frequencies),
                        "transfer_repeat": repeat_index,
                        "transfer_repeats": repeats,
                    },
                })

        scan_config["scan_dimensions"] = 1
        scan_config["dim2_enabled"] = False
        scan_config["dim3_enabled"] = False
        scan_config["randomize"] = False
        scan_config["averages"] = 1
        scan_config["transfer_settling_time_s"] = 5.0
        scan_config["transfer_frequency_values_hz"] = frequencies
        scan_config["transfer_repeats"] = repeats
        return parameters

    def _build_ac_stark_execution(self, scan_config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self._resolve_scan_dimensions(scan_config) != 1:
            raise ValueError("AC Stark Centering only supports a 1D live scan")
        if scan_config.get('randomize'):
            raise ValueError("AC Stark Centering does not support Randomize")
        self._validate_ac_stark_sequence()

        source_xml = Path(config.DDS_TABLE_UPLOAD_PATH)
        if not source_xml.is_file():
            raise ValueError("Upload a DDS .xml table before starting AC Stark Centering")
        writer_path = Path(str(self.settings.get('dds_writetable_path') or '')).expanduser()
        if not config.USE_SIMULATION and not writer_path.is_file():
            raise ValueError(f"writetable.py not found: {writer_path}")

        group = str(scan_config.get('ac_stark_raman_group') or 'up').strip().lower()
        if group not in {'up', 'down'}:
            raise ValueError("AC Stark Raman group must be UP or DOWN")
        calibration_r1 = self.settings.get(f'raman_{group}_r1_calibration')
        calibration_r2 = self.settings.get(f'raman_{group}_r2_calibration')
        if not isinstance(calibration_r1, dict) or not isinstance(calibration_r2, dict):
            raise ValueError(f"Raman {group.upper()} R1/R2 calibrations are missing in Settings")

        ratios = dds_table.generate_ratio_values(
            scan_config.get('ac_stark_ratio_start', 0.5),
            scan_config.get('ac_stark_ratio_stop', 2.0),
            scan_config.get('ac_stark_ratio_step', 0.1),
        )
        generated_xml = Path(config.BASE_DIR) / 'temp' / 'dds_ac_stark_scan.xml'
        ratio_plans = dds_table.build_ac_stark_table(
            source_xml,
            generated_xml,
            ratios,
            float(scan_config.get('ac_stark_total_power', 0)),
            calibration_r1,
            calibration_r2,
        )

        left_p0 = int(scan_config.get('ac_stark_left_p0', 0))
        right_p0 = int(scan_config.get('ac_stark_right_p0', 0))
        base_points: List[Dict[str, Any]] = []
        for plan in ratio_plans:
            plan_data = plan.to_dict()
            for side, p0_value in (('left', left_p0), ('right', right_p0)):
                metadata = {
                    'ac_stark_ratio': plan.ratio,
                    'ac_stark_side': side,
                    'ac_stark_dds_element': plan.element,
                    'ac_stark_power_r1': plan.requested_power_r1,
                    'ac_stark_power_r2': plan.requested_power_r2,
                    'ac_stark_amplitude_r1': plan.amplitude_r1,
                    'ac_stark_amplitude_r2': plan.amplitude_r2,
                    'ac_stark_actual_power_r1': plan.actual_power_r1,
                    'ac_stark_actual_power_r2': plan.actual_power_r2,
                    'ac_stark_actual_ratio': plan.actual_ratio,
                    'ac_stark_actual_total_power': plan.actual_total_power,
                    'ac_stark_raman_group': group,
                }
                base_points.append({
                    'sequence_parameters': [p0_value, plan.element],
                    'metadata': metadata,
                    'plan': plan_data,
                })

        parameters: List[Dict[str, Any]] = []
        averages = max(1, int(scan_config.get('averages', 1)))
        for repeat_index in range(averages):
            for point in base_points:
                copied_point = {
                    **point,
                    'sequence_parameters': list(point['sequence_parameters']),
                    'metadata': {**point['metadata'], 'ac_stark_repeat_index': repeat_index + 1},
                }
                parameters.append(copied_point)

        plan_dicts = [plan.to_dict() for plan in ratio_plans]
        scan_config['scan_dimensions'] = 1
        scan_config['dim2_enabled'] = False
        scan_config['dim3_enabled'] = False
        scan_config['randomize'] = False
        scan_config['_ac_stark_ratio_plan'] = plan_dicts
        return parameters, {
            'original_xml': source_xml,
            'generated_xml': generated_xml,
            'writer_path': writer_path,
            'ratio_plan': plan_dicts,
        }

    def _restore_ac_stark_dds(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        if not context or config.USE_SIMULATION:
            return None
        try:
            self.status.message = 'Restoring original DDS table...'
            dds_table.write_and_verify_dds_table(
                context['writer_path'],
                context['original_xml'],
            )
            print("[AC Stark] Original DDS table restored and verified.")
            return None
        except Exception as exc:
            message = f"DDS restore/verify failed: {exc}"
            print(f"[AC Stark] {message}")
            return message

    def _validate_auto_marker_execution(self, scan_config: Dict[str, Any], parameters: List[Any]) -> None:
        if str(scan_config.get("parameter_source") or "classic").lower() != "markers":
            return
        if str(scan_config.get("mode") or "standard").lower() != "standard":
            raise ValueError("Auto Markers only support Standard scan logic")
        scan_dimensions = self._resolve_scan_dimensions(scan_config)
        axes = [str(value or "").strip().upper() for value in scan_config.get("marker_axes", [])]
        axes = [value for value in axes if value]
        if len(axes) != scan_dimensions:
            raise ValueError(f"Auto Markers require {scan_dimensions} selected marker axes")
        if len(set(axes)) != len(axes):
            raise ValueError("Auto Marker axes must be unique")
        template_path = Path(
            config.SEQUENCE_TEMPLATE_PATH_WIN if config.USE_SIMULATION else self.settings["template_path"]
        )
        if not template_path.is_file():
            raise ValueError(f"Sequence template not found: {template_path}")
        template_content, _ = decode_mot_bytes(template_path.read_bytes())
        definitions = marker_definitions_for_sequence(
            self.settings, scan_config.get("sequence_name") or "sequence.mot"
        )
        validate_auto_marker_scan(template_content, axes, parameters, definitions)
        scan_config["marker_axes"] = axes

    def build_scan_parameter_plan(self, scan_config: Dict[str, Any]) -> List[Any]:
        payload = dict(scan_config or {})
        mode = str(payload.get('mode') or 'standard').strip().lower()
        supported_modes = {'standard', 'timing', 'rabi', 'half', 'link', 'bragg_rabi'}
        if mode not in supported_modes:
            raise ValueError(
                "Sync mode supports Standard, Timing, Rabi, Half, Link and Bragg Rabi scan logic"
            )
        return self._generate_parameters(payload)

    def start_scan(
        self,
        scan_config: Dict[str, Any],
        parameters_override: Optional[List[Any]] = None,
    ) -> Dict[str, str]:
        if not hasattr(self, 'status'):
            self.status = ExperimentStatus()

        acquired, busy_message = self.acquire_run_slot('scan')
        if not acquired:
            return {'status': 'error', 'message': busy_message}

        self.refresh_runtime_settings_from_disk()
        ac_stark_context: Optional[Dict[str, Any]] = None
        try:
            if parameters_override is not None:
                parameters = list(parameters_override)
            elif scan_config.get('mode') == 'ac_stark':
                parameters, ac_stark_context = self._build_ac_stark_execution(scan_config)
            elif scan_config.get('mode') == 'lock_in':
                parameters = self._build_lock_in_execution(scan_config)
            elif scan_config.get('mode') == 'transfer_function':
                parameters = self._build_transfer_function_execution(scan_config)
            else:
                parameters = self._generate_parameters(scan_config)
            if parameters_override is None or not scan_config.get('_sync_slave'):
                validation_parameters = [
                    item.get("sequence_parameters")
                    if isinstance(item, dict) and "sequence_parameters" in item
                    else item
                    for item in parameters
                ]
                self._validate_auto_marker_execution(scan_config, validation_parameters)
        except Exception as exc:
            self.release_run_slot('scan')
            return {'status': 'error', 'message': f'Param generation failed: {str(exc)}'}

        try:
            fit_config = self.build_fit_config(scan_config)
        except Exception as exc:
            self.release_run_slot('scan')
            return {'status': 'error', 'message': str(exc)}

        self.stop_flag = False
        self._scan_finalize_error = None
        self.status = ExperimentStatus(is_running=True, total_steps=len(parameters), message='Starting...')

        try:
            requested_phase_calibration = scan_config.get('interferometer_phase_calibration_override')
            self._active_phase_calibration_for_run = (
                deepcopy(requested_phase_calibration)
                if isinstance(requested_phase_calibration, dict)
                else self.get_active_bragg_phase_calibration()
            )
            scan_config['_system_settings_snapshot'] = self.settings
            scan_config['_interferometer_phase_calibration_snapshot'] = self._active_phase_calibration_for_run
            self.data_manager.init_run(scan_config)
            if ac_stark_context:
                archived = self.data_manager.archive_ac_stark_plan(
                    Path(ac_stark_context['original_xml']),
                    Path(ac_stark_context['generated_xml']),
                    ac_stark_context['ratio_plan'],
                )
                ac_stark_context['original_xml'] = Path(archived['original_xml'])
                ac_stark_context['generated_xml'] = Path(archived['generated_xml'])
                if not config.USE_SIMULATION:
                    self.status.message = 'Writing and verifying AC Stark DDS table...'
                    dds_table.write_and_verify_dds_table(
                        ac_stark_context['writer_path'],
                        ac_stark_context['generated_xml'],
                    )
                    print("[AC Stark] Generated DDS scan table written and verified.")
        except Exception as exc:
            restore_error = self._restore_ac_stark_dds(ac_stark_context)
            self.data_manager.close_run()
            self.status = ExperimentStatus(message=restore_error or 'IDLE')
            self.release_run_slot('scan')
            message = f'AC Stark preparation failed: {exc}' if ac_stark_context else f'Data Init Failed: {exc}'
            if restore_error:
                message += f'; {restore_error}'
            return {'status': 'error', 'message': message}

        with self.data_queue.mutex:
            self.data_queue.queue.clear()

        self.proc_thread = threading.Thread(target=self._processing_loop, args=(fit_config, scan_config))
        self.proc_thread.daemon = True
        self.proc_thread.start()

        self.acq_thread = threading.Thread(
            target=self._acquisition_loop,
            args=(parameters, scan_config, ac_stark_context),
        )
        self.acq_thread.daemon = True
        self.acq_thread.start()

        return {'status': 'success', 'message': 'Scan started (Parallel)'}

    def stop_scan(self) -> Dict[str, str]:
        active_mode = self.get_active_mode()
        if active_mode != 'scan' or not self.status.is_running:
            if active_mode == 'optimization':
                return {'status': 'warning', 'message': 'Optimization is running. Use the optimization stop control.'}
            return {'status': 'warning', 'message': 'No experiment running'}

        self.stop_flag = True
        self.status.message = 'Stopping...'
        return {'status': 'success', 'message': 'Stop signal sent'}

    def _resolve_scan_dimensions(self, config: Dict[str, Any]) -> int:
        legacy_dims = 1
        if config.get('dim2_enabled', False):
            legacy_dims = 2
        if config.get('dim3_enabled', False):
            legacy_dims = 3

        try:
            scan_dimensions = int(config.get('scan_dimensions', legacy_dims))
        except (TypeError, ValueError):
            scan_dimensions = legacy_dims

        return max(1, min(3, scan_dimensions))

    def _generate_parameters(self, config: Dict[str, Any]) -> List[Any]:
        max_base_points = int(config.get('_max_base_points', 0) or 0)
        max_base_points_label = str(config.get('_max_base_points_label') or 'Bragg').strip() or 'Bragg'
        def normalize_single_value(value: float, target_type: str) -> Any:
            if target_type == 'int':
                return int(round(float(value)))
            return round(float(value), 6)

        def normalize_values(values: List[float], target_type: str) -> List[Any]:
            return [normalize_single_value(x, target_type) for x in values]

        def get_values(scan_type, method, start, stop, step_or_count, clist, target_type='float'):
            if scan_type == 'list':
                try:
                    raw_vals = [float(x.strip()) for x in clist.split(',') if x.strip()]
                except ValueError:
                    raise ValueError(f"Invalid list format: {clist}")
                if not raw_vals:
                    raise ValueError("Scan list cannot be empty")
                if max_base_points and len(raw_vals) > max_base_points:
                    raise ValueError(f"{max_base_points_label} ZIP export is limited to {max_base_points} files")
                return normalize_values(raw_vals, target_type)

            start = float(start)
            stop = float(stop)
            raw_vals: List[float] = []

            if method == 'n_points':
                try:
                    n_points = int(float(step_or_count))
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid point count: {step_or_count}")
                if n_points <= 0:
                    raise ValueError("Point count must be positive")
                if max_base_points and n_points > max_base_points:
                    raise ValueError(f"{max_base_points_label} ZIP export is limited to {max_base_points} files")
                if n_points == 1:
                    raw_vals = [start]
                else:
                    raw_vals = np.linspace(start, stop, n_points).tolist()
            else:
                step = float(step_or_count)
                if step == 0:
                    raw_vals = [start]
                else:
                    direction = 1.0 if stop >= start else -1.0
                    effective_step = abs(step) * direction
                    current = start
                    tolerance = abs(effective_step) * 1e-9 + 1e-12
                    compare = (lambda value: value <= stop + tolerance) if direction > 0 else (lambda value: value >= stop - tolerance)
                    while compare(current):
                        raw_vals.append(current)
                        if max_base_points and len(raw_vals) > max_base_points:
                            raise ValueError(f"{max_base_points_label} ZIP export is limited to {max_base_points} files")
                        current += effective_step
                    if not raw_vals:
                        raw_vals = [start]

            return normalize_values(raw_vals, target_type)

        scan_dimensions = self._resolve_scan_dimensions(config)
        mode = config.get('mode', 'standard')
        if scan_dimensions > 1 and mode != 'standard':
            raise ValueError("2D/3D scans only support Standard mode with independent P0/P1/P2 axes")

        dim_specs = [
            {
                'scan_type': config.get('dim1_type', 'range'),
                'method': config.get('dim1_method', 'step_size'),
                'start': config.get('start', 0),
                'stop': config.get('stop', 10),
                'step': config.get('step', 1),
                'list': config.get('custom_list', ''),
                'target_type': config.get('param_type', 'float'),
            },
            {
                'scan_type': config.get('dim2_type', 'range'),
                'method': config.get('dim2_method', 'step_size'),
                'start': config.get('dim2_start', 0),
                'stop': config.get('dim2_stop', 10),
                'step': config.get('dim2_step', 1),
                'list': config.get('dim2_list', ''),
                'target_type': config.get('dim2_param_type', config.get('param_type', 'float')),
            },
            {
                'scan_type': config.get('dim3_type', 'range'),
                'method': config.get('dim3_method', 'step_size'),
                'start': config.get('dim3_start', 0),
                'stop': config.get('dim3_stop', 10),
                'step': config.get('dim3_step', 1),
                'list': config.get('dim3_list', ''),
                'target_type': config.get('dim3_param_type', config.get('param_type', 'float')),
            },
        ]

        if scan_dimensions == 1:
            vals_1 = get_values(
                dim_specs[0]['scan_type'],
                dim_specs[0]['method'],
                dim_specs[0]['start'],
                dim_specs[0]['stop'],
                dim_specs[0]['step'],
                dim_specs[0]['list'],
                target_type=dim_specs[0]['target_type'],
            )
            target_type = dim_specs[0]['target_type']
            mode_param_raw = config.get('mode_param')
            if mode in {'timing', 'rabi'} and mode_param_raw in (None, ''):
                raise ValueError('Timing/Rabi mode requires a numeric Total Parameter')
            mode_param = float(mode_param_raw or 0.0)
            link_formulas = config.get('link_formulas', [])
            final_list = []
            for v in vals_1:
                s = [v]
                if mode == 'timing':
                    s.append(normalize_single_value(mode_param - v, target_type))
                elif mode == 'rabi':
                    s.append(normalize_single_value(mode_param - v / 2.0, target_type))
                elif mode == 'half':
                    s.append(normalize_single_value(v / 2.0, target_type))
                elif mode == 'link':
                    eval_ctx = {"P0": v, "math": math, "np": np}
                    for i, formula_str in enumerate(link_formulas):
                        try:
                            val = normalize_single_value(
                                float(eval(formula_str, {"__builtins__": {}}, eval_ctx)),
                                target_type
                            )
                            eval_ctx[f"P{i+1}"] = val
                            s.append(val)
                        except Exception:
                            raise ValueError(f"Formula Error: {formula_str}")
                final_list.append(s)
        else:
            axis_values = []
            for dim_idx in range(scan_dimensions):
                spec = dim_specs[dim_idx]
                axis_values.append(
                    get_values(
                        spec['scan_type'],
                        spec['method'],
                        spec['start'],
                        spec['stop'],
                        spec['step'],
                        spec['list'],
                        target_type=spec['target_type'],
                    )
                )
            final_list = [list(param_set) for param_set in product(*axis_values)]

        full_scan = []
        for _ in range(int(config.get('averages', 1))):
            full_scan.extend(final_list)
        if config.get('randomize', False):
            random.shuffle(full_scan)
        return full_scan

    def build_bragg_export_fwhm_values(self, scan_config: Dict[str, Any]) -> List[float]:
        payload = dict(scan_config or {})
        if str(payload.get('mode') or '').strip().lower() != 'bragg_rabi':
            raise ValueError("Bragg ZIP export requires Bragg Rabi mode")
        if self._resolve_scan_dimensions(payload) != 1:
            raise ValueError("Bragg ZIP export only supports a 1D scan")
        payload['averages'] = 1
        payload['randomize'] = False
        payload['_max_base_points'] = 200
        parameter_sets = self._generate_parameters(payload)
        values = []
        for parameter_set in parameter_sets:
            normalized = self._normalize_parameter_list(parameter_set)
            if not normalized:
                continue
            value = float(normalized[0])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Bragg FWHM values must be positive finite numbers")
            values.append(value)
        if not values:
            raise ValueError("Bragg scan contains no FWHM values")
        return values

    def build_link_export_parameter_sets(
        self,
        scan_config: Dict[str, Any],
        p0: Optional[float] = None,
    ) -> List[List[Any]]:
        payload = dict(scan_config or {})
        if str(payload.get('mode') or '').strip().lower() != 'link':
            raise ValueError("Link export requires Link mode")
        if self._resolve_scan_dimensions(payload) != 1:
            raise ValueError("Link export only supports a 1D scan")
        payload['averages'] = 1
        payload['randomize'] = False
        payload['scan_dimensions'] = 1
        payload['dim2_enabled'] = False
        payload['dim3_enabled'] = False
        payload['_max_base_points'] = 200
        payload['_max_base_points_label'] = 'Link'
        if p0 is not None:
            numeric_p0 = float(p0)
            if not math.isfinite(numeric_p0):
                raise ValueError("Link export P0 must be a finite number")
            payload['dim1_type'] = 'list'
            payload['custom_list'] = str(numeric_p0)
        parameter_sets = [
            self._normalize_parameter_list(parameter_set)
            for parameter_set in self._generate_parameters(payload)
        ]
        if not parameter_sets:
            raise ValueError("Link export contains no parameter sets")
        return parameter_sets

    '''# [UPDATED] Robust VCD Parser with Debug Prints
    def _calculate_delay_from_vcd(self, vcd_path: str, launch_id: str, trigger_id: str) -> float:
        try:
            # 1. Clean inputs (Remove potential parens or whitespace)
            launch_id_clean = str(launch_id).strip().strip("()")
            trigger_id_clean = str(trigger_id).strip().strip("()")
            
            print(f"[VCD] Analyzing {vcd_path}")
            
            # 2. Build Code Map from Header
            code_map = {} 
            with open(vcd_path, 'r') as f:
                lines = f.readlines()
            
            header_end = 0
            timescale = 1e-9 # Default 1ns

            for i, line in enumerate(lines):
                parts = line.strip().split()
                
                # Timescale
                if len(parts) >= 2 and parts[0] == '$timescale':
                    if 'ns' in line: timescale = 1e-9
                    elif 'us' in line: timescale = 1e-6
                    elif 'ms' in line: timescale = 1e-3
                    elif 'ps' in line: timescale = 1e-12
                
                # Variable definition: $var type size code name $end
                # e.g. $var reg 1 60 TTL2_D0 $end
                if len(parts) >= 4 and parts[0] == '$var':
                    # Standard VCD: $var wire 1 ! D0 $end  -> code is !
                    # CMOT VCD: $var reg 1 60 D0 $end      -> code is 60
                    # We look for the code at index 3
                    code = parts[3]
                    name = parts[4]
                    code_map[code] = name
                
                if line.startswith('$enddefinitions'):
                    header_end = i
                    break
            
            # 3. Resolve Codes
            # Try to find code by Name, then by Code itself
            name_to_code = {v: k for k, v in code_map.items()}
            
            code_launch = name_to_code.get(launch_id_clean)
            if not code_launch and launch_id_clean in code_map:
                code_launch = launch_id_clean
                
            code_trigger = name_to_code.get(trigger_id_clean)
            if not code_trigger and trigger_id_clean in code_map:
                code_trigger = trigger_id_clean
            
            print(f"[VCD] Targets -> Launch: '{launch_id_clean}' (Code: {code_launch}), Trigger: '{trigger_id_clean}' (Code: {code_trigger})")
            
            if not code_launch or not code_trigger:
                print("[VCD Error] Could not resolve signal codes.")
                return 0.0

            # 4. Scan Data for Rising Edges
            current_time = 0.0
            t_launch = None
            t_trigger = None
            state_launch = '0'
            state_trigger = '0'

            for line in lines[header_end:]:
                line = line.strip()
                if not line: continue
                
                # Time marker
                if line.startswith('#'):
                    try:
                        current_time = float(line[1:]) * timescale
                    except: pass
                    continue
                
                # Value Change
                # Standard scalar: '160' (Value 1, Code 60)
                # Vector/Real: 'b100 2' or 'r1.5 33' (Space separated)
                
                if ' ' in line:
                    # Vector/Real format: "val code"
                    parts = line.split()
                    if len(parts) != 2: continue
                    val_part, code_part = parts[0], parts[1]
                else:
                    # Scalar format: "vCode" (e.g. 160)
                    # We don't know length of code, but we know our target codes
                    # Optimization: Check if line ends with our target codes
                    
                    if line.endswith(code_launch):
                        val_part = line[:-len(code_launch)]
                        code_part = code_launch
                    elif line.endswith(code_trigger):
                        val_part = line[:-len(code_trigger)]
                        code_part = code_trigger
                    else:
                        continue

                # Check Logic
                if code_part == code_launch:
                    # Scalar '1' or binary 'b1' or '1'
                    # Simply check if value represents logic high
                    is_high = (val_part == '1' or val_part == 'b1')
                    if is_high and state_launch == '0':
                         if t_launch is None:
                             t_launch = current_time
                             print(f"[VCD] Launch Rising Edge found at {t_launch:.6f}s")
                    state_launch = '1' if is_high else '0'

                elif code_part == code_trigger:
                    is_high = (val_part == '1' or val_part == 'b1')
                    if is_high and state_trigger == '0':
                         if t_trigger is None:
                             t_trigger = current_time
                             print(f"[VCD] Trigger Rising Edge found at {t_trigger:.6f}s")
                    state_trigger = '1' if is_high else '0'

                if t_launch is not None and t_trigger is not None:
                    break
            
            # Fallback
            if t_launch is None: t_launch = 0.0
            if t_trigger is None: t_trigger = 0.0
            
            delta = t_trigger - t_launch
            print(f"[VCD] Calculated Delta T = {delta:.6f} s")
            return delta

        except Exception as e:
            print(f"[VCD Critical Error] {e}")
            traceback.print_exc()
            return 0.0 '''
    # [UPDATED] Universal VCD Parser: Compatible with TTL and DAC
    def _calculate_delay_from_vcd(self, vcd_path: str, launch_id: str, trigger_id: str) -> float:
        try:
            # 1. Clean inputs
            launch_id_clean = str(launch_id).strip().strip("()")
            trigger_id_clean = str(trigger_id).strip().strip("()")
            
            print(f"[VCD] Analyzing {vcd_path} for Launch='{launch_id_clean}' & Trigger='{trigger_id_clean}'")
            
            # --- Internal Helper: Convert VCD value to voltage float ---
            def _parse_vcd_val(val_str: str) -> float:
                # Case 1: Standard logic '1' or '0'
                if val_str == '1': return 3.3 # Treated as high level voltage
                if val_str == '0': return 0.0
                
                # Case 2: Real type (e.g., r3.3 or r5.0)
                if val_str.lower().startswith('r'):
                    try: return float(val_str[1:])
                    except: return 0.0
                
                # Case 3: Binary bus (e.g., b101) - rarely used for triggers, but included for safety
                if val_str.lower().startswith('b'):
                    try: return float(int(val_str[1:], 2)) # Convert to integer value
                    except: return 0.0
                
                # Case 4: Direct numeric string
                try: return float(val_str)
                except: return 0.0
            # ---------------------------------------------

            # 2. Parse Header to build Code Map
            code_map = {} 
            with open(vcd_path, 'r') as f: lines = f.readlines()
            
            header_end = 0
            timescale = 1e-9 # Default 1ns

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == '$timescale':
                    if 'ns' in line: timescale = 1e-9
                    elif 'us' in line: timescale = 1e-6
                    elif 'ms' in line: timescale = 1e-3
                
                if len(parts) >= 4 and parts[0] == '$var':
                    # $var real 1 ! DAC_A0 $end  -> code is !
                    code = parts[3]
                    name = parts[4]
                    code_map[code] = name
                
                if line.startswith('$enddefinitions'):
                    header_end = i; break
            
            # 3. Resolve Target Codes
            name_to_code = {v: k for k, v in code_map.items()}
            
            code_launch = name_to_code.get(launch_id_clean)
            if not code_launch and launch_id_clean in code_map: code_launch = launch_id_clean
                
            code_trigger = name_to_code.get(trigger_id_clean)
            if not code_trigger and trigger_id_clean in code_map: code_trigger = trigger_id_clean
            
            if not code_launch or not code_trigger:
                print(f"[VCD Error] Could not find signals. Launch code: {code_launch}, Trigger code: {code_trigger}")
                return 0.0

            # 4. Scan Data (Using Threshold Detection)
            current_time = 0.0
            t_launch = None; t_trigger = None
            
            # Set detection threshold (e.g., 1.5V)
            # Any voltage above 1.5V is considered High
            THRESHOLD_V = 1.5 
            
            state_launch = False   # False=Low, True=High
            state_trigger = False

            for line in lines[header_end:]:
                line = line.strip()
                if not line: continue
                
                if line.startswith('#'):
                    try: current_time = float(line[1:]) * timescale
                    except: pass
                    continue
                
                # Parse Line: Formats like "1!", "r3.3 !", "b10 @"
                val_part = ""; code_part = ""
                
                if ' ' in line: # Vector/Real: "val code"
                    parts = line.split()
                    if len(parts) != 2: continue
                    val_part, code_part = parts[0], parts[1]
                else: # Scalar: "vCode"
                    # Attempt to match suffix
                    if line.endswith(code_launch):
                        val_part = line[:-len(code_launch)]; code_part = code_launch
                    elif line.endswith(code_trigger):
                        val_part = line[:-len(code_trigger)]; code_part = code_trigger
                    else: continue

                # === Core Improved Logic ===
                if code_part == code_launch or code_part == code_trigger:
                    # 1. Get voltage value
                    voltage = _parse_vcd_val(val_part)
                    # 2. Determine if High Level
                    is_high = voltage > THRESHOLD_V
                    
                    if code_part == code_launch:
                        # Detect Rising Edge: Was Low, Now High
                        if is_high and not state_launch:
                             if t_launch is None: t_launch = current_time
                        state_launch = is_high

                    elif code_part == code_trigger:
                        if is_high and not state_trigger:
                             if t_trigger is None: t_trigger = current_time
                        state_trigger = is_high

                if t_launch is not None and t_trigger is not None:
                    break
            
            if t_launch is None: t_launch = 0.0
            if t_trigger is None: t_trigger = 0.0
            
            delta = t_trigger - t_launch
            # If negative (Trigger before Launch), usually take abs or zero, depends on needs
            # Keeping as is, allowing negative delay (physically rare, but possible in simulation)
            print(f"[VCD] Launch@{t_launch*1e3:.3f}ms, Trig@{t_trigger*1e3:.3f}ms -> Delay={delta:.6f}s")
            return delta

        except Exception as e:
            print(f"[VCD Critical Error] {e}")
            traceback.print_exc()
            return 0.0


    def _normalize_parameter_list(self, param_set: Any) -> List[Any]:
        if isinstance(param_set, list):
            return list(param_set)
        if isinstance(param_set, tuple):
            return list(param_set)
        return [param_set]

    def _prepare_sequence_parameters(self, params_to_write: List[Any], execution_config: Dict[str, Any]) -> List[Any]:
        mode = execution_config.get('mode', 'standard')
        if mode == 'bragg_rabi':
            fwhm_val = params_to_write[0]
            pulse_code, comp_time = generate_bragg_pulse(
                fwhm=fwhm_val,
                shape=execution_config.get('bragg_shape', 'blackman'),
                base_timing=int(execution_config.get('bragg_base_timing', 331119)),
                calibration=self.settings.get('bragg_power_calibration'),
            )
            return [pulse_code, comp_time]
        return params_to_write

    def execute_single_measurement(
        self,
        param_set: Any,
        execution_config: Dict[str, Any],
        *,
        idx: int,
        total_steps: int,
        scan_dimensions: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params_to_write = self._normalize_parameter_list(param_set)
        actual_params_to_write = self._prepare_sequence_parameters(params_to_write, execution_config)
        metadata = dict(metadata or {})

        try:
            template_path = execution_config.get('_template_path_override') or (config.SEQUENCE_TEMPLATE_PATH_WIN if config.USE_SIMULATION else self.settings['template_path'])
            self.seq_editor.generate_sequence(
                template_path,
                config.SEQUENCE_OUTPUT_PATH,
                actual_params_to_write,
                bragg_mode=execution_config.get('mode') == 'bragg_rabi',
                marker_axes=(execution_config.get('marker_axes') or [])
                    if execution_config.get('parameter_source') == 'markers' else None,
                marker_definitions=marker_definitions_for_sequence(
                    self.settings, execution_config.get('sequence_name') or 'sequence.mot'
                ),
            )

            cmot_bin = config.CMOT_BINARY_PATH_WIN if config.USE_SIMULATION else self.settings['cmot_path']
            self.driver.compile_vcd(config.SEQUENCE_OUTPUT_PATH, config.VCD_OUTPUT_PATH, binary_path=cmot_bin)

            if config.USE_SIMULATION:
                start_delay = 0.78
            else:
                start_delay = self._calculate_delay_from_vcd(
                    config.VCD_OUTPUT_PATH,
                    str(self.settings['chan_launch']),
                    str(self.settings['chan_trigger']),
                )

            tmot_bin = config.TMOT_BINARY_PATH_WIN if config.USE_SIMULATION else self.settings['tmot_path']
            ext_trigger_enabled = bool(execution_config.get('ext_trigger', False))
            tmot_args = '' if config.USE_SIMULATION else ('-e' if ext_trigger_enabled else '')
            print(f"[TMOT] Effective settings: path={tmot_bin!r}, ext_trigger={ext_trigger_enabled}, args={tmot_args!r}")
            success = self.driver.run_sequence(
                config.SEQUENCE_OUTPUT_PATH,
                binary_path=tmot_bin,
                extra_args=tmot_args,
            )
            if not success:
                return {
                    'idx': idx,
                    'total': total_steps,
                    'params': params_to_write,
                    'scan_dimensions': scan_dimensions,
                    'start_delay': start_delay,
                    'volt_up': [],
                    'volt_dw': [],
                    'timestamp': time.time(),
                    'metadata': metadata,
                    'error': 'Sequence execution failed',
                }

            _, volt_up_raw = self.rp_driver_red.acquire_channel('ch1')
            _, volt_dw_raw = self.rp_driver_red.acquire_channel('ch2')
            return {
                'idx': idx,
                'total': total_steps,
                'params': params_to_write,
                'scan_dimensions': scan_dimensions,
                'start_delay': start_delay,
                'volt_up': volt_up_raw,
                'volt_dw': volt_dw_raw,
                'timestamp': time.time(),
                'metadata': metadata,
            }
        except Exception as exc:
            print(f"[Acq Error] Step {idx + 1}: {traceback.format_exc()}")
            return {
                'idx': idx,
                'total': total_steps,
                'params': params_to_write,
                'scan_dimensions': scan_dimensions,
                'start_delay': 0.0,
                'volt_up': [],
                'volt_dw': [],
                'timestamp': time.time(),
                'metadata': metadata,
                'error': str(exc),
            }

    def _build_error_payload(
        self,
        job: Dict[str, Any],
        message: str,
        *,
        stream_type: str,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = job.get('metadata') or {}
        params = metadata.get('display_parameters') if isinstance(metadata.get('display_parameters'), list) else (job.get('params') or [])
        payload = {
            'stream_type': stream_type,
            'parameter': params[0] if params else None,
            'all_parameters': params,
            'scan_dimensions': int(job.get('scan_dimensions', 1) or 1),
            'error': message,
            'current_step': int(job.get('idx', 0)) + 1,
            'total_steps': int(job.get('total', 1) or 1),
        }
        payload.update(metadata)
        if extra_payload:
            payload.update(extra_payload)
        return payload

    def process_measurement_job(
        self,
        job: Dict[str, Any],
        fit_config: Dict[str, Any],
        *,
        data_manager: Optional[DataManager] = None,
        save_step_index: Optional[int] = None,
        stream_type: str = 'scan_point',
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ScanResult], Dict[str, Any]]:
        settings = self.settings
        storage_step = 1
        selected_fit_model = fitting.get_fit_model_by_key(
            fit_config.get('models'),
            fit_config.get('model_key'),
        )

        idx = int(job.get('idx', 0))
        total = int(job.get('total', 1) or 1)
        params = job.get('params') or []
        metadata = job.get('metadata') or {}
        scan_dimensions = int(job.get('scan_dimensions', 1) or 1)
        if isinstance(metadata.get('sync_parameters'), list):
            display_params = metadata['sync_parameters']
        elif isinstance(metadata.get('display_parameters'), list):
            display_params = metadata['display_parameters']
        else:
            display_params = params
        primary_param = display_params[0] if display_params else None
        start_delay = float(job.get('start_delay', 0.0) or 0.0)
        volt_up_raw = job.get('volt_up') or []
        volt_dw_raw = job.get('volt_dw') or []

        if job.get('error'):
            return None, self._build_error_payload(job, str(job['error']), stream_type=stream_type, extra_payload=extra_payload)

        if not volt_up_raw or not volt_dw_raw:
            return None, self._build_error_payload(job, 'No Data', stream_type=stream_type, extra_payload=extra_payload)

        try:
            if len(volt_up_raw) > 300:
                offset_up = float(np.mean(volt_up_raw[-200:]))
                offset_dw = float(np.mean(volt_dw_raw[-200:]))
            else:
                offset_up = 0.0
                offset_dw = 0.0

            tail_mean_up_raw = float(offset_up)
            tail_mean_dw_raw = float(offset_dw)
            val_up_clean = np.array(volt_up_raw) - offset_up
            val_dw_clean = np.array(volt_dw_raw) - offset_dw

            voltage_limit = float(settings.get('voltage_limit', 9.5))
            max_amp_up = np.max(np.abs(val_up_clean))
            max_amp_dw = np.max(np.abs(val_dw_clean))
            if max_amp_up > voltage_limit or max_amp_dw > voltage_limit:
                msg = f"Signal Amplitude > {voltage_limit}V (UP={max_amp_up:.2f}V, DW={max_amp_dw:.2f}V)"
                print(f"[Filter] Rejected (Step {idx + 1}): {msg}")
                return None, self._build_error_payload(job, msg, stream_type=stream_type, extra_payload=extra_payload)

            gain_up = float(settings.get('gain_up', 1.0)) or 1.0
            gain_dw = float(settings.get('gain_dw', 1.0)) or 1.0
            volt_up = val_up_clean / gain_up
            volt_dw = val_dw_clean / gain_dw

            if config.USE_SIMULATION:
                time_axis = np.linspace(0, 0.1, len(volt_up))
            else:
                decimation = int(settings.get('decimation', 8192))
                dt = 8e-9 * decimation
                time_axis = np.array([i * dt for i in range(len(volt_up))])

            tof_axis = time_axis + start_delay

            def get_fit_data(time_values, voltage_values, center_ms, width_ms):
                if width_ms > 0:
                    cen_s = center_ms * 1e-3
                    wid_s = width_ms * 1e-3
                    start = cen_s - wid_s / 2
                    end = cen_s + wid_s / 2
                    mask = (time_values >= start) & (time_values <= end)
                    if np.any(mask):
                        return time_values[mask], voltage_values[mask], (start, end)
                return time_values, voltage_values, None

            fit_t_up, fit_v_up, win_up = get_fit_data(tof_axis, volt_up, fit_config.get('center_up', 0), fit_config.get('width_up', 0))
            fit_t_dw, fit_v_dw, win_dw = get_fit_data(tof_axis, volt_dw, fit_config.get('center_dw', 0), fit_config.get('width_dw', 0))
            fit_result_up = fitting.perform_configured_fit(selected_fit_model, fit_t_up, fit_v_up, eval_x=tof_axis)
            fit_result_dw = fitting.perform_configured_fit(selected_fit_model, fit_t_dw, fit_v_dw, eval_x=tof_axis)
            fit_curve_up = fit_result_up.fit_curve if fit_result_up is not None else np.zeros_like(tof_axis)
            fit_curve_dw = fit_result_dw.fit_curve if fit_result_dw is not None else np.zeros_like(tof_axis)
            area_method = str(settings.get('atom_area_method', 'legacy')).strip().lower()
            baseline_points = int(settings.get('atom_area_baseline_points', 2))

            amp_up = fit_result_up.amplitude if fit_result_up is not None else 0
            sig_up = fit_result_up.width if fit_result_up is not None else 0
            cen_up = fit_result_up.center if fit_result_up is not None else 0
            amp_dw = fit_result_dw.amplitude if fit_result_dw is not None else 0
            sig_dw = fit_result_dw.width if fit_result_dw is not None else 0
            cen_dw = fit_result_dw.center if fit_result_dw is not None else 0
            amp_up_nf = np.max(fit_v_up) if len(fit_v_up) > 0 else 0
            amp_dw_nf = np.max(fit_v_dw) if len(fit_v_dw) > 0 else 0
            sig_up_nf = fitting.calc_sigma(fit_v_up, fit_t_up) or 0
            sig_dw_nf = fitting.calc_sigma(fit_v_dw, fit_t_dw) or 0
            cen_up_nf = fit_t_up[np.argmax(fit_v_up)] if len(fit_v_up) > 0 else 0
            cen_dw_nf = fit_t_dw[np.argmax(fit_v_dw)] if len(fit_v_dw) > 0 else 0

            if area_method == 'edge_line':
                area_up_nf = fitting.calculate_area_with_edge_baseline(fit_t_up, fit_v_up, baseline_points)
                area_dw_nf = fitting.calculate_area_with_edge_baseline(fit_t_dw, fit_v_dw, baseline_points)
                area_up = fitting.calculate_area_with_edge_baseline(fit_t_up, fit_result_up.fit_window_curve, baseline_points) if fit_result_up is not None else 0
                area_dw = fitting.calculate_area_with_edge_baseline(fit_t_dw, fit_result_dw.fit_window_curve, baseline_points) if fit_result_dw is not None else 0
            else:
                area_up_nf = abs(np.trapz(fit_v_up, fit_t_up)) if len(fit_v_up) > 1 else 0
                area_dw_nf = abs(np.trapz(fit_v_dw, fit_t_dw)) if len(fit_v_dw) > 1 else 0
                area_up = fit_result_up.area if fit_result_up is not None else 0
                area_dw = fit_result_dw.area if fit_result_dw is not None else 0

            n_f2, n_f1 = physics.calculate_atom_numbers(
                area_up,
                area_dw,
                max_vol_up=amp_up,
                max_vol_dw=amp_dw,
                alpha=settings['alpha'],
                beta=settings['beta'],
                R=settings['R'],
                K=settings['K'],
                max_low=settings['max_low'],
            )
            n_f2_nf, n_f1_nf = physics.calculate_atom_numbers(
                area_up_nf,
                area_dw_nf,
                max_vol_up=amp_up_nf,
                max_vol_dw=amp_dw_nf,
                alpha=settings['alpha'],
                beta=settings['beta'],
                R=settings['R'],
                K=settings['K'],
                max_low=settings['max_low'],
            )

            launch_velocity = float(settings['launch_velocity'])
            t_flight_up = cen_up
            t_flight_dw = cen_dw
            t_flight_up_nf = cen_up_nf
            t_flight_dw_nf = cen_dw_nf
            temp_up = physics.calculate_temperature(sig_up, t_flight_up, launch_velocity, is_sigma_in_ms=False)
            temp_dw = physics.calculate_temperature(sig_dw, t_flight_dw, launch_velocity, is_sigma_in_ms=False)
            temp_up_nf = physics.calculate_temperature(sig_up_nf, t_flight_up_nf, launch_velocity, is_sigma_in_ms=False)
            temp_dw_nf = physics.calculate_temperature(sig_dw_nf, t_flight_dw_nf, launch_velocity, is_sigma_in_ms=False)
            prob_up, prob_dw = physics.calculate_probabilities(n_f2, n_f1)
            prob_up_nf, prob_dw_nf = physics.calculate_probabilities(n_f2_nf, n_f1_nf)
            i_n1, i_n2, i_p1, i_p2 = physics.calculate_interferometer_output(
                n_f1,
                n_f2,
                settings.get('intf_alpha', 0.35),
                settings.get('intf_beta', 0.076),
                settings.get('intf_gamma', 0.25),
            )
            i_n1_nf, i_n2_nf, i_p1_nf, i_p2_nf = physics.calculate_interferometer_output(
                n_f1_nf,
                n_f2_nf,
                settings.get('intf_alpha', 0.35),
                settings.get('intf_beta', 0.076),
                settings.get('intf_gamma', 0.25),
            )

            phase_input = {
                'atom_number_up': n_f2, 'atom_number_dw': n_f1,
                'atom_number_up_nofit': n_f2_nf, 'atom_number_dw_nofit': n_f1_nf,
                'transition_probability_up': prob_up, 'transition_probability_dw': prob_dw,
                'transition_probability_up_nofit': prob_up_nf, 'transition_probability_dw_nofit': prob_dw_nf,
                'intf_p1': i_p1, 'intf_p2': i_p2,
                'intf_p1_nofit': i_p1_nf, 'intf_p2_nofit': i_p2_nf,
            }
            phase_manager = data_manager or self.data_manager
            phase_result = interferometer_phase.calculate_phase(
                phase_input, getattr(phase_manager, 'phase_calibration_snapshot', None)
            )

            manager_for_save = data_manager or self.data_manager
            volt_up_store = volt_up[::storage_step]
            volt_dw_store = volt_dw[::storage_step]
            fit_up_store = fit_curve_up[::storage_step]
            fit_dw_store = fit_curve_dw[::storage_step]
            time_axis_store = tof_axis[::storage_step]
            run_id = manager_for_save.current_run_id_str if manager_for_save else ''
            result = ScanResult(
                parameter=primary_param if primary_param is not None else 0.0,
                timestamp=job['timestamp'],
                scan_dimensions=scan_dimensions,
                current_step=idx + 1,
                total_steps=total,
                detected_delay=start_delay,
                run_id=run_id,
                raw_data_up=volt_up_store,
                raw_data_dw=volt_dw_store,
                fit_data_up=fit_up_store,
                fit_data_dw=fit_dw_store,
                time_axis=time_axis_store,
                all_parameters=display_params,
                transfer_frequency_hz=metadata.get('transfer_frequency_hz'),
                transfer_repeat=metadata.get('transfer_repeat'),
                ac_stark_ratio=metadata.get('ac_stark_ratio'),
                ac_stark_side=metadata.get('ac_stark_side'),
                ac_stark_dds_element=metadata.get('ac_stark_dds_element'),
                ac_stark_power_r1=metadata.get('ac_stark_power_r1'),
                ac_stark_power_r2=metadata.get('ac_stark_power_r2'),
                ac_stark_amplitude_r1=metadata.get('ac_stark_amplitude_r1'),
                ac_stark_amplitude_r2=metadata.get('ac_stark_amplitude_r2'),
                ac_stark_actual_power_r1=metadata.get('ac_stark_actual_power_r1'),
                ac_stark_actual_power_r2=metadata.get('ac_stark_actual_power_r2'),
                lock_in_block_index=metadata.get('lock_in_block_index'),
                lock_in_position=metadata.get('lock_in_position'),
                lock_in_state=metadata.get('lock_in_state'),
                lock_in_reference=metadata.get('lock_in_reference'),
                workflow_step=metadata.get('workflow_step'),
                workflow_marker=metadata.get('workflow_marker'),
                workflow_point=metadata.get('workflow_point'),
                workflow_repeat=metadata.get('workflow_repeat'),
                workflow_shot=metadata.get('workflow_shot'),
                workflow_randomized=metadata.get('workflow_randomized'),
                tail_mean_up_raw=tail_mean_up_raw,
                tail_mean_dw_raw=tail_mean_dw_raw,
                window_up=win_up,
                window_dw=win_dw,
                atom_number_up=n_f2,
                atom_number_dw=n_f1,
                amplitude_up=amp_up,
                amplitude_dw=amp_dw,
                sigma_up=sig_up * 1000.0,
                sigma_dw=sig_dw * 1000.0,
                temperature_up=temp_up,
                temperature_dw=temp_dw,
                arrival_time_up=t_flight_up,
                arrival_time_dw=t_flight_dw,
                transition_probability_up=prob_up,
                transition_probability_dw=prob_dw,
                atom_number_up_nofit=n_f2_nf,
                atom_number_dw_nofit=n_f1_nf,
                amplitude_up_nofit=amp_up_nf,
                amplitude_dw_nofit=amp_dw_nf,
                sigma_up_nofit=sig_up_nf * 1000.0,
                sigma_dw_nofit=sig_dw_nf * 1000.0,
                temperature_up_nofit=temp_up_nf,
                temperature_dw_nofit=temp_dw_nf,
                arrival_time_up_nofit=t_flight_up_nf,
                arrival_time_dw_nofit=t_flight_dw_nf,
                transition_probability_up_nofit=prob_up_nf,
                transition_probability_dw_nofit=prob_dw_nf,
                intf_n1=i_n1,
                intf_n2=i_n2,
                intf_p1=i_p1,
                intf_p2=i_p2,
                intf_n1_nofit=i_n1_nf,
                intf_n2_nofit=i_n2_nf,
                intf_p1_nofit=i_p1_nf,
                intf_p2_nofit=i_p2_nf,
                interferometer_phase=phase_result['interferometer_phase'],
                interferometer_phase_valid=phase_result['interferometer_phase_valid'],
                interferometer_phase_source_value=phase_result['interferometer_phase_source_value'],
                interferometer_phase_calibration_id=phase_result['interferometer_phase_calibration_id'],
                interferometer_phase_calibration_name=phase_result['interferometer_phase_calibration_name'],
                interferometer_phase_reference_t2_us2=phase_result['interferometer_phase_reference_t2_us2'],
            )
            if manager_for_save is not None:
                manager_for_save.save_point(result, save_step_index or (idx + 1))

            step_size = max(1, len(tof_axis) // 2000)
            frontend_data = asdict(result)
            frontend_data['stream_type'] = stream_type
            frontend_data['raw_data_up'] = volt_up[::step_size].tolist()
            frontend_data['raw_data_dw'] = volt_dw[::step_size].tolist()
            frontend_data['fit_data_up'] = fit_curve_up[::step_size].tolist()
            frontend_data['fit_data_dw'] = fit_curve_dw[::step_size].tolist()
            frontend_data['time_axis'] = tof_axis[::step_size].tolist()
            frontend_data.update(job.get('metadata') or {})
            if extra_payload:
                frontend_data.update(extra_payload)
            return result, frontend_data
        except Exception:
            print(f"Processing Error step {idx + 1}: {traceback.format_exc()}")
            return None, self._build_error_payload(job, 'Processing error', stream_type=stream_type, extra_payload=extra_payload)

    # --- THREAD 1: ACQUISITION (PRODUCER) ---
    def _acquisition_loop(
        self,
        parameter_list: List[Any],
        scan_config: Dict[str, Any],
        ac_stark_context: Optional[Dict[str, Any]] = None,
    ):
        print(f"--- Acquisition Started: {len(parameter_list)} points ---")
        total_steps = len(parameter_list)
        scan_dimensions = self._resolve_scan_dimensions(scan_config)
        transfer_mode = str(scan_config.get("mode") or "").strip().lower() == "transfer_function"
        tti_client: Optional[TtiGeneratorClient] = None
        active_transfer_frequency: Optional[float] = None

        try:
            if transfer_mode and not config.USE_SIMULATION:
                tti_client = TtiGeneratorClient(TtiConnectionSettings(
                    host=str(self.settings.get("tti_host") or "").strip(),
                    port=int(self.settings.get("tti_port", 9221)),
                    timeout_s=float(self.settings.get("tti_timeout_s", 3.0)),
                ))
                identity = tti_client.connect()
                print(f"[Transfer Function] Connected to {identity}")
            for idx, param_set in enumerate(parameter_list):
                if self.stop_flag:
                    break
                metadata: Optional[Dict[str, Any]] = None
                sequence_parameters = param_set
                if isinstance(param_set, dict) and 'sequence_parameters' in param_set:
                    sequence_parameters = param_set['sequence_parameters']
                    metadata = param_set.get('metadata') or {}
                if transfer_mode:
                    frequency = float((metadata or {}).get("transfer_frequency_hz"))
                    if active_transfer_frequency is None or frequency != active_transfer_frequency:
                        self.status.message = f"Setting TG5012A CH1 to {frequency:g} Hz..."
                        if tti_client is not None:
                            tti_client.set_ch1_frequency(frequency)
                        active_transfer_frequency = frequency
                        settling_time = float(scan_config.get("transfer_settling_time_s", 5.0))
                        if not config.USE_SIMULATION and settling_time > 0:
                            self.status.message = f"TG5012A confirmed {frequency:g} Hz; settling {settling_time:g} s..."
                            deadline = time.monotonic() + settling_time
                            while not self.stop_flag and time.monotonic() < deadline:
                                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                        if self.stop_flag:
                            break
                job = self.execute_single_measurement(
                    sequence_parameters,
                    scan_config,
                    idx=idx,
                    total_steps=total_steps,
                    scan_dimensions=scan_dimensions,
                    metadata=metadata,
                )
                self.data_queue.put(job)
                if config.USE_SIMULATION:
                    time.sleep(0.1)
        except Exception as exc:
            self._scan_finalize_error = f"Acquisition loop failed: {exc}"
            print(f"[Acq Error] {traceback.format_exc()}")
        finally:
            if tti_client is not None:
                tti_client.close()
            restore_error = self._restore_ac_stark_dds(ac_stark_context)
            if restore_error:
                self._scan_finalize_error = restore_error
            self.data_queue.put(None)
            print("--- Acquisition Finished ---")

    def _build_ac_stark_summary(self, results: List[ScanResult]) -> List[Dict[str, Any]]:
        metrics = (
            'atom_number_up',
            'atom_number_dw',
            'transition_probability_up',
            'transition_probability_dw',
            'atom_number_up_nofit',
            'atom_number_dw_nofit',
            'transition_probability_up_nofit',
            'transition_probability_dw_nofit',
        )
        grouped: Dict[float, Dict[str, List[ScanResult]]] = {}
        for result in results:
            if result.ac_stark_ratio is None or result.ac_stark_side not in {'left', 'right'}:
                continue
            key = round(float(result.ac_stark_ratio), 12)
            grouped.setdefault(key, {'left': [], 'right': []})[result.ac_stark_side].append(result)

        summary: List[Dict[str, Any]] = []
        for ratio in sorted(grouped):
            sides = grouped[ratio]
            representative = (sides['left'] or sides['right'])[0]
            row: Dict[str, Any] = {
                'ratio': float(ratio),
                'dds_element': representative.ac_stark_dds_element,
                'requested_power_r1': representative.ac_stark_power_r1,
                'requested_power_r2': representative.ac_stark_power_r2,
                'amplitude_r1': representative.ac_stark_amplitude_r1,
                'amplitude_r2': representative.ac_stark_amplitude_r2,
                'actual_power_r1': representative.ac_stark_actual_power_r1,
                'actual_power_r2': representative.ac_stark_actual_power_r2,
                'left_count': len(sides['left']),
                'right_count': len(sides['right']),
            }
            for metric in metrics:
                statistics_by_side: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
                for side in ('left', 'right'):
                    values = [
                        float(value)
                        for value in (getattr(item, metric, None) for item in sides[side])
                        if value is not None and math.isfinite(float(value))
                    ]
                    if values:
                        mean = float(np.mean(values))
                        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        sem = std / math.sqrt(len(values)) if values else 0.0
                    else:
                        mean = std = sem = None
                    statistics_by_side[side] = (mean, std, sem)
                    row[f'{metric}_{side}_mean'] = mean
                    row[f'{metric}_{side}_std'] = std
                    row[f'{metric}_{side}_sem'] = sem
                left_mean, _, left_sem = statistics_by_side['left']
                right_mean, _, right_sem = statistics_by_side['right']
                row[f'{metric}_difference'] = (
                    right_mean - left_mean
                    if left_mean is not None and right_mean is not None
                    else None
                )
                row[f'{metric}_difference_sem'] = (
                    math.sqrt((left_sem or 0.0) ** 2 + (right_sem or 0.0) ** 2)
                    if left_mean is not None and right_mean is not None
                    else None
                )
            summary.append(row)
        return summary

    # --- THREAD 2: PROCESSING (CONSUMER) ---
    def _processing_loop(self, fit_config: Dict[str, Any], scan_config: Optional[Dict[str, Any]] = None):
        print("--- Processing Thread Started ---")
        selected_fit_model = fitting.get_fit_model_by_key(
            fit_config.get('models'),
            fit_config.get('model_key'),
        )
        print(f"[Fit] Using model: {selected_fit_model['label']} ({selected_fit_model['key']})")
        ac_stark_results: List[ScanResult] = []
        lock_in_results: List[ScanResult] = []
        transfer_function_results: List[ScanResult] = []

        try:
            while True:
                try:
                    job = self.data_queue.get(timeout=5)
                except queue.Empty:
                    continue
                if job is None:
                    break

                params = job.get('params') or []
                self.status.current_step = int(job.get('idx', 0)) + 1
                self.status.total_steps = int(job.get('total', 1) or 1)
                self.status.message = f"Processing: {params} (Queue: {self.data_queue.qsize()})"

                result, payload = self.process_measurement_job(
                    job,
                    fit_config,
                    data_manager=self.data_manager,
                    stream_type='scan_point',
                )
                if result is not None and result.ac_stark_ratio is not None:
                    ac_stark_results.append(result)
                if result is not None and result.lock_in_block_index is not None:
                    lock_in_results.append(result)
                if result is not None and result.transfer_frequency_hz is not None:
                    transfer_function_results.append(result)
                self.publish_data(payload)
        finally:
            if ac_stark_results:
                try:
                    self.data_manager.save_ac_stark_summary(
                        self._build_ac_stark_summary(ac_stark_results)
                    )
                except Exception as exc:
                    self._scan_finalize_error = f"AC Stark summary save failed: {exc}"
                    print(f"[AC Stark] {self._scan_finalize_error}")
            if scan_config and scan_config.get('mode') == 'lock_in':
                try:
                    analysis = build_lock_in_analysis(
                        [asdict(result) for result in lock_in_results],
                        expected_blocks=max(1, int(scan_config.get('averages', 1))),
                    )
                    self.data_manager.save_lock_in_analysis(analysis)
                except Exception as exc:
                    self._scan_finalize_error = f"Lock-in analysis save failed: {exc}"
                    print(f"[Lock-in] {self._scan_finalize_error}")
            if scan_config and scan_config.get('mode') == 'transfer_function':
                try:
                    self.data_manager.save_transfer_function_summary(
                        build_transfer_function_summary(transfer_function_results)
                    )
                except Exception as exc:
                    self._scan_finalize_error = f"Transfer Function summary save failed: {exc}"
                    print(f"[Transfer Function] {self._scan_finalize_error}")
            self.data_manager.close_run()
            self.status.is_running = False
            if self._scan_finalize_error:
                self.status.message = self._scan_finalize_error
            else:
                self.status.message = 'Stopped' if self.stop_flag else 'Done'
            self.release_run_slot('scan')
            print("--- Processing Finished ---")
