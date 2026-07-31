import time
import threading
import traceback
import json
import random
import math
import queue
import os
import shutil
import shlex
import subprocess
import numpy as np
from pathlib import Path
from itertools import product
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import asdict

import config
from app.drivers.hardware import SequenceEditor, ExperimentDriver, RedPitayaDriver
from app.drivers.vcd_parser import VCDParser
from app.analysis import fitting, physics
from app.models.schemas import ScanConfig
from app.core.data_manager import DataManager
from app.core.structures import ExperimentStatus, ScanResult
from app.core.pulse_generator import generate_bragg_pulse

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
        self._activity_lock = threading.Lock()
        self._active_mode: Optional[str] = None

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

    def load_run_preset(self, year: str, month: str, day: str, run_id: str) -> Dict[str, Any]:
        if getattr(self.status, "is_running", False):
            raise ValueError("Cannot load a previous run while an experiment is running")

        run_dir = Path(config.DATA_BASE_DIR) / year / month / day / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        restored_config = self._load_run_preset_config(run_dir)
        source_sequence = run_dir / "sequence.mot"
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

        return self._normalize_atom_area_settings(
            self._normalize_fit_settings(
                self._normalize_update_settings(
                    self._normalize_hardware_settings(self._normalize_tmot_settings(base_settings))
                ),
                strict=False
            )
        )

    def _save_settings_to_disk(self):
        try:
            with open(config.SETTINGS_FILE_PATH, 'w') as f:
                json.dump(self.settings, f, indent=4)
            print(f"[Settings] Saved to {config.SETTINGS_FILE_PATH}")
        except Exception as e:
            print(f"[Settings] Save failed: {e}")

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

    def update_settings(self, new_settings: Dict[str, Any]):
        if new_settings.get('tmot_args') is None:
            new_settings['tmot_args'] = self.settings.get('tmot_args', self._default_tmot_args())
        new_settings = self._normalize_atom_area_settings(
            self._normalize_fit_settings(
                self._normalize_update_settings(
                    self._normalize_hardware_settings(self._normalize_tmot_settings(new_settings))
                ),
                strict=True
            )
        )
        self.settings.update(new_settings)
        self._apply_runtime_settings()
        self._save_settings_to_disk()
        print(f">>> System Settings Updated: {self.settings}")
    

    def get_analysis_config(self) -> Dict[str, Any]:
        return self.settings

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

    def start_scan(self, scan_config: Dict[str, Any]) -> Dict[str, str]:
        if not hasattr(self, 'status'):
            self.status = ExperimentStatus()

        acquired, busy_message = self.acquire_run_slot('scan')
        if not acquired:
            return {'status': 'error', 'message': busy_message}

        self.refresh_runtime_settings_from_disk()

        try:
            parameters = self._generate_parameters(scan_config)
        except Exception as exc:
            self.release_run_slot('scan')
            return {'status': 'error', 'message': f'Param generation failed: {str(exc)}'}

        try:
            fit_config = self.build_fit_config(scan_config)
        except Exception as exc:
            self.release_run_slot('scan')
            return {'status': 'error', 'message': str(exc)}

        self.stop_flag = False
        self.status = ExperimentStatus(is_running=True, total_steps=len(parameters), message='Starting...')

        try:
            scan_config['_system_settings_snapshot'] = self.settings
            self.data_manager.init_run(scan_config)
        except Exception as exc:
            self.status = ExperimentStatus()
            self.release_run_slot('scan')
            return {'status': 'error', 'message': f'Data Init Failed: {str(exc)}'}

        with self.data_queue.mutex:
            self.data_queue.queue.clear()

        self.proc_thread = threading.Thread(target=self._processing_loop, args=(fit_config,))
        self.proc_thread.daemon = True
        self.proc_thread.start()

        self.acq_thread = threading.Thread(target=self._acquisition_loop, args=(parameters, scan_config))
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
            template_path = config.SEQUENCE_TEMPLATE_PATH_WIN if config.USE_SIMULATION else self.settings['template_path']
            self.seq_editor.generate_sequence(template_path, config.SEQUENCE_OUTPUT_PATH, actual_params_to_write)

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
        params = job.get('params') or []
        payload = {
            'stream_type': stream_type,
            'parameter': params[0] if params else None,
            'all_parameters': params,
            'scan_dimensions': int(job.get('scan_dimensions', 1) or 1),
            'error': message,
            'current_step': int(job.get('idx', 0)) + 1,
            'total_steps': int(job.get('total', 1) or 1),
        }
        payload.update(job.get('metadata') or {})
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
        scan_dimensions = int(job.get('scan_dimensions', 1) or 1)
        primary_param = params[0] if params else None
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
                all_parameters=params,
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
    def _acquisition_loop(self, parameter_list: List[Any], scan_config: Dict[str, Any]):
        print(f"--- Acquisition Started: {len(parameter_list)} points ---")
        total_steps = len(parameter_list)
        scan_dimensions = self._resolve_scan_dimensions(scan_config)

        for idx, param_set in enumerate(parameter_list):
            if self.stop_flag:
                break
            job = self.execute_single_measurement(
                param_set,
                scan_config,
                idx=idx,
                total_steps=total_steps,
                scan_dimensions=scan_dimensions,
            )
            self.data_queue.put(job)
            if config.USE_SIMULATION:
                time.sleep(0.1)

        self.data_queue.put(None)
        print("--- Acquisition Finished ---")

    # --- THREAD 2: PROCESSING (CONSUMER) ---
    def _processing_loop(self, fit_config: Dict[str, Any]):
        print("--- Processing Thread Started ---")
        selected_fit_model = fitting.get_fit_model_by_key(
            fit_config.get('models'),
            fit_config.get('model_key'),
        )
        print(f"[Fit] Using model: {selected_fit_model['label']} ({selected_fit_model['key']})")

        try:
            while True:
                try:
                    job = self.data_queue.get(timeout=5)
                except queue.Empty:
                    if self.stop_flag:
                        break
                    continue
                if job is None:
                    break

                params = job.get('params') or []
                self.status.current_step = int(job.get('idx', 0)) + 1
                self.status.total_steps = int(job.get('total', 1) or 1)
                self.status.message = f"Processing: {params} (Queue: {self.data_queue.qsize()})"

                _, payload = self.process_measurement_job(
                    job,
                    fit_config,
                    data_manager=self.data_manager,
                    stream_type='scan_point',
                )
                if self.on_data_ready:
                    self.on_data_ready(payload)
        finally:
            self.data_manager.close_run()
            self.status.is_running = False
            self.status.message = 'Stopped' if self.stop_flag else 'Done'
            self.release_run_slot('scan')
            print("--- Processing Finished ---")
