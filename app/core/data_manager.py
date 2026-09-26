import os
import csv
import json
import shutil
import math
import threading
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import config
from app.core.structures import ScanResult


RESULTS_CSV_HEADER = [
    "Step", "Timestamp", "Parameter_P0", "All_Parameters",
    "Atom_UP", "Atom_DW", "Temp_UP", "Temp_DW", "Sigma_UP", "Sigma_DW",
    "Center_UP", "Center_DW", "Amp_UP", "Amp_DW", "Prob_UP_F2", "Prob_DW_F1",
    "Intf_N1", "Intf_N2", "Intf_P1", "Intf_P2",
    "NF_Atom_UP", "NF_Atom_DW", "NF_Temp_UP", "NF_Temp_DW", "NF_Sigma_UP", "NF_Sigma_DW",
    "NF_Center_UP", "NF_Center_DW", "NF_Amp_UP", "NF_Amp_DW", "NF_Prob_UP", "NF_Prob_DW",
    "NF_Intf_N1", "NF_Intf_N2", "NF_Intf_P1", "NF_Intf_P2",
    "Interferometer_Phase_Rad", "Interferometer_Phase_Valid", "Interferometer_Phase_Source_Value",
    "Interferometer_Phase_Calibration_ID", "Interferometer_Phase_Calibration_Name", "Interferometer_Phase_Reference_T2_us2",
    "TailMean_UP", "TailMean_DW",
    "AC_Stark_Ratio", "AC_Stark_Side", "AC_Stark_DDS_Element",
    "AC_Stark_Power_R1", "AC_Stark_Power_R2",
    "AC_Stark_Amplitude_R1", "AC_Stark_Amplitude_R2",
    "AC_Stark_Actual_Power_R1", "AC_Stark_Actual_Power_R2",
    "LockIn_Block", "LockIn_Position", "LockIn_State", "LockIn_Reference",
    "Workflow_Step", "Workflow_Marker", "Workflow_Point", "Workflow_Repeat",
    "Workflow_Shot", "Workflow_Randomized",
    "TTI_Frequency_Hz", "Transfer_Repeat", "TTI_Phase_Deg",
    "Transfer_Zero_Phase_Baseline", "Transfer_Zero_Phase_Block", "Transfer_Zero_Phase_Repeat",
    "Interferometer_Phase_Raw_Rad",
    "Ramsey_Delta_F_MHz", "Ramsey_Repeat", "Ramsey_Center_Frequency_MHz",
    "Ramsey_CH1_Frequency_MHz", "Ramsey_CH2_Frequency_MHz",
    "Ramsey_CH1_Power_dBm", "Ramsey_CH2_Power_dBm",
    "Power_Meter_W", "Power_Meter_Measured_At", "Power_Meter_Wavelength_nm",
    "Power_Meter_Serial", "Power_Meter_Reference_W", "Power_Meter_Deviation_Percent",
    "Power_Meter_Valid", "Power_Meter_Invalid_Reason", "Transfer_Frequency_Attempt",
    "I_Alpha_Applied", "I_Alpha_Calibration_Block", "I_Alpha_Calibration_Shot",
    "Phase_Noise_T2_us2", "Phase_Noise_Repeat", "Phase_Noise_Total_Repeats",
    "Phase_Noise_T_Index", "Phase_Noise_T_Count", "Phase_Noise_Science_Shot",
    "Bragg_Calibration_Stage",
]


class DataManager:
    def __init__(self):
        self.current_run_dir: Path = None
        self.waveforms_dir: Path = None
        self.csv_file: Path = None
        self.csv_writer = None
        self.csv_handle = None
        self.phase_calibration_snapshot = None
        # [NEW] Track Current Run ID for display
        self.current_run_id_str = "run00"
        self.run_log_file: Optional[Path] = None
        self._run_log_lock = threading.Lock()

    @staticmethod
    def _log_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, BaseException):
            return {"type": type(value).__name__, "message": str(value)}
        try:
            json.dumps(value, allow_nan=False)
            return value
        except (TypeError, ValueError):
            return repr(value)

    def log_event(self, event: str, level: str = "INFO", message: str = "", **fields: Any) -> None:
        """Append one durable JSON-lines event. Logging must never interrupt a run."""
        path = self.run_log_file
        if path is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "timestamp_unix_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
            "level": str(level or "INFO").upper(),
            "event": str(event),
            "message": str(message or ""),
            "run_id": self.current_run_id_str,
            **{key: self._log_value(value) for key, value in fields.items()},
        }
        try:
            with self._run_log_lock:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
        except Exception:
            print(f"[Run Log] Failed to write {event}: {traceback.format_exc()}")
    def _get_next_id(self, base_dir: Path) -> int:
        if not base_dir.exists(): return 0
        max_id = -1
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.startswith("run"):
                try:
                    # Extract numeric part after "run" (supports run05 or run05_2026...)
                    num_part = ""
                    for char in item.name[3:]:
                        if char.isdigit(): num_part += char
                        else: break # Stop at first non-digit (like '_')
                    
                    if num_part:
                        cid = int(num_part)
                        if cid > max_id: max_id = cid
                except: continue
        return max_id + 1   
        

    # [NEW] Predict next run ID without creating folder
    def get_next_run_id_str(self) -> str:
        now = datetime.now()
        year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")
        base_day_dir = Path(config.DATA_BASE_DIR) / year / month / day
        
        # Get next ID using smart logic
        run_id = self._get_next_id(base_day_dir)
        
        # Format: runXX_YYYYMMDD
        return f"run{run_id:02d}_{year}{month}{day}"

    def init_run(self, scan_config: Dict[str, Any]):
        now = datetime.now()
        year, month, day = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")
        
        base_day_dir = Path(config.DATA_BASE_DIR) / year / month / day
        os.makedirs(base_day_dir, exist_ok=True)

        # Determine next ID
        run_id = self._get_next_id(base_day_dir)
        
        # Create folder name with date
        run_name = f"run{run_id:02d}_{year}{month}{day}"
        
        self.current_run_dir = base_day_dir / run_name
        self.phase_calibration_snapshot = scan_config.get('_interferometer_phase_calibration_snapshot')
        self.current_run_id_str = run_name
        
        # Ensure directory exists
        if not self.current_run_dir.exists():
            os.makedirs(self.current_run_dir, exist_ok=True)
        self.run_log_file = self.current_run_dir / "run.log"
        self.log_event(
            "run.initialized",
            message="Run archive initialized",
            mode=scan_config.get("mode") or "standard",
            run_label=scan_config.get("run_label") or "",
            sequence_name=scan_config.get("sequence_name") or "",
            sync_run_id=scan_config.get("sync_run_id") or "",
            sync_role=scan_config.get("sync_role") or "",
            sync_node_id=scan_config.get("sync_node_id") or "",
        )
            
        # ... (Rest remains unchanged: waveforms, config.json, etc.) ...
        self.waveforms_dir = self.current_run_dir / "waveforms"
        os.makedirs(self.waveforms_dir, exist_ok=True)
        
        with open(self.current_run_dir / "config.json", 'w') as f:
            json.dump(scan_config, f, indent=4)
            
        src_seq = scan_config.get('_template_path_override') or (
            config.SEQUENCE_TEMPLATE_PATH_WIN if config.USE_SIMULATION else config.SEQUENCE_TEMPLATE_PATH_LINUX
        )
        if os.path.exists(src_seq):
            shutil.copy(src_seq, self.current_run_dir / "sequence.mot")
            
        self.csv_file = self.current_run_dir / "results.csv"
        self._init_csv(self.csv_file)
        
        print(f"[DataManager] Run initialized at: {self.current_run_dir}")

    def archive_ac_stark_plan(
        self,
        original_xml_path: Path,
        generated_xml_path: Path,
        ratio_plan: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        if not self.current_run_dir:
            raise RuntimeError("Run directory is not initialized")
        original_target = self.current_run_dir / "dds_original.xml"
        generated_target = self.current_run_dir / "dds_ac_stark_scan.xml"
        plan_json = self.current_run_dir / "ac_stark_ratio_plan.json"
        plan_csv = self.current_run_dir / "ac_stark_ratio_plan.csv"
        shutil.copy2(original_xml_path, original_target)
        shutil.copy2(generated_xml_path, generated_target)
        with open(plan_json, "w", encoding="utf-8") as handle:
            json.dump(ratio_plan, handle, ensure_ascii=False, indent=2)
        fieldnames = list(ratio_plan[0].keys()) if ratio_plan else []
        with open(plan_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(ratio_plan)
        return {
            "original_xml": str(original_target),
            "generated_xml": str(generated_target),
            "ratio_plan_json": str(plan_json),
            "ratio_plan_csv": str(plan_csv),
        }

    def save_ac_stark_summary(self, summary_rows: List[Dict[str, Any]]) -> None:
        if not self.current_run_dir or not summary_rows:
            return
        json_path = self.current_run_dir / "ac_stark_summary.json"
        csv_path = self.current_run_dir / "ac_stark_summary.csv"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary_rows, handle, ensure_ascii=False, indent=2)
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    def save_lock_in_analysis(self, analysis: Dict[str, Any]) -> None:
        if not self.current_run_dir:
            return
        with open(self.current_run_dir / "lock_in_analysis.json", "w", encoding="utf-8") as handle:
            json.dump(analysis, handle, ensure_ascii=False, indent=2)
        rows = analysis.get("blocks") if isinstance(analysis, dict) else None
        if not isinstance(rows, list) or not rows:
            return
        with open(self.current_run_dir / "lock_in_blocks.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def save_transfer_function_summary(self, summary_rows: List[Dict[str, Any]]) -> None:
        if not self.current_run_dir:
            return
        with open(self.current_run_dir / "transfer_function_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary_rows, handle, ensure_ascii=False, indent=2)
        if not summary_rows:
            return
        with open(self.current_run_dir / "transfer_function_summary.csv", "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                key for key in summary_rows[0].keys()
                if key != "interferometer_phase_s2_components"
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)

    def save_power_monitor_events(self, events: List[Dict[str, Any]]) -> None:
        if not self.current_run_dir:
            return
        with open(self.current_run_dir / "power_monitor_events.json", "w", encoding="utf-8") as handle:
            json.dump(events, handle, ensure_ascii=False, indent=2)

    def save_intf_alpha_calibrations(self, events: List[Dict[str, Any]]) -> None:
        if not self.current_run_dir:
            return
        with open(self.current_run_dir / "intf_alpha_calibrations.json", "w", encoding="utf-8") as handle:
            json.dump(events, handle, ensure_ascii=False, indent=2)

    def invalidate_transfer_attempt(self, frequency_hz: float, attempt: int, reason: str) -> None:
        if not self.waveforms_dir:
            return
        for path in self.waveforms_dir.glob("step_*.npz"):
            try:
                with np.load(path, allow_pickle=True) as archive:
                    values = {key: archive[key] for key in archive.files}
                stored_frequency = float(values.get("transfer_frequency_hz", np.nan))
                stored_attempt = int(values.get("transfer_frequency_attempt", 1))
                if math.isclose(stored_frequency, float(frequency_hz), rel_tol=0.0, abs_tol=1e-9) and stored_attempt == int(attempt):
                    values["power_meter_valid"] = 0
                    values["power_meter_invalid_reason"] = str(reason)
                    np.savez_compressed(path, **values)
            except (OSError, ValueError, TypeError):
                continue
        if self.csv_file and self.csv_file.is_file():
            if self.csv_handle:
                self.csv_handle.flush()
                self.csv_handle.close()
            with open(self.csv_file, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                try:
                    matches = math.isclose(float(row.get("TTI_Frequency_Hz") or "nan"), float(frequency_hz), rel_tol=0.0, abs_tol=1e-9)
                    same_attempt = int(row.get("Transfer_Frequency_Attempt") or 1) == int(attempt)
                except (TypeError, ValueError):
                    matches = same_attempt = False
                if matches and same_attempt:
                    row["Power_Meter_Valid"] = "0"
                    row["Power_Meter_Invalid_Reason"] = str(reason)
            with open(self.csv_file, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_HEADER)
                writer.writeheader()
                writer.writerows(rows)
            self.csv_handle = open(self.csv_file, "a", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_handle)

    def save_phase_noise_summary(self, summary_rows: List[Dict[str, Any]]) -> None:
        if not self.current_run_dir:
            return
        with open(self.current_run_dir / "phase_noise_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary_rows, handle, ensure_ascii=False, indent=2)
        if summary_rows:
            with open(self.current_run_dir / "phase_noise_summary.csv", "w", newline="", encoding="utf-8") as handle:
                fieldnames = [key for key in summary_rows[0].keys() if key != "allan_deviations"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(summary_rows)
            allan_rows = [
                {"t2_us2": row.get("t2_us2"), "t_ms": row.get("t_ms"), **allan}
                for row in summary_rows
                for allan in (row.get("allan_deviations") or [])
                if isinstance(allan, dict)
            ]
            if allan_rows:
                with open(self.current_run_dir / "phase_noise_allan.csv", "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(allan_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(allan_rows)

    def save_bragg_fringe_calibration_result(
        self, result: Dict[str, Any], mot_payload: Optional[bytes] = None, mot_filename: str = ""
    ) -> None:
        if not self.current_run_dir:
            return
        with open(self.current_run_dir / "bragg_fringe_calibration.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        if mot_payload is not None:
            safe_name = Path(mot_filename or "bragg_mid_fringe.mot").name
            (self.current_run_dir / safe_name).write_bytes(mot_payload)

    def _init_csv(self, path: Path):
        self.csv_handle = open(path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_handle)
        self.csv_writer.writerow(RESULTS_CSV_HEADER)
        self.csv_handle.flush()

    def save_point(self, result: ScanResult, step_index: int):
        if not self.current_run_dir: return
        self._write_csv_row(result, step_index)
        
        w_up = result.window_up if result.window_up is not None else [-1, -1]
        w_dw = result.window_dw if result.window_dw is not None else [-1, -1]

        np.savez_compressed(
            self.waveforms_dir / f"step_{step_index:04d}.npz",
            raw_up=result.raw_data_up, 
            raw_dw=result.raw_data_dw,
            fit_up=result.fit_data_up if result.fit_data_up is not None else [],
            fit_dw=result.fit_data_dw if result.fit_data_dw is not None else [],
            time_axis=result.time_axis if result.time_axis is not None else [],
            time_stamp=result.timestamp,
            params=result.all_parameters,
            window_up=w_up, window_dw=w_dw,
            ac_stark_ratio=result.ac_stark_ratio if result.ac_stark_ratio is not None else np.nan,
            ac_stark_side=result.ac_stark_side or "",
            ac_stark_dds_element=result.ac_stark_dds_element if result.ac_stark_dds_element is not None else -1,
            ac_stark_power_r1=result.ac_stark_power_r1 if result.ac_stark_power_r1 is not None else np.nan,
            ac_stark_power_r2=result.ac_stark_power_r2 if result.ac_stark_power_r2 is not None else np.nan,
            ac_stark_amplitude_r1=result.ac_stark_amplitude_r1 if result.ac_stark_amplitude_r1 is not None else -1,
            ac_stark_amplitude_r2=result.ac_stark_amplitude_r2 if result.ac_stark_amplitude_r2 is not None else -1,
            ac_stark_actual_power_r1=result.ac_stark_actual_power_r1 if result.ac_stark_actual_power_r1 is not None else np.nan,
            ac_stark_actual_power_r2=result.ac_stark_actual_power_r2 if result.ac_stark_actual_power_r2 is not None else np.nan,
            lock_in_block_index=result.lock_in_block_index if result.lock_in_block_index is not None else -1,
            lock_in_position=result.lock_in_position if result.lock_in_position is not None else -1,
            lock_in_state=result.lock_in_state or "",
            lock_in_reference=result.lock_in_reference if result.lock_in_reference is not None else 0,
            workflow_step=result.workflow_step if result.workflow_step is not None else -1,
            workflow_marker=result.workflow_marker or "",
            workflow_point=result.workflow_point if result.workflow_point is not None else -1,
            workflow_repeat=result.workflow_repeat if result.workflow_repeat is not None else -1,
            workflow_shot=result.workflow_shot if result.workflow_shot is not None else -1,
            workflow_randomized=(1 if result.workflow_randomized else 0) if result.workflow_randomized is not None else -1,
            transfer_frequency_hz=result.transfer_frequency_hz if result.transfer_frequency_hz is not None else np.nan,
            transfer_repeat=result.transfer_repeat if result.transfer_repeat is not None else -1,
            transfer_phase_deg=result.transfer_phase_deg if result.transfer_phase_deg is not None else np.nan,
            transfer_zero_phase_baseline=1 if result.transfer_zero_phase_baseline else 0,
            transfer_zero_phase_block_id=result.transfer_zero_phase_block_id if result.transfer_zero_phase_block_id is not None else -1,
            transfer_zero_phase_repeat=result.transfer_zero_phase_repeat if result.transfer_zero_phase_repeat is not None else -1,
            interferometer_phase_raw=result.interferometer_phase_raw if result.interferometer_phase_raw is not None else np.nan,
            ramsey_delta_f_mhz=result.ramsey_delta_f_mhz if result.ramsey_delta_f_mhz is not None else np.nan,
            ramsey_repeat=result.ramsey_repeat if result.ramsey_repeat is not None else -1,
            ramsey_center_frequency_mhz=result.ramsey_center_frequency_mhz if result.ramsey_center_frequency_mhz is not None else np.nan,
            ramsey_ch1_frequency_mhz=result.ramsey_ch1_frequency_mhz if result.ramsey_ch1_frequency_mhz is not None else np.nan,
            ramsey_ch2_frequency_mhz=result.ramsey_ch2_frequency_mhz if result.ramsey_ch2_frequency_mhz is not None else np.nan,
            ramsey_ch1_power_dbm=result.ramsey_ch1_power_dbm if result.ramsey_ch1_power_dbm is not None else np.nan,
            ramsey_ch2_power_dbm=result.ramsey_ch2_power_dbm if result.ramsey_ch2_power_dbm is not None else np.nan,
            power_meter_power_w=result.power_meter_power_w if result.power_meter_power_w is not None else np.nan,
            power_meter_measured_at=result.power_meter_measured_at or "",
            power_meter_wavelength_nm=result.power_meter_wavelength_nm if result.power_meter_wavelength_nm is not None else np.nan,
            power_meter_serial_number=result.power_meter_serial_number or "",
            power_meter_reference_w=result.power_meter_reference_w if result.power_meter_reference_w is not None else np.nan,
            power_meter_deviation_percent=result.power_meter_deviation_percent if result.power_meter_deviation_percent is not None else np.nan,
            power_meter_valid=(1 if result.power_meter_valid else 0) if result.power_meter_valid is not None else -1,
            power_meter_invalid_reason=result.power_meter_invalid_reason or "",
            intf_alpha_applied=result.intf_alpha_applied if result.intf_alpha_applied is not None else np.nan,
            intf_alpha_calibration_block_id=result.intf_alpha_calibration_block_id if result.intf_alpha_calibration_block_id is not None else -1,
            intf_alpha_calibration_shot=result.intf_alpha_calibration_shot if result.intf_alpha_calibration_shot is not None else -1,
            transfer_frequency_attempt=result.transfer_frequency_attempt,
        )

    def _write_csv_row(self, result: ScanResult, step_index: int):
        def f(val, prec=8): return f"{val:.{prec}f}" if val is not None else ""
        all_params_str = ";" .join([str(p) for p in result.all_parameters]) if result.all_parameters else ""
        
        row = [
            step_index, f"{result.timestamp:.4f}", f"{result.parameter:.6f}", all_params_str,
            f(result.atom_number_up), f(result.atom_number_dw),
            f(result.temperature_up), f(result.temperature_dw),
            f(result.sigma_up,6), f(result.sigma_dw,6), f(result.arrival_time_up,6), f(result.arrival_time_dw,6),
            f(result.amplitude_up), f(result.amplitude_dw), f(result.transition_probability_up,2), f(result.transition_probability_dw,2),
            # [New] 
            f(result.intf_n1), f(result.intf_n2), f(result.intf_p1, 2), f(result.intf_p2, 2),
            # No Fit
            f(result.atom_number_up_nofit), f(result.atom_number_dw_nofit),
            f(result.temperature_up_nofit), f(result.temperature_dw_nofit),
            f(result.sigma_up_nofit,6), f(result.sigma_dw_nofit,6),
            f(result.arrival_time_up_nofit,6), f(result.arrival_time_dw_nofit,6),
            f(result.amplitude_up_nofit), f(result.amplitude_dw_nofit),
            f(result.transition_probability_up_nofit,2), f(result.transition_probability_dw_nofit,2),
            f(result.intf_n1_nofit), f(result.intf_n2_nofit), f(result.intf_p1_nofit, 2), f(result.intf_p2_nofit, 2),
            f(result.interferometer_phase, 10), 1 if result.interferometer_phase_valid else 0,
            f(result.interferometer_phase_source_value), result.interferometer_phase_calibration_id or "",
            result.interferometer_phase_calibration_name or "", f(result.interferometer_phase_reference_t2_us2, 6),
            f(result.tail_mean_up_raw), f(result.tail_mean_dw_raw),
            f(result.ac_stark_ratio), result.ac_stark_side or "", result.ac_stark_dds_element if result.ac_stark_dds_element is not None else "",
            f(result.ac_stark_power_r1), f(result.ac_stark_power_r2),
            result.ac_stark_amplitude_r1 if result.ac_stark_amplitude_r1 is not None else "",
            result.ac_stark_amplitude_r2 if result.ac_stark_amplitude_r2 is not None else "",
            f(result.ac_stark_actual_power_r1), f(result.ac_stark_actual_power_r2),
            result.lock_in_block_index if result.lock_in_block_index is not None else "",
            result.lock_in_position if result.lock_in_position is not None else "",
            result.lock_in_state or "",
            result.lock_in_reference if result.lock_in_reference is not None else "",
            result.workflow_step if result.workflow_step is not None else "",
            result.workflow_marker or "",
            result.workflow_point if result.workflow_point is not None else "",
            result.workflow_repeat if result.workflow_repeat is not None else "",
            result.workflow_shot if result.workflow_shot is not None else "",
            (1 if result.workflow_randomized else 0) if result.workflow_randomized is not None else "",
            f(result.transfer_frequency_hz, 6),
            result.transfer_repeat if result.transfer_repeat is not None else "",
            f(result.transfer_phase_deg, 6),
            1 if result.transfer_zero_phase_baseline else 0,
            result.transfer_zero_phase_block_id if result.transfer_zero_phase_block_id is not None else "",
            result.transfer_zero_phase_repeat if result.transfer_zero_phase_repeat is not None else "",
            f(result.interferometer_phase_raw, 10),
            f(result.ramsey_delta_f_mhz, 9),
            result.ramsey_repeat if result.ramsey_repeat is not None else "",
            f(result.ramsey_center_frequency_mhz, 9),
            f(result.ramsey_ch1_frequency_mhz, 9),
            f(result.ramsey_ch2_frequency_mhz, 9),
            f(result.ramsey_ch1_power_dbm, 6),
            f(result.ramsey_ch2_power_dbm, 6),
            f(result.power_meter_power_w, 12), result.power_meter_measured_at or "",
            f(result.power_meter_wavelength_nm, 6), result.power_meter_serial_number or "",
            f(result.power_meter_reference_w, 12), f(result.power_meter_deviation_percent, 6),
            (1 if result.power_meter_valid else 0) if result.power_meter_valid is not None else "",
            result.power_meter_invalid_reason or "", result.transfer_frequency_attempt,
            f(result.intf_alpha_applied, 10),
            result.intf_alpha_calibration_block_id if result.intf_alpha_calibration_block_id is not None else "",
            result.intf_alpha_calibration_shot if result.intf_alpha_calibration_shot is not None else "",
            f(result.phase_noise_t2_us2, 9),
            result.phase_noise_repeat if result.phase_noise_repeat is not None else "",
            result.phase_noise_total_repeats if result.phase_noise_total_repeats is not None else "",
            result.phase_noise_t_index if result.phase_noise_t_index is not None else "",
            result.phase_noise_t_count if result.phase_noise_t_count is not None else "",
            result.phase_noise_science_shot if result.phase_noise_science_shot is not None else "",
            result.bragg_calibration_stage or "",
        ]
        self.csv_writer.writerow(row)
        self.csv_handle.flush()

    def close_run(self, status: str = "closed", message: str = ""):
        self.log_event("run.closed", message=message or status, status=status)
        if self.csv_handle:
            self.csv_handle.close()
            self.csv_handle = None
        print("[DataManager] Run saved and closed.")

    def overwrite_run(
        self,
        year,
        month,
        day,
        run_id,
        new_settings: Dict,
        new_data: List[Dict],
        transfer_function_summary: Optional[List[Dict[str, Any]]] = None,
    ):
        target_dir = Path(config.DATA_BASE_DIR) / year / month / day / run_id
        if not target_dir.exists(): raise FileNotFoundError(f"Run {run_id} not found")

        config_path = target_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f: data = json.load(f)
            existing_system = data.get('_system_settings_snapshot') if isinstance(data.get('_system_settings_snapshot'), dict) else {}
            existing_analysis = data.get('_analysis_snapshot') if isinstance(data.get('_analysis_snapshot'), dict) else {}
            data['_system_settings_snapshot'] = {**existing_system, **new_settings}
            data['_analysis_snapshot'] = {**existing_analysis, **new_settings}
            if str(data.get("mode") or "").strip().lower() in {"transfer_function", "transfer_burst_time_scan"}:
                for key in (
                    "transfer_frequency_modulation_mhz",
                    "transfer_atom_mirror_distance_m",
                    "transfer_phase_noise_sigma_mrad",
                ):
                    if key in new_settings:
                        data[key] = new_settings[key]
            with open(config_path, 'w') as f: json.dump(data, f, indent=4)

        csv_path = target_dir / "results.csv"
        shutil.copy(csv_path, str(csv_path) + ".bak")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(RESULTS_CSV_HEADER)
            
            for pt in new_data:
                def f(key, prec=8): val = pt.get(key); return f"{val:.{prec}f}" if val is not None else ""
                all_params = pt.get('all_parameters', []); all_params_str = ";" .join([str(p) for p in all_params])
                
                row = [
                    pt.get('step', 0), pt.get('timestamp', 0), f"{pt.get('parameter', 0):.6f}", all_params_str,
                    f('atom_number_up'), f('atom_number_dw'), f('temperature_up'), f('temperature_dw'),
                    f('sigma_up',6), f('sigma_dw',6), f('arrival_time_up',6), f('arrival_time_dw',6),
                    f('amplitude_up'), f('amplitude_dw'), f('transition_probability_up',2), f('transition_probability_dw',2),
                    f('intf_n1'), f('intf_n2'), f('intf_p1', 2), f('intf_p2', 2),
                    f('atom_number_up_nofit'), f('atom_number_dw_nofit'), f('temperature_up_nofit'), f('temperature_dw_nofit'),
                    f('sigma_up_nofit',6), f('sigma_dw_nofit',6), f('arrival_time_up_nofit',6), f('arrival_time_dw_nofit',6),
                    f('amplitude_up_nofit'), f('amplitude_dw_nofit'), f('transition_probability_up_nofit',2), f('transition_probability_dw_nofit',2),
                    f('intf_n1_nofit'), f('intf_n2_nofit'), f('intf_p1_nofit', 2), f('intf_p2_nofit', 2),
                    f('interferometer_phase', 10), 1 if pt.get('interferometer_phase_valid') else 0,
                    f('interferometer_phase_source_value'), str(pt.get('interferometer_phase_calibration_id') or ''),
                    str(pt.get('interferometer_phase_calibration_name') or ''), f('interferometer_phase_reference_t2_us2', 6),
                    f('tail_mean_up_raw'), f('tail_mean_dw_raw'),
                    f('ac_stark_ratio'), str(pt.get('ac_stark_side') or ''),
                    pt.get('ac_stark_dds_element', ''),
                    f('ac_stark_power_r1'), f('ac_stark_power_r2'),
                    pt.get('ac_stark_amplitude_r1', ''), pt.get('ac_stark_amplitude_r2', ''),
                    f('ac_stark_actual_power_r1'), f('ac_stark_actual_power_r2'),
                    pt.get('lock_in_block_index', ''), pt.get('lock_in_position', ''),
                    str(pt.get('lock_in_state') or ''), pt.get('lock_in_reference', ''),
                    pt.get('workflow_step', ''), str(pt.get('workflow_marker') or ''),
                    pt.get('workflow_point', ''), pt.get('workflow_repeat', ''),
                    pt.get('workflow_shot', ''), pt.get('workflow_randomized', ''),
                    f('transfer_frequency_hz', 6), pt.get('transfer_repeat', ''),
                    f('transfer_phase_deg', 6),
                    1 if pt.get('transfer_zero_phase_baseline') else 0,
                    pt.get('transfer_zero_phase_block_id', ''), pt.get('transfer_zero_phase_repeat', ''),
                    f('interferometer_phase_raw', 10),
                    f('ramsey_delta_f_mhz', 9), pt.get('ramsey_repeat', ''),
                    f('ramsey_center_frequency_mhz', 9),
                    f('ramsey_ch1_frequency_mhz', 9), f('ramsey_ch2_frequency_mhz', 9),
                    f('ramsey_ch1_power_dbm', 6), f('ramsey_ch2_power_dbm', 6),
                    f('power_meter_power_w', 12), str(pt.get('power_meter_measured_at') or ''),
                    f('power_meter_wavelength_nm', 6), str(pt.get('power_meter_serial_number') or ''),
                    f('power_meter_reference_w', 12), f('power_meter_deviation_percent', 6),
                    pt.get('power_meter_valid', ''), str(pt.get('power_meter_invalid_reason') or ''),
                    pt.get('transfer_frequency_attempt', 1), f('intf_alpha_applied', 10),
                    pt.get('intf_alpha_calibration_block_id', ''), pt.get('intf_alpha_calibration_shot', ''),
                    f('phase_noise_t2_us2', 9), pt.get('phase_noise_repeat', ''),
                    pt.get('phase_noise_total_repeats', ''), pt.get('phase_noise_t_index', ''),
                    pt.get('phase_noise_t_count', ''), pt.get('phase_noise_science_shot', ''),
                    str(pt.get('bragg_calibration_stage') or ''),
                ]
                writer.writerow(row)

        if transfer_function_summary is not None:
            with open(target_dir / "transfer_function_summary.json", "w", encoding="utf-8") as handle:
                json.dump(transfer_function_summary, handle, ensure_ascii=False, indent=2)
            if transfer_function_summary:
                fieldnames = [
                    key for key in transfer_function_summary[0].keys()
                    if key != "interferometer_phase_s2_components"
                ]
                with open(target_dir / "transfer_function_summary.csv", "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(transfer_function_summary)
        
        print(f"[DataManager] Run {run_id} overwritten.")
    
    # =========================================================
    # [新增] 数据清洗与加载功能 (对应 archive.html 的加载请求)
    # =========================================================

    def _sanitize_data(self, data):
        """
        递归遍历数据，将所有的 NaN (Not a Number) 和 Infinity (无穷大)
        替换为 0.0。这是为了解决 JSON 序列化报错导致的前端 500 错误。
        """
        import math # 确保导入 math 库
        
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(v) for v in data]
        elif isinstance(data, float):
            # 核心逻辑：检查是否为坏死数据
            if math.isnan(data) or math.isinf(data):
                return 0.0
        return data

    def load_run(self, run_id_str: str) -> Dict[str, Any]:
        """
        加载指定 Run ID 的数据，并自动清洗 NaN。
        对应 archive.html 中 loadRun() 调用的后端接口。
        """
        try:
            # 1. 根据 ID 解析日期 (格式: runXX_YYYYMMDD)
            # 例如: run68_20260123 -> 2026, 01, 23
            parts = run_id_str.split('_')
            if len(parts) < 2:
                # 如果格式不对，尝试直接在当天目录找（根据您的逻辑调整）
                print(f"[DataManager] Invalid Run ID format: {run_id_str}")
                return {"error": "Invalid Run ID format"}
            
            date_str = parts[-1] # 获取 YYYYMMDD
            year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
            
            # 2. 构造文件路径
            # 路径规则: config.DATA_BASE_DIR / YYYY / MM / DD / runID / runID.json
            run_dir = Path(config.DATA_BASE_DIR) / year / month / day / run_id_str
            file_path = run_dir / f"{run_id_str}.json"
            
            # 兼容性检查：如果主 JSON 不在，尝试找 config.json (旧版本可能只有 config.json)
            if not file_path.exists():
                fallback_path = run_dir / "config.json"
                if fallback_path.exists():
                    file_path = fallback_path
                else:
                    print(f"[DataManager] File not found: {file_path}")
                    return {"error": "File not found"}

            # 3. 读取文件
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
            
            # 4. [关键步骤] 清洗数据！
            # 这会将文件里存储的 "NaN" 替换为 0.0，修复 500 错误
            clean_data = self._sanitize_data(raw_data)
            
            print(f"[DataManager] Loaded and sanitized {run_id_str}")
            return clean_data

        except Exception as e:
            print(f"[DataManager Error] Load failed for {run_id_str}: {e}")
            # 这里抛出异常以便 FastAPI 返回 500，但此时我们已经在控制台打印了原因
            raise e
