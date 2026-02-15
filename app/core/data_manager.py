import os
import csv
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import config
from app.core.structures import ScanResult

class DataManager:
    def __init__(self):
        self.current_run_dir: Path = None
        self.waveforms_dir: Path = None
        self.csv_file: Path = None
        self.csv_writer = None
        self.csv_handle = None
        # [NEW] Track Current Run ID for display
        self.current_run_id_str = "run00"
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
        self.current_run_id_str = run_name
        
        # Ensure directory exists
        if not self.current_run_dir.exists():
            os.makedirs(self.current_run_dir, exist_ok=True)
            
        # ... (Rest remains unchanged: waveforms, config.json, etc.) ...
        self.waveforms_dir = self.current_run_dir / "waveforms"
        os.makedirs(self.waveforms_dir, exist_ok=True)
        
        with open(self.current_run_dir / "config.json", 'w') as f:
            json.dump(scan_config, f, indent=4)
            
        src_seq = config.SEQUENCE_TEMPLATE_PATH_WIN if config.USE_SIMULATION else config.SEQUENCE_TEMPLATE_PATH_LINUX
        if os.path.exists(src_seq):
            shutil.copy(src_seq, self.current_run_dir / "sequence.mot")
            
        self.csv_file = self.current_run_dir / "results.csv"
        self._init_csv(self.csv_file)
        
        print(f"[DataManager] Run initialized at: {self.current_run_dir}")
    def _init_csv(self, path: Path):
        self.csv_handle = open(path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_handle)
        # Expanded Header
        header = [
            "Step", "Timestamp", "Parameter_P0", "All_Parameters",
            "Atom_UP", "Atom_DW", "Temp_UP", "Temp_DW", "Sigma_UP", "Sigma_DW", "Center_UP", "Center_DW", "Amp_UP", "Amp_DW", "Prob_UP_F2", "Prob_DW_F1",
            "NF_Atom_UP", "NF_Atom_DW", "NF_Temp_UP", "NF_Temp_DW", "NF_Sigma_UP", "NF_Sigma_DW", "NF_Center_UP", "NF_Center_DW", "NF_Amp_UP", "NF_Amp_DW", "NF_Prob_UP", "NF_Prob_DW"
        ]
        self.csv_writer.writerow(header)
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
            window_up=w_up, window_dw=w_dw
        )

    def _write_csv_row(self, result: ScanResult, step_index: int):
        def f(val, prec=4): return f"{val:.{prec}f}" if val is not None else ""
        all_params_str = ";" .join([str(p) for p in result.all_parameters]) if result.all_parameters else ""
        
        row = [
            step_index, f"{result.timestamp:.4f}", f"{result.parameter:.6f}", all_params_str,
            f(result.atom_number_up), f(result.atom_number_dw),
            f(result.temperature_up), f(result.temperature_dw),
            f(result.sigma_up,6), f(result.sigma_dw,6), f(result.arrival_time_up,6), f(result.arrival_time_dw,6),
            f(result.amplitude_up), f(result.amplitude_dw), f(result.transition_probability_up,2), f(result.transition_probability_dw,2),
            # No Fit
            f(result.atom_number_up_nofit), f(result.atom_number_dw_nofit),
            f(result.temperature_up_nofit), f(result.temperature_dw_nofit),
            f(result.sigma_up_nofit,6), f(result.sigma_dw_nofit,6),
            f(result.arrival_time_up_nofit,6), f(result.arrival_time_dw_nofit,6),
            f(result.amplitude_up_nofit), f(result.amplitude_dw_nofit),
            f(result.transition_probability_up_nofit,2), f(result.transition_probability_dw_nofit,2),
        ]
        self.csv_writer.writerow(row)
        self.csv_handle.flush()

    def close_run(self):
        if self.csv_handle:
            self.csv_handle.close()
            self.csv_handle = None
        print("[DataManager] Run saved and closed.")

    def overwrite_run(self, year, month, day, run_id, new_settings: Dict, new_data: List[Dict]):
        target_dir = Path(config.DATA_BASE_DIR) / year / month / day / run_id
        if not target_dir.exists(): raise FileNotFoundError(f"Run {run_id} not found")

        config_path = target_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f: data = json.load(f)
            data['_system_settings_snapshot'] = new_settings
            data['_analysis_snapshot'] = new_settings 
            with open(config_path, 'w') as f: json.dump(data, f, indent=4)

        csv_path = target_dir / "results.csv"
        shutil.copy(csv_path, str(csv_path) + ".bak")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                "Step", "Timestamp", "Parameter_P0", "All_Parameters",
                "Atom_UP", "Atom_DW", "Temp_UP", "Temp_DW", "Sigma_UP", "Sigma_DW", "Center_UP", "Center_DW", "Amp_UP", "Amp_DW", "Prob_UP_F2", "Prob_DW_F1",
                "NF_Atom_UP", "NF_Atom_DW", "NF_Temp_UP", "NF_Temp_DW", "NF_Sigma_UP", "NF_Sigma_DW", "NF_Center_UP", "NF_Center_DW", "NF_Amp_UP", "NF_Amp_DW", "NF_Prob_UP", "NF_Prob_DW"
            ]
            writer.writerow(header)
            
            for pt in new_data:
                def f(key, prec=4): val = pt.get(key); return f"{val:.{prec}f}" if val is not None else ""
                all_params = pt.get('all_parameters', []); all_params_str = ";" .join([str(p) for p in all_params])
                
                row = [
                    pt.get('step', 0), pt.get('timestamp', 0), f"{pt.get('parameter', 0):.6f}", all_params_str,
                    f('atom_number_up'), f('atom_number_dw'), f('temperature_up'), f('temperature_dw'),
                    f('sigma_up',6), f('sigma_dw',6), f('arrival_time_up',6), f('arrival_time_dw',6),
                    f('amplitude_up'), f('amplitude_dw'), f('transition_probability_up',2), f('transition_probability_dw',2),
                    f('atom_number_up_nofit'), f('atom_number_dw_nofit'), f('temperature_up_nofit'), f('temperature_dw_nofit'),
                    f('sigma_up_nofit',6), f('sigma_dw_nofit',6), f('arrival_time_up_nofit',6), f('arrival_time_dw_nofit',6),
                    f('amplitude_up_nofit'), f('amplitude_dw_nofit'), f('transition_probability_up_nofit',2), f('transition_probability_dw_nofit',2),
                ]
                writer.writerow(row)
        
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
