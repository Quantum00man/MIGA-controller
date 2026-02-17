import os
import csv
import json
import math  # <--- [修复1] 必须导入 math 库
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import config

class DataLoader:
    def __init__(self):
        self.base_dir = config.DATA_BASE_DIR

    def get_archive_tree(self) -> Dict[str, Any]:
        tree = {}
        if not self.base_dir.exists(): return tree
        for year_dir in sorted(self.base_dir.iterdir()):
            if year_dir.is_dir():
                year = year_dir.name
                tree[year] = {}
                for month_dir in sorted(year_dir.iterdir()):
                    if month_dir.is_dir():
                        month = month_dir.name
                        tree[year][month] = {}
                        for day_dir in sorted(month_dir.iterdir()):
                            if day_dir.is_dir():
                                day = day_dir.name
                                runs = [r.name for r in sorted(day_dir.iterdir()) if r.is_dir() and r.name.startswith("run")]
                                tree[year][month][day] = runs
        return tree

    # [新增] 递归清洗函数
    def _sanitize_structure(self, data):
        """递归将 NaN/Infinity 转换为 None 或 0.0"""
        if isinstance(data, dict):
            return {k: self._sanitize_structure(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_structure(v) for v in data]
        elif isinstance(data, float):
            if math.isnan(data) or math.isinf(data):
                return 0.0 # 或者 None
        return data

    def load_run(self, year: str, month: str, day: str, run_id: str) -> Dict[str, Any]:
        run_dir = self.base_dir / year / month / day / run_id
        if not run_dir.exists(): raise FileNotFoundError(f"Run not found: {run_dir}")

        config_data = {}
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f: 
                    raw_config = json.load(f)
                    # [修复2] 清洗 Config 数据，防止 settings 里有 NaN
                    config_data = self._sanitize_structure(raw_config)
            except: pass

        data_points = []
        csv_path = run_dir / "results.csv"
        
        MAX_DISPLAY_POINTS = 3000
        
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = list(csv.DictReader(f)) 
                total_rows = len(reader)
                
                step = 1
                if total_rows > MAX_DISPLAY_POINTS:
                    step = total_rows // MAX_DISPLAY_POINTS
                
                for row in reader[::step]:
                    # 这里的 safe_float 之前会因为缺少 math 库而报错
                    def safe_float(key): 
                        val = row.get(key)
                        if not val or val.strip() == "": return None
                        try:
                            f = float(val)
                            if math.isnan(f) or math.isinf(f): return None
                            return f
                        except ValueError: return None
                    
                    point = {
                        "step": int(row["Step"]) if row.get("Step") else 0,
                        "parameter": safe_float("Parameter_P0") if safe_float("Parameter_P0") is not None else 0.0,
                        "all_parameters": [float(x) for x in row["All_Parameters"].split(";")] if row.get("All_Parameters") else [],
                        
                        "atom_number_up": safe_float("Atom_UP"), "atom_number_dw": safe_float("Atom_DW"),
                        "temperature_up": safe_float("Temp_UP"), "temperature_dw": safe_float("Temp_DW"),
                        "sigma_up": safe_float("Sigma_UP"), "sigma_dw": safe_float("Sigma_DW"),
                        "amplitude_up": safe_float("Amp_UP"), "amplitude_dw": safe_float("Amp_DW"),
                        "arrival_time_up": safe_float("Center_UP"), "arrival_time_dw": safe_float("Center_DW"),
                        "transition_probability_up": safe_float("Prob_UP_F2"), "transition_probability_dw": safe_float("Prob_DW_F1"),
                        
                        "atom_number_up_nofit": safe_float("NF_Atom_UP"), "atom_number_dw_nofit": safe_float("NF_Atom_DW"),
                        "temperature_up_nofit": safe_float("NF_Temp_UP"), "temperature_dw_nofit": safe_float("NF_Temp_DW"),
                        "sigma_up_nofit": safe_float("NF_Sigma_UP"), "sigma_dw_nofit": safe_float("NF_Sigma_DW"),
                        "amplitude_up_nofit": safe_float("NF_Amp_UP"), "amplitude_dw_nofit": safe_float("NF_Amp_DW"),
                        "arrival_time_up_nofit": safe_float("NF_Center_UP"), "arrival_time_dw_nofit": safe_float("NF_Center_DW"),
                        "transition_probability_up_nofit": safe_float("NF_Prob_UP"), "transition_probability_dw_nofit": safe_float("NF_Prob_DW"),
                    }
                    data_points.append(point)

        return { "config": config_data, "data": data_points }

    def load_waveform(self, year: str, month: str, day: str, run_id: str, step_index: int) -> Dict[str, Any]:
        npz_path = self.base_dir / year / month / day / run_id / "waveforms" / f"step_{step_index:04d}.npz"
        if not npz_path.exists(): raise FileNotFoundError(f"Waveform not found")
        try:
            with np.load(npz_path) as data:
                def get_arr(k): 
                    # [修复3] 波形数据也要清洗
                    arr = data[k].tolist() if k in data else []
                    return [0.0 if (math.isnan(x) or math.isinf(x)) else x for x in arr]
                
                def get_val(k): return data[k].tolist() if k in data else None
                
                return {
                    "time_axis": get_arr('time_axis'),
                    "raw_up": get_arr('raw_up'), "raw_dw": get_arr('raw_dw'),
                    "fit_up": get_arr('fit_up'), "fit_dw": get_arr('fit_dw'),
                    "window_up": get_val('window_up'), "window_dw": get_val('window_dw')
                }
        except Exception as e: raise RuntimeError(f"Failed to load waveform: {str(e)}")