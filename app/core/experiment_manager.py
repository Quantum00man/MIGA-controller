import time
import threading
import traceback
import json
import random
import math
import queue
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import asdict

import config
from app.drivers.hardware import SequenceEditor, ExperimentDriver, RedPitayaDriver
from app.drivers.vcd_parser import VCDParser
from app.analysis import fitting, physics
from app.core.data_manager import DataManager
from app.core.structures import ExperimentStatus, ScanResult

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

        self.status = ExperimentStatus()
        self.stop_flag = False
        
        self.data_queue = queue.Queue()
        self.acq_thread: Optional[threading.Thread] = None
        self.proc_thread: Optional[threading.Thread] = None
        self.on_data_ready: Optional[Callable[[Dict[str, Any]], None]] = None

    def _load_initial_settings(self) -> Dict[str, Any]:
        base_settings = {
            "voltage_limit": .015,
            "rp_ip_red": config.RP_IP_RED_REAL,
            "rp_ip_green": config.RP_IP_GREEN_REAL,
            "network_timeout": config.NETWORK_TIMEOUT,
            **config.DEFAULT_ANALYSIS_SETTINGS,
            "g_const": config.G_CONST,
            "link_total_time": config.LINK_TOTAL_TIME,
            "tmot_path": config.TMOT_BINARY_PATH_WIN if config.IS_WINDOWS else config.TMOT_BINARY_PATH_LINUX,
            "cmot_path": config.CMOT_BINARY_PATH_WIN if config.IS_WINDOWS else config.CMOT_BINARY_PATH_LINUX,
            "template_path": config.SEQUENCE_TEMPLATE_PATH_WIN if config.IS_WINDOWS else config.SEQUENCE_TEMPLATE_PATH_LINUX,
            "intf_alpha": 0.35,
            "intf_beta": 0.07636,
            "intf_gamma": 0.25
            
        }
        
        settings_path = Path(config.SETTINGS_FILE_PATH)
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    saved_settings = json.load(f)
                for k, v in saved_settings.items():
                    if k in base_settings: base_settings[k] = v
                print(f"[Settings] Loaded user settings from {settings_path}")
            except Exception as e:
                print(f"[Settings] Failed to load user settings: {e}")
        
        return base_settings

    def _save_settings_to_disk(self):
        try:
            with open(config.SETTINGS_FILE_PATH, 'w') as f:
                json.dump(self.settings, f, indent=4)
            print(f"[Settings] Saved to {config.SETTINGS_FILE_PATH}")
        except Exception as e:
            print(f"[Settings] Save failed: {e}")

    def get_settings(self) -> Dict[str, Any]: return self.settings

    def update_settings(self, new_settings: Dict[str, Any]):
        self.settings.update(new_settings)
        self.rp_driver_red.real_ip = self.settings['rp_ip_red']
        self.rp_driver_red.timeout = int(self.settings['network_timeout'])
        self.rp_driver_green.real_ip = self.settings['rp_ip_green']
        self.rp_driver_green.timeout = int(self.settings['network_timeout'])
        self._save_settings_to_disk()
        print(f">>> System Settings Updated: {self.settings}")
    
    def get_analysis_config(self) -> Dict[str, Any]: return self.settings
    def update_analysis_config(self, new_config: Dict[str, Any]): self.update_settings(new_config)
    def set_simulation_mode(self, enabled: bool):
        config.USE_SIMULATION = enabled
        print(f">>> System Mode Switched: {'SIMULATION' if enabled else 'REAL HARDWARE'}")

    def start_scan(self, scan_config: Dict[str, Any]) -> Dict[str, str]:
        if not hasattr(self, 'status'): self.status = ExperimentStatus()
        if self.status.is_running: return {"status": "error", "message": "Experiment already running"}

        # ========================================================
        # [关键修复]：在开始前，强制重新读取配置文件并应用到驱动
        # ========================================================
        try:
            print("[Manager] Auto-configuring driver before scan...")
            
            # 1. 重新从硬盘加载最新配置
            # 这一步确保即使重启程序没有点Save，也能读到上次保存的配置
            current_settings = self._load_initial_settings()
            
            # 2. 更新内存中的 settings
            self.settings.update(current_settings)
            
            # 3. 强制配置 RP 驱动 (RedPitayaDriver)
            # 这会触发 IP 切换 (RP <-> DAQ) 和 采样率下发给 Server
            # 注意：我们主要用 rp_driver_red 来控制 DAQ Server
            if hasattr(self, 'rp_driver_red'):
                self.rp_driver_red.configure(self.settings)
                print("[Manager] Driver configured successfully.")
                
        except Exception as e:
            print(f"[Manager Error] Failed to auto-configure driver: {e}")
            # 不阻断流程，万一配置失败也尝试继续跑
        # ========================================================

        try: parameters = self._generate_parameters(scan_config)
        except Exception as e: return {"status": "error", "message": f"Param generation failed: {str(e)}"}

        self.stop_flag = False
        self.status = ExperimentStatus(is_running=True, total_steps=len(parameters), message="Starting...")
        
        try:
            scan_config['_system_settings_snapshot'] = self.settings
            self.data_manager.init_run(scan_config)
        except Exception as e: return {"status": "error", "message": f"Data Init Failed: {str(e)}"}

        fit_config = {
            'center_up': float(scan_config.get('fit_center_up', 0)),
            'width_up': float(scan_config.get('fit_width_up', 0)),
            'center_dw': float(scan_config.get('fit_center_dw', 0)),
            'width_dw': float(scan_config.get('fit_width_dw', 0))
        }

        with self.data_queue.mutex: self.data_queue.queue.clear()

        self.proc_thread = threading.Thread(target=self._processing_loop, args=(fit_config,))
        self.proc_thread.daemon = True
        self.proc_thread.start()

        self.acq_thread = threading.Thread(target=self._acquisition_loop, args=(parameters,))
        self.acq_thread.daemon = True
        self.acq_thread.start()

        return {"status": "success", "message": "Scan started (Parallel)"}

    def stop_scan(self) -> Dict[str, str]:
        if self.status.is_running:
            self.stop_flag = True
            self.status.message = "Stopping..."
            return {"status": "success", "message": "Stop signal sent"}
        return {"status": "warning", "message": "No experiment running"}

    def _generate_parameters(self, config: Dict[str, Any]) -> List[Any]:
        def get_values(ptype, start, stop, step, clist, target_type='float'):
            raw_vals = []
            if ptype == 'list':
                try: return [float(x.strip()) for x in clist.split(',') if x.strip()]
                except ValueError: raise ValueError(f"Invalid list format: {clist}")
            else:
                start, stop, step = float(start), float(stop), float(step)
                if step == 0: raw_vals = [start]
                else:
                    num = int(abs((stop - start) / step)) + 1
                    raw_vals = np.linspace(start, start + (num-1)*step, num).tolist()
            if target_type == 'int': return [int(x) for x in raw_vals]
            else: return [round(x, 6) for x in raw_vals]

        param_type = config.get('param_type', 'float')
        vals_1 = get_values(config.get('dim1_type', 'range'), config.get('start', 0), config.get('stop', 10), config.get('step', 1), config.get('custom_list', ''), target_type=param_type)
        mode = config.get('mode', 'standard')
        mode_param = float(config.get('mode_param') or 0.0)
        link_formulas = config.get('link_formulas', [])
        sets_1 = []
        for v in vals_1:
            s = [v] 
            if mode == 'timing': s.append(round(mode_param - v, 6))
            elif mode == 'rabi': s.append(round(mode_param - v/2.0, 6))
            elif mode == 'half': s.append(round(v/2.0, 6))
            elif mode == 'link':
                eval_ctx = {"P0": v, "math": math, "np": np}
                for i, formula_str in enumerate(link_formulas):
                    try: val = float(eval(formula_str, {"__builtins__": {}}, eval_ctx)); eval_ctx[f"P{i+1}"] = val; s.append(round(val, 6))
                    except Exception: raise ValueError(f"Formula Error: {formula_str}")
            sets_1.append(s)
        sets_2 = [[]]
        if config.get('dim2_enabled', False):
            vals_2 = get_values(config.get('dim2_type', 'range'), config.get('dim2_start', 0), config.get('dim2_stop', 10), config.get('dim2_step', 1), config.get('dim2_list', ''), target_type='float')
            sets_2 = [[v] for v in vals_2]
        final_list = []
        for s1 in sets_1:
            for s2 in sets_2:
                final_list.append(s1 + s2)
        full_scan = []
        for _ in range(int(config.get('averages', 1))): full_scan.extend(final_list)
        if config.get('randomize', False): random.shuffle(full_scan)
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

    # --- THREAD 1: ACQUISITION (PRODUCER) ---
    def _acquisition_loop(self, parameter_list: List[Any]):
        print(f"--- Acquisition Started: {len(parameter_list)} points ---")
        S = self.settings 
        total_steps = len(parameter_list)

        for idx, param_set in enumerate(parameter_list):
            if self.stop_flag: break

            if not isinstance(param_set, list): params_to_write = [param_set]
            else: params_to_write = param_set
            
            tpl = config.SEQUENCE_TEMPLATE_PATH_WIN if config.USE_SIMULATION else S['template_path']
            self.seq_editor.generate_sequence(tpl, config.SEQUENCE_OUTPUT_PATH, params_to_write)
            
            cmot_bin = config.CMOT_BINARY_PATH_WIN if config.USE_SIMULATION else S['cmot_path']
            self.driver.compile_vcd(config.SEQUENCE_OUTPUT_PATH, config.VCD_OUTPUT_PATH, binary_path=cmot_bin)
            
            # [MODIFIED] Use new robust calculator
            if config.USE_SIMULATION: 
                start_delay = 0.78 
            else:
                start_delay = self._calculate_delay_from_vcd(
                    config.VCD_OUTPUT_PATH, 
                    str(S['chan_launch']), 
                    str(S['chan_trigger'])
                )

            tmot_bin = config.TMOT_BINARY_PATH_WIN if config.USE_SIMULATION else S['tmot_path']
            success = self.driver.run_sequence(config.SEQUENCE_OUTPUT_PATH, binary_path=tmot_bin)
            
            if not success:
                print(f"[Acq] Failed at step {idx+1}")
                continue

            t_trig, volt_up_raw = self.rp_driver_red.acquire_channel("ch1")
            _, volt_dw_raw = self.rp_driver_red.acquire_channel("ch2")

            job = {
                'idx': idx, 'total': total_steps, 'params': params_to_write,
                'start_delay': start_delay, 'volt_up': volt_up_raw, 'volt_dw': volt_dw_raw,
                'timestamp': time.time()
            }
            self.data_queue.put(job)
            if config.USE_SIMULATION: time.sleep(0.1)
        
        self.data_queue.put(None)
        print("--- Acquisition Finished ---")

    # --- THREAD 2: PROCESSING (CONSUMER) ---
    def _processing_loop(self, fit_config: Dict[str, float]):
        print("--- Processing Thread Started ---")
        S = self.settings
        STORAGE_STEP = 1 
        
        while True:
            try: job = self.data_queue.get(timeout=5)
            except queue.Empty:
                if self.stop_flag: break
                continue
            if job is None: break 

            idx = job['idx']; total = job['total']; params = job['params']
            primary_param = params[0]; start_delay = job['start_delay']
            volt_up_raw = job['volt_up']; volt_dw_raw = job['volt_dw']
            
            self.status.current_step = idx + 1
            self.status.total_steps = total
            self.status.message = f"Processing: {params} (Queue: {self.data_queue.qsize()})"

            if not volt_up_raw or not volt_dw_raw:
                 if self.on_data_ready: self.on_data_ready({'parameter': primary_param, 'error': 'No Data', 'current_step': idx+1, 'total_steps': total})
                 continue

            try:
                # ==========================================================
                # 1. 先计算 Offset (基线)
                # ==========================================================
                if len(volt_up_raw) > 300: 
                    offset_up = np.mean(volt_up_raw[-200:])
                    offset_dw = np.mean(volt_dw_raw[-200:])
                else: 
                    offset_up = 0.0; offset_dw = 0.0

                # ==========================================================
                # 2. 计算“净电压” (Raw - Offset)
                #    注意：此时还没有除以 Gain，是真实的物理电压波动
                # ==========================================================
                val_up_clean = np.array(volt_up_raw) - offset_up
                val_dw_clean = np.array(volt_dw_raw) - offset_dw

                # ==========================================================
                # 3. 执行 Voltage Limit 检查 (基于净电压)
                # ==========================================================
                # 获取阈值 (默认为 9.5V)
                v_limit = float(S.get('voltage_limit', 9.5))
                
                # 计算最大波动幅度 (取绝对值)
                max_amp_up = np.max(np.abs(val_up_clean))
                max_amp_dw = np.max(np.abs(val_dw_clean))

                if max_amp_up > v_limit or max_amp_dw > v_limit:
                    msg = f"Signal Amplitude > {v_limit}V (UP={max_amp_up:.2f}V, DW={max_amp_dw:.2f}V)"
                    print(f"[Filter] Rejected (Step {idx+1}): {msg}")
                    
                    if self.on_data_ready:
                        self.on_data_ready({
                            'parameter': primary_param, 
                            'error': msg, 
                            'current_step': idx+1, 
                            'total_steps': total
                        })
                    continue # 关键：跳过后续处理，丢弃这组数据

                # ==========================================================
                # 4. 应用 Gain (增益)
                # ==========================================================
                g_up = float(S.get('gain_up', 1.0)); g_dw = float(S.get('gain_dw', 1.0))
                if g_up == 0: g_up = 1.0
                if g_dw == 0: g_dw = 1.0
                
                # 使用刚才算好的 clean 数据除以 gain，得到最终物理量
                volt_up = val_up_clean / g_up
                volt_dw = val_dw_clean / g_dw

                # ==========================================================
                # 5. 后续常规拟合流程 (保持不变)
                # ==========================================================
                if config.USE_SIMULATION:
                    time_axis = np.linspace(0, 0.1, len(volt_up))
                else:
                    deci = int(S.get('decimation', 8192))
                    dt = 8e-9 * deci
                    time_axis = np.array([i * dt for i in range(len(volt_up))])
                
                tof_axis = time_axis + start_delay

                def get_fit_data(t, v, center_ms, width_ms):
                    if width_ms > 0:
                        cen_s = center_ms * 1e-3; wid_s = width_ms * 1e-3
                        start = cen_s - wid_s/2; end = cen_s + wid_s/2
                        mask = (t >= start) & (t <= end)
                        if np.any(mask): return t[mask], v[mask], (start, end)
                    return t, v, None

                fit_t_up, fit_v_up, win_up = get_fit_data(tof_axis, volt_up, fit_config.get('center_up', 0), fit_config.get('width_up', 0))
                fit_t_dw, fit_v_dw, win_dw = get_fit_data(tof_axis, volt_dw, fit_config.get('center_dw', 0), fit_config.get('width_dw', 0))

                popt_up, _ = fitting.perform_odr_fit(fitting.MODEL_GAUSSIAN, fit_t_up, fit_v_up)
                popt_dw, _ = fitting.perform_odr_fit(fitting.MODEL_GAUSSIAN, fit_t_dw, fit_v_dw)
                fit_curve_up = fitting.fit_funcs(popt_up, tof_axis) if popt_up is not None else np.zeros_like(tof_axis)
                fit_curve_dw = fitting.fit_funcs(popt_dw, tof_axis) if popt_dw is not None else np.zeros_like(tof_axis)

                amp_up = popt_up[1] if popt_up is not None else 0; sig_up = popt_up[3] if popt_up is not None else 0; cen_up = popt_up[4] if popt_up is not None else 0
                amp_dw = popt_dw[1] if popt_dw is not None else 0; sig_dw = popt_dw[3] if popt_dw is not None else 0; cen_dw = popt_dw[4] if popt_dw is not None else 0
                amp_up_nf = np.max(fit_v_up) if len(fit_v_up) > 0 else 0; amp_dw_nf = np.max(fit_v_dw) if len(fit_v_dw) > 0 else 0
                sig_up_nf = fitting.calc_sigma(fit_v_up, fit_t_up) or 0; sig_dw_nf = fitting.calc_sigma(fit_v_dw, fit_t_dw) or 0
                cen_up_nf = fit_t_up[np.argmax(fit_v_up)] if len(fit_v_up)>0 else 0; cen_dw_nf = fit_t_dw[np.argmax(fit_v_dw)] if len(fit_v_dw)>0 else 0
                area_up_nf = abs(np.trapz(fit_v_up, fit_t_up)) if len(fit_v_up)>1 else 0; area_dw_nf = abs(np.trapz(fit_v_dw, fit_t_dw)) if len(fit_v_dw)>1 else 0
                area_up = amp_up * abs(sig_up) * np.sqrt(2 * np.pi); area_dw = amp_dw * abs(sig_dw) * np.sqrt(2 * np.pi)

                n_f2, n_f1 = physics.calculate_atom_numbers(area_up, area_dw, max_vol_up=amp_up, max_vol_dw=amp_dw, alpha=S['alpha'], beta=S['beta'], R=S['R'], K=S['K'], max_low=S['max_low'])
                n_f2_nf, n_f1_nf = physics.calculate_atom_numbers(area_up_nf, area_dw_nf, max_vol_up=amp_up_nf, max_vol_dw=amp_dw_nf, alpha=S['alpha'], beta=S['beta'], R=S['R'], K=S['K'], max_low=S['max_low'])

                v_launch = float(S['launch_velocity'])
                
                t_flight_up = cen_up; t_flight_dw = cen_dw
                t_flight_up_nf = cen_up_nf; t_flight_dw_nf = cen_dw_nf
                
                temp_up = physics.calculate_temperature(sig_up, t_flight_up, v_launch, is_sigma_in_ms=False)
                temp_dw = physics.calculate_temperature(sig_dw, t_flight_dw, v_launch, is_sigma_in_ms=False)
                temp_up_nf = physics.calculate_temperature(sig_up_nf, t_flight_up_nf, v_launch, is_sigma_in_ms=False)
                temp_dw_nf = physics.calculate_temperature(sig_dw_nf, t_flight_dw_nf, v_launch, is_sigma_in_ms=False)

                prob_up, prob_dw = physics.calculate_probabilities(n_f2, n_f1)
                prob_up_nf, prob_dw_nf = physics.calculate_probabilities(n_f2_nf, n_f1_nf)
                # ============================================================
                # [New] Calculate Interferometer Output (Using Dynamic Settings)
                # ============================================================
                
                # Fit Data
                i_n1, i_n2, i_p1, i_p2 = physics.calculate_interferometer_output(
                    n_f1, n_f2, 
                    S.get('intf_alpha', 0.35), 
                    S.get('intf_beta', 0.076), 
                    S.get('intf_gamma', 0.25)
                )
                
                # NoFit Data
                i_n1_nf, i_n2_nf, i_p1_nf, i_p2_nf = physics.calculate_interferometer_output(
                    n_f1_nf, n_f2_nf, 
                    S.get('intf_alpha', 0.35), 
                    S.get('intf_beta', 0.076), 
                    S.get('intf_gamma', 0.25)
                )

                volt_up_store = volt_up[::STORAGE_STEP]; volt_dw_store = volt_dw[::STORAGE_STEP]
                fit_up_store = fit_curve_up[::STORAGE_STEP]; fit_dw_store = fit_curve_dw[::STORAGE_STEP]
                time_axis_store = tof_axis[::STORAGE_STEP]

                result = ScanResult(
                    parameter=primary_param, timestamp=job['timestamp'],
                    current_step=idx + 1, total_steps=total,
                    detected_delay=start_delay,
                    run_id=self.data_manager.current_run_id_str,
                    raw_data_up=volt_up_store, raw_data_dw=volt_dw_store,
                    fit_data_up=fit_up_store, fit_data_dw=fit_dw_store,
                    time_axis=time_axis_store, 
                    all_parameters=params, window_up=win_up, window_dw=win_dw,
                    atom_number_up=n_f2, atom_number_dw=n_f1, amplitude_up=amp_up, amplitude_dw=amp_dw,
                    sigma_up=sig_up * 1000.0, sigma_dw=sig_dw * 1000.0,
                    temperature_up=temp_up, temperature_dw=temp_dw, arrival_time_up=t_flight_up, arrival_time_dw=t_flight_dw,
                    transition_probability_up=prob_up, transition_probability_dw=prob_dw,
                    atom_number_up_nofit=n_f2_nf, atom_number_dw_nofit=n_f1_nf, amplitude_up_nofit=amp_up_nf, amplitude_dw_nofit=amp_dw_nf,
                    sigma_up_nofit=sig_up_nf * 1000.0, sigma_dw_nofit=sig_dw_nf * 1000.0,
                    temperature_up_nofit=temp_up_nf, temperature_dw_nofit=temp_dw_nf,
                    arrival_time_up_nofit=t_flight_up_nf, arrival_time_dw_nofit=t_flight_dw_nf,
                    transition_probability_up_nofit=prob_up_nf, transition_probability_dw_nofit=prob_dw_nf,
                    intf_n1=i_n1, intf_n2=i_n2, intf_p1=i_p1, intf_p2=i_p2,
                    intf_n1_nofit=i_n1_nf, intf_n2_nofit=i_n2_nf, intf_p1_nofit=i_p1_nf, intf_p2_nofit=i_p2_nf
                )
                self.data_manager.save_point(result, idx + 1)
                if self.on_data_ready:
                    step_size = max(1, len(tof_axis) // 2000)
                    frontend_data = asdict(result)
                    frontend_data['raw_data_up'] = volt_up[::step_size].tolist(); frontend_data['raw_data_dw'] = volt_dw[::step_size].tolist()
                    frontend_data['fit_data_up'] = fit_curve_up[::step_size].tolist(); frontend_data['fit_data_dw'] = fit_curve_dw[::step_size].tolist()
                    frontend_data['time_axis'] = tof_axis[::step_size].tolist()
                    self.on_data_ready(frontend_data)
            
            except Exception as e: 
                print(f"Processing Error step {idx+1}: {traceback.format_exc()}")

        self.data_manager.close_run()
        self.status.is_running = False; self.status.message = "Done"
        print("--- Processing Finished ---")
