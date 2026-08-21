import os
import time
import subprocess
import shlex
import requests
import numpy as np
from typing import List, Tuple
import config

class SequenceEditor:
    @staticmethod
    def render_sequence_content(content: str, parameters: List[float]) -> str:
        rendered = str(content)
        for i, param in enumerate(parameters):
            placeholder = f"<PARAMETER{i}>"
            val_str = f"{param:.6f}" if isinstance(param, float) else str(param)
            rendered = rendered.replace(placeholder, val_str)
        return rendered

    @staticmethod
    def render_bragg_sequence_content(content: str, pulse_code: str, compensation: str) -> str:
        placeholder = "<PARAMETER0>"
        lines = str(content).splitlines(keepends=True)
        placeholder_lines = [index for index, line in enumerate(lines) if placeholder in line]
        bragg_lines = [index for index in placeholder_lines if "bragg" in lines[index].lower()]
        if len(bragg_lines) == 1:
            target_index = bragg_lines[0]
        elif not bragg_lines and len(placeholder_lines) == 1:
            target_index = placeholder_lines[0]
        elif not placeholder_lines:
            raise ValueError("Bragg sequence template must contain <PARAMETER0>")
        else:
            raise ValueError(
                "Bragg sequence template must have exactly one <PARAMETER0> line containing 'Bragg'"
            )

        pulse_text = str(pulse_code)
        if not pulse_text.startswith("\n"):
            pulse_text = "\n" + pulse_text
        for index in placeholder_lines:
            lines[index] = lines[index].replace(
                placeholder,
                pulse_text if index == target_index else "",
            )
        return SequenceEditor.render_sequence_content("".join(lines), ["", compensation])

    @staticmethod
    def generate_sequence(
        template_path: str,
        output_path: str,
        parameters: List[float],
        *,
        bragg_mode: bool = False,
        marker_axes: List[str] | None = None,
        marker_definitions: List[dict] | None = None,
    ):
        if not os.path.exists(template_path):
            if config.USE_SIMULATION:
                os.makedirs(os.path.dirname(template_path), exist_ok=True)
                with open(template_path, 'w') as f:
                    f.write("%% MOCK TEMPLATE %%\nPARAM0 = <PARAMETER0>\n")
            else:
                raise FileNotFoundError(f"Sequence template not found: {template_path}")

        with open(template_path, 'r') as f:
            content = f.read()

        if marker_axes:
            from app.core.sequence_markers import render_auto_marker_sequence
            content = render_auto_marker_sequence(
                content,
                marker_axes,
                parameters,
                marker_definitions or [],
            )
        elif bragg_mode:
            if len(parameters) < 2:
                raise ValueError("Bragg sequence generation requires pulse and compensation parameters")
            content = SequenceEditor.render_bragg_sequence_content(content, parameters[0], parameters[1])
        else:
            content = SequenceEditor.render_sequence_content(content, parameters)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
        return output_path

class ExperimentDriver:
    def __init__(self): pass

    @staticmethod
    def _build_command(binary_path: str, sequence_file_path: str, extra_args: str = "") -> List[str]:
        binary = (binary_path or "").strip()
        if not binary:
            raise ValueError("TMOT binary path is empty")

        # Support both the new split config (`tmot_path` + `tmot_args`)
        # and legacy values such as "/path/to/tmot4 -e".
        if os.path.exists(binary):
            command = [binary]
        else:
            command = shlex.split(binary)
            if not command:
                raise ValueError("TMOT command is empty")

        if extra_args:
            command.extend(shlex.split(extra_args))

        command.extend(["-f", sequence_file_path])
        return command

    def run_sequence(self, sequence_file_path: str, binary_path: str = None, extra_args: str = "") -> bool:
        if config.USE_SIMULATION:
            print(f"[SIMULATION] Executing sequence: {sequence_file_path}")
            try: requests.post(f"http://{config.RP_IP_MOCK}:{config.RP_PORT_MOCK}/trigger", timeout=1)
            except: pass
            time.sleep(0.1)
            return True
        else:
            time.sleep(0.2)
            lock_file = "/var/lock/mot4"
            if os.path.exists(lock_file):
                try: os.remove(lock_file)
                except OSError as e: print(f"Warning: Could not remove lock file: {e}")

            binary = binary_path if binary_path else config.TMOT_BINARY_PATH_LINUX
            try:
                command = self._build_command(binary, sequence_file_path, extra_args=extra_args)
                print(f"[TMOT] Running command: {' '.join(shlex.quote(part) for part in command)}")
                subprocess.run(command, check=True, capture_output=True, text=True)
                return True
            except subprocess.CalledProcessError as e:
                cmd_str = " ".join(shlex.quote(part) for part in e.cmd)
                print(f"Hardware Execution Error: command failed: {cmd_str}")
                if e.stderr:
                    print(e.stderr.strip())
                elif e.stdout:
                    print(e.stdout.strip())
                return False
            except Exception as e:
                print(f"Hardware Execution Error: {e}")
                return False

    def compile_vcd(self, sequence_file_path: str, output_vcd_path: str, binary_path: str = None) -> bool:
        if config.USE_SIMULATION: return True
        else:
            binary = binary_path if binary_path else config.CMOT_BINARY_PATH_LINUX
            try:
                # 1. Run CMOT
                subprocess.run([binary, "-f", sequence_file_path], check=True, capture_output=True, text=True)
                
                # 2. Locate the generated VCD file
                # cmot usually outputs to CWD (./seq.vcd) ignoring input path (temp/seq.mot)
                base_name = os.path.basename(sequence_file_path) # e.g. "seq.mot"
                vcd_name = os.path.splitext(base_name)[0] + ".vcd" # e.g. "seq.vcd"
                
                # Candidate locations
                cwd_vcd = os.path.join(os.getcwd(), vcd_name)       # ./seq.vcd
                input_dir_vcd = sequence_file_path.replace(".mot", ".vcd") # temp/seq.vcd
                
                source_vcd = None
                if os.path.exists(cwd_vcd):
                    source_vcd = cwd_vcd
                elif os.path.exists(input_dir_vcd):
                    source_vcd = input_dir_vcd
                
                # 3. Move to target location
                if source_vcd:
                    # Only move if paths differ
                    if os.path.abspath(source_vcd) != os.path.abspath(output_vcd_path):
                        if os.path.exists(output_vcd_path):
                            os.remove(output_vcd_path)
                        os.rename(source_vcd, output_vcd_path)
                    return True
                else:
                    print(f"VCD Error: Generated file {vcd_name} not found in CWD or temp folder.")
                    return False

            except Exception as e:
                print(f"VCD Compilation Error (cmot4): {e}")
                return False

class RedPitayaDriver:
    def __init__(self, ip_address: str, timeout: int = 2):
        self.real_ip = ip_address
        self.timeout = timeout

    def update_settings(self, ip_address: str, timeout: int):
        self.real_ip = ip_address
        self.timeout = timeout

    def acquire_channel(self, channel_name: str) -> Tuple[float, List[float]]:
        """
        Retrieves data from a specific channel (ch1 or ch2).
        Returns (trigger_timestamp, voltage_list).
        """
        if config.USE_SIMULATION:
            # Simulation logic remains same...
            return time.time(), [np.sin(i/10) for i in range(100)]
            
        # [MODIFIED] Smart URL construction
        # Checks if the user provided a custom port (e.g., "127.0.0.1:8001")
        if ':' in self.real_ip:
            base_url = f"http://{self.real_ip}"
        else:
            base_url = f"http://{self.real_ip}:{config.RP_PORT_REAL}"

        target_url = f"{base_url}/{channel_name}.dat"
        time.sleep(0.05)
        trigger_time = time.time() 
	
        try:
            # ... (Rest of the code remains exactly the same) ...
            response = requests.get(target_url, timeout=self.timeout)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            
            # Handle legacy .dat format
            data_lines = lines
            if len(lines) > 0:
                try:
                    possible_timestamp = float(lines[0].strip())
                    trigger_time = possible_timestamp
                    data_lines = lines[1:] 
                except ValueError:
                    pass

            voltage_data = []
            for line in data_lines:
                if not line.strip(): continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    try: voltage_data.append(float(parts[1]))
                    except ValueError: continue
                elif len(parts) == 1:
                    try: voltage_data.append(float(parts[0]))
                    except ValueError: continue
            
            return trigger_time, voltage_data

        except requests.RequestException as e:
            if not config.USE_SIMULATION:
                print(f"Network Error connecting to {target_url}: {e}")
            return time.time(), []
