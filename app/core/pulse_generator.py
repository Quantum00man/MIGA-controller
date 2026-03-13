import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Define robust path resolution
CURRENT_DIR = Path(__file__).resolve().parent      # Expected: app/core/
PROJECT_ROOT = CURRENT_DIR.parent.parent             # Expected: project root

# List of possible locations for the calibration file
POSSIBLE_PATHS = [
    CURRENT_DIR / "calibration.csv",       # 1. app/core/calibration.csv
    PROJECT_ROOT / "calibration.csv",      # 2. Project Root/calibration.csv
    Path.cwd() / "calibration.csv"         # 3. Current Working Directory
]

def get_inverse_calibration_func():
    """
    Reads the calibration data and returns an inverse interpolation function:
    DAC_value = f(Target_Optical_Power)
    """
    csv_path = None
    # Smart search for the calibration file
    for path in POSSIBLE_PATHS:
        if path.exists():
            csv_path = path
            print(f"[Pulse Generator] Successfully loaded calibration from: {csv_path}")
            break
            
    # If the file is strictly required, raise a hard error instead of failing silently
    if not csv_path:
        error_msg = (
            "\n[CRITICAL ERROR] Calibration file 'calibration.csv' NOT FOUND!\n"
            "The system looked in the following locations:\n"
            f"  1. {POSSIBLE_PATHS[0]}\n"
            f"  2. {POSSIBLE_PATHS[1]}\n"
            f"  3. {POSSIBLE_PATHS[2]}\n"
            "Please place the file in one of these directories."
        )
        raise FileNotFoundError(error_msg)
        
    # Read the monotonic, offset-removed calibration data
    data = pd.read_csv(csv_path, header=None)
    optical_power = data.iloc[:, 0].values
    dac_values = data.iloc[:, 1].values
    
    # Use 'linear' interpolation and force 0.0V output for negative or zero target power
    inv_func = interp1d(
        optical_power, 
        dac_values, 
        kind='linear', 
        bounds_error=False, 
        fill_value=(0.0, dac_values[-1])
    )
    return inv_func

# Load the calibration function into memory once when the module is imported
_CALIB_FUNC = get_inverse_calibration_func()

def generate_bragg_pulse(fwhm: float, shape: str = 'blackman', base_timing: int = 331119, clock_res: float = 0.2, target_amp: float = 0.1733):
    """
    Generates the timing sequence code for the .mot file and calculates the compensation timing.
    Returns: (pulse_code_string, compensation_time_string)
    """
    shape = shape.lower()
    
    # 1. Calculate pulse array based on chosen shape
    if shape == 'gaussian':
        std_dev = fwhm / (2 * np.sqrt(2 * np.log(2)))
        # Extend to 4 sigma 
        mean = 4.0 * std_dev
        x_end = 2.0 * mean
        num_points = int(x_end / clock_res)
        x_values = np.linspace(0, x_end, num_points)
        ideal_optical_shape = target_amp * np.exp(-((x_values - mean) ** 2) / (2 * std_dev ** 2))
        
    elif shape == 'blackman':
        # The FWHM of a standard Blackman window is roughly 0.405 * total_duration
        total_duration = fwhm / 0.405
        num_points = int(total_duration / clock_res)
        ideal_optical_shape = target_amp * np.blackman(num_points)
        
    else:
        raise ValueError(f"Unsupported shape '{shape}'. Please choose 'gaussian' or 'blackman'.")

    # 2. Map ideal optical power to actual DAC voltage using the calibration function
    y_values = _CALIB_FUNC(ideal_optical_shape)
    y_values = np.clip(y_values, 0, None)
    
    # 3. Generate command list matching the original hardware logic
    pulse_commands = []
    pulse_name = f"{shape.capitalize()}_pulse"
    
    # Insert 0.0V command at the beginning and hold for 500us
    pulse_commands.append(f"+500.0us {pulse_name} = 0.000\t\t(32)")
    
    # Main waveform body
    for y in y_values:
        cmd = f"+{clock_res:.1f}us {pulse_name} = {y:.3f}\t\t(32)"
        pulse_commands.append(cmd)
        
    # Append 0.0V command at the end to ensure the AOM is completely off
    pulse_commands.append(f"+{clock_res:.1f}us {pulse_name} = 0.000\t\t(32)")
    
    # Combine commands with line breaks
    pulse_code = "\n".join(pulse_commands)
    
    # 4. Calculate total points and PARAMETER1 compensation timing
    total_generated_points = num_points + 2
    pulse_logic_duration = total_generated_points * clock_res
    param1_compensation = base_timing - pulse_logic_duration
    
    # Return string formats ready for template replacement
    return pulse_code, f"{param1_compensation:.1f}"