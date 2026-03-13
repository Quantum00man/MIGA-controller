# app/core/pulse_generator.py
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def get_inverse_calibration_func(csv_path="calibration.csv"):
    """
    Reads the calibration data and returns an inverse interpolation function.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Calibration file '{csv_path}' not found. Using ideal linear output.")
        return lambda x: x 
        
    data = pd.read_csv(csv_path, header=None)
    optical_power = data.iloc[:, 0].values
    dac_values = data.iloc[:, 1].values
    
    inv_func = interp1d(
        optical_power, dac_values, kind='linear', bounds_error=False, fill_value=(0.0, dac_values[-1])
    )
    return inv_func

# Load the calibration function once into memory
_CALIB_FUNC = get_inverse_calibration_func()

def generate_bragg_pulse(fwhm: float, shape: str = 'blackman', base_timing: int = 331119, clock_res: float = 0.2, target_amp: float = 0.1733):
    """
    Generates the DAC sequence code for the .mot file and calculates the compensation timing.
    Returns: (pulse_code_string, compensation_time_string)
    """
    shape = shape.lower()
    
    # 1. Calculate time array and ideal Y values based on shape
    if shape == 'gaussian':
        std_dev = fwhm / (2 * np.sqrt(2 * np.log(2)))
        cutoff = 3 * std_dev
        t_arr = np.arange(-cutoff, cutoff, clock_res)
        y_ideal = target_amp * np.exp(-0.5 * (t_arr / std_dev)**2)
    else: # blackman
        duration = 2.4 * fwhm
        t_arr = np.arange(0, duration, clock_res)
        y_ideal = target_amp * np.blackman(len(t_arr))
        
    # 2. Map to actual DAC values using calibration
    y_dac = _CALIB_FUNC(y_ideal)
    
    # 3. Generate Pulse Code for <PARAMETER0>
    pulse_commands = [f"          set dac0 {val:.3f}" for val in y_dac]
    pulse_code = "\n".join(pulse_commands)
    
    # 4. Calculate Compensation Time for <PARAMETER1>
    total_generated_points = len(t_arr) + 2
    pulse_logic_duration = total_generated_points * clock_res
    param1_compensation = int(base_timing - pulse_logic_duration)
    
    return pulse_code, str(param1_compensation)