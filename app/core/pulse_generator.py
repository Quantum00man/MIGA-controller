import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Retrieve the absolute directory path where this pulse_generator.py is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the calibration file
DEFAULT_CALIB_PATH = os.path.join(CURRENT_DIR, "calibration.csv")

def get_inverse_calibration_func(csv_path=DEFAULT_CALIB_PATH):
    """
    Reads the calibration data and returns an inverse interpolation function:
    DAC_value = f(Target_Optical_Power)
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Calibration file '{csv_path}' not found. Using ideal linear output.")
        return lambda x: x 
        
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
        # Extend to 4 sigma (ideal optical power drops to ~0.03% of peak, smoothly reaching 0)
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