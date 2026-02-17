import numpy as np
from typing import Tuple, Optional

# --- Physical Constants ---
G = 9.81              
M_RB87 = 1.443e-25    
LAMBDA = 780.241e-9   
KB = 1.38e-23         

def calc_arrival_time(v_launch: float, z_target: float, direction_sign: int = 1) -> Optional[float]:
    discriminant = v_launch**2 - 2 * G * z_target
    if discriminant < 0: return None 
    return (v_launch + direction_sign * np.sqrt(discriminant)) / G

def calculate_atom_numbers(
    area_up: float,     # Integral UP
    area_dw: float,     # Integral DOWN
    max_vol_up: float,  # Max Voltage UP
    max_vol_dw: float,  # Max Voltage DOWN
    alpha: float, 
    beta: float, 
    R: float, 
    K: float,
    max_low: float      # Threshold
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates calibrated atom numbers based on Legacy Python Code logic.
    Strictly implements:
    if maxvol1 > max_low or maxvol2 > max_low:
        atomF2 = (atomUp - atomDw * alfa) * K
        atomF1 = (atomDw - atomUp * beta) * R * K
    else:
        None
    """
    # Check threshold (OR logic as per legacy code)
    if max_vol_up > max_low or max_vol_dw > max_low:
        atom_f2 = (area_up - area_dw * alpha) * K
        atom_f1 = (area_dw - area_up * beta) * R * K
        return atom_f2, atom_f1
    else:
        # Signal too weak
        return None, None

def calculate_probabilities(n_f2: Optional[float], n_f1: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates transition probabilities.
    probaF2 = 100 * atoms[0] / (atoms[0] + atoms[1])
    """
    if n_f2 is None or n_f1 is None:
        return None, None
        
    total = n_f2 + n_f1
    if total == 0:
        return 0.0, 0.0
        
    prob_f2 = 100.0 * n_f2 / total
    prob_f1 = 100.0 * n_f1 / total
    
    return prob_f2, prob_f1

def calculate_temperature(sigma_t: float, t_flight: float, v_launch: float, is_sigma_in_ms: bool = False) -> Optional[float]:
    if t_flight <= 1e-6: return None
    sigma_seconds = sigma_t * 1e-3 if is_sigma_in_ms else sigma_t
    expansion_factor = (v_launch / t_flight - G)
    temp_kelvin = (M_RB87 / KB) * (expansion_factor**2) * (sigma_seconds**2)
    return temp_kelvin * 1e6 

def calc_velocity_from_frequency(freq_khz: float, correction_factor: float = 1.0) -> float:
    df = freq_khz * 1e3 
    return np.sqrt(3) * LAMBDA * df * correction_factor