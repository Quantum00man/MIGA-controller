import numpy as np
from scipy.odr import ODR, Model, Data
from scipy.special import erf
import warnings
from typing import List, Tuple, Optional, Any

# --- Fitting Model Constants ---
MODEL_GAUSSIAN = 1
MODEL_MOD_GAUSSIAN_1 = 2
MODEL_MOD_GAUSSIAN_2 = 3
MODEL_LORENTZIAN = 4
MODEL_SINC_SQ = 5

def fit_funcs(Ks: List[float], x: np.ndarray) -> np.ndarray:
    """
    Mathematical models for fitting.
    Vectorized implementation replacing the original loop-based logic.
    
    :param Ks: List of parameters [type, amp, mod1, width, center, mod2, mod3, offset, ?]
    :param x: Independent variable array (time or frequency)
    """
    # Ks[0] determines the model type
    model_type = int(Ks[0])
    
    if model_type == MODEL_GAUSSIAN:
        # Standard Gaussian: A * exp(-(x-x0)^2 / (2*sigma^2)) + offset
        # Ks map: 1=Amp, 3=Width(Sigma), 4=Center, 7=Offset
        return Ks[1] * np.exp(-(x - Ks[4])**2 / (2 * Ks[3]**2)) + Ks[7]
    
    elif model_type == MODEL_MOD_GAUSSIAN_1:
        # Modified Gaussian with Error Function term
        t1 = erf((x - Ks[4]) / Ks[5])
        return Ks[1] * np.exp(-(x - Ks[4] + Ks[2] * t1)**2 / (2 * Ks[3]**2)) + Ks[7]
    
    elif model_type == MODEL_MOD_GAUSSIAN_2:
        # Modified Gaussian with Error Function and Quadratic term
        t1 = erf((x - Ks[4]) / Ks[5])
        t2 = (x - Ks[4])**2
        return Ks[1] * np.exp(-(x - Ks[4] + Ks[2] * t1 - Ks[6] * t2)**2 / (2 * Ks[3]**2)) + Ks[7]
    
    elif model_type == MODEL_LORENTZIAN:
        # Lorentzian profile
        return Ks[1] * 1. / (1 + ((x - Ks[4]) / Ks[3])**2)**Ks[6] + Ks[7]
    
    elif model_type == MODEL_SINC_SQ:
        # Sinc squared profile: (sin(u)/u)^2
        # Handle singularity at x == center
        delta = x - Ks[4]
        # Avoid division by zero
        # Create a safe divisor
        safe_delta = np.where(np.abs(delta) < 1e-9, 1e-9, delta)
        arg = safe_delta * 1.22 / Ks[3]
        
        # Calculate sinc values
        val = Ks[1] * (np.sin(arg) / arg)**2 + Ks[7]
        
        # Correct the peak value where delta was ~0
        # Limit (sin(x)/x)^2 as x->0 is 1. 
        # But wait, the original logic had a specific peak value handling?
        # Original: y.append(Ks[1]/((1.22/Ks[3]))**2+Ks[7]) which seems wrong for standard sinc?
        # Let's stick to standard behavior: peak is Amp + Offset
        mask = np.abs(delta) < 1e-9
        val[mask] = Ks[1] + Ks[7] 
        
        return val
        
    return np.zeros_like(x)

def calc_sigma(func: np.ndarray, x: np.ndarray, pkidx: int = None) -> Optional[float]:
    """
    Calculate Sigma based on Full Width at Half Maximum (FWHM).
    Assumes Gaussian-like distribution.
    """
    if func is None or x is None: return None
    func = np.array(func)
    x = np.array(x)
    
    if len(func) == 0: return None

    # Normalize to zero baseline
    f_min = np.min(func)
    f = func - f_min
    peak_height = np.max(f)
    
    if pkidx is None:
        pkidx = np.argmax(f)
        
    # Find Half Maximum point to the right
    idx = pkidx
    target = peak_height * 0.5
    
    # Search right side
    while idx < len(f) - 2 and f[idx] > target:
        idx += 1
    
    # Calculate HWHM (Half Width Half Max)
    hwhm = x[idx] - x[pkidx]
    
    # Convert HWHM to Sigma: Sigma = HWHM / sqrt(2*ln(2))
    sigma = hwhm / np.sqrt(2 * np.log(2))
    
    # Check for validity
    return sigma if sigma > 0 else 1e-4

def calc_std(func: np.ndarray, x: np.ndarray) -> Optional[float]:
    """
    Calculate Standard Deviation using statistical moments.
    """
    if func is None or x is None: return None
    func = np.array(func) - np.min(func)
    x = np.array(x)
    
    norm = np.trapz(func, x)
    if norm == 0: return 0
    
    func_norm = func / norm
    
    # Expectation E[x]
    ex = np.trapz(x * func_norm, x)
    # Expectation E[x^2]
    ex2 = np.trapz(x**2 * func_norm, x)
    
    var = ex2 - ex**2
    return np.sqrt(var) if var > 0 else 0

def perform_odr_fit(ifunc: int, x_data: np.ndarray, y_data: np.ndarray, initial_guess: List[float] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Executes Orthogonal Distance Regression (ODR) fit.
    Replaces the original fit() function.
    
    :param ifunc: Model ID (e.g., MODEL_GAUSSIAN)
    :param x_data: Time/Frequency array
    :param y_data: Voltage/Signal array
    :return: (best_parameters, residual_variance)
    """
    if x_data is None or y_data is None or len(x_data) != len(y_data):
        return None, None

    # 1. Generate Initial Guess if not provided
    if initial_guess is None:
        max_y = np.max(y_data)
        max_idx = np.argmax(y_data)
        center_x = x_data[max_idx]
        sigma_guess = calc_sigma(y_data, x_data, max_idx) or 1e-4
        
        # Standard guess structure based on legacy code:
        # [type, amp, mod1, width, center, mod2, mod3, offset, ?]
        initial_guess = [float(ifunc), max_y, 10.0, sigma_guess, center_x, 0.01, 0.003, 0.0, 0.0]

    # 2. Define Fixed Parameters (ifix)
    # 0 means fixed, 1 means fitted
    # Mapping based on legacy fit() function
    ifix_map = {
        MODEL_GAUSSIAN:       [0, 1, 0, 1, 1, 0, 0, 0, 0], #[0, 1, 0, 1, 1, 0, 0, ''0'', 0] the  ''0''means offset is fixed  to  0. ''1'' offset no fixed
        MODEL_MOD_GAUSSIAN_1: [0, 1, 1, 1, 1, 1, 0, 1, 0],
        MODEL_MOD_GAUSSIAN_2: [0, 1, 1, 1, 1, 1, 1, 1, 0],
        MODEL_LORENTZIAN:     [0, 1, 0, 1, 1, 0, 1, 1, 0],
        MODEL_SINC_SQ:        [0, 1, 0, 1, 1, 0, 0, 1, 0],
    }
    # Default to fitting everything except type if unknown
    ifix = ifix_map.get(ifunc, [0, 1, 1, 1, 1, 1, 1, 1, 1])

    # 3. Setup ODR
    model = Model(fit_funcs)
    data = Data(x_data, y_data)
    
    # Ensure the first parameter (Model Type) matches ifunc
    initial_guess[0] = float(ifunc)
    
    try:
        # fit_type=2 corresponds to Least Squares (ODR reduces to OLS if x errors are not provided)
        odr = ODR(data, model, beta0=initial_guess, ifixb=ifix)
        odr.set_job(fit_type=2) 
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = odr.run()
            
        return output.beta, output.res_var
    except Exception as e:
        print(f"Fit failed for func {ifunc}: {e}")
        return None, None
