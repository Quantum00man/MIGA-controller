import ast
import copy
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.odr import ODR, Model, Data
from scipy.optimize import curve_fit, least_squares, minimize_scalar
from scipy.sparse import lil_matrix
from scipy.special import erf

# --- Fitting Model Constants ---
MODEL_GAUSSIAN = 1
MODEL_MOD_GAUSSIAN_1 = 2
MODEL_MOD_GAUSSIAN_2 = 3
MODEL_LORENTZIAN = 4
MODEL_SINC_SQ = 5

_ROLE_KEYS = ("amplitude", "width", "center", "offset")


@dataclass
class FitExecutionResult:
    model_key: str
    model_label: str
    parameter_values: Dict[str, float]
    fit_curve: np.ndarray
    fit_window_curve: np.ndarray
    residual_variance: Optional[float]
    amplitude: float
    width: float
    center: float
    offset: float
    area: float


@dataclass
class BraggFringeFitResult:
    parameter_values: Dict[str, float]
    fit_curve: np.ndarray
    fit_window_curve: np.ndarray
    residual_variance: Optional[float]
    effective_wavevector_rad_m: float
    angular_frequency_rad_per_us2: float
    mid_fringe_x: List[float]
    mid_fringe_spacing_us2: float


@dataclass
class DifferentialEllipseFitResult:
    parameter_values: Dict[str, float]
    parameter_uncertainties: Dict[str, Optional[float]]
    fit_x: np.ndarray
    fit_y: np.ndarray
    point_fit_x: np.ndarray
    point_fit_y: np.ndarray
    normalized_rms_residual: float
    residual_variance: Optional[float]
    r_squared: Optional[float]
    phase_coverage_rad: float
    condition_number: Optional[float]
    warnings: List[str]


DEFAULT_FIT_MODELS: List[Dict[str, Any]] = [
    {
        "key": "gaussian",
        "label": "GAUSSIAN",
        "formula": "amp * exp(-((x - center)**2) / (2 * sigma**2)) + offset",
        "parameters": [
            {"name": "amp", "guess": "y_max - y_min", "fixed": False},
            {"name": "sigma", "guess": "max(sigma_est, 1e-4)", "fixed": False},
            {"name": "center", "guess": "x_peak", "fixed": False},
            {"name": "offset", "guess": "0.0", "fixed": True},
        ],
        "roles": {"amplitude": "amp", "width": "sigma", "center": "center", "offset": "offset"},
        "area_mode": "gaussian_sigma",
    },
    {
        "key": "mod_gaussian_1",
        "label": "MOD_GAUSSIAN_1",
        "formula": "amp * exp(-((x - center + skew * erf((x - center) / erf_width))**2) / (2 * sigma**2)) + offset",
        "parameters": [
            {"name": "amp", "guess": "y_max - y_min", "fixed": False},
            {"name": "skew", "guess": "0.0", "fixed": False},
            {"name": "sigma", "guess": "max(sigma_est, 1e-4)", "fixed": False},
            {"name": "center", "guess": "x_peak", "fixed": False},
            {"name": "erf_width", "guess": "max(sigma_est, 1e-4)", "fixed": False},
            {"name": "offset", "guess": "0.0", "fixed": False},
        ],
        "roles": {"amplitude": "amp", "width": "sigma", "center": "center", "offset": "offset"},
        "area_mode": "window_integral",
    },
    {
        "key": "mod_gaussian_2",
        "label": "MOD_GAUSSIAN_2",
        "formula": "amp * exp(-((x - center + skew * erf((x - center) / erf_width) - quad * (x - center)**2)**2) / (2 * sigma**2)) + offset",
        "parameters": [
            {"name": "amp", "guess": "y_max - y_min", "fixed": False},
            {"name": "skew", "guess": "0.0", "fixed": False},
            {"name": "sigma", "guess": "max(sigma_est, 1e-4)", "fixed": False},
            {"name": "center", "guess": "x_peak", "fixed": False},
            {"name": "erf_width", "guess": "max(sigma_est, 1e-4)", "fixed": False},
            {"name": "quad", "guess": "0.0", "fixed": False},
            {"name": "offset", "guess": "0.0", "fixed": False},
        ],
        "roles": {"amplitude": "amp", "width": "sigma", "center": "center", "offset": "offset"},
        "area_mode": "window_integral",
    },
    {
        "key": "lorentzian",
        "label": "LORENTZIAN",
        "formula": "amp / (1 + ((x - center) / width)**2)**power + offset",
        "parameters": [
            {"name": "amp", "guess": "y_max - y_min", "fixed": False},
            {"name": "width", "guess": "max(sigma_est, 1e-4)", "fixed": False},
            {"name": "center", "guess": "x_peak", "fixed": False},
            {"name": "power", "guess": "1.0", "fixed": False},
            {"name": "offset", "guess": "y_min", "fixed": False},
        ],
        "roles": {"amplitude": "amp", "width": "width", "center": "center", "offset": "offset"},
        "area_mode": "window_integral",
    },
    {
        "key": "sinc_sq",
        "label": "SINC_SQ",
        "formula": "amp * sinc_sq((x - center) * 1.22 / width) + offset",
        "parameters": [
            {"name": "amp", "guess": "y_max - y_min", "fixed": False},
            {"name": "width", "guess": "max(sigma_est, 1e-4)", "fixed": False},
            {"name": "center", "guess": "x_peak", "fixed": False},
            {"name": "offset", "guess": "y_min", "fixed": False},
        ],
        "roles": {"amplitude": "amp", "width": "width", "center": "center", "offset": "offset"},
        "area_mode": "window_integral",
    },
]


_GAUSSIAN_AREA_FORMULA_SIGNATURES = {
    "amp*exp(-((x-center)**2)/(2*sigma**2))+offset",
    "offset+amp*exp(-((x-center)**2)/(2*sigma**2))",
}

_EFFECTIVE_SIGMA_MODEL_KEYS = {"mod_gaussian_1", "mod_gaussian_2", "lorentzian", "sinc_sq"}


def _normalize_formula_signature(formula: Any) -> str:
    return re.sub(r"\s+", "", str(formula or "")).lower()


def _supports_gaussian_sigma_area_mode(model_key: str, formula: str) -> bool:
    return str(model_key or "").strip().lower() == "gaussian" or _normalize_formula_signature(formula) in _GAUSSIAN_AREA_FORMULA_SIGNATURES


DEFAULT_SCAN_FIT_MODELS: List[Dict[str, Any]] = [
    {
        "key": "gaussian",
        "label": "Gaussian",
        "formula": "offset + amp * exp(-((x - center)**2) / (2 * sigma**2))",
        "parameters": [
            {"name": "amp", "guess": "max(y_range, 1e-6)", "fixed": False},
            {"name": "sigma", "guess": "max(x_span / 6, 1e-4)", "fixed": False},
            {"name": "center", "guess": "x_peak", "fixed": False},
            {"name": "offset", "guess": "y_min", "fixed": False},
        ],
        "roles": {"amplitude": "amp", "width": "sigma", "center": "center", "offset": "offset"},
        "area_mode": "window_integral",
    },
    {
        "key": "lorentzian",
        "label": "Lorentzian",
        "formula": "offset + amp / (1 + ((x - center) / width)**2)",
        "parameters": [
            {"name": "amp", "guess": "max(y_range, 1e-6)", "fixed": False},
            {"name": "width", "guess": "max(x_span / 8, 1e-4)", "fixed": False},
            {"name": "center", "guess": "x_peak", "fixed": False},
            {"name": "offset", "guess": "y_min", "fixed": False},
        ],
        "roles": {"amplitude": "amp", "width": "width", "center": "center", "offset": "offset"},
        "area_mode": "window_integral",
    },
    {
        "key": "rabi_oscillation",
        "label": "Rabi Oscillation",
        "formula": "offset + amp * sin(2 * pi * x / period + phase)**2",
        "parameters": [
            {"name": "amp", "guess": "max(y_range, 1e-6)", "fixed": False},
            {"name": "period", "guess": "max(x_span / 3, 1e-4)", "fixed": False},
            {"name": "phase", "guess": "0.0", "fixed": False},
            {"name": "offset", "guess": "y_min", "fixed": False},
        ],
        "roles": {"amplitude": "amp", "width": "period", "offset": "offset"},
        "area_mode": "window_integral",
    },
    {
        "key": "ai_fringes",
        "label": "AI Fringes",
        "formula": "offset + amp * cos(2 * pi * x / period + phase)",
        "parameters": [
            {"name": "amp", "guess": "max(y_range / 2, 1e-6)", "fixed": False},
            {"name": "period", "guess": "max(x_span / 3, 1e-4)", "fixed": False},
            {"name": "phase", "guess": "0.0", "fixed": False},
            {"name": "offset", "guess": "y_mean", "fixed": False},
        ],
        "roles": {"amplitude": "amp", "width": "period", "offset": "offset"},
        "area_mode": "window_integral",
    },
    {
        "key": "bragg_fringes",
        "label": "Bragg Fringes",
        "formula": "C + A * cos(omega * x + phi0)",
        "parameters": [
            {"name": "A", "guess": "max(y_range / 2, 1e-6)", "fixed": False},
            {"name": "omega", "guess": "2 * pi / max(x_span, 1e-4)", "fixed": False},
            {"name": "phi0", "guess": "0.0", "fixed": False},
            {"name": "C", "guess": "y_mean", "fixed": False},
        ],
        "roles": {"amplitude": "A", "width": "omega", "offset": "C"},
        "area_mode": "window_integral",
    },
    {
        "key": "custom",
        "label": "Custom",
        "formula": "offset + amp * exp(-((x - center)**2) / (2 * width**2))",
        "parameters": [
            {"name": "amp", "guess": "max(y_range, 1e-6)", "fixed": False},
            {"name": "width", "guess": "max(x_span / 6, 1e-4)", "fixed": False},
            {"name": "center", "guess": "x_peak", "fixed": False},
            {"name": "offset", "guess": "y_min", "fixed": False},
        ],
        "roles": {"amplitude": "amp", "width": "width", "center": "center", "offset": "offset"},
        "area_mode": "window_integral",
    },
]

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


def _sinc_sq(arg: np.ndarray) -> np.ndarray:
    arr = np.asarray(arg, dtype=float)
    safe = np.where(np.abs(arr) < 1e-12, 1e-12, arr)
    values = (np.sin(safe) / safe) ** 2
    values[np.abs(arr) < 1e-12] = 1.0
    return values


_FORMULA_ENV: Dict[str, Any] = {
    "exp": np.exp,
    "erf": erf,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "log": np.log,
    "log10": np.log10,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "sinc_sq": _sinc_sq,
    "pi": np.pi,
    "e": np.e,
}

_GUESS_ENV: Dict[str, Any] = {
    **_FORMULA_ENV,
    "max": max,
    "min": min,
}

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


def calculate_area_with_edge_baseline(x_data: np.ndarray, y_data: np.ndarray, baseline_points: int = 2) -> float:
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    if len(x_data) != len(y_data) or len(x_data) == 0:
        return 0.0
    if len(x_data) == 1:
        return 0.0

    n = max(1, int(baseline_points or 1))
    n = min(n, len(x_data))

    left_x = float(np.mean(x_data[:n]))
    right_x = float(np.mean(x_data[-n:]))
    left_y = float(np.mean(y_data[:n]))
    right_y = float(np.mean(y_data[-n:]))

    if np.isclose(right_x, left_x):
        baseline = np.full_like(y_data, 0.5 * (left_y + right_y), dtype=float)
    else:
        slope = (right_y - left_y) / (right_x - left_x)
        baseline = left_y + slope * (x_data - left_x)

    return float(abs(np.trapz(y_data - baseline, x_data)))


def get_default_fit_models() -> List[Dict[str, Any]]:
    return copy.deepcopy(DEFAULT_FIT_MODELS)


def get_default_scan_fit_models() -> List[Dict[str, Any]]:
    return copy.deepcopy(DEFAULT_SCAN_FIT_MODELS)


def _wrap_phase_radians(value: float) -> float:
    wrapped = (float(value) + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.pi if np.isclose(wrapped, -np.pi) else wrapped)


def _ellipse_theta_initial(
    x_data: np.ndarray,
    y_data: np.ndarray,
    center_x: float,
    center_y: float,
    amplitude_x: float,
    amplitude_y: float,
    phase_difference: float,
) -> np.ndarray:
    normalized_x = np.clip((x_data - center_x) / amplitude_x, -1.0, 1.0)
    normalized_y = (y_data - center_y) / amplitude_y
    positive = np.arccos(normalized_x)
    negative = -positive
    positive_error = np.abs(np.cos(positive + phase_difference) - normalized_y)
    negative_error = np.abs(np.cos(negative + phase_difference) - normalized_y)
    return np.where(positive_error <= negative_error, positive, negative)


def _ellipse_global_covariance(
    theta: np.ndarray,
    amplitude_x: float,
    amplitude_y: float,
    phase_difference: float,
    scale_x: float,
    scale_y: float,
    residual_variance: float,
) -> Tuple[np.ndarray, float]:
    information = np.zeros((5, 5), dtype=float)
    for angle in theta:
        cos_x = float(np.cos(angle))
        sin_x = float(np.sin(angle))
        cos_y = float(np.cos(angle + phase_difference))
        sin_y = float(np.sin(angle + phase_difference))
        global_jacobian = np.asarray([
            [1.0 / scale_x, 0.0, amplitude_x * cos_x / scale_x, 0.0, 0.0],
            [0.0, 1.0 / scale_y, 0.0, amplitude_y * cos_y / scale_y, -amplitude_y * sin_y / scale_y],
        ])
        theta_jacobian = np.asarray([
            -amplitude_x * sin_x / scale_x,
            -amplitude_y * sin_y / scale_y,
        ])
        denominator = float(np.dot(theta_jacobian, theta_jacobian))
        projection = np.eye(2)
        if denominator > np.finfo(float).eps:
            projection -= np.outer(theta_jacobian, theta_jacobian) / denominator
        information += global_jacobian.T @ projection @ global_jacobian
    condition_number = float(np.linalg.cond(information))
    covariance = np.linalg.pinv(information) * float(residual_variance)
    return covariance, condition_number


def perform_sync_differential_ellipse_fit(
    x_data: np.ndarray,
    y_data: np.ndarray,
    eval_points: int = 361,
) -> Optional[DifferentialEllipseFitResult]:
    """Fit X=Cx+Ax*cos(theta), Y=Cy+Ay*cos(theta+delta), with delta in [0, pi]."""
    if x_data is None or y_data is None or len(x_data) != len(y_data):
        return None
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    finite = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[finite]
    y_data = y_data[finite]
    point_count = len(x_data)
    if point_count < 8 or len(np.unique(x_data)) < 4 or len(np.unique(y_data)) < 4:
        return None

    x_range = float(np.ptp(x_data))
    y_range = float(np.ptp(y_data))
    if x_range <= np.finfo(float).eps or y_range <= np.finfo(float).eps:
        return None
    if point_count >= 20:
        x_low, x_high = np.percentile(x_data, [2.0, 98.0])
        y_low, y_high = np.percentile(y_data, [2.0, 98.0])
    else:
        x_low, x_high = float(np.min(x_data)), float(np.max(x_data))
        y_low, y_high = float(np.min(y_data)), float(np.max(y_data))
    center_x_initial = float((x_low + x_high) / 2.0)
    center_y_initial = float((y_low + y_high) / 2.0)
    amplitude_x_initial = max(float((x_high - x_low) / 2.0), x_range * 0.1)
    amplitude_y_initial = max(float((y_high - y_low) / 2.0), y_range * 0.1)
    scale_x = max(float(np.std(x_data)), x_range / 6.0, np.finfo(float).eps)
    scale_y = max(float(np.std(y_data)), y_range / 6.0, np.finfo(float).eps)

    correlation = float(np.corrcoef(x_data, y_data)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    correlation_phase = float(np.arccos(np.clip(correlation, -0.98, 0.98)))
    phase_starts = []
    for candidate in (correlation_phase, np.pi / 2.0, np.pi / 4.0, 3.0 * np.pi / 4.0):
        if all(abs(candidate - existing) > 0.05 for existing in phase_starts):
            phase_starts.append(float(candidate))

    parameter_count = 5 + point_count
    sparsity = lil_matrix((2 * point_count, parameter_count), dtype=int)
    for index in range(point_count):
        sparsity[2 * index, 0] = 1
        sparsity[2 * index, 2] = 1
        sparsity[2 * index, 5 + index] = 1
        sparsity[2 * index + 1, 1] = 1
        sparsity[2 * index + 1, 3] = 1
        sparsity[2 * index + 1, 4] = 1
        sparsity[2 * index + 1, 5 + index] = 1

    minimum_amplitude_x = max(x_range * 1e-3, np.finfo(float).eps)
    minimum_amplitude_y = max(y_range * 1e-3, np.finfo(float).eps)
    lower = np.concatenate((
        [float(np.min(x_data) - 2.0 * x_range), float(np.min(y_data) - 2.0 * y_range),
         np.log(minimum_amplitude_x), np.log(minimum_amplitude_y), 1e-5],
        np.full(point_count, -np.inf),
    ))
    upper = np.concatenate((
        [float(np.max(x_data) + 2.0 * x_range), float(np.max(y_data) + 2.0 * y_range),
         np.log(x_range * 20.0), np.log(y_range * 20.0), np.pi - 1e-5],
        np.full(point_count, np.inf),
    ))

    def residuals(parameters: np.ndarray) -> np.ndarray:
        center_x, center_y, log_amplitude_x, log_amplitude_y, phase_difference = parameters[:5]
        theta = parameters[5:]
        amplitude_x = float(np.exp(log_amplitude_x))
        amplitude_y = float(np.exp(log_amplitude_y))
        output = np.empty(2 * point_count, dtype=float)
        output[0::2] = (center_x + amplitude_x * np.cos(theta) - x_data) / scale_x
        output[1::2] = (center_y + amplitude_y * np.cos(theta + phase_difference) - y_data) / scale_y
        return output

    best_result = None
    best_sum_squares = float("inf")
    for phase_start in phase_starts:
        theta_initial = _ellipse_theta_initial(
            x_data, y_data,
            center_x_initial, center_y_initial,
            amplitude_x_initial, amplitude_y_initial,
            phase_start,
        )
        initial = np.concatenate(([
            center_x_initial,
            center_y_initial,
            np.log(amplitude_x_initial),
            np.log(amplitude_y_initial),
            phase_start,
        ], theta_initial))
        try:
            candidate = least_squares(
                residuals,
                initial,
                bounds=(lower, upper),
                jac_sparsity=sparsity,
                loss="soft_l1",
                f_scale=0.1,
                max_nfev=500,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
        except (ValueError, FloatingPointError):
            continue
        sum_squares = float(np.dot(residuals(candidate.x), residuals(candidate.x)))
        if candidate.success and sum_squares < best_sum_squares:
            best_result = candidate
            best_sum_squares = sum_squares
    if best_result is None:
        return None

    center_x, center_y, log_amplitude_x, log_amplitude_y, phase_difference = best_result.x[:5]
    theta = best_result.x[5:]
    amplitude_x = float(np.exp(log_amplitude_x))
    amplitude_y = float(np.exp(log_amplitude_y))
    phase_difference = float(np.clip(phase_difference, 0.0, np.pi))
    point_fit_x = center_x + amplitude_x * np.cos(theta)
    point_fit_y = center_y + amplitude_y * np.cos(theta + phase_difference)
    normalized_residuals = residuals(best_result.x)
    degrees_of_freedom = max(1, point_count - 5)
    residual_variance = float(np.dot(normalized_residuals, normalized_residuals) / degrees_of_freedom)
    normalized_rms = float(np.sqrt(np.mean(
        normalized_residuals[0::2] ** 2 + normalized_residuals[1::2] ** 2
    )))
    centered_total = float(np.sum(((x_data - np.mean(x_data)) / scale_x) ** 2)
                           + np.sum(((y_data - np.mean(y_data)) / scale_y) ** 2))
    r_squared = float(1.0 - best_sum_squares / centered_total) if centered_total > 0 else None

    wrapped_theta = np.sort(np.mod(theta, 2.0 * np.pi))
    gaps = np.diff(np.concatenate((wrapped_theta, [wrapped_theta[0] + 2.0 * np.pi])))
    phase_coverage = float(2.0 * np.pi - np.max(gaps))
    covariance, condition_number = _ellipse_global_covariance(
        theta,
        amplitude_x,
        amplitude_y,
        phase_difference,
        scale_x,
        scale_y,
        residual_variance,
    )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    uncertainties: Dict[str, Optional[float]] = {
        "center_x": float(standard_errors[0]),
        "center_y": float(standard_errors[1]),
        "amplitude_x": float(amplitude_x * standard_errors[2]),
        "amplitude_y": float(amplitude_y * standard_errors[3]),
        "phase_difference_rad": float(standard_errors[4]),
    }

    warnings_out: List[str] = []
    if min(phase_difference, np.pi - phase_difference) < 0.1:
        warnings_out.append("The fitted ellipse is close to a line, so the phase difference is weakly constrained.")
    if phase_coverage < np.pi:
        warnings_out.append("The data cover less than half of the ellipse; fitted center, amplitudes, and phase may be correlated.")
    if not np.isfinite(condition_number) or condition_number > 1e10:
        warnings_out.append("The fit is ill-conditioned; interpret parameter uncertainties with caution.")
    if r_squared is not None and r_squared < 0.8:
        warnings_out.append("The selected X/Y data do not closely follow the common-phase ellipse model.")

    evaluation_theta = np.linspace(0.0, 2.0 * np.pi, max(64, min(int(eval_points), 4000)))
    fit_x = center_x + amplitude_x * np.cos(evaluation_theta)
    fit_y = center_y + amplitude_y * np.cos(evaluation_theta + phase_difference)
    return DifferentialEllipseFitResult(
        parameter_values={
            "center_x": float(center_x),
            "center_y": float(center_y),
            "amplitude_x": amplitude_x,
            "amplitude_y": amplitude_y,
            "phase_difference_rad": phase_difference,
            "phase_difference_deg": float(np.degrees(phase_difference)),
        },
        parameter_uncertainties=uncertainties,
        fit_x=fit_x,
        fit_y=fit_y,
        point_fit_x=np.asarray(point_fit_x, dtype=float),
        point_fit_y=np.asarray(point_fit_y, dtype=float),
        normalized_rms_residual=normalized_rms,
        residual_variance=residual_variance,
        r_squared=r_squared,
        phase_coverage_rad=phase_coverage,
        condition_number=condition_number,
        warnings=warnings_out,
    )


def _fringe_linear_solution(x_centered: np.ndarray, y_data: np.ndarray, omega: float) -> Tuple[np.ndarray, float]:
    design = np.column_stack(
        (
            np.ones_like(x_centered),
            np.cos(omega * x_centered),
            np.sin(omega * x_centered),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, y_data, rcond=None)
    residuals = y_data - design @ coefficients
    return coefficients, float(np.dot(residuals, residuals))


def perform_bragg_fringe_fit(
    x_data: np.ndarray,
    y_data: np.ndarray,
    wavelength_nm: float = 780.0,
    bragg_order: int = 1,
    eval_x: Optional[np.ndarray] = None,
) -> Optional[BraggFringeFitResult]:
    """Fit C + A cos(k_eff a T^2 + phi0), with x=T^2 in (microseconds)^2."""
    if x_data is None or y_data is None or len(x_data) != len(y_data) or len(x_data) < 4:
        return None
    wavelength_nm = float(wavelength_nm)
    bragg_order = int(bragg_order)
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0 or bragg_order < 1:
        return None

    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    finite = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[finite]
    y_data = y_data[finite]
    if len(x_data) < 4:
        return None

    order = np.argsort(x_data)
    x_data = x_data[order]
    y_data = y_data[order]
    span = float(x_data[-1] - x_data[0])
    if span <= 0:
        return None

    unique_x = np.unique(x_data)
    if len(unique_x) < 4:
        return None
    spacings = np.diff(unique_x)
    spacings = spacings[spacings > 0]
    if not len(spacings):
        return None

    x_reference = float(np.mean(x_data))
    x_centered = x_data - x_reference
    wavelength_m = wavelength_nm * 1e-9
    effective_wavevector = 4.0 * np.pi * bragg_order / wavelength_m
    standard_gravity = 9.80665
    # Search down to 1/100 of a cycle across the selected window. The upper
    # bound uses the median sampling interval so a single near-duplicate P0
    # value cannot create an unrealistic Nyquist range. Acceleration caused by
    # a projection of gravity cannot exceed g, which also removes unphysical
    # high-frequency aliases from noisy scans.
    gravity_omega = effective_wavevector * standard_gravity * 1e-12
    min_omega = min(2.0 * np.pi / (100.0 * span), gravity_omega * 1e-4)
    max_omega = min(np.pi / float(np.median(spacings)), gravity_omega)
    if not np.isfinite(max_omega) or max_omega <= min_omega:
        return None

    grid_size = int(min(12000, max(2000, 24 * len(x_data))))
    omega_grid = np.linspace(min_omega, max_omega, grid_size)
    errors = np.empty_like(omega_grid)
    for index, omega in enumerate(omega_grid):
        _, errors[index] = _fringe_linear_solution(x_centered, y_data, float(omega))

    candidate_indices = np.argsort(errors)[: min(12, len(errors))]
    best_omega = float(omega_grid[int(candidate_indices[0])])
    best_error = float(errors[int(candidate_indices[0])])
    grid_step = float(omega_grid[1] - omega_grid[0]) if len(omega_grid) > 1 else min_omega
    for raw_index in candidate_indices:
        index = int(raw_index)
        lower = max(min_omega, float(omega_grid[index] - 2.0 * grid_step))
        upper = min(max_omega, float(omega_grid[index] + 2.0 * grid_step))
        if upper <= lower:
            continue
        optimized = minimize_scalar(
            lambda omega: _fringe_linear_solution(x_centered, y_data, float(omega))[1],
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": max(grid_step * 1e-5, np.finfo(float).eps)},
        )
        if optimized.success and float(optimized.fun) < best_error:
            best_omega = float(optimized.x)
            best_error = float(optimized.fun)

    coefficients, best_error = _fringe_linear_solution(x_centered, y_data, best_omega)
    offset = float(coefficients[0])
    cosine_coefficient = float(coefficients[1])
    sine_coefficient = float(coefficients[2])
    amplitude = float(np.hypot(cosine_coefficient, sine_coefficient))
    centered_phase = float(np.arctan2(-sine_coefficient, cosine_coefficient))
    phase = _wrap_phase_radians(centered_phase - best_omega * x_reference)

    acceleration = best_omega * 1e12 / effective_wavevector
    angle_mrad = float(np.arcsin(np.clip(acceleration / standard_gravity, -1.0, 1.0)) * 1e3)

    eval_x = np.asarray(eval_x if eval_x is not None else x_data, dtype=float)
    fit_window_curve = offset + amplitude * np.cos(best_omega * x_data + phase)
    fit_curve = offset + amplitude * np.cos(best_omega * eval_x + phase)
    residual_variance = float(best_error / len(x_data)) if len(x_data) else None
    first_index = int(np.ceil((best_omega * x_data[0] + phase - np.pi / 2.0) / np.pi))
    last_index = int(np.floor((best_omega * x_data[-1] + phase - np.pi / 2.0) / np.pi))
    mid_fringe_x = [
        float((np.pi / 2.0 + index * np.pi - phase) / best_omega)
        for index in range(first_index, last_index + 1)
    ]
    mid_fringe_spacing = float(np.pi / best_omega)
    return BraggFringeFitResult(
        parameter_values={
            "a": float(acceleration),
            "alpha": angle_mrad,
            "phi0": phase,
            "A": amplitude,
            "C": offset,
        },
        fit_curve=fit_curve,
        fit_window_curve=fit_window_curve,
        residual_variance=residual_variance,
        effective_wavevector_rad_m=float(effective_wavevector),
        angular_frequency_rad_per_us2=float(best_omega),
        mid_fringe_x=mid_fringe_x,
        mid_fringe_spacing_us2=mid_fringe_spacing,
    )


def sanitize_model_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "fit_model"


def normalize_fit_model_definition(model: Dict[str, Any], fallback_key: str = "gaussian") -> Dict[str, Any]:
    source = copy.deepcopy(model or {})
    key = sanitize_model_key(source.get("key") or source.get("label") or fallback_key)
    label = str(source.get("label") or key.upper()).strip() or key.upper()
    formula = str(source.get("formula") or "").strip()

    parameters = []
    for idx, param in enumerate(source.get("parameters") or []):
        if isinstance(param, str):
            name = param.strip()
            parameters.append({"name": name or f"p{idx}", "guess": "", "fixed": False})
            continue
        if not isinstance(param, dict):
            continue
        name = str(param.get("name") or f"p{idx}").strip()
        if not name:
            name = f"p{idx}"
        parameters.append(
            {
                "name": name,
                "guess": str(param.get("guess") or "").strip(),
                "fixed": bool(param.get("fixed", False)),
            }
        )

    if not parameters:
        parameters = [
            {"name": "amp", "guess": "y_max - y_min", "fixed": False},
            {"name": "sigma", "guess": "max(sigma_est, 1e-4)", "fixed": False},
            {"name": "center", "guess": "x_peak", "fixed": False},
            {"name": "offset", "guess": "0.0", "fixed": True},
        ]

    roles = {}
    raw_roles = source.get("roles") or {}
    for role_name in _ROLE_KEYS:
        role_value = raw_roles.get(role_name) if isinstance(raw_roles, dict) else None
        roles[role_name] = str(role_value).strip() if role_value else None

    area_mode = str(source.get("area_mode") or "window_integral").strip() or "window_integral"
    if area_mode not in {"gaussian_sigma", "window_integral"}:
        area_mode = "window_integral"
    if not _supports_gaussian_sigma_area_mode(key, formula):
        area_mode = "window_integral"

    return {
        "key": key,
        "label": label,
        "formula": formula,
        "parameters": parameters,
        "roles": roles,
        "area_mode": area_mode,
    }


def normalize_fit_model_list(models: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    source_models = models or get_default_fit_models()

    for idx, model in enumerate(source_models):
        norm = normalize_fit_model_definition(model, fallback_key=f"fit_model_{idx}")
        base_key = norm["key"]
        key = base_key
        suffix = 1
        while key in seen:
            suffix += 1
            key = f"{base_key}_{suffix}"
        norm["key"] = key
        seen.add(key)
        normalized.append(norm)

    if not normalized:
        return get_default_fit_models()
    return normalized


def get_fit_model_by_key(models: Optional[List[Dict[str, Any]]], model_key: Optional[str]) -> Dict[str, Any]:
    normalized = normalize_fit_model_list(models)
    wanted = sanitize_model_key(model_key or "")
    for model in normalized:
        if model["key"] == wanted:
            return model
    return normalized[0]


def validate_fit_model_definition(model_definition: Dict[str, Any]) -> Optional[str]:
    try:
        model = normalize_fit_model_definition(model_definition)
    except Exception as exc:
        return f"Model normalization failed: {exc}"

    if not model["formula"]:
        return "Formula is empty"

    param_names = [param["name"] for param in model["parameters"]]
    if len(param_names) != len(set(param_names)):
        return "Parameter names must be unique"

    dummy_ctx = {
        "x_peak": 0.5,
        "x_start": 0.0,
        "x_end": 1.0,
        "x_span": 1.0,
        "x_mean": 0.5,
        "y_max": 1.0,
        "y_min": 0.0,
        "y_mean": 0.5,
        "y_range": 1.0,
        "sigma_est": 0.1,
    }
    guess_values: Dict[str, float] = {}
    for param in model["parameters"]:
        expr = str(param.get("guess") or "").strip()
        if not expr:
            guess_values[param["name"]] = _heuristic_guess(param["name"], dummy_ctx)
            continue
        try:
            guess_values[param["name"]] = float(_safe_eval_expression(expr, {**_GUESS_ENV, **dummy_ctx, **guess_values}))
        except Exception as exc:
            return f"Invalid guess for '{param['name']}': {exc}"

    for role_name in _ROLE_KEYS:
        role_value = model.get("roles", {}).get(role_name)
        if role_value and role_value not in guess_values:
            return f"Role '{role_name}' points to unknown parameter '{role_value}'"

    try:
        test_curve = _evaluate_formula(model["formula"], np.linspace(0.0, 1.0, 8), guess_values or {"amp": 1.0})
    except Exception as exc:
        return f"Invalid formula: {exc}"
    if not np.all(np.isfinite(test_curve)):
        return "Formula evaluates to non-finite values"
    return None


def _normalize_expression(expr: str) -> str:
    return str(expr or "").replace("^", "**").strip()


def _validate_expression_ast(node: ast.AST, allowed_names: Set[str]) -> None:
    if not isinstance(node, _ALLOWED_AST_NODES):
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in allowed_names:
            raise ValueError("Unsupported function in expression")
    if isinstance(node, ast.Name) and node.id not in allowed_names:
        raise ValueError(f"Unknown symbol: {node.id}")
    for child in ast.iter_child_nodes(node):
        _validate_expression_ast(child, allowed_names)


def _safe_eval_expression(expr: str, env: Dict[str, Any]) -> Any:
    normalized_expr = _normalize_expression(expr)
    if not normalized_expr:
        raise ValueError("Expression is empty")
    parsed = ast.parse(normalized_expr, mode="eval")
    _validate_expression_ast(parsed, set(env.keys()))
    compiled = compile(parsed, "<fit-expr>", "eval")
    return eval(compiled, {"__builtins__": {}}, env)


def _build_guess_context(x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
    y_data = np.asarray(y_data, dtype=float)
    x_data = np.asarray(x_data, dtype=float)
    if len(x_data) == 0:
        return {
            "x_peak": 0.0,
            "x_start": 0.0,
            "x_end": 0.0,
            "x_span": 0.0,
            "x_mean": 0.0,
            "y_max": 0.0,
            "y_min": 0.0,
            "y_mean": 0.0,
            "y_range": 0.0,
            "sigma_est": 1e-4,
        }

    peak_idx = int(np.argmax(y_data))
    sigma_est = calc_sigma(y_data, x_data, peak_idx) or 1e-4
    return {
        "x_peak": float(x_data[peak_idx]),
        "x_start": float(x_data[0]),
        "x_end": float(x_data[-1]),
        "x_span": float(x_data[-1] - x_data[0]),
        "x_mean": float(np.mean(x_data)),
        "y_max": float(np.max(y_data)),
        "y_min": float(np.min(y_data)),
        "y_mean": float(np.mean(y_data)),
        "y_range": float(np.max(y_data) - np.min(y_data)),
        "sigma_est": float(max(abs(sigma_est), 1e-4)),
    }


def _heuristic_guess(param_name: str, ctx: Dict[str, float]) -> float:
    name = param_name.strip().lower()
    span = max(abs(ctx.get("x_span", 0.0)), 1e-4)

    if "offset" in name or "baseline" in name or name in {"bg", "background"}:
        return float(ctx.get("y_min", 0.0))
    if name in {"amp", "amplitude", "a"} or "amp" in name:
        return float(max(ctx.get("y_range", 0.0), 1e-6))
    if "center" in name or name in {"mu", "x0", "t0"}:
        return float(ctx.get("x_peak", 0.0))
    if "sigma" in name or "width" in name or "gamma" in name or "tau" in name:
        return float(max(ctx.get("sigma_est", 1e-4), span / 10.0, 1e-6))
    if "power" in name or name in {"n", "order"}:
        return 1.0
    if "skew" in name or "shift" in name or "asym" in name or "quad" in name or "mod" in name:
        return 0.0
    return 1.0


def _evaluate_parameter_guesses(model: Dict[str, Any], x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
    ctx = _build_guess_context(x_data, y_data)
    guesses: Dict[str, float] = {}
    for param in model["parameters"]:
        name = param["name"]
        expr = str(param.get("guess") or "").strip()
        try:
            if expr:
                raw_value = _safe_eval_expression(expr, {**_GUESS_ENV, **ctx, **guesses})
                guess_value = float(raw_value)
            else:
                guess_value = _heuristic_guess(name, ctx)
        except Exception:
            guess_value = _heuristic_guess(name, ctx)
        if not np.isfinite(guess_value):
            guess_value = _heuristic_guess(name, ctx)
        guesses[name] = float(guess_value)
    return guesses


def _evaluate_formula(formula: str, x_values: np.ndarray, param_values: Dict[str, float]) -> np.ndarray:
    result = _safe_eval_expression(
        formula,
        {
            **_FORMULA_ENV,
            "x": np.asarray(x_values, dtype=float),
            **{name: float(value) for name, value in param_values.items()},
        },
    )
    array = np.asarray(result, dtype=float)
    if array.shape == ():
        array = np.full_like(np.asarray(x_values, dtype=float), float(array), dtype=float)
    return array


def _safe_curve_values(formula: str, x_values: np.ndarray, param_values: Dict[str, float]) -> np.ndarray:
    try:
        values = _evaluate_formula(formula, x_values, param_values)
    except Exception:
        return np.full_like(np.asarray(x_values, dtype=float), 1e12, dtype=float)
    if not np.all(np.isfinite(values)):
        return np.nan_to_num(values, nan=1e12, posinf=1e12, neginf=-1e12)
    return values


def _role_value(role_name: str, model: Dict[str, Any], param_values: Dict[str, float]) -> Optional[float]:
    role_key = model.get("roles", {}).get(role_name)
    if role_key and role_key in param_values:
        return float(param_values[role_key])
    return None


def perform_configured_fit(
    model_definition: Dict[str, Any],
    x_data: np.ndarray,
    y_data: np.ndarray,
    eval_x: Optional[np.ndarray] = None,
) -> Optional[FitExecutionResult]:
    if x_data is None or y_data is None or len(x_data) != len(y_data) or len(x_data) == 0:
        return None

    model = normalize_fit_model_definition(model_definition)
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    eval_x = np.asarray(eval_x if eval_x is not None else x_data, dtype=float)

    initial_values = _evaluate_parameter_guesses(model, x_data, y_data)
    free_params = [param for param in model["parameters"] if not param.get("fixed", False)]
    fixed_params = [param for param in model["parameters"] if param.get("fixed", False)]

    if free_params and len(x_data) < len(free_params):
        return None

    parameter_values = {name: float(value) for name, value in initial_values.items()}

    def curve_wrapper(x_values: np.ndarray, *free_values: float) -> np.ndarray:
        full_values = dict(parameter_values)
        for param, value in zip(free_params, free_values):
            full_values[param["name"]] = float(value)
        return _safe_curve_values(model["formula"], x_values, full_values)

    try:
        if free_params:
            p0 = [parameter_values[param["name"]] for param in free_params]
            opt_values, _ = curve_fit(curve_wrapper, x_data, y_data, p0=p0, maxfev=20000)
            for param, value in zip(free_params, opt_values):
                parameter_values[param["name"]] = float(value)
        else:
            for param in fixed_params:
                parameter_values[param["name"]] = float(parameter_values.get(param["name"], 0.0))
    except Exception as exc:
        print(f"Configured fit failed for model {model['key']}: {exc}")
        return None

    try:
        fit_window_curve = _evaluate_formula(model["formula"], x_data, parameter_values)
        fit_curve = _evaluate_formula(model["formula"], eval_x, parameter_values)
    except Exception as exc:
        print(f"Configured fit evaluation failed for model {model['key']}: {exc}")
        return None

    if not np.all(np.isfinite(fit_window_curve)) or not np.all(np.isfinite(fit_curve)):
        return None

    offset = _role_value("offset", model, parameter_values)
    if offset is None:
        offset = float(np.min(fit_window_curve))

    amplitude = _role_value("amplitude", model, parameter_values)
    if amplitude is None:
        amplitude = float(np.max(fit_window_curve) - offset)

    center = _role_value("center", model, parameter_values)
    if center is None:
        center = float(x_data[int(np.argmax(fit_window_curve))])

    width = _role_value("width", model, parameter_values)
    if model.get("key") in _EFFECTIVE_SIGMA_MODEL_KEYS:
        width = float(calc_sigma(fit_window_curve, x_data) or 0.0)
    elif width is None:
        width = float(calc_sigma(fit_window_curve, x_data) or 0.0)
    else:
        width = float(width)

    residuals = y_data - fit_window_curve
    res_var = float(np.mean(residuals ** 2)) if len(residuals) else None

    if model.get("area_mode") == "gaussian_sigma":
        area = float(abs(amplitude) * abs(width) * np.sqrt(2 * np.pi))
    else:
        baseline_removed = fit_window_curve - offset
        area = float(abs(np.trapz(baseline_removed, x_data))) if len(x_data) > 1 else float(abs(baseline_removed[0]))

    return FitExecutionResult(
        model_key=model["key"],
        model_label=model["label"],
        parameter_values=parameter_values,
        fit_curve=fit_curve,
        fit_window_curve=fit_window_curve,
        residual_variance=res_var,
        amplitude=float(amplitude),
        width=float(width),
        center=float(center),
        offset=float(offset),
        area=float(area),
    )

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
