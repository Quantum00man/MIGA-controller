from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from app.analysis.fitting import perform_bragg_fringe_fit


def _finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _wrap(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def grouped_intf_p1(points: Iterable[Any], stage: str) -> List[Dict[str, Any]]:
    """Group valid INTF P1 Fit shots by P0 for one calibration stage."""
    grouped: Dict[float, List[float]] = {}
    for point in points:
        getter = point.get if isinstance(point, dict) else lambda key, default=None: getattr(point, key, default)
        if str(getter("bragg_calibration_stage", "") or "") != stage:
            continue
        p0 = _finite(getter("parameter"))
        signal = _finite(getter("intf_p1"))
        if p0 is None or signal is None:
            continue
        grouped.setdefault(round(p0, 12), []).append(signal)
    rows = []
    for p0 in sorted(grouped):
        values = np.asarray(grouped[p0], dtype=float)
        rows.append({
            "p0_t2_us2": float(p0),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "sem": float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else None,
            "count": int(values.size),
        })
    return rows


def coarse_fit(points: Iterable[Any], target_fringe: int) -> Dict[str, Any]:
    rows = grouped_intf_p1(points, "coarse")
    if len(rows) < 4:
        raise ValueError("Bragg fringe calibration requires at least four valid coarse P0 values")
    x = np.asarray([row["p0_t2_us2"] for row in rows], dtype=float)
    y = np.asarray([row["mean"] for row in rows], dtype=float)
    y_std = np.asarray([row["std"] for row in rows], dtype=float)
    counts = np.asarray([row["count"] for row in rows], dtype=float)
    fit = perform_bragg_fringe_fit(
        x, y, fit_method="weighted_sem", y_std=y_std, counts=counts,
    )
    if fit is None:
        raise ValueError("Unable to fit the coarse Bragg fringe")
    candidates = [
        value for value in fit.mid_fringe_x
        if float(x[0]) < float(value) < float(x[-1])
    ]
    selected_index = int(target_fringe)
    if selected_index < 1:
        raise ValueError("Target mid fringe must be numbered from one")
    if selected_index > len(candidates):
        raise ValueError(
            f"Requested mid fringe #{selected_index}, but only {len(candidates)} fully bracketed mid fringes were found"
        )
    dense_x = np.linspace(float(x[0]), float(x[-1]), max(240, len(rows) * 20))
    params = fit.parameter_values
    dense_y = params["C"] + params["A"] * np.cos(
        fit.angular_frequency_rad_per_us2 * dense_x + params["phi0"]
    )
    return {
        "rows": rows,
        "parameter_values": dict(params),
        "angular_frequency_rad_per_us2": float(fit.angular_frequency_rad_per_us2),
        "mid_fringe_x": [float(value) for value in candidates],
        "selected_fringe_number": selected_index,
        "selected_mid_fringe_t2_us2": float(candidates[selected_index - 1]),
        "fit_x": dense_x.tolist(),
        "fit_y": dense_y.tolist(),
        "residual_variance": fit.residual_variance,
    }


def fine_phase_offsets(half_range_rad: float, point_count: int) -> List[float]:
    half_range = float(half_range_rad)
    count = int(point_count)
    if not math.isfinite(half_range) or half_range <= 0 or half_range >= math.pi / 2:
        raise ValueError("Fine phase half-range must be greater than zero and less than pi/2")
    if count < 5 or count > 101 or count % 2 == 0:
        raise ValueError("Fine scan point count must be an odd number between 5 and 101")
    values = np.linspace(-half_range, half_range, count).tolist()
    return sorted((float(value) for value in values), key=lambda value: (abs(value), value < 0))


def build_fine_p0_values(
    center_t2_us2: float,
    omega_rad_per_us2: float,
    half_range_rad: float,
    point_count: int,
) -> List[float]:
    center = float(center_t2_us2)
    omega = float(omega_rad_per_us2)
    if not math.isfinite(center) or center < 0 or not math.isfinite(omega) or omega <= 0:
        raise ValueError("Fine scan requires a non-negative center and positive angular frequency")
    values = [center + phase / omega for phase in fine_phase_offsets(half_range_rad, point_count)]
    if any(value < 0 for value in values):
        raise ValueError("Fine scan extends below P0 = T^2 = 0")
    return values


def fine_fit(
    points: Iterable[Any],
    coarse: Dict[str, Any],
    *,
    half_range_rad: float,
    minimum_contrast: float = 0.0,
    maximum_t2_uncertainty_us2: float = 0.0,
) -> Dict[str, Any]:
    rows = grouped_intf_p1(points, "fine")
    if len(rows) < 5:
        raise ValueError("Local mid-fringe fit requires at least five valid fine-scan P0 values")
    center = float(coarse["selected_mid_fringe_t2_us2"])
    omega = float(coarse["angular_frequency_rad_per_us2"])
    x = np.asarray([row["p0_t2_us2"] for row in rows], dtype=float)
    y = np.asarray([row["mean"] for row in rows], dtype=float)
    delta = omega * (x - center)
    design = np.column_stack((np.ones_like(delta), np.cos(delta), np.sin(delta)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if int(rank) < 3:
        raise ValueError("Fine-scan points do not constrain offset and local phase independently")
    offset, cosine_term, sine_term = (float(value) for value in coefficients)
    amplitude_local = math.hypot(cosine_term, sine_term)
    local_phase = math.atan2(-sine_term, cosine_term)
    coarse_params = coarse["parameter_values"]
    predicted_phase = omega * center + float(coarse_params["phi0"])
    target_phase = math.pi / 2.0 + round((predicted_phase - math.pi / 2.0) / math.pi) * math.pi
    local_target = math.pi / 2.0 + round((local_phase - math.pi / 2.0) / math.pi) * math.pi
    correction_rad = local_target - local_phase
    corrected_t2 = center + correction_rad / omega

    fitted = design @ coefficients
    residuals = y - fitted
    degrees = max(1, len(y) - 3)
    residual_variance = float(np.sum(residuals ** 2) / degrees)
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    radius2 = cosine_term ** 2 + sine_term ** 2
    phase_variance = None
    if radius2 > np.finfo(float).eps:
        gradient = np.asarray([0.0, sine_term / radius2, -cosine_term / radius2])
        phase_variance = max(0.0, float(gradient @ covariance @ gradient))
    phase_uncertainty = math.sqrt(phase_variance) if phase_variance is not None else None
    t2_uncertainty = phase_uncertainty / omega if phase_uncertainty is not None else None

    minimum = max(0.0, float(minimum_contrast))
    noise_floor = 3.0 * math.sqrt(max(0.0, residual_variance))
    contrast_limit = max(minimum, noise_floor)
    auto_uncertainty_limit = (float(half_range_rad) / omega) / 4.0
    uncertainty_limit = (
        float(maximum_t2_uncertainty_us2)
        if float(maximum_t2_uncertainty_us2) > 0
        else auto_uncertainty_limit
    )
    checks = {
        "inside_fine_range": abs(correction_rad) <= float(half_range_rad),
        "contrast": amplitude_local >= contrast_limit,
        "finite_uncertainty": t2_uncertainty is not None and math.isfinite(t2_uncertainty),
        "uncertainty": t2_uncertainty is not None and t2_uncertainty <= uncertainty_limit,
    }
    passed = all(checks.values())

    final_amplitude = float(coarse_params["A"])
    final_phi0 = _wrap(target_phase - omega * corrected_t2)
    fit_x = np.asarray(coarse["fit_x"], dtype=float)
    fit_y = offset + final_amplitude * np.cos(omega * fit_x + final_phi0)
    mid_fringe_x = [
        float((math.pi / 2.0 + index * math.pi - final_phi0) / omega)
        for index in range(
            math.ceil((omega * float(fit_x[0]) + final_phi0 - math.pi / 2.0) / math.pi),
            math.floor((omega * float(fit_x[-1]) + final_phi0 - math.pi / 2.0) / math.pi) + 1,
        )
    ]
    slope = "positive" if -final_amplitude * omega * math.sin(target_phase) > 0 else "negative"
    return {
        "rows": rows,
        "local_parameter_values": {"A": amplitude_local, "C": offset, "phase_at_center": local_phase},
        "phase_correction_rad": float(correction_rad),
        "mid_fringe_t2_us2": float(corrected_t2),
        "phase_uncertainty_rad": phase_uncertainty,
        "t2_uncertainty_us2": t2_uncertainty,
        "residual_standard_deviation": math.sqrt(max(0.0, residual_variance)),
        "quality_checks": checks,
        "quality_passed": passed,
        "quality_limits": {
            "minimum_contrast": contrast_limit,
            "maximum_t2_uncertainty_us2": uncertainty_limit,
        },
        "calibration_fit": {
            "model_key": "bragg_fringes",
            "model_label": "Bragg Fringes",
            "metric_tab": "intf",
            "metric_label": "Interferometer",
            "source_key": "intf_p",
            "source_label": "INTF P1 Fit",
            "source_mode": "fit",
            "source_field": "intf_p1",
            "channel": "up",
            "channel_label": "P1",
            "fit_min": float(fit_x[0]),
            "fit_max": float(fit_x[-1]),
            "fit_x": fit_x.tolist(),
            "fit_y": fit_y.tolist(),
            "parameter_values": {
                **dict(coarse_params), "A": final_amplitude, "C": offset, "phi0": final_phi0,
            },
            "bragg": {
                "angular_frequency_rad_per_us2": omega,
                "mid_fringe_x": mid_fringe_x,
            },
            "reference_input_mode": "t2",
            "reference_t_unit": "us",
            "reference_value": float(corrected_t2),
            "reference_t2_us2": float(corrected_t2),
            "monotonic_slope": slope,
            "phase_conversion_mode": "monotonic_half_fringe",
        },
    }
