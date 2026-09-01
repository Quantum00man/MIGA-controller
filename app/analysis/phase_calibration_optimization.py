from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from app.analysis import interferometer_phase


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _calibration_parameters(calibration: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    parameters = calibration.get("parameter_values") or {}
    bragg = calibration.get("bragg") or {}
    amplitude = _finite(parameters.get("A"))
    offset = _finite(parameters.get("C"))
    omega = _finite(bragg.get("angular_frequency_rad_per_us2"))
    phi0 = _finite(parameters.get("phi0"))
    reference = interferometer_phase.reference_t2_us2(calibration)
    if None in {amplitude, offset, omega, phi0, reference} or amplitude <= 0 or omega <= 0:
        raise ValueError("Both nodes require complete, positive-amplitude Bragg phase calibrations")
    return float(amplitude), float(offset), float(omega), float(phi0), float(reference)


def _reference_phase(calibration: Dict[str, Any]) -> float:
    _, _, omega, phi0, reference = _calibration_parameters(calibration)
    sign = -1.0 if interferometer_phase.monotonic_slope(calibration) == "positive" else 1.0
    return sign * math.acos(math.cos(omega * reference + phi0))


def _phase_values(
    signals: np.ndarray,
    amplitude: float,
    offset: float,
    sign: float,
    reference_phase: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = (signals - offset) / amplitude
    valid = np.isfinite(normalized) & (normalized >= -1.0) & (normalized <= 1.0)
    phases = np.full(len(signals), np.nan, dtype=float)
    phases[valid] = sign * np.arccos(normalized[valid]) - reference_phase
    overflow = np.maximum(np.abs(normalized) - 1.0, 0.0)
    return phases, valid, overflow


def _allan_one(values: np.ndarray) -> float:
    if len(values) < 2:
        return math.inf
    return float(np.sqrt(np.mean(np.diff(values) ** 2) / 2.0))


def _metrics(values: np.ndarray) -> Dict[str, Optional[float]]:
    if not len(values):
        return {"mean_rad": None, "std_rad": None, "allan_n1_rad": None}
    return {
        "mean_rad": float(np.mean(values)),
        "std_rad": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "allan_n1_rad": _allan_one(values) if len(values) > 1 else None,
    }


def _objective_metric(values: np.ndarray, objective: str, allan_weight: float) -> float:
    std = float(np.std(values, ddof=1)) if len(values) > 1 else math.inf
    allan = _allan_one(values)
    if objective == "std":
        return std
    if objective == "combined":
        return float(np.sqrt(allan_weight * allan ** 2 + (1.0 - allan_weight) * std ** 2))
    return allan


def _curve_constraint(calibration: Dict[str, Any], amplitude: float, offset: float) -> Tuple[float, str]:
    fit_x = np.asarray(calibration.get("fit_x") or [], dtype=float)
    fit_y = np.asarray(calibration.get("fit_y") or [], dtype=float)
    _, _, omega, phi0, _ = _calibration_parameters(calibration)
    finite = np.isfinite(fit_x) & np.isfinite(fit_y) if len(fit_x) == len(fit_y) else np.zeros(0, dtype=bool)
    if len(fit_x) >= 2 and np.any(finite):
        residual = offset + amplitude * np.cos(omega * fit_x[finite] + phi0) - fit_y[finite]
        return float(np.sqrt(np.mean(residual ** 2))), "saved_fit_curve"
    original_amplitude, original_offset, _, _, _ = _calibration_parameters(calibration)
    fallback = math.hypot(amplitude - original_amplitude, offset - original_offset)
    return float(fallback), "parameter_fallback"


def _optimized_calibration(calibration: Dict[str, Any], amplitude: float, offset: float) -> Dict[str, Any]:
    result = deepcopy(calibration)
    result["parameter_values"] = {
        **(result.get("parameter_values") or {}),
        "A": float(amplitude),
        "C": float(offset),
    }
    fit_x = np.asarray(result.get("fit_x") or [], dtype=float)
    if len(fit_x):
        _, _, omega, phi0, _ = _calibration_parameters(result)
        result["fit_y"] = (offset + amplitude * np.cos(omega * fit_x + phi0)).tolist()
    result["sync_optimized_preview"] = True
    return result


def optimize_sync_phase_calibrations(
    pairs: Sequence[Dict[str, Any]],
    reference_calibration: Dict[str, Any],
    target_calibration: Dict[str, Any],
    objective: str = "allan",
    combined_allan_weight: float = 0.5,
    parameter_bound_fraction: float = 0.1,
    fringe_weight: float = 1.0,
    prior_weight: float = 0.05,
) -> Dict[str, Any]:
    """Jointly tune A/C for two phase calibrations using constant SYNC differential phase."""
    objective = str(objective or "allan").strip().lower()
    if objective not in {"allan", "std", "combined"}:
        raise ValueError("Objective must be allan, std, or combined")
    if len(pairs) < 8:
        raise ValueError("At least 8 paired, finite SYNC shots are required")

    reference_signals = np.asarray([item["reference_signal"] for item in pairs], dtype=float)
    target_signals = np.asarray([item["target_signal"] for item in pairs], dtype=float)
    finite = np.isfinite(reference_signals) & np.isfinite(target_signals)
    reference_signals = reference_signals[finite]
    target_signals = target_signals[finite]
    selected_pairs = [item for item, keep in zip(pairs, finite) if keep]
    if len(selected_pairs) < 8:
        raise ValueError("At least 8 paired, finite SYNC shots are required")

    ref_a0, ref_c0, _, _, _ = _calibration_parameters(reference_calibration)
    target_a0, target_c0, _, _, _ = _calibration_parameters(target_calibration)
    bound_fraction = float(parameter_bound_fraction)
    initial = np.asarray([ref_a0, ref_c0, target_a0, target_c0], dtype=float)
    scales = np.asarray([ref_a0, ref_a0, target_a0, target_a0], dtype=float)
    lower = np.asarray([
        ref_a0 * (1.0 - bound_fraction), ref_c0 - ref_a0 * bound_fraction,
        target_a0 * (1.0 - bound_fraction), target_c0 - target_a0 * bound_fraction,
    ])
    upper = np.asarray([
        ref_a0 * (1.0 + bound_fraction), ref_c0 + ref_a0 * bound_fraction,
        target_a0 * (1.0 + bound_fraction), target_c0 + target_a0 * bound_fraction,
    ])
    bounds = list(zip(lower, upper))

    reference_sign = -1.0 if interferometer_phase.monotonic_slope(reference_calibration) == "positive" else 1.0
    target_sign = -1.0 if interferometer_phase.monotonic_slope(target_calibration) == "positive" else 1.0
    reference_reference_phase = _reference_phase(reference_calibration)
    target_reference_phase = _reference_phase(target_calibration)

    def evaluate(parameters: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ref_phase, ref_valid, ref_overflow = _phase_values(
            reference_signals, parameters[0], parameters[1], reference_sign, reference_reference_phase
        )
        target_phase, target_valid, target_overflow = _phase_values(
            target_signals, parameters[2], parameters[3], target_sign, target_reference_phase
        )
        valid = ref_valid & target_valid
        difference = np.unwrap(target_phase[valid] - ref_phase[valid])
        return difference, valid, ref_overflow, target_overflow

    baseline_difference, baseline_valid, _, _ = evaluate(initial)
    if np.count_nonzero(baseline_valid) < 8:
        raise ValueError("The current calibrations produce fewer than 8 valid paired phases")
    baseline_scale = max(
        _objective_metric(baseline_difference, objective, combined_allan_weight), 1e-9
    )

    def loss(parameters: np.ndarray) -> float:
        difference, valid, ref_overflow, target_overflow = evaluate(parameters)
        valid_count = int(np.count_nonzero(valid))
        invalid_count = len(valid) - valid_count
        if valid_count < 8:
            return 1e9 + (8 - valid_count) * 1e8
        sync_loss = (_objective_metric(difference, objective, combined_allan_weight) / baseline_scale) ** 2
        ref_curve_rmse, _ = _curve_constraint(reference_calibration, parameters[0], parameters[1])
        target_curve_rmse, _ = _curve_constraint(target_calibration, parameters[2], parameters[3])
        curve_loss = (ref_curve_rmse / ref_a0) ** 2 + (target_curve_rmse / target_a0) ** 2
        prior_loss = float(np.mean(((parameters - initial) / (scales * bound_fraction)) ** 2))
        overflow_loss = float(np.mean(ref_overflow ** 2 + target_overflow ** 2))
        invalid_loss = invalid_count / len(valid)
        return float(sync_loss + fringe_weight * curve_loss + prior_weight * prior_loss + 1e4 * overflow_loss + 100.0 * invalid_loss)

    starts = [initial]
    for direction in (-0.35, 0.35):
        starts.append(np.clip(initial + direction * (upper - lower), lower, upper))
    best = None
    for start in starts:
        candidate = minimize(loss, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1200, "ftol": 1e-13})
        if best is None or float(candidate.fun) < float(best.fun):
            best = candidate
    if best is None or not np.all(np.isfinite(best.x)):
        raise ValueError("SYNC phase calibration optimization failed")

    optimized = np.asarray(best.x, dtype=float)
    final_difference, final_valid, _, _ = evaluate(optimized)
    ref_curve_before, ref_constraint_source = _curve_constraint(reference_calibration, ref_a0, ref_c0)
    target_curve_before, target_constraint_source = _curve_constraint(target_calibration, target_a0, target_c0)
    ref_curve_after, _ = _curve_constraint(reference_calibration, optimized[0], optimized[1])
    target_curve_after, _ = _curve_constraint(target_calibration, optimized[2], optimized[3])

    warnings_out: List[str] = []
    tolerance = 1e-4 * np.maximum(upper - lower, 1e-12)
    names = ("reference A", "reference C", "target A", "target C")
    for index, name in enumerate(names):
        if optimized[index] <= lower[index] + tolerance[index] or optimized[index] >= upper[index] - tolerance[index]:
            warnings_out.append(f"Optimized {name} reached its allowed boundary.")
    if np.count_nonzero(final_valid) != len(final_valid):
        warnings_out.append("Some paired shots are outside an optimized fringe envelope.")
    warnings_out.append("At mid-fringe, C is better constrained by SYNC data than A; use the fringe constraint when judging A changes.")

    def parameter_payload(values: np.ndarray) -> Dict[str, Dict[str, float]]:
        return {
            "reference": {"A": float(values[0]), "C": float(values[1])},
            "target": {"A": float(values[2]), "C": float(values[3])},
        }

    before_metrics = _metrics(baseline_difference)
    after_metrics = _metrics(final_difference)
    before_metrics.update({
        "valid_pair_count": int(np.count_nonzero(baseline_valid)),
        "invalid_pair_count": int(len(baseline_valid) - np.count_nonzero(baseline_valid)),
    })
    after_metrics.update({
        "valid_pair_count": int(np.count_nonzero(final_valid)),
        "invalid_pair_count": int(len(final_valid) - np.count_nonzero(final_valid)),
    })

    def full_difference_series(parameters: np.ndarray) -> np.ndarray:
        ref_phase, ref_valid, _ = _phase_values(
            reference_signals, parameters[0], parameters[1], reference_sign, reference_reference_phase
        )
        target_phase, target_valid, _ = _phase_values(
            target_signals, parameters[2], parameters[3], target_sign, target_reference_phase
        )
        valid = ref_valid & target_valid
        values = np.full(len(valid), np.nan, dtype=float)
        values[valid] = np.unwrap(target_phase[valid] - ref_phase[valid])
        return values

    def full_phase_series(parameters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        reference_values, _, _ = _phase_values(
            reference_signals, parameters[0], parameters[1], reference_sign, reference_reference_phase
        )
        target_values, _, _ = _phase_values(
            target_signals, parameters[2], parameters[3], target_sign, target_reference_phase
        )
        return reference_values, target_values

    before_series = full_difference_series(initial)
    after_series = full_difference_series(optimized)
    reference_before_series, target_before_series = full_phase_series(initial)
    reference_after_series, target_after_series = full_phase_series(optimized)
    return {
        "objective": objective,
        "combined_allan_weight": float(combined_allan_weight),
        "success": bool(best.success),
        "message": str(best.message),
        "objective_value": float(best.fun),
        "pair_count": len(selected_pairs),
        "parameters_before": parameter_payload(initial),
        "parameters_after": parameter_payload(optimized),
        "bounds": {
            "reference": {"A": [float(lower[0]), float(upper[0])], "C": [float(lower[1]), float(upper[1])]},
            "target": {"A": [float(lower[2]), float(upper[2])], "C": [float(lower[3]), float(upper[3])]},
        },
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "fringe_constraint": {
            "reference": {"source": ref_constraint_source, "rmse_before": ref_curve_before, "rmse_after": ref_curve_after},
            "target": {"source": target_constraint_source, "rmse_before": target_curve_before, "rmse_after": target_curve_after},
        },
        "optimized_reference_calibration": _optimized_calibration(reference_calibration, optimized[0], optimized[1]),
        "optimized_target_calibration": _optimized_calibration(target_calibration, optimized[2], optimized[3]),
        "series": [
            {
                "shot": item.get("shot"),
                "p0": item.get("p0"),
                "before_rad": float(before_series[index]) if np.isfinite(before_series[index]) else None,
                "after_rad": float(after_series[index]) if np.isfinite(after_series[index]) else None,
                "reference_before_rad": float(reference_before_series[index]) if np.isfinite(reference_before_series[index]) else None,
                "reference_after_rad": float(reference_after_series[index]) if np.isfinite(reference_after_series[index]) else None,
                "target_before_rad": float(target_before_series[index]) if np.isfinite(target_before_series[index]) else None,
                "target_after_rad": float(target_after_series[index]) if np.isfinite(target_after_series[index]) else None,
            }
            for index, item in enumerate(selected_pairs)
        ],
        "warnings": warnings_out,
    }
