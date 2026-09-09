from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from app.analysis import interferometer_phase


def calibration_at_mid_fringe(calibration: Dict[str, Any], t2_us2: float) -> Dict[str, Any]:
    """Return a calibration whose zero and monotonic branch match one mid fringe."""
    reference = float(t2_us2)
    if not math.isfinite(reference) or reference < 0:
        raise ValueError("Phase-noise mid fringe must be a non-negative finite T² value")
    result = deepcopy(calibration)
    result.update({
        "reference_input_mode": "t2",
        "reference_t_unit": "us",
        "reference_value": reference,
        "reference_t2_us2": reference,
        "phase_conversion_mode": "monotonic_half_fringe",
    })
    result.pop("monotonic_slope", None)
    result["monotonic_slope"] = interferometer_phase.monotonic_slope(result)
    return result


def validate_mid_fringe_values(calibration: Dict[str, Any], values: Iterable[float]) -> List[float]:
    available = [
        float(value) for value in (calibration.get("bragg") or {}).get("mid_fringe_x") or []
        if math.isfinite(float(value)) and float(value) >= 0
    ]
    selected: List[float] = []
    for raw in values:
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Selected phase-noise mid fringes must be non-negative and finite")
        tolerance = 1e-9 * max(1.0, abs(value))
        if not any(abs(value - candidate) <= tolerance for candidate in available):
            raise ValueError(f"Selected T²={value:g} is not part of the active Bragg calibration")
        if not any(abs(value - existing) <= tolerance for existing in selected):
            selected.append(value)
    if not selected:
        raise ValueError("Select at least one mid fringe for Phase Noise Analyze")
    return sorted(selected)


def _finite(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def normalize_allan_orders(orders: Optional[Iterable[int]]) -> List[int]:
    """Return unique positive Allan orders in ascending order."""
    normalized: List[int] = []
    for raw in orders or []:
        try:
            order = int(raw)
        except (TypeError, ValueError):
            continue
        if order >= 1 and order not in normalized:
            normalized.append(order)
    return sorted(normalized)


def overlapping_allan_deviation(
    values: Iterable[Optional[float]], order: int,
) -> tuple[Optional[float], int]:
    """Calculate overlapping Allan deviation without bridging invalid shots."""
    n = int(order)
    if n < 1:
        raise ValueError("Allan order must be at least 1")
    samples = np.asarray([
        value if (value := _finite(raw)) is not None else np.nan
        for raw in values
    ], dtype=float)
    window_count = samples.size - 2 * n + 1
    if window_count <= 0:
        return None, 0
    valid = np.isfinite(samples)
    safe = np.where(valid, samples, 0.0)
    value_prefix = np.concatenate(([0.0], np.cumsum(safe)))
    valid_prefix = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    valid_a = valid_prefix[n:n + window_count] - valid_prefix[:window_count]
    valid_b = valid_prefix[2 * n:2 * n + window_count] - valid_prefix[n:n + window_count]
    valid_windows = (valid_a == n) & (valid_b == n)
    valid_window_count = int(np.count_nonzero(valid_windows))
    if valid_window_count == 0:
        return None, 0
    mean_a = (value_prefix[n:n + window_count] - value_prefix[:window_count]) / n
    mean_b = (value_prefix[2 * n:2 * n + window_count] - value_prefix[n:n + window_count]) / n
    differences = (mean_b - mean_a) / math.sqrt(2.0)
    return float(np.sqrt(np.mean(np.square(differences[valid_windows])))), valid_window_count


def build_phase_noise_summary(
    points: Iterable[Any],
    calibration: Optional[Dict[str, Any]],
    detection_noise_signal: float,
    laser_phase_noise_mrad: float,
    allan_orders: Optional[Iterable[int]] = (1,),
) -> List[Dict[str, Any]]:
    """Aggregate measured and configured noise for every scanned mid fringe."""
    params = (calibration or {}).get("parameter_values") or {}
    amplitude = _finite(params.get("A"))
    detection = _finite(detection_noise_signal)
    laser_mrad = _finite(laser_phase_noise_mrad)
    detection_rad = (
        abs(detection / amplitude)
        if detection is not None and amplitude is not None and amplitude > 0
        else None
    )
    laser_rad = abs(laser_mrad) / 1000.0 if laser_mrad is not None else None
    expected_rad = (
        math.hypot(detection_rad, laser_rad)
        if detection_rad is not None and laser_rad is not None
        else None
    )

    grouped: Dict[float, List[Optional[float]]] = {}
    counts: Dict[float, int] = {}
    for point in points:
        getter = point.get if isinstance(point, dict) else lambda key, default=None: getattr(point, key, default)
        reference = _finite(getter("interferometer_phase_reference_t2_us2"))
        if reference is None:
            reference = _finite(getter("parameter"))
        if reference is None:
            continue
        key = round(reference, 9)
        counts[key] = counts.get(key, 0) + 1
        phase = _finite(getter("interferometer_phase"))
        valid = bool(getter("interferometer_phase_valid", False))
        grouped.setdefault(key, []).append(phase if valid and phase is not None else None)

    result: List[Dict[str, Any]] = []
    requested_orders = normalize_allan_orders(allan_orders)
    for t2 in sorted(counts):
        sequence = grouped.get(t2, [])
        phases = [value for value in sequence if value is not None]
        measured = float(np.std(phases, ddof=1)) if len(phases) >= 2 else None
        allan_rows = []
        for order in requested_orders:
            allan_measured, valid_windows = overlapping_allan_deviation(sequence, order)
            order_scale = math.sqrt(order)
            allan_detection = detection_rad / order_scale if detection_rad is not None else None
            allan_laser = laser_rad / order_scale if laser_rad is not None else None
            allan_expected = expected_rad / order_scale if expected_rad is not None else None
            allan_rows.append({
                "order": order,
                "measured_phase_noise_rad": allan_measured,
                "valid_window_count": valid_windows,
                "detection_phase_noise_rad": allan_detection,
                "laser_phase_noise_rad": allan_laser,
                "expected_total_phase_noise_rad": allan_expected,
            })
        result.append({
            "t2_us2": float(t2),
            "t_ms": math.sqrt(t2) / 1000.0,
            "shot_count": int(counts[t2]),
            "valid_phase_count": len(phases),
            "measured_phase_noise_rad": measured,
            "detection_phase_noise_rad": detection_rad,
            "laser_phase_noise_rad": laser_rad,
            "expected_total_phase_noise_rad": expected_rad,
            "available_allan_max_order": len(sequence) // 2,
            "allan_deviations": allan_rows,
        })
    return result
