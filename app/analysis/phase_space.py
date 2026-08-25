from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import numpy as np


def wrap_phase(value: float) -> float:
    wrapped = (float(value) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if math.isclose(wrapped, -math.pi) else wrapped


def nearest_cosine_phase(normalized_value: float, predicted_phase: float) -> float:
    principal = math.acos(float(np.clip(normalized_value, -1.0, 1.0)))
    candidates = []
    for base in (principal, -principal):
        cycle = round((predicted_phase - base) / (2.0 * math.pi))
        candidates.extend(base + 2.0 * math.pi * (cycle + shift) for shift in (-1, 0, 1))
    return float(min(candidates, key=lambda value: abs(value - predicted_phase)))


def _phase_statistics(points: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    values = np.asarray(
        [float(point["phase_deviation_rad"]) for point in points],
        dtype=float,
    )
    if not len(values):
        return {"count": 0, "mean_rad": None, "rms_rad": None, "std_rad": None}
    return {
        "count": int(len(values)),
        "mean_rad": float(np.mean(values)),
        "rms_rad": float(np.sqrt(np.mean(values ** 2))),
        "std_rad": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def convert_bragg_points_to_phase_space(
    points: Iterable[Dict[str, Any]],
    *,
    amplitude: float,
    offset: float,
    angular_frequency_rad_per_us2: float,
    phase_offset_rad: float,
    mid_fringe_fraction: float = 0.5,
) -> Dict[str, Any]:
    amplitude = float(amplitude)
    offset = float(offset)
    omega = float(angular_frequency_rad_per_us2)
    phase_offset = float(phase_offset_rad)
    mid_fraction = float(mid_fringe_fraction)
    if not all(math.isfinite(value) for value in (amplitude, offset, omega, phase_offset, mid_fraction)):
        raise ValueError("Bragg phase-space parameters must be finite")
    if amplitude <= 0:
        raise ValueError("Bragg phase-space amplitude must be positive")
    if omega <= 0:
        raise ValueError("Bragg phase-space angular frequency must be positive")
    if not 0 <= mid_fraction <= 1:
        raise ValueError("Mid-fringe fraction must be between 0 and 1")

    converted: List[Dict[str, Any]] = []
    for index, raw_point in enumerate(points):
        p0 = float(raw_point.get("p0"))
        value = float(raw_point.get("value"))
        if not math.isfinite(p0) or not math.isfinite(value):
            continue
        predicted_phase = omega * p0 + phase_offset
        raw_normalized = (value - offset) / amplitude
        clipped = raw_normalized < -1.0 or raw_normalized > 1.0
        normalized = float(np.clip(raw_normalized, -1.0, 1.0))
        measured_phase = nearest_cosine_phase(normalized, predicted_phase)
        deviation = wrap_phase(measured_phase - predicted_phase)
        sensitivity = abs(math.sin(predicted_phase))
        is_mid_fringe = abs(math.cos(predicted_phase)) <= mid_fraction
        is_high_quality = bool(is_mid_fringe and not clipped)
        converted.append({
            "index": index,
            "p0": p0,
            "value": value,
            "shot": raw_point.get("shot"),
            "key": str(raw_point.get("key") or ""),
            "predicted_phase_rad": float(predicted_phase),
            "measured_phase_rad": measured_phase,
            "phase_deviation_rad": deviation,
            "normalized_signal": normalized,
            "raw_normalized_signal": float(raw_normalized),
            "sensitivity": float(sensitivity),
            "is_mid_fringe": bool(is_mid_fringe),
            "is_clipped": bool(clipped),
            "is_high_quality": is_high_quality,
        })

    high_quality = [point for point in converted if point["is_high_quality"]]
    return {
        "points": converted,
        "statistics": {
            "all": _phase_statistics(converted),
            "mid_fringe": _phase_statistics(high_quality),
        },
        "mid_fringe_fraction": mid_fraction,
        "mid_fringe_lower": offset - mid_fraction * amplitude,
        "mid_fringe_upper": offset + mid_fraction * amplitude,
    }
