"""Time-resolved interferometer labeling-efficiency calibration."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import numpy as np
from scipy.interpolate import UnivariateSpline


INTERPOLATION_METHODS = {"linear", "weighted_smoothing_spline", "nearest"}


def _finite_float_or_nan(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return converted if math.isfinite(converted) else float("nan")


def alpha_from_probability_percent(probability_percent: Any, gamma: Any) -> float:
    probability = float(probability_percent) / 100.0
    gamma_value = float(gamma)
    denominator = 1.0 - gamma_value
    if not math.isfinite(probability) or not math.isfinite(gamma_value) or denominator <= 0:
        raise ValueError("I_alpha calibration inputs are not finite or I_Gamma is not below one")
    alpha = probability / denominator
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"calibrated I_alpha {alpha:.6g} is outside [0, 1]")
    return alpha


def accepted_points(events: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    points = []
    for event in events:
        try:
            timestamp = float(event.get("representative_time"))
            alpha = float(event.get("intf_alpha"))
        except (TypeError, ValueError):
            continue
        if event.get("accepted") is True and math.isfinite(timestamp) and 0.0 <= alpha <= 1.0:
            points.append({**event, "representative_time": timestamp, "intf_alpha": alpha})
    return sorted(points, key=lambda item: item["representative_time"])


def build_alpha_interpolator(events: Iterable[Dict[str, Any]], method: str = "linear"):
    points = accepted_points(events)
    if not points:
        return None, method
    requested_method = str(method or "linear").strip().lower()
    if requested_method not in INTERPOLATION_METHODS:
        raise ValueError(f"Unsupported I_alpha interpolation method: {method}")
    effective_method = requested_method
    spline = None
    if requested_method == "weighted_smoothing_spline" and len(points) >= 4:
        timestamps = np.asarray([point["representative_time"] for point in points], dtype=float)
        span = float(timestamps[-1] - timestamps[0])
        if span > 0 and np.all(np.diff(timestamps) > 0):
            x = (timestamps - timestamps[0]) / span
            y = np.asarray([point["intf_alpha"] for point in points], dtype=float)
            raw_sem = np.asarray([
                _finite_float_or_nan(point.get("intf_alpha_sem")) for point in points
            ], dtype=float)
            positive_sem = raw_sem[np.isfinite(raw_sem) & (raw_sem > 0)]
            fallback_sem = float(np.median(positive_sem)) if positive_sem.size else 1.0
            sigma = np.where(np.isfinite(raw_sem) & (raw_sem > 0), raw_sem, fallback_sem)
            sigma = np.maximum(sigma, max(fallback_sem * 0.1, np.finfo(float).eps))
            spline = UnivariateSpline(x, y, w=1.0 / sigma, k=3, s=float(len(points)))
        else:
            effective_method = "linear"
    elif requested_method == "weighted_smoothing_spline":
        effective_method = "linear"

    def evaluate(timestamp: Any) -> Dict[str, Any]:
        value = float(timestamp)
        if len(points) == 1 or value <= points[0]["representative_time"]:
            point = points[0]
            return {"value": point["intf_alpha"], "left_id": point["calibration_id"], "right_id": point["calibration_id"], "extrapolated": value != point["representative_time"], "method": effective_method}
        if value >= points[-1]["representative_time"]:
            point = points[-1]
            return {"value": point["intf_alpha"], "left_id": point["calibration_id"], "right_id": point["calibration_id"], "extrapolated": value != point["representative_time"], "method": effective_method}
        for left, right in zip(points, points[1:]):
            if left["representative_time"] <= value <= right["representative_time"]:
                if requested_method == "nearest":
                    selected = left if value - left["representative_time"] <= right["representative_time"] - value else right
                    interpolated = selected["intf_alpha"]
                elif spline is not None:
                    normalized_time = (value - points[0]["representative_time"]) / (points[-1]["representative_time"] - points[0]["representative_time"])
                    interpolated = float(spline(normalized_time))
                else:
                    fraction = (value - left["representative_time"]) / (right["representative_time"] - left["representative_time"])
                    interpolated = float(left["intf_alpha"] + fraction * (right["intf_alpha"] - left["intf_alpha"]))
                return {"value": float(np.clip(interpolated, 0.0, 1.0)), "left_id": left["calibration_id"], "right_id": right["calibration_id"], "extrapolated": False, "method": effective_method}
        raise ValueError("I_alpha interpolation interval could not be resolved")

    return evaluate, effective_method


def interpolate_alpha(timestamp: Any, events: Iterable[Dict[str, Any]], method: str = "linear") -> Optional[Dict[str, Any]]:
    interpolator, _ = build_alpha_interpolator(events, method)
    return interpolator(timestamp) if interpolator is not None else None


def summarize_block(
    probabilities: Iterable[Any], timestamps: Iterable[Any], gamma: Any,
    calibration_id: str, requested_shots: int,
    accepted_min: Any = 0.0, accepted_max: Any = 1.0,
) -> Dict[str, Any]:
    values = np.asarray([float(value) for value in probabilities if value is not None and math.isfinite(float(value))], dtype=float)
    times = np.asarray([float(value) for value in timestamps if value is not None and math.isfinite(float(value))], dtype=float)
    event: Dict[str, Any] = {
        "calibration_id": calibration_id, "requested_shots": int(requested_shots),
        "valid_shots": int(values.size), "accepted": False,
    }
    if values.size < 2 or times.size < 2:
        event["rejection_reason"] = "fewer_than_two_valid_shots"
        return event
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    event.update({
        "representative_time": float(np.mean(times)),
        "transition_probability_up_mean_percent": mean,
        "transition_probability_up_std_percent": std,
        "transition_probability_up_sem_percent": float(std / math.sqrt(values.size)),
        "intf_gamma": float(gamma),
        "accepted_min": float(accepted_min),
        "accepted_max": float(accepted_max),
    })
    try:
        event["intf_alpha"] = alpha_from_probability_percent(mean, gamma)
        event["intf_alpha_sem"] = float((std / math.sqrt(values.size)) / 100.0 / (1.0 - float(gamma)))
        if not float(accepted_min) <= event["intf_alpha"] <= float(accepted_max):
            raise ValueError(
                f"calibrated I_alpha {event['intf_alpha']:.6g} is outside accepted range "
                f"[{float(accepted_min):.6g}, {float(accepted_max):.6g}]"
            )
        event["accepted"] = True
        event["rejection_reason"] = ""
    except (TypeError, ValueError) as exc:
        event["rejection_reason"] = str(exc)
    return event
