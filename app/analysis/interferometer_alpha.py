"""Time-resolved interferometer labeling-efficiency calibration."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import numpy as np


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


def interpolate_alpha(timestamp: Any, events: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    points = accepted_points(events)
    if not points:
        return None
    value = float(timestamp)
    if len(points) == 1 or value <= points[0]["representative_time"]:
        point = points[0]
        return {"value": point["intf_alpha"], "left_id": point["calibration_id"], "right_id": point["calibration_id"], "extrapolated": value != point["representative_time"]}
    if value >= points[-1]["representative_time"]:
        point = points[-1]
        return {"value": point["intf_alpha"], "left_id": point["calibration_id"], "right_id": point["calibration_id"], "extrapolated": value != point["representative_time"]}
    for left, right in zip(points, points[1:]):
        if left["representative_time"] <= value <= right["representative_time"]:
            fraction = (value - left["representative_time"]) / (right["representative_time"] - left["representative_time"])
            return {
                "value": float(left["intf_alpha"] + fraction * (right["intf_alpha"] - left["intf_alpha"])),
                "left_id": left["calibration_id"], "right_id": right["calibration_id"], "extrapolated": False,
            }
    return None


def summarize_block(probabilities: Iterable[Any], timestamps: Iterable[Any], gamma: Any, calibration_id: str, requested_shots: int) -> Dict[str, Any]:
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
    })
    try:
        event["intf_alpha"] = alpha_from_probability_percent(mean, gamma)
        event["intf_alpha_sem"] = float((std / math.sqrt(values.size)) / 100.0 / (1.0 - float(gamma)))
        event["accepted"] = True
        event["rejection_reason"] = ""
    except (TypeError, ValueError) as exc:
        event["rejection_reason"] = str(exc)
    return event
