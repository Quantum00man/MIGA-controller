from __future__ import annotations

import math
from typing import Any, Dict, Optional

from app.analysis.phase_space import nearest_cosine_phase, wrap_phase


SOURCE_FIELDS = {
    ("atoms", "up", "fit"): "atom_number_up",
    ("atoms", "dw", "fit"): "atom_number_dw",
    ("atoms", "up", "raw"): "atom_number_up_nofit",
    ("atoms", "dw", "raw"): "atom_number_dw_nofit",
    ("prob", "up", "fit"): "transition_probability_up",
    ("prob", "dw", "fit"): "transition_probability_dw",
    ("prob", "up", "raw"): "transition_probability_up_nofit",
    ("prob", "dw", "raw"): "transition_probability_dw_nofit",
    ("intf", "up", "fit"): "intf_p1",
    ("intf", "dw", "fit"): "intf_p2",
    ("intf", "up", "raw"): "intf_p1_nofit",
    ("intf", "dw", "raw"): "intf_p2_nofit",
}


def _finite(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def source_mode(calibration: Dict[str, Any]) -> str:
    explicit = str(calibration.get("source_mode") or "").strip().lower()
    if explicit in {"fit", "raw"}:
        return explicit
    source_key = str(calibration.get("source_key") or "").strip().lower()
    return "raw" if source_key.startswith("nf_") or "nofit" in source_key else "fit"


def source_field(calibration: Dict[str, Any]) -> str:
    explicit = str(calibration.get("source_field") or "").strip()
    if explicit:
        return explicit
    metric = str(calibration.get("metric_tab") or "intf").strip().lower()
    channel = "dw" if str(calibration.get("channel") or "up").strip().lower() == "dw" else "up"
    return SOURCE_FIELDS.get((metric, channel, source_mode(calibration)), "intf_p1")


def reference_t2_us2(calibration: Dict[str, Any]) -> Optional[float]:
    direct = _finite(calibration.get("reference_t2_us2"))
    if direct is not None and direct >= 0:
        return direct
    value = _finite(calibration.get("reference_value"))
    if value is None or value < 0:
        return None
    mode = str(calibration.get("reference_input_mode") or "t2").strip().lower()
    unit = str(calibration.get("reference_t_unit") or "us").strip().lower()
    scale = 1000.0 if unit == "ms" else 1.0
    return (value * scale) ** 2 if mode == "t" else value * (scale ** 2)


def calculate_phase(point: Dict[str, Any], calibration: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "interferometer_phase": None,
        "interferometer_phase_valid": False,
        "interferometer_phase_source_value": None,
        "interferometer_phase_calibration_id": "",
        "interferometer_phase_calibration_name": "",
        "interferometer_phase_reference_t2_us2": None,
    }
    if not isinstance(calibration, dict):
        return result

    params = calibration.get("parameter_values") or {}
    bragg = calibration.get("bragg") or {}
    amplitude = _finite(params.get("A"))
    offset = _finite(params.get("C"))
    phi0 = _finite(params.get("phi0"))
    omega = _finite(bragg.get("angular_frequency_rad_per_us2"))
    t2 = reference_t2_us2(calibration)
    field = source_field(calibration)
    signal = _finite(point.get(field))
    result.update({
        "interferometer_phase_source_value": signal,
        "interferometer_phase_calibration_id": str(calibration.get("id") or ""),
        "interferometer_phase_calibration_name": str(calibration.get("name") or ""),
        "interferometer_phase_reference_t2_us2": t2,
    })
    if None in {amplitude, offset, phi0, omega, t2, signal} or amplitude <= 0 or omega <= 0:
        return result

    normalized = (signal - offset) / amplitude
    # Values outside the calibrated fringe envelope are deliberately invalid.
    # Clipping them to an extremum would bias ordinary and Allan statistics.
    if normalized < -1.0 or normalized > 1.0:
        return result
    reference_phase = omega * t2 + phi0
    measured_phase = nearest_cosine_phase(normalized, reference_phase)
    result["interferometer_phase"] = wrap_phase(measured_phase - reference_phase)
    result["interferometer_phase_valid"] = True
    return result


def apply_phase(point: Dict[str, Any], calibration: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**point, **calculate_phase(point, calibration)}
