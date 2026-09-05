"""Per-frequency sample statistics for Transfer Function scans."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


SPEED_OF_LIGHT_M_S = 299_792_458.0
ATOM_MIRROR_DISTANCE_M = 2.23


METRIC_FIELDS = {
    "atom_number_up_fit_std": ("atom_number_up",),
    "atom_number_dw_fit_std": ("atom_number_dw",),
    "atom_number_total_fit_std": ("atom_number_up", "atom_number_dw"),
    "atom_number_up_nofit_std": ("atom_number_up_nofit",),
    "atom_number_dw_nofit_std": ("atom_number_dw_nofit",),
    "atom_number_total_nofit_std": ("atom_number_up_nofit", "atom_number_dw_nofit"),
    "intf_p1_fit_std": ("intf_p1",),
    "intf_p2_fit_std": ("intf_p2",),
    "intf_p1_nofit_std": ("intf_p1_nofit",),
    "intf_p2_nofit_std": ("intf_p2_nofit",),
    "interferometer_phase_std": ("interferometer_phase",),
}


def _value(result: Any, fields: Iterable[str]) -> Optional[float]:
    field_names = tuple(fields)
    if field_names == ("interferometer_phase",):
        phase_valid = (
            result.get("interferometer_phase_valid")
            if isinstance(result, dict)
            else getattr(result, "interferometer_phase_valid", False)
        )
        if phase_valid is not True and phase_valid not in (1, "1", "true", "True"):
            return None
    values: List[float] = []
    for field in field_names:
        raw = result.get(field) if isinstance(result, dict) else getattr(result, field, None)
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        values.append(numeric)
    return float(sum(values))


def bragg_phase_modulation_rad(frequency_modulation_mhz: Any) -> Optional[float]:
    """Return the Eq. (14) Bragg phase amplitude for actual 780-nm FM."""
    try:
        modulation_hz = float(frequency_modulation_mhz) * 1_000_000.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(modulation_hz) or modulation_hz <= 0:
        return None
    return (4.0 * math.pi * ATOM_MIRROR_DISTANCE_M / SPEED_OF_LIGHT_M_S) * modulation_hz


def build_transfer_function_summary(
    results: Iterable[Any],
    frequency_modulation_mhz: Any = None,
    phase_degrees: Any = None,
) -> List[Dict[str, Any]]:
    phase_amplitude = bragg_phase_modulation_rad(frequency_modulation_mhz)
    try:
        expected_phases = [float(value) for value in phase_degrees] if phase_degrees is not None else []
    except (TypeError, ValueError):
        expected_phases = []
    grouped: Dict[float, List[Any]] = {}
    for result in results:
        raw_frequency = (
            result.get("transfer_frequency_hz")
            if isinstance(result, dict)
            else getattr(result, "transfer_frequency_hz", None)
        )
        try:
            frequency = float(raw_frequency)
        except (TypeError, ValueError):
            continue
        if math.isfinite(frequency):
            grouped.setdefault(frequency, []).append(result)

    rows: List[Dict[str, Any]] = []
    for frequency in sorted(grouped):
        samples = grouped[frequency]
        row: Dict[str, Any] = {
            "frequency_hz": frequency,
            "shot_count": len(samples),
        }
        for output_name, fields in METRIC_FIELDS.items():
            values = [value for item in samples if (value := _value(item, fields)) is not None]
            row[output_name.replace("_std", "_count")] = len(values)
            row[output_name.replace("_std", "_mean")] = float(np.mean(values)) if values else None
            row[output_name] = float(np.std(values, ddof=1)) if len(values) >= 2 else None
        phase_mean = row.get("interferometer_phase_mean")
        row["frequency_modulation_mhz"] = (
            float(frequency_modulation_mhz) if phase_amplitude is not None else None
        )
        row["bragg_phase_modulation_rad"] = phase_amplitude
        for phase_label in ("0deg", "90deg"):
            row[f"interferometer_phase_{phase_label}_count"] = 0
            row[f"interferometer_phase_{phase_label}_mean_rad"] = None
            row[f"interferometer_phase_{phase_label}_std_rad"] = None
            row[f"interferometer_phase_{phase_label}_s2"] = None
        phase_components: List[Dict[str, Any]] = []
        for phase_deg in expected_phases:
            phase_samples = []
            for item in samples:
                raw_phase = item.get("transfer_phase_deg") if isinstance(item, dict) else getattr(item, "transfer_phase_deg", None)
                try:
                    matches_phase = math.isclose(float(raw_phase), phase_deg, abs_tol=1e-9)
                except (TypeError, ValueError):
                    matches_phase = False
                if matches_phase:
                    value = _value(item, ("interferometer_phase",))
                    if value is not None:
                        phase_samples.append(value)
            component_mean = float(np.mean(phase_samples)) if phase_samples else None
            component_std = float(np.std(phase_samples, ddof=1)) if len(phase_samples) >= 2 else None
            component_s2 = (
                float((component_mean / phase_amplitude) ** 2)
                if component_mean is not None and phase_amplitude is not None
                else None
            )
            phase_components.append({
                "phase_deg": phase_deg,
                "count": len(phase_samples),
                "mean_rad": component_mean,
                "std_rad": component_std,
                "s2": component_s2,
            })
            if phase_deg in {0.0, 90.0}:
                phase_label = f"{int(phase_deg)}deg"
                row[f"interferometer_phase_{phase_label}_count"] = len(phase_samples)
                row[f"interferometer_phase_{phase_label}_mean_rad"] = component_mean
                row[f"interferometer_phase_{phase_label}_std_rad"] = component_std
                row[f"interferometer_phase_{phase_label}_s2"] = component_s2
        row["interferometer_phase_s2_components"] = phase_components
        if len(phase_components) == 2:
            component_values = [component.get("s2") for component in phase_components]
            row["interferometer_phase_s2"] = (
                float(sum(component_values)) if all(value is not None for value in component_values) else None
            )
        elif not phase_components:
            row["interferometer_phase_s2"] = (
                float((phase_mean / phase_amplitude) ** 2)
                if phase_mean is not None and phase_amplitude is not None
                else None
            )
        else:
            row["interferometer_phase_s2"] = None
        rows.append(row)
    return rows
