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
) -> List[Dict[str, Any]]:
    phase_amplitude = bragg_phase_modulation_rad(frequency_modulation_mhz)
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
        row["interferometer_phase_s2"] = (
            float((phase_mean / phase_amplitude) ** 2)
            if phase_mean is not None and phase_amplitude is not None
            else None
        )
        rows.append(row)
    return rows
