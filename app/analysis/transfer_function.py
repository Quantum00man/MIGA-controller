"""Per-frequency sample statistics for Transfer Function scans."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


SPEED_OF_LIGHT_M_S = 299_792_458.0
DEFAULT_ATOM_MIRROR_DISTANCE_M = 2.23
# Backward-compatible name for integrations that imported the former constant.
ATOM_MIRROR_DISTANCE_M = DEFAULT_ATOM_MIRROR_DISTANCE_M
SYNC_DIFFERENTIAL_FORMULA_VERSION = 3


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


def bragg_phase_modulation_rad(
    frequency_modulation_mhz: Any,
    atom_mirror_distance_m: Any = DEFAULT_ATOM_MIRROR_DISTANCE_M,
) -> Optional[float]:
    """Return the Eq. (14) Bragg phase amplitude for actual 780-nm FM."""
    try:
        modulation_hz = float(frequency_modulation_mhz) * 1_000_000.0
        distance_m = float(atom_mirror_distance_m)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(modulation_hz)
        or modulation_hz <= 0
        or not math.isfinite(distance_m)
        or distance_m <= 0
    ):
        return None
    return (4.0 * math.pi * distance_m / SPEED_OF_LIGHT_M_S) * modulation_hz


def build_transfer_function_summary(
    results: Iterable[Any],
    frequency_modulation_mhz: Any = None,
    phase_degrees: Any = None,
    atom_mirror_distance_m: Any = DEFAULT_ATOM_MIRROR_DISTANCE_M,
    phase_noise_sigma_mrad: Any = 100.0,
) -> List[Dict[str, Any]]:
    phase_amplitude = bragg_phase_modulation_rad(
        frequency_modulation_mhz,
        atom_mirror_distance_m,
    )
    try:
        distance_m = float(atom_mirror_distance_m)
    except (TypeError, ValueError):
        distance_m = None
    if distance_m is not None and (not math.isfinite(distance_m) or distance_m <= 0):
        distance_m = None
    try:
        noise_sigma_mrad = float(phase_noise_sigma_mrad)
    except (TypeError, ValueError):
        noise_sigma_mrad = None
    if noise_sigma_mrad is not None and (
        not math.isfinite(noise_sigma_mrad) or noise_sigma_mrad < 0
    ):
        noise_sigma_mrad = None
    noise_sigma_rad = noise_sigma_mrad / 1000.0 if noise_sigma_mrad is not None else None
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
    if not expected_phases:
        inferred_phases = set()
        for samples in grouped.values():
            for item in samples:
                raw_phase = item.get("transfer_phase_deg") if isinstance(item, dict) else getattr(item, "transfer_phase_deg", None)
                try:
                    numeric_phase = float(raw_phase)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric_phase) and numeric_phase in {0.0, 90.0}:
                    inferred_phases.add(numeric_phase)
        expected_phases = sorted(inferred_phases)

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
        row["atom_mirror_distance_m"] = distance_m if phase_amplitude is not None else None
        row["bragg_phase_modulation_rad"] = phase_amplitude
        row["transfer_phase_noise_sigma_mrad"] = noise_sigma_mrad
        for output_name in METRIC_FIELDS:
            if output_name == "interferometer_phase_std":
                continue
            metric_name = output_name.removesuffix("_std")
            for phase_label in ("0deg", "90deg"):
                row[f"{metric_name}_{phase_label}_count"] = 0
                row[f"{metric_name}_{phase_label}_mean"] = None
                row[f"{metric_name}_{phase_label}_std"] = None
        for phase_label in ("0deg", "90deg"):
            row[f"interferometer_phase_{phase_label}_count"] = 0
            row[f"interferometer_phase_{phase_label}_mean_rad"] = None
            row[f"interferometer_phase_{phase_label}_std_rad"] = None
            row[f"interferometer_phase_{phase_label}_s2"] = None
            row[f"interferometer_phase_{phase_label}_phase2_rad2"] = None
        phase_components: List[Dict[str, Any]] = []
        for phase_deg in expected_phases:
            selected_samples = []
            for item in samples:
                raw_phase = item.get("transfer_phase_deg") if isinstance(item, dict) else getattr(item, "transfer_phase_deg", None)
                try:
                    matches_phase = math.isclose(float(raw_phase), phase_deg, abs_tol=1e-9)
                except (TypeError, ValueError):
                    matches_phase = False
                if matches_phase:
                    selected_samples.append(item)
            if phase_deg in {0.0, 90.0}:
                phase_label = f"{int(phase_deg)}deg"
                for output_name, fields in METRIC_FIELDS.items():
                    if output_name == "interferometer_phase_std":
                        continue
                    metric_name = output_name.removesuffix("_std")
                    values = [
                        value for item in selected_samples
                        if (value := _value(item, fields)) is not None
                    ]
                    row[f"{metric_name}_{phase_label}_count"] = len(values)
                    row[f"{metric_name}_{phase_label}_mean"] = float(np.mean(values)) if values else None
                    row[f"{metric_name}_{phase_label}_std"] = (
                        float(np.std(values, ddof=1)) if len(values) >= 2 else None
                    )
            phase_samples = [
                value for item in selected_samples
                if (value := _value(item, ("interferometer_phase",))) is not None
            ]
            component_mean = float(np.mean(phase_samples)) if phase_samples else None
            component_std = float(np.std(phase_samples, ddof=1)) if len(phase_samples) >= 2 else None
            component_s2 = (
                float((component_mean / phase_amplitude) ** 2)
                if component_mean is not None and phase_amplitude is not None
                else None
            )
            component_phase2 = float(component_mean ** 2) if component_mean is not None else None
            phase_components.append({
                "phase_deg": phase_deg,
                "count": len(phase_samples),
                "mean_rad": component_mean,
                "std_rad": component_std,
                "s2": component_s2,
                "phase2_rad2": component_phase2,
            })
            if phase_deg in {0.0, 90.0}:
                phase_label = f"{int(phase_deg)}deg"
                row[f"interferometer_phase_{phase_label}_count"] = len(phase_samples)
                row[f"interferometer_phase_{phase_label}_mean_rad"] = component_mean
                row[f"interferometer_phase_{phase_label}_std_rad"] = component_std
                row[f"interferometer_phase_{phase_label}_s2"] = component_s2
                row[f"interferometer_phase_{phase_label}_phase2_rad2"] = component_phase2
        row["interferometer_phase_s2_components"] = phase_components
        phase2_values = [component.get("phase2_rad2") for component in phase_components]
        row["interferometer_phase_phase2_rad2"] = (
            float(sum(phase2_values))
            if len(phase2_values) == 2 and all(value is not None for value in phase2_values)
            else None
        )
        row["interferometer_phase_noise_phase2_rad2"] = (
            float(len(phase_components) * noise_sigma_rad ** 2)
            if phase_components and noise_sigma_rad is not None
            else None
        )
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
            row["interferometer_phase_phase2_rad2"] = (
                float(phase_mean ** 2) if phase_mean is not None else None
            )
            row["interferometer_phase_noise_phase2_rad2"] = (
                float(noise_sigma_rad ** 2) if noise_sigma_rad is not None else None
            )
        else:
            row["interferometer_phase_s2"] = None
        rows.append(row)
    return rows


def build_differential_transfer_function_summary(
    pairs: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build paired differential quadratures normalized by the baseline response."""
    grouped: Dict[tuple[str, float], Dict[float, List[float]]] = {}
    for pair in pairs:
        master = pair.get("master") or {}
        slave = pair.get("slave") or {}
        if not (_value(master, ("interferometer_phase",)) is not None
                and _value(slave, ("interferometer_phase",)) is not None):
            continue
        try:
            frequency = float(master.get("transfer_frequency_hz", slave.get("transfer_frequency_hz")))
            phase_deg = float(master.get("transfer_phase_deg", slave.get("transfer_phase_deg")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(frequency) or phase_deg not in {0.0, 90.0}:
            continue

        try:
            modulation_hz = float(master.get("transfer_frequency_modulation_mhz")) * 1_000_000.0
            master_distance_m = float(master.get("transfer_atom_mirror_distance_m"))
            slave_distance_m = float(slave.get("transfer_atom_mirror_distance_m"))
        except (TypeError, ValueError):
            continue
        baseline_m = master_distance_m - slave_distance_m
        denominator = (4.0 * math.pi * modulation_hz / SPEED_OF_LIGHT_M_S) * baseline_m
        if not math.isfinite(denominator) or denominator == 0.0:
            continue
        master_phase = _value(master, ("interferometer_phase",))
        slave_phase = _value(slave, ("interferometer_phase",))
        if master_phase is None or slave_phase is None:
            continue
        slave_id = str(pair.get("slave_node_id") or slave.get("sync_node_id") or "slave")
        grouped.setdefault((slave_id, frequency), {0.0: [], 90.0: []})[phase_deg].append(
            float((master_phase - slave_phase) / denominator)
        )

    rows: List[Dict[str, Any]] = []
    for (slave_id, frequency), components in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        row: Dict[str, Any] = {"slave_node_id": slave_id, "frequency_hz": frequency}
        means: List[Optional[float]] = []
        for phase_deg in (0.0, 90.0):
            values = components[phase_deg]
            mean = float(np.mean(values)) if values else None
            std = float(np.std(values, ddof=1)) if len(values) >= 2 else None
            label = f"{int(phase_deg)}deg"
            row[f"delta_s_{label}_count"] = len(values)
            row[f"delta_s_{label}_mean"] = mean
            row[f"delta_s_{label}_std"] = std
            row[f"delta_s_{label}_sem"] = std / math.sqrt(len(values)) if std is not None else None
            row[f"delta_s_{label}_s2"] = float(mean * mean) if mean is not None else None
            row[f"delta_s_{label}_s2_sem"] = (
                float(2.0 * abs(mean) * row[f"delta_s_{label}_sem"])
                if mean is not None and row[f"delta_s_{label}_sem"] is not None
                else None
            )
            means.append(mean)
        available = [value for value in means if value is not None]
        row["differential_s2"] = float(sum(value * value for value in available)) if available else None
        row["differential_magnitude"] = math.sqrt(row["differential_s2"]) if row["differential_s2"] is not None else None
        row["quadrature_complete"] = all(value is not None for value in means)
        rows.append(row)
    return rows
