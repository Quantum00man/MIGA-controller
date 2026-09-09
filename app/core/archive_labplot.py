"""Translate an Archive run into a LabPlot 2.12.1 project."""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.core.labplot_export import Curve, Plot, Worksheet, build_project


METRICS: Dict[str, Dict[str, str]] = {
    "atoms": {"label": "Atom Number", "fit": "atoms", "raw": "nf_atoms"},
    "amp": {"label": "Max Voltage (V)", "fit": "amp", "raw": "nf_amp"},
    "tail": {"label": "Tail Mean (V)", "fit": "tail", "raw": "nf_tail"},
    "sigma": {"label": "Width (ms)", "fit": "sigma", "raw": "nf_sigma"},
    "temp": {"label": "Temperature (uK)", "fit": "temp", "raw": "nf_temp"},
    "arrival": {"label": "Arrival Time (s)", "fit": "arrival", "raw": "nf_arrival"},
    "prob": {"label": "Transition Probability (%)", "fit": "prob", "raw": "nf_prob"},
    "intf": {"label": "Interferometer P (%)", "fit": "intf_p", "raw": "nf_intf_p"},
    "phase": {"label": "Interferometer Phase (rad)", "fit": "phase", "raw": "phase"},
}

COLORS = [
    (0, 114, 178), (213, 94, 0), (0, 158, 115), (230, 159, 0),
    (204, 121, 167), (86, 180, 233), (0, 0, 0), (117, 117, 117),
]

TRANSFER_PHASE_COLORS = {
    0.0: (0, 114, 178),
    90.0: (213, 94, 0),
}
FIT_COLOR = (35, 35, 35)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _series(rows: Iterable[Dict[str, Any]], field: str) -> Tuple[List[float], List[float]]:
    x_values: List[float] = []
    y_values: List[float] = []
    for row in rows:
        x = _finite(row.get("x"))
        y = _finite(row.get(field))
        if x is not None and y is not None:
            x_values.append(x)
            y_values.append(y)
    return x_values, y_values


def _node_label(manifest: Optional[Dict[str, Any]], node_id: str) -> str:
    if node_id == "master":
        return str((manifest or {}).get("runtime", {}).get("master_node_id") or "Master")
    entry = ((manifest or {}).get("archive_nodes") or {}).get(node_id) or {}
    if entry.get("name"):
        return str(entry["name"])
    for slave in ((manifest or {}).get("runtime") or {}).get("slaves") or []:
        if str(slave.get("node_id") or "") == node_id:
            return str(slave.get("name") or node_id)
    return node_id


def _archive_node_ids(manifest: Optional[Dict[str, Any]]) -> List[str]:
    if not manifest:
        return [""]
    nodes = list((manifest.get("archive_nodes") or {}).keys())
    ordered = (["master"] if "master" in nodes else []) + [item for item in nodes if item != "master"]
    return ordered or ["master"]


def _calibration_curve(calibration: Any, channel: str, label: str) -> Optional[Curve]:
    if not isinstance(calibration, dict):
        return None
    if str(calibration.get("channel") or "up").lower() != channel:
        return None
    x = calibration.get("fit_x") or []
    y = calibration.get("fit_y") or []
    if len(x) < 2 or len(x) != len(y):
        return None
    return Curve(
        f"{label} fringe fit", x, y, FIT_COLOR,
        line=True, symbols=False, line_style=2,
    )


def _current_fit_curve(current_fit: Any, channel: str) -> Optional[Curve]:
    if not isinstance(current_fit, dict) or current_fit.get("model_key") != "bragg_fringes":
        return None
    if str(current_fit.get("channel") or "up").lower() != channel:
        return None
    x = current_fit.get("fit_x") or []
    y = current_fit.get("fit_y") or []
    if len(x) < 2 or len(x) != len(y):
        return None
    return Curve(
        "Current Bragg fringe fit", x, y, FIT_COLOR,
        line=True, symbols=False, line_style=2,
    )


def _metric_worksheet(
    metric_key: str,
    source: str,
    node_payloads: Sequence[Dict[str, Any]],
    include_fits: bool,
    current_fit: Optional[Dict[str, Any]],
) -> Optional[Worksheet]:
    metric = METRICS[metric_key]
    prefix = metric["raw" if source == "nofit" else "fit"]
    x_label = "Parameter P0"
    if metric_key == "phase":
        curves: List[Curve] = []
        for index, node in enumerate(node_payloads):
            x, y = _series(node["stats"], "phase_up")
            if x:
                curves.append(Curve(f"{node['label']} phase", x, y, COLORS[index % len(COLORS)]))
        return Worksheet(metric["label"], [Plot(metric["label"], x_label, metric["label"], curves)]) if curves else None

    plots: List[Plot] = []
    for channel, channel_label in (("up", "UP"), ("dw", "DOWN")):
        curves = []
        for index, node in enumerate(node_payloads):
            x, y = _series(node["stats"], f"{prefix}_{channel}")
            if x:
                curves.append(Curve(f"{node['label']} {channel_label} {source.upper()}", x, y, COLORS[index % len(COLORS)]))
            if metric_key == "intf" and include_fits:
                fit_curve = _calibration_curve(node.get("calibration"), channel, node["label"])
                if fit_curve:
                    curves.append(fit_curve)
        if metric_key == "intf" and include_fits:
            fit_curve = _current_fit_curve(current_fit, channel)
            if fit_curve:
                curves.append(fit_curve)
        plots.append(Plot(f"{metric['label']} - {channel_label}", x_label, metric["label"], curves))
    return Worksheet(metric["label"], plots) if any(plot.curves for plot in plots) else None


def _transfer_phase_degrees(rows: Sequence[Dict[str, Any]]) -> List[float]:
    phases = set()
    for row in rows:
        for component in row.get("interferometer_phase_s2_components") or []:
            phase = _finite(component.get("phase_deg")) if isinstance(component, dict) else None
            if phase is not None:
                phases.add(phase)
        for phase, label in ((0.0, "0deg"), (90.0, "90deg")):
            if any(f"_{label}_" in str(key) and row.get(key) is not None for key in row):
                phases.add(phase)
    return sorted(phases)


def _transfer_component(row: Dict[str, Any], phase_deg: float) -> Dict[str, Any]:
    for component in row.get("interferometer_phase_s2_components") or []:
        if not isinstance(component, dict):
            continue
        component_phase = _finite(component.get("phase_deg"))
        if component_phase is not None and math.isclose(component_phase, phase_deg, abs_tol=1e-9):
            return component
    return {}


def _transfer_curve(
    rows: Sequence[Dict[str, Any]],
    field: str,
    label: str,
    color: Tuple[int, int, int],
    *,
    symbols: bool = True,
    line_style: int = 1,
) -> Optional[Curve]:
    x_values: List[float] = []
    y_values: List[float] = []
    for row in rows:
        x = _finite(row.get("frequency_hz"))
        y = _finite(row.get(field))
        if x is not None and y is not None:
            x_values.append(x)
            y_values.append(y)
    return Curve(
        label, x_values, y_values, color,
        symbols=symbols, line_style=line_style,
    ) if x_values else None


def _transfer_phase_curve(
    rows: Sequence[Dict[str, Any]],
    phase_deg: float,
    statistic: str,
) -> Optional[Curve]:
    phase_label = f"{int(round(phase_deg))}deg"
    nested_key = {"mean": "mean_rad", "std": "std_rad", "s2": "s2"}[statistic]
    flat_key = {
        "mean": f"interferometer_phase_{phase_label}_mean_rad",
        "std": f"interferometer_phase_{phase_label}_std_rad",
        "s2": f"interferometer_phase_{phase_label}_s2",
    }[statistic]
    x_values: List[float] = []
    y_values: List[float] = []
    for row in rows:
        x = _finite(row.get("frequency_hz"))
        component = _transfer_component(row, phase_deg)
        y = _finite(component.get(nested_key))
        if y is None:
            y = _finite(row.get(flat_key))
        if x is not None and y is not None:
            x_values.append(x)
            y_values.append(y)
    color = TRANSFER_PHASE_COLORS.get(phase_deg, COLORS[len(TRANSFER_PHASE_COLORS) % len(COLORS)])
    return Curve(f"{phase_deg:g} deg", x_values, y_values, color) if x_values else None


def _transfer_statistic_curves(
    rows: Sequence[Dict[str, Any]],
    field_base: str,
    statistic: str,
    phases: Sequence[float],
) -> List[Curve]:
    curves: List[Curve] = []
    for phase_index, phase_deg in enumerate(phases):
        phase_label = f"{int(round(phase_deg))}deg"
        curve = _transfer_curve(
            rows,
            f"{field_base}_{phase_label}_{statistic}",
            f"{phase_deg:g} deg",
            TRANSFER_PHASE_COLORS.get(phase_deg, COLORS[phase_index % len(COLORS)]),
        )
        if curve:
            curves.append(curve)
    if not curves:
        curve = _transfer_curve(rows, f"{field_base}_{statistic}", "All shots", COLORS[0])
        if curve:
            curves.append(curve)
    return curves


def _transfer_function_worksheets(
    metrics: Sequence[str], source: str, rows: Sequence[Dict[str, Any]]
) -> List[Worksheet]:
    ordered_rows = sorted(
        (row for row in rows if isinstance(row, dict)),
        key=lambda row: _finite(row.get("frequency_hz")) or 0.0,
    )
    phases = _transfer_phase_degrees(ordered_rows)
    worksheets: List[Worksheet] = []
    x_label = "TTI carrier frequency (Hz)"

    if "atoms" in metrics:
        suffix = "nofit" if source == "nofit" else "fit"
        channels = (("up", "UP"), ("dw", "DOWN"), ("total", "TOTAL (UP+DOWN)"))
        for statistic, statistic_label in (("mean", "Mean"), ("std", "Standard Deviation")):
            plots = [
                Plot(
                    f"Atom Number {channel_label} - {statistic_label}",
                    x_label,
                    f"Atom number {statistic_label.lower()}",
                    _transfer_statistic_curves(
                        ordered_rows, f"atom_number_{channel}_{suffix}", statistic, phases
                    ),
                )
                for channel, channel_label in channels
            ]
            worksheets.append(Worksheet(f"Atom Number - {statistic_label}", plots))

    if "intf" in metrics:
        suffix = "nofit" if source == "nofit" else "fit"
        channels = (("p1", "P1"), ("p2", "P2"))
        for statistic, statistic_label in (("mean", "Mean"), ("std", "Standard Deviation")):
            plots = [
                Plot(
                    f"Interferometer {channel_label} - {statistic_label}",
                    x_label,
                    f"Interferometer probability {statistic_label.lower()}",
                    _transfer_statistic_curves(
                        ordered_rows, f"intf_{channel}_{suffix}", statistic, phases
                    ),
                )
                for channel, channel_label in channels
            ]
            worksheets.append(Worksheet(f"Interferometer P - {statistic_label}", plots))

    if "phase" in metrics:
        for statistic, statistic_label in (("mean", "Mean"), ("std", "Standard Deviation")):
            curves = [
                curve for phase_deg in phases
                if (curve := _transfer_phase_curve(ordered_rows, phase_deg, statistic))
            ]
            if not curves:
                fallback = _transfer_curve(
                    ordered_rows,
                    f"interferometer_phase_{statistic}",
                    "All shots",
                    COLORS[0],
                )
                curves = [fallback] if fallback else []
            worksheets.append(Worksheet(
                f"Interferometer Phase - {statistic_label}",
                [Plot(
                    f"Interferometer Phase - {statistic_label}", x_label,
                    f"Interferometer phase {statistic_label.lower()} (rad)", curves,
                )],
            ))
        s2_curves = [
            curve for phase_deg in phases
            if (curve := _transfer_phase_curve(ordered_rows, phase_deg, "s2"))
        ]
        quadrature = _transfer_curve(
            ordered_rows, "interferometer_phase_s2", "Quadrature sum", COLORS[2]
        )
        if quadrature:
            s2_curves.append(quadrature)
        noise_floor = _transfer_curve(
            ordered_rows,
            "interferometer_phase_noise_s2",
            "Phase-noise floor",
            FIT_COLOR,
            symbols=False,
            line_style=2,
        )
        if noise_floor:
            s2_curves.append(noise_floor)
        worksheets.append(Worksheet(
            "Transfer Function S2",
            [Plot("Transfer Function S2", x_label, "S2 (dimensionless)", s2_curves)],
        ))

    return [worksheet for worksheet in worksheets if any(plot.curves for plot in worksheet.plots)]


def _pair_records(manifest: Dict[str, Any], slave_id: str) -> List[Dict[str, Any]]:
    return [
        row for row in (manifest.get("pairs") or [])
        if isinstance(row, dict) and str(row.get("slave_node_id") or "") == slave_id
    ]


def _differential_points(
    rows: Sequence[Dict[str, Any]], aggregate: bool, source: str
) -> Tuple[List[float], List[float]]:
    points: List[Tuple[float, float, Tuple[Any, ...]]] = []
    for row in rows:
        master = row.get("master") or {}
        slave = row.get("slave") or {}
        field = "intf_p1_nofit" if source == "nofit" else "intf_p1"
        x = _finite(master.get(field))
        y = _finite(slave.get(field))
        if x is None or y is None or master.get("error") or slave.get("error"):
            continue
        raw_params = (
            row.get("sync_parameters") or master.get("sync_master_parameters")
            or slave.get("sync_master_parameters") or [row.get("sync_p0")]
        )
        key = tuple(raw_params) if isinstance(raw_params, list) else (raw_params,)
        points.append((x, y, key))
    if not aggregate:
        return [item[0] for item in points], [item[1] for item in points]
    groups: Dict[Tuple[Any, ...], List[Tuple[float, float]]] = defaultdict(list)
    for x, y, key in points:
        groups[key].append((x, y))
    return (
        [sum(item[0] for item in group) / len(group) for group in groups.values()],
        [sum(item[1] for item in group) / len(group) for group in groups.values()],
    )


def _matching_ellipse_fit(
    fits: Sequence[Dict[str, Any]], slave_id: str, aggregation: str, source_mode: str
) -> Optional[Dict[str, Any]]:
    matches = []
    for fit in fits:
        source = fit.get("source") or {}
        if source.get("aggregation") != aggregation or str(source.get("slave_id") or "") != slave_id:
            continue
        if str(source.get("data_mode") or "fit") != source_mode:
            continue
        expected_field = "intf_p1_nofit" if source_mode == "nofit" else "intf_p1"
        if str(source.get("x_field") or "intf_p1") != expected_field:
            continue
        if str(source.get("y_field") or "intf_p1") != expected_field:
            continue
        matches.append(fit)
    return matches[-1] if matches else None


def _differential_worksheets(
    manifest: Optional[Dict[str, Any]], fits: Sequence[Dict[str, Any]], include_fits: bool, source: str
) -> List[Worksheet]:
    if not manifest:
        return []
    slaves = [item for item in _archive_node_ids(manifest) if item != "master"]
    if not slaves:
        return []
    slave_id = slaves[0]
    slave_label = _node_label(manifest, slave_id)
    rows = _pair_records(manifest, slave_id)
    worksheets = []
    for aggregation, aggregate, title_suffix in (("shots", False, "Every Shot"), ("average", True, "Average")):
        x, y = _differential_points(rows, aggregate, source)
        curves = [Curve("Measured pairs", x, y, COLORS[0], line=False, symbols=True)] if x else []
        if include_fits:
            fit = _matching_ellipse_fit(fits, slave_id, aggregation, source)
            if fit:
                curves.append(Curve(
                    "Ellipse fit", fit.get("fit_x") or [], fit.get("fit_y") or [],
                    FIT_COLOR, True, False, line_style=2,
                ))
        if curves:
            name = f"Differential - {title_suffix}"
            worksheets.append(Worksheet(name, [Plot(
                f"Master vs {slave_label} - Interferometer P UP ({source.upper()}) - {title_suffix}",
                "Master Interferometer P UP (%)", f"{slave_label} Interferometer P UP (%)", curves,
            )]))
    return worksheets


def build_archive_project(
    loader: Any,
    year: str,
    month: str,
    day: str,
    run_id: str,
    metrics: Sequence[str],
    source: str = "fit",
    include_fits: bool = True,
    include_differential: bool = True,
    current_phase_calibration: Optional[Dict[str, Any]] = None,
    current_fit: Optional[Dict[str, Any]] = None,
    transfer_function_summary: Optional[Sequence[Dict[str, Any]]] = None,
) -> bytes:
    loaded_root = loader.load_run(
        year, month, day, run_id, current_phase_calibration=current_phase_calibration
    )
    config_data = loaded_root.get("config") or {}
    is_transfer_function = str(config_data.get("mode") or "").strip().lower() == "transfer_function"
    if is_transfer_function:
        selected = [item for item in metrics if item in {"atoms", "intf", "phase"}]
        summary = (
            list(transfer_function_summary)
            if transfer_function_summary is not None
            else loaded_root.get("transfer_function_summary") or []
        )
        worksheets = _transfer_function_worksheets(selected, source, summary)
        project_name = f"MIGA Transfer Function {day} {run_id}"
        normalization = summary[0] if summary else {}
        modulation_mhz = _finite(normalization.get("frequency_modulation_mhz"))
        distance_m = _finite(normalization.get("atom_mirror_distance_m"))
        phase_amplitude = _finite(normalization.get("bragg_phase_modulation_rad"))
        normalization_comment = "; ".join(filter(None, (
            f"780nm FM={modulation_mhz:g} MHz" if modulation_mhz is not None else "",
            f"L={distance_m:g} m" if distance_m is not None else "",
            f"phi={phase_amplitude:g} rad" if phase_amplitude is not None else "",
        )))
        comment = (
            f"Transfer Function archive {year}-{month}-{day}/{run_id}; source={source}; "
            f"x=TTI carrier frequency{'; ' + normalization_comment if normalization_comment else ''}; "
            "generated for LabPlot 2.12.1"
        )
        return build_project(project_name, worksheets, comment=comment)

    manifest = loaded_root.get("sync_manifest")
    node_payloads = []
    for node_id in _archive_node_ids(manifest):
        loaded = loader.load_run(
            year, month, day, run_id,
            node_id=node_id or None,
            current_phase_calibration=current_phase_calibration,
        )
        node_payloads.append({
            "id": node_id or "archive",
            "label": _node_label(manifest, node_id) if manifest else "Archive",
            "stats": loaded.get("stats") or [],
            "calibration": deepcopy(loaded.get("interferometer_phase_calibration")),
        })

    selected = [item for item in metrics if item in METRICS]
    worksheets = []
    for metric in selected:
        worksheet = _metric_worksheet(metric, source, node_payloads, include_fits, current_fit)
        if worksheet:
            worksheets.append(worksheet)
    if include_differential and manifest:
        worksheets.extend(_differential_worksheets(
            manifest, loaded_root.get("sync_differential_fits") or [], include_fits, source
        ))
    project_name = f"MIGA Archive {day} {run_id}"
    comment = f"Archive {year}-{month}-{day}/{run_id}; source={source}; generated for LabPlot 2.12.1"
    return build_project(project_name, worksheets, comment=comment)
