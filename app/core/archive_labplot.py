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
    (220, 53, 69), (13, 110, 253), (25, 135, 84), (255, 128, 0),
    (111, 66, 193), (32, 150, 160), (108, 117, 125), (214, 51, 132),
]


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
    return Curve(f"{label} fringe fit", x, y, (255, 128, 0), line=True, symbols=False)


def _current_fit_curve(current_fit: Any, channel: str) -> Optional[Curve]:
    if not isinstance(current_fit, dict) or current_fit.get("model_key") != "bragg_fringes":
        return None
    if str(current_fit.get("channel") or "up").lower() != channel:
        return None
    x = current_fit.get("fit_x") or []
    y = current_fit.get("fit_y") or []
    if len(x) < 2 or len(x) != len(y):
        return None
    return Curve("Current Bragg fringe fit", x, y, (255, 128, 0), line=True, symbols=False)


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
        curves = [Curve("Measured pairs", x, y, (13, 110, 253), line=False, symbols=True)] if x else []
        if include_fits:
            fit = _matching_ellipse_fit(fits, slave_id, aggregation, source)
            if fit:
                curves.append(Curve("Ellipse fit", fit.get("fit_x") or [], fit.get("fit_y") or [], (220, 53, 69), True, False))
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
) -> bytes:
    loaded_root = loader.load_run(
        year, month, day, run_id, current_phase_calibration=current_phase_calibration
    )
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
