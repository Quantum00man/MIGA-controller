import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.analysis import fitting, physics
from app.analysis.lock_in import build_lock_in_analysis
import config


MAX_DISPLAY_POINTS = 3000
MAX_WAVEFORM_PREVIEW_STEPS = 48


class DataLoader:
    def __init__(self):
        self.base_dir = config.DATA_BASE_DIR

    def _clean_run_label(self, value: Any) -> str:
        return str(value or "").strip()

    def _get_run_summary(self, config_data: Dict[str, Any]) -> str:
        scan_dimensions = self._resolve_scan_dimensions(config_data)
        mode = str(config_data.get("mode") or "standard").strip().lower() or "standard"
        averages = max(1, self._parse_int(config_data.get("averages"), 1))
        randomize = bool(config_data.get("randomize", False))
        sequence_name = str(config_data.get("sequence_name") or "").strip()

        summary_parts = [f"{scan_dimensions}D"]
        if scan_dimensions == 1:
            summary_parts.append(mode)
        elif scan_dimensions == 2:
            summary_parts.append("P0-P1")
        else:
            summary_parts.append("P0-P1-P2")
        if randomize:
            summary_parts.append("random")
        if averages > 1:
            summary_parts.append(f"avg×{averages}")
        if sequence_name:
            summary_parts.append(sequence_name)
        return " · ".join(summary_parts)

    def _build_run_entry(self, run_dir: Path) -> Dict[str, Any]:
        config_data = self._load_config_data(run_dir)
        has_marker_optimization = (run_dir / "marker_optimization_report.json").is_file()
        has_sync = (run_dir / "sync_manifest.json").is_file()
        if has_marker_optimization:
            config_data = {**config_data, "mode": "marker_optimization"}
        run_id = run_dir.name
        run_label = self._clean_run_label(config_data.get("run_label"))
        display_name = f"{run_id} | {run_label}" if run_label else run_id
        scan_dimensions = self._resolve_scan_dimensions(config_data)
        sequence_path = run_dir / "sequence.mot"
        return {
            "id": run_id,
            "run_id": run_id,
            "run_label": run_label,
            "display_name": display_name,
            "summary": self._get_run_summary(config_data),
            "sequence_name": str(config_data.get("sequence_name") or "").strip(),
            "has_sequence_file": sequence_path.exists(),
            "scan_dimensions": scan_dimensions,
            "randomize": bool(config_data.get("randomize", False)),
            "mode": str(config_data.get("mode") or "standard").strip().lower() or "standard",
            "has_marker_optimization": has_marker_optimization,
            "has_sync": has_sync,
        }

    def get_run_entry(self, year: str, month: str, day: str, run_id: str) -> Dict[str, Any]:
        run_dir = self._get_run_dir(year, month, day, run_id)
        return self._build_run_entry(run_dir)

    def _get_sequence_download_name(self, run_id: str, config_data: Optional[Dict[str, Any]] = None) -> str:
        raw_name = Path(str((config_data or {}).get("sequence_name") or "").strip()).name
        if not raw_name:
            raw_name = "sequence.mot"
        if not raw_name.lower().endswith('.mot'):
            raw_name = f"{raw_name}.mot"
        prefix = f"{run_id}_"
        return raw_name if raw_name.startswith(prefix) else f"{prefix}{raw_name}"

    def get_archived_sequence_file(self, year: str, month: str, day: str, run_id: str) -> Tuple[Path, str]:
        run_dir = self._get_run_dir(year, month, day, run_id)
        sequence_path = run_dir / "sequence.mot"
        if not sequence_path.exists():
            raise FileNotFoundError(f"Sequence file not found for run {run_id}")
        config_data = self._load_config_data(run_dir)
        return sequence_path, self._get_sequence_download_name(run_id, config_data)

    def get_archive_tree(self) -> Dict[str, Any]:
        tree = {}
        if not self.base_dir.exists():
            return tree
        for year_dir in sorted(self.base_dir.iterdir()):
            if year_dir.is_dir():
                year = year_dir.name
                tree[year] = {}
                for month_dir in sorted(year_dir.iterdir()):
                    if month_dir.is_dir():
                        month = month_dir.name
                        tree[year][month] = {}
                        for day_dir in sorted(month_dir.iterdir()):
                            if day_dir.is_dir():
                                day = day_dir.name
                                runs = [self._build_run_entry(r) for r in sorted(day_dir.iterdir()) if r.is_dir() and r.name.startswith("run")]
                                tree[year][month][day] = runs
        return tree

    def _sanitize_structure(self, data):
        if isinstance(data, dict):
            return {k: self._sanitize_structure(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._sanitize_structure(v) for v in data]
        if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
            return 0.0
        return data

    def _get_run_dir(self, year: str, month: str, day: str, run_id: str) -> Path:
        run_dir = self.base_dir / year / month / day / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_dir}")
        return run_dir

    def _load_config_data(self, run_dir: Path) -> Dict[str, Any]:
        config_path = run_dir / "config.json"
        if not config_path.exists():
            return {}
        try:
            with open(config_path, "r") as handle:
                return self._sanitize_structure(json.load(handle))
        except Exception:
            return {}

    def _parse_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed

    def _parse_int(self, value: Any, default: int = 0) -> int:
        parsed = self._parse_float(value)
        return int(parsed) if parsed is not None else default

    def _parse_parameters(self, value: Any) -> List[float]:
        if value is None:
            return []
        raw_parts = str(value).split(";")
        parsed: List[float] = []
        for part in raw_parts:
            number = self._parse_float(part)
            if number is not None:
                parsed.append(number)
        return parsed

    def _row_to_point(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "step": self._parse_int(row.get("Step"), 0),
            "timestamp": self._parse_float(row.get("Timestamp")) or 0.0,
            "parameter": self._parse_float(row.get("Parameter_P0")) or 0.0,
            "all_parameters": self._parse_parameters(row.get("All_Parameters")),
            "atom_number_up": self._parse_float(row.get("Atom_UP")),
            "atom_number_dw": self._parse_float(row.get("Atom_DW")),
            "temperature_up": self._parse_float(row.get("Temp_UP")),
            "temperature_dw": self._parse_float(row.get("Temp_DW")),
            "sigma_up": self._parse_float(row.get("Sigma_UP")),
            "sigma_dw": self._parse_float(row.get("Sigma_DW")),
            "amplitude_up": self._parse_float(row.get("Amp_UP")),
            "amplitude_dw": self._parse_float(row.get("Amp_DW")),
            "arrival_time_up": self._parse_float(row.get("Center_UP")),
            "arrival_time_dw": self._parse_float(row.get("Center_DW")),
            "transition_probability_up": self._parse_float(row.get("Prob_UP_F2")),
            "transition_probability_dw": self._parse_float(row.get("Prob_DW_F1")),
            "intf_n1": self._parse_float(row.get("Intf_N1")),
            "intf_n2": self._parse_float(row.get("Intf_N2")),
            "intf_p1": self._parse_float(row.get("Intf_P1")),
            "intf_p2": self._parse_float(row.get("Intf_P2")),
            "atom_number_up_nofit": self._parse_float(row.get("NF_Atom_UP")),
            "atom_number_dw_nofit": self._parse_float(row.get("NF_Atom_DW")),
            "temperature_up_nofit": self._parse_float(row.get("NF_Temp_UP")),
            "temperature_dw_nofit": self._parse_float(row.get("NF_Temp_DW")),
            "sigma_up_nofit": self._parse_float(row.get("NF_Sigma_UP")),
            "sigma_dw_nofit": self._parse_float(row.get("NF_Sigma_DW")),
            "amplitude_up_nofit": self._parse_float(row.get("NF_Amp_UP")),
            "amplitude_dw_nofit": self._parse_float(row.get("NF_Amp_DW")),
            "arrival_time_up_nofit": self._parse_float(row.get("NF_Center_UP")),
            "arrival_time_dw_nofit": self._parse_float(row.get("NF_Center_DW")),
            "transition_probability_up_nofit": self._parse_float(row.get("NF_Prob_UP")),
            "transition_probability_dw_nofit": self._parse_float(row.get("NF_Prob_DW")),
            "intf_n1_nofit": self._parse_float(row.get("NF_Intf_N1")),
            "intf_n2_nofit": self._parse_float(row.get("NF_Intf_N2")),
            "intf_p1_nofit": self._parse_float(row.get("NF_Intf_P1")),
            "intf_p2_nofit": self._parse_float(row.get("NF_Intf_P2")),
            "tail_mean_up_raw": self._parse_float(row.get("TailMean_UP")),
            "tail_mean_dw_raw": self._parse_float(row.get("TailMean_DW")),
            "ac_stark_ratio": self._parse_float(row.get("AC_Stark_Ratio")),
            "ac_stark_side": str(row.get("AC_Stark_Side") or "").strip().lower(),
            "ac_stark_dds_element": self._parse_int(row.get("AC_Stark_DDS_Element"), -1),
            "ac_stark_power_r1": self._parse_float(row.get("AC_Stark_Power_R1")),
            "ac_stark_power_r2": self._parse_float(row.get("AC_Stark_Power_R2")),
            "ac_stark_amplitude_r1": self._parse_int(row.get("AC_Stark_Amplitude_R1"), -1),
            "ac_stark_amplitude_r2": self._parse_int(row.get("AC_Stark_Amplitude_R2"), -1),
            "ac_stark_actual_power_r1": self._parse_float(row.get("AC_Stark_Actual_Power_R1")),
            "ac_stark_actual_power_r2": self._parse_float(row.get("AC_Stark_Actual_Power_R2")),
            "lock_in_block_index": self._parse_int(row.get("LockIn_Block"), -1),
            "lock_in_position": self._parse_int(row.get("LockIn_Position"), -1),
            "lock_in_state": str(row.get("LockIn_State") or "").strip().lower(),
            "lock_in_reference": self._parse_int(row.get("LockIn_Reference"), 0),
            "workflow_step": self._parse_int(row.get("Workflow_Step"), -1),
            "workflow_marker": str(row.get("Workflow_Marker") or "").strip(),
            "workflow_point": self._parse_int(row.get("Workflow_Point"), -1),
            "workflow_repeat": self._parse_int(row.get("Workflow_Repeat"), -1),
            "workflow_shot": self._parse_int(row.get("Workflow_Shot"), -1),
            "workflow_randomized": self._parse_int(row.get("Workflow_Randomized"), -1),
        }

    def _read_results_csv(self, run_dir: Path, max_points: Optional[int] = MAX_DISPLAY_POINTS) -> List[Dict[str, Any]]:
        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            return []

        with open(csv_path, "r") as handle:
            rows = list(csv.DictReader(handle))

        sampled_rows = self._sample_sequence(rows, max_points)

        return [self._row_to_point(row) for row in sampled_rows]

    def _sample_sequence(self, items: List[Any], max_points: Optional[int]) -> List[Any]:
        if max_points is None or max_points <= 0 or len(items) <= max_points:
            return list(items)
        step = max(1, math.ceil(len(items) / max_points))
        return list(items[::step])[:max_points]

    def _calc_stats(self, values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        count = len(values)
        mean = float(sum(values) / count)
        if count < 2:
            return mean, 0.0
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        return mean, float(math.sqrt(variance))

    def _resolve_scan_dimensions(
        self,
        config_data: Optional[Dict[str, Any]] = None,
        fallback: int = 1,
    ) -> int:
        if isinstance(config_data, dict):
            try:
                scan_dimensions = int(config_data.get("scan_dimensions", fallback))
            except (TypeError, ValueError):
                scan_dimensions = fallback
            if config_data.get("dim3_enabled", False):
                scan_dimensions = max(scan_dimensions, 3)
            elif config_data.get("dim2_enabled", False):
                scan_dimensions = max(scan_dimensions, 2)
            return max(1, min(3, scan_dimensions))
        return max(1, min(3, int(fallback or 1)))

    def _get_group_params(self, point: Dict[str, Any], scan_dimensions: int) -> List[float]:
        raw_params = point.get("all_parameters") if isinstance(point, dict) else None
        if isinstance(raw_params, list) and raw_params:
            params = [self._safe_scalar(value) for value in raw_params[:scan_dimensions]]
        else:
            params = [self._safe_scalar(point.get("parameter"))]
        if not params:
            params = [0.0]
        return params

    def _make_group_key(self, params: List[float]) -> str:
        return "|".join(f"{float(value):.6f}" for value in params)

    def _build_stats_array(self, points: List[Dict[str, Any]], scan_dimensions: int = 1) -> List[Dict[str, Any]]:
        metric_fields = {
            "atoms": ("atom_number_up", "atom_number_dw"),
            "temp": ("temperature_up", "temperature_dw"),
            "sigma": ("sigma_up", "sigma_dw"),
            "amp": ("amplitude_up", "amplitude_dw"),
            "arrival": ("arrival_time_up", "arrival_time_dw"),
            "prob": ("transition_probability_up", "transition_probability_dw"),
            "intf_p": ("intf_p1", "intf_p2"),
            "tail": ("tail_mean_up_raw", "tail_mean_dw_raw"),
            "nf_atoms": ("atom_number_up_nofit", "atom_number_dw_nofit"),
            "nf_temp": ("temperature_up_nofit", "temperature_dw_nofit"),
            "nf_sigma": ("sigma_up_nofit", "sigma_dw_nofit"),
            "nf_amp": ("amplitude_up_nofit", "amplitude_dw_nofit"),
            "nf_arrival": ("arrival_time_up_nofit", "arrival_time_dw_nofit"),
            "nf_prob": ("transition_probability_up_nofit", "transition_probability_dw_nofit"),
            "nf_intf_p": ("intf_p1_nofit", "intf_p2_nofit"),
            "nf_tail": ("tail_mean_up_raw", "tail_mean_dw_raw"),
        }

        grouped: Dict[str, Dict[str, Any]] = {}
        for point in points:
            params = self._get_group_params(point, scan_dimensions)
            key = self._make_group_key(params)
            if key not in grouped:
                grouped[key] = {
                    "key": key,
                    "params": params,
                    "x": float(params[0]),
                    "values": {
                        metric: {"up": [], "dw": []}
                        for metric in metric_fields
                    },
                }
            group = grouped[key]["values"]
            for metric, (field_up, field_dw) in metric_fields.items():
                value_up = point.get(field_up)
                value_dw = point.get(field_dw)
                if value_up is not None:
                    group[metric]["up"].append(float(value_up))
                if value_dw is not None:
                    group[metric]["dw"].append(float(value_dw))

        stats_rows: List[Dict[str, Any]] = []
        for key in sorted(grouped.keys(), key=lambda item: grouped[item]["params"]):
            group = grouped[key]
            row: Dict[str, Any] = {"key": key, "params": list(group["params"]), "x": group["x"]}
            for metric in metric_fields:
                up_values = group["values"][metric]["up"]
                dw_values = group["values"][metric]["dw"]
                mean_up, std_up = self._calc_stats(up_values) if up_values else (None, None)
                mean_dw, std_dw = self._calc_stats(dw_values) if dw_values else (None, None)
                row[f"{metric}_up"] = mean_up
                row[f"{metric}_up_std"] = std_up
                row[f"{metric}_dw"] = mean_dw
                row[f"{metric}_dw_std"] = std_dw
            stats_rows.append(row)
        return stats_rows

    def _build_preview_map(self, points: List[Dict[str, Any]], scan_dimensions: int = 1) -> Dict[str, Dict[str, Any]]:
        preview_map: Dict[str, Dict[str, Any]] = {}
        for point in points:
            params = self._get_group_params(point, scan_dimensions)
            key = self._make_group_key(params)
            step = int(point.get("step", 0))
            if key not in preview_map:
                preview_map[key] = {"key": key, "params": params, "x": float(params[0]), "count": 0, "stepIndices": []}
            preview_map[key]["count"] += 1
            preview_map[key]["stepIndices"].append(step)

        for key, item in preview_map.items():
            item["stepIndices"] = self._sample_sequence(item["stepIndices"], MAX_WAVEFORM_PREVIEW_STEPS)
        return preview_map

    def _build_ac_stark_summary(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        metrics = (
            "atom_number_up", "atom_number_dw",
            "transition_probability_up", "transition_probability_dw",
            "atom_number_up_nofit", "atom_number_dw_nofit",
            "transition_probability_up_nofit", "transition_probability_dw_nofit",
        )
        grouped: Dict[float, Dict[str, List[Dict[str, Any]]]] = {}
        for point in points:
            ratio = self._parse_float(point.get("ac_stark_ratio"))
            side = str(point.get("ac_stark_side") or "").strip().lower()
            if ratio is None or side not in {"left", "right"}:
                continue
            grouped.setdefault(round(ratio, 12), {"left": [], "right": []})[side].append(point)

        summary: List[Dict[str, Any]] = []
        for ratio in sorted(grouped):
            sides = grouped[ratio]
            representative = (sides["left"] or sides["right"])[0]
            row: Dict[str, Any] = {
                "ratio": float(ratio),
                "dds_element": representative.get("ac_stark_dds_element"),
                "requested_power_r1": representative.get("ac_stark_power_r1"),
                "requested_power_r2": representative.get("ac_stark_power_r2"),
                "amplitude_r1": representative.get("ac_stark_amplitude_r1"),
                "amplitude_r2": representative.get("ac_stark_amplitude_r2"),
                "actual_power_r1": representative.get("ac_stark_actual_power_r1"),
                "actual_power_r2": representative.get("ac_stark_actual_power_r2"),
                "left_count": len(sides["left"]),
                "right_count": len(sides["right"]),
                "left_key": self._make_group_key(self._get_group_params(sides["left"][0], 1)) if sides["left"] else None,
                "right_key": self._make_group_key(self._get_group_params(sides["right"][0], 1)) if sides["right"] else None,
            }
            for metric in metrics:
                side_stats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
                for side in ("left", "right"):
                    values = [
                        float(point[metric]) for point in sides[side]
                        if point.get(metric) is not None and math.isfinite(float(point[metric]))
                    ]
                    mean, std = self._calc_stats(values) if values else (None, None)
                    sem = std / math.sqrt(len(values)) if values and std is not None else None
                    row[f"{metric}_{side}_mean"] = mean
                    row[f"{metric}_{side}_std"] = std
                    row[f"{metric}_{side}_sem"] = sem
                    side_stats[side] = (mean, sem)
                left_mean, left_sem = side_stats["left"]
                right_mean, right_sem = side_stats["right"]
                has_pair = left_mean is not None and right_mean is not None
                row[f"{metric}_difference"] = right_mean - left_mean if has_pair else None
                row[f"{metric}_difference_sem"] = (
                    math.sqrt((left_sem or 0.0) ** 2 + (right_sem or 0.0) ** 2)
                    if has_pair else None
                )
            summary.append(row)
        return summary

    def _load_marker_optimization_report(self, run_dir: Path) -> Dict[str, Any]:
        report_path = run_dir / "marker_optimization_report.json"
        if not report_path.is_file():
            return {}
        try:
            return self._sanitize_structure(json.loads(report_path.read_text(encoding="utf-8")))
        except Exception:
            return {}

    def _marker_optimization_artifact_paths(self, run_dir: Path) -> Dict[str, Path]:
        artifacts: Dict[str, Path] = {}
        exact = {
            "report_pdf": run_dir / "marker_optimization_report.pdf",
            "report_json": run_dir / "marker_optimization_report.json",
            "workflow_preset": run_dir / "workflow_preset.json",
        }
        for kind, path in exact.items():
            if path.is_file():
                artifacts[kind] = path
        patterns = {
            "original_sequence": "*_original.mot",
            "optimized_sequence": "*_optimized.mot",
            "report_bundle": "*_marker_optimization_report.zip",
        }
        for kind, pattern in patterns.items():
            matches = sorted(run_dir.glob(pattern))
            if matches:
                artifacts[kind] = matches[0]
        return artifacts

    def get_marker_optimization_artifact(
        self, year: str, month: str, day: str, run_id: str, kind: str
    ) -> Tuple[Path, str]:
        run_dir = self._get_run_dir(year, month, day, run_id)
        artifact = self._marker_optimization_artifact_paths(run_dir).get(str(kind or "").strip())
        if artifact is None or not artifact.is_file():
            raise FileNotFoundError(f"Marker optimization artifact not found: {kind}")
        return artifact, artifact.name

    def _build_marker_optimization_archive(
        self, run_dir: Path, full_points: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        report = self._load_marker_optimization_report(run_dir)
        report_steps = report.get("steps") if isinstance(report.get("steps"), list) else []
        points_by_step: Dict[int, List[Dict[str, Any]]] = {}
        has_persisted_steps = any(int(point.get("workflow_step", -1)) > 0 for point in full_points)
        if has_persisted_steps:
            for point in full_points:
                workflow_step = int(point.get("workflow_step", -1))
                if workflow_step > 0:
                    points_by_step.setdefault(workflow_step, []).append(point)
        else:
            # Legacy optimization CSV files stored steps contiguously. The
            # scientific report retains the exact number of acquired repeats.
            cursor = 0
            for fallback_index, step in enumerate(report_steps, start=1):
                step_index = int(step.get("index") or fallback_index)
                shot_count = sum(
                    max(1, len(point.get("repeats") or []))
                    for point in (step.get("points") or [])
                )
                if shot_count <= 0:
                    continue
                points_by_step[step_index] = full_points[cursor:cursor + shot_count]
                cursor += shot_count

        known_indices = {
            int(step.get("index") or index)
            for index, step in enumerate(report_steps, start=1)
        } | set(points_by_step)
        report_lookup = {
            int(step.get("index") or index): step
            for index, step in enumerate(report_steps, start=1)
        }
        step_payloads: List[Dict[str, Any]] = []
        for step_index in sorted(known_indices):
            step = report_lookup.get(step_index, {})
            step_points = points_by_step.get(step_index, [])
            step_payloads.append({
                "index": step_index,
                "marker_id": step.get("marker_id") or (
                    step_points[0].get("workflow_marker") if step_points else ""
                ),
                "marker_name": step.get("marker_name") or step.get("marker_id") or f"Step {step_index}",
                "marker_kind": step.get("marker_kind") or "",
                "status": step.get("status") or "unknown",
                "objective": step.get("objective") or "",
                "metric_key": step.get("metric_key") or "",
                "metric_label": step.get("metric_label") or "",
                "metric_source": step.get("metric_source") or "fit",
                "average_count": step.get("average_count") or 1,
                "randomize": bool(step.get("randomize", False)),
                "minimum_r_squared": step.get("minimum_r_squared"),
                "start": step.get("start"),
                "stop": step.get("stop"),
                "step": step.get("step"),
                "scan_method": step.get("scan_method") or "step_size",
                "applied_value": step.get("applied_value"),
                "digital_conditions": step.get("digital_conditions") or [],
                "error": step.get("error"),
                "analysis": step.get("analysis") if isinstance(step.get("analysis"), dict) else {},
                "data": self._sample_sequence(step_points, MAX_DISPLAY_POINTS),
                "stats": self._build_stats_array(step_points, scan_dimensions=1),
                "preview_map": self._build_preview_map(step_points, scan_dimensions=1),
                "total_points": len(step_points),
            })

        artifact_labels = {
            "original_sequence": "Original MOT",
            "optimized_sequence": "Optimized MOT",
            "report_pdf": "PDF Report",
            "report_json": "JSON Report",
            "workflow_preset": "Preset JSON",
            "report_bundle": "Complete Package",
        }
        artifacts = [
            {"kind": kind, "filename": path.name, "label": artifact_labels.get(kind, kind)}
            for kind, path in self._marker_optimization_artifact_paths(run_dir).items()
        ]
        configuration = report.get("configuration") if isinstance(report.get("configuration"), dict) else {}
        return {
            "workflow_name": report.get("workflow_name") or "",
            "run_label": report.get("run_label") or configuration.get("run_label") or "",
            "phase": report.get("phase") or "unknown",
            "stop_reason": report.get("stop_reason"),
            "error": report.get("error"),
            "started_at_ms": report.get("started_at_ms"),
            "ended_at_ms": report.get("ended_at_ms"),
            "completed_steps": report.get("completed_steps") or 0,
            "total_steps": report.get("total_steps") or len(step_payloads),
            "applied_values": report.get("applied_values") or {},
            "steps": step_payloads,
            "artifacts": artifacts,
        }

    def build_collection_preview(
        self,
        year: str,
        month: str,
        day: str,
        run_id: str,
        metric: str = "prob",
        workflow_step: Optional[int] = None,
        max_points: int = 48,
    ) -> Dict[str, Any]:
        """Build a compact 1D statistics snapshot for the Collections browser."""
        loaded = self.load_run(year, month, day, run_id)
        metric = str(metric or "prob").strip().lower()
        allowed = {"atoms", "amp", "tail", "sigma", "temp", "arrival", "prob", "intf"}
        if metric not in allowed:
            metric = "prob"
        stats = loaded.get("stats") or []
        selected_step = None
        optimization = loaded.get("marker_optimization") or {}
        steps = optimization.get("steps") or []
        if steps:
            selected_step = next(
                (step for step in steps if int(step.get("index") or 0) == int(workflow_step or 0)),
                steps[0],
            )
            stats = selected_step.get("stats") or []
        field = "intf_p" if metric == "intf" else metric
        rows = [row for row in stats if isinstance(row, dict)]
        if len(rows) > max_points:
            indices = np.linspace(0, len(rows) - 1, max_points, dtype=int)
            rows = [rows[int(index)] for index in indices]
        labels = {
            "atoms": "Atom Number", "amp": "Max Voltage", "tail": "Tail Mean",
            "sigma": "Width", "temp": "Temperature", "arrival": "Arrival Time",
            "prob": "Transition Probability", "intf": "Interferometer",
        }
        return self._sanitize_structure({
            "metric": metric,
            "metric_label": labels[metric],
            "x": [row.get("x") for row in rows],
            "up": [row.get(f"{field}_up") for row in rows],
            "down": [row.get(f"{field}_dw") for row in rows],
            "scan_dimensions": int(loaded.get("scan_dimensions") or 1),
            "step_index": selected_step.get("index") if selected_step else None,
            "step_name": selected_step.get("marker_name") if selected_step else "",
        })

    def load_run(self, year: str, month: str, day: str, run_id: str) -> Dict[str, Any]:
        run_dir = self._get_run_dir(year, month, day, run_id)
        config_data = self._load_config_data(run_dir)
        scan_dimensions = self._resolve_scan_dimensions(config_data)
        full_points = self._read_results_csv(run_dir, max_points=None)
        marker_optimization = self._build_marker_optimization_archive(run_dir, full_points)
        is_marker_optimization = bool(marker_optimization.get("steps")) or (
            run_dir / "marker_optimization_report.json"
        ).is_file()
        if is_marker_optimization:
            config_data = {**config_data, "mode": "marker_optimization", "scan_dimensions": 1}
            scan_dimensions = 1
        initial_step = (marker_optimization.get("steps") or [{}])[0]
        sampled_points = (
            initial_step.get("data", [])
            if is_marker_optimization
            else self._sample_sequence(full_points, MAX_DISPLAY_POINTS)
        )
        is_lock_in = str(config_data.get("mode") or "").strip().lower() == "lock_in"
        expected_lock_in_blocks = self._parse_int(config_data.get("averages"), 0) if is_lock_in else 0
        lock_in_analysis = build_lock_in_analysis(full_points, expected_blocks=expected_lock_in_blocks) if is_lock_in else {}
        sync_manifest = None
        sync_manifest_path = run_dir / "sync_manifest.json"
        if sync_manifest_path.is_file():
            try:
                sync_manifest = json.loads(sync_manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                sync_manifest = {"runtime": {"status": "invalid", "message": "Sync manifest could not be read"}, "pairs": []}
        return {
            "config": config_data,
            "run_entry": self._build_run_entry(run_dir),
            "scan_dimensions": scan_dimensions,
            "data": sampled_points,
            "stats": (
                initial_step.get("stats", [])
                if is_marker_optimization
                else self._build_stats_array(full_points, scan_dimensions=scan_dimensions)
            ),
            "ac_stark_summary": self._build_ac_stark_summary(full_points),
            "lock_in_analysis": lock_in_analysis,
            "preview_map": (
                initial_step.get("preview_map", {})
                if is_marker_optimization
                else self._build_preview_map(full_points, scan_dimensions=scan_dimensions)
            ),
            "total_points": len(full_points),
            "marker_optimization": marker_optimization if is_marker_optimization else None,
            "sync_manifest": sync_manifest,
        }


    def _get_allan_metric_fields(self) -> Dict[str, Dict[str, Dict[str, Tuple[str, ...]]]]:
        return {
            "atoms": {
                "fit": {"up": ("atom_number_up",), "dw": ("atom_number_dw",), "total": ("atom_number_up", "atom_number_dw")},
                "raw": {"up": ("atom_number_up_nofit",), "dw": ("atom_number_dw_nofit",), "total": ("atom_number_up_nofit", "atom_number_dw_nofit")},
            },
            "amp": {
                "fit": {"up": ("amplitude_up",), "dw": ("amplitude_dw",)},
                "raw": {"up": ("amplitude_up_nofit",), "dw": ("amplitude_dw_nofit",)},
            },
            "sigma": {
                "fit": {"up": ("sigma_up",), "dw": ("sigma_dw",)},
                "raw": {"up": ("sigma_up_nofit",), "dw": ("sigma_dw_nofit",)},
            },
            "temp": {
                "fit": {"up": ("temperature_up",), "dw": ("temperature_dw",)},
                "raw": {"up": ("temperature_up_nofit",), "dw": ("temperature_dw_nofit",)},
            },
            "arrival": {
                "fit": {"up": ("arrival_time_up",), "dw": ("arrival_time_dw",)},
                "raw": {"up": ("arrival_time_up_nofit",), "dw": ("arrival_time_dw_nofit",)},
            },
            "prob": {
                "fit": {"up": ("transition_probability_up",), "dw": ("transition_probability_dw",)},
                "raw": {"up": ("transition_probability_up_nofit",), "dw": ("transition_probability_dw_nofit",)},
            },
            "intf": {
                "fit": {"up": ("intf_p1",), "dw": ("intf_p2",)},
                "raw": {"up": ("intf_p1_nofit",), "dw": ("intf_p2_nofit",)},
            },
            "tail": {
                "fit": {"up": ("tail_mean_up_raw",), "dw": ("tail_mean_dw_raw",)},
                "raw": {"up": ("tail_mean_up_raw",), "dw": ("tail_mean_dw_raw",)},
            },
        }

    def _extract_allan_p0(self, point: Dict[str, Any]) -> float:
        parsed_parameter = self._parse_float(point.get("parameter"))
        if parsed_parameter is not None:
            return float(parsed_parameter)

        all_parameters = point.get("all_parameters")
        if isinstance(all_parameters, (list, tuple)) and all_parameters:
            parsed_first = self._parse_float(all_parameters[0])
            if parsed_first is not None:
                return float(parsed_first)

        return float(self._safe_scalar(point.get("parameter"), 0.0))

    def _filter_allan_points_by_p0_range(
        self,
        points: List[Dict[str, Any]],
        p0_min: Optional[float] = None,
        p0_max: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[float], Optional[float], Optional[float], Optional[float]]:
        if not points:
            return [], None, None, None, None

        p0_values = [self._extract_allan_p0(point) for point in points]
        available_p0_min = float(min(p0_values))
        available_p0_max = float(max(p0_values))
        selected_p0_min = available_p0_min if p0_min is None else float(p0_min)
        selected_p0_max = available_p0_max if p0_max is None else float(p0_max)

        if selected_p0_min > selected_p0_max:
            selected_p0_min, selected_p0_max = selected_p0_max, selected_p0_min

        selected_p0_min = max(available_p0_min, min(selected_p0_min, available_p0_max))
        selected_p0_max = min(available_p0_max, max(selected_p0_max, available_p0_min))
        epsilon = 1e-12
        filtered_points = [
            point
            for point, p0_value in zip(points, p0_values)
            if (selected_p0_min - epsilon) <= p0_value <= (selected_p0_max + epsilon)
        ]

        return filtered_points, available_p0_min, available_p0_max, float(selected_p0_min), float(selected_p0_max)

    def _build_allan_curve_meta(self, points: List[Dict[str, Any]], requested_order: int) -> Dict[str, Any]:
        sequence = [
            {
                "step": int(point.get("step", index)),
                "p0": self._extract_allan_p0(point),
            }
            for index, point in enumerate(points)
        ]
        sequence_length = len(sequence)
        max_order = sequence_length // 2
        requested_max_order = max(1, int(requested_order or max_order or 1))
        used_max_order = min(requested_max_order, max_order) if max_order > 0 else 0
        orders = list(range(1, used_max_order + 1))
        return {
            "requested_order": requested_max_order,
            "max_order": max_order,
            "used_max_order": used_max_order,
            "sequence_length": sequence_length,
            "order_count": len(orders),
            "orders": orders,
        }

    def _extract_allan_value(self, point: Dict[str, Any], field_names: Tuple[str, ...]) -> Optional[float]:
        values: List[float] = []
        for field_name in field_names:
            parsed = self._parse_float(point.get(field_name))
            if parsed is None:
                return None
            values.append(float(parsed))
        return float(sum(values))

    def _build_allan_channel(self, points: List[Dict[str, Any]], field_names: Tuple[str, ...], orders: List[int]) -> Dict[str, Any]:
        values = np.asarray([self._extract_allan_value(point, field_names) for point in points], dtype=float)
        finite_values = values[np.isfinite(values)]
        mean_value = float(np.mean(finite_values)) if finite_values.size else None
        rms_value = float(np.sqrt(np.mean(np.square(finite_values)))) if finite_values.size else None
        standard_deviation = float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else (0.0 if finite_values.size else None)
        sequence_statistics = {
            "mean": mean_value,
            "rms": rms_value,
            "standard_deviation": standard_deviation,
            "sample_count": int(finite_values.size),
        }
        if values.size == 0 or not orders:
            return {
                "y": [],
                "valid_window_counts": [],
                "mean_value": mean_value,
                "sequence_statistics": sequence_statistics,
            }

        valid = np.isfinite(values)
        safe_values = np.where(valid, values, 0.0)
        value_prefix = np.concatenate(([0.0], np.cumsum(safe_values)))
        valid_prefix = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))

        sigma_values: List[Optional[float]] = []
        valid_window_counts: List[int] = []
        for order in orders:
            window_count = values.size - (2 * order) + 1
            if window_count <= 0:
                sigma_values.append(None)
                valid_window_counts.append(0)
                continue

            valid_a = valid_prefix[order:order + window_count] - valid_prefix[:window_count]
            valid_b = valid_prefix[2 * order:2 * order + window_count] - valid_prefix[order:order + window_count]
            valid_windows = (valid_a == order) & (valid_b == order)
            current_valid_count = int(np.count_nonzero(valid_windows))
            valid_window_counts.append(current_valid_count)
            if current_valid_count == 0:
                sigma_values.append(None)
                continue

            mean_a = (value_prefix[order:order + window_count] - value_prefix[:window_count]) / order
            mean_b = (value_prefix[2 * order:2 * order + window_count] - value_prefix[order:order + window_count]) / order
            diffs = (mean_b - mean_a) / math.sqrt(2.0)
            sigma_values.append(float(np.sqrt(np.mean(np.square(diffs[valid_windows])))))

        return {
            "y": sigma_values,
            "valid_window_counts": valid_window_counts,
            "mean_value": mean_value,
            "sequence_statistics": sequence_statistics,
        }

    def _build_allan_payload(self, points: List[Dict[str, Any]], requested_order: int) -> Dict[str, Any]:
        payload = self._build_allan_curve_meta(points, requested_order)
        orders = payload["orders"]
        metrics: Dict[str, Any] = {}
        for metric_name, source_map in self._get_allan_metric_fields().items():
            metrics[metric_name] = {}
            for source_name, channel_map in source_map.items():
                metrics[metric_name][source_name] = {}
                for channel_name, field_names in channel_map.items():
                    metrics[metric_name][source_name][channel_name] = self._build_allan_channel(points, field_names, orders)
        payload["metrics"] = metrics
        payload["overlapping"] = True
        return payload

    def _load_allan_points(
        self,
        run_dir: Path,
        config_data: Dict[str, Any],
        display_mode: str,
        new_settings: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        mode = str(display_mode or "saved").strip().lower()
        if mode != "recalculated":
            return self._read_results_csv(run_dir, max_points=None)

        original_settings = (
            config_data.get("_system_settings_snapshot")
            or config_data.get("_analysis_snapshot")
            or {}
        )
        settings = self._normalize_archive_settings(new_settings or {}, fallback=original_settings)
        points = self._read_results_csv(run_dir, max_points=None)
        recalculated_points: List[Dict[str, Any]] = []
        for point in points:
            waveform = self._load_waveform_arrays(run_dir, int(point["step"]))
            result = self._recalculate_point(point, waveform, settings, original_settings)
            if result is not None:
                recalculated_points.append(result)
        return recalculated_points

    def calculate_allan_run(
        self,
        year: str,
        month: str,
        day: str,
        run_id: str,
        requested_order: int,
        display_mode: str,
        new_settings: Optional[Dict[str, Any]] = None,
        p0_min: Optional[float] = None,
        p0_max: Optional[float] = None,
    ) -> Dict[str, Any]:
        run_dir = self._get_run_dir(year, month, day, run_id)
        config_data = self._load_config_data(run_dir)
        scan_dimensions = self._resolve_scan_dimensions(config_data)
        randomized = bool(config_data.get("randomize", False))
        if scan_dimensions != 1:
            raise ValueError("Allan deviation is only available for 1D scans")
        if randomized:
            raise ValueError("Allan deviation is only available for non-random scans")

        normalized_mode = "recalculated" if str(display_mode or "saved").strip().lower() == "recalculated" else "saved"
        all_points = self._load_allan_points(run_dir, config_data, normalized_mode, new_settings=new_settings)
        filtered_points, available_p0_min, available_p0_max, selected_p0_min, selected_p0_max = self._filter_allan_points_by_p0_range(
            all_points,
            p0_min=p0_min,
            p0_max=p0_max,
        )
        payload = self._build_allan_payload(filtered_points, requested_order)
        payload.update(
            {
                "display_mode": normalized_mode,
                "scan_dimensions": scan_dimensions,
                "randomize": randomized,
                "total_points": len(all_points),
                "filtered_points": len(filtered_points),
                "available_p0_min": available_p0_min,
                "available_p0_max": available_p0_max,
                "selected_p0_min": selected_p0_min,
                "selected_p0_max": selected_p0_max,
            }
        )
        return payload

    def _load_waveform_arrays(self, run_dir: Path, step_index: int) -> Dict[str, Any]:
        npz_path = run_dir / "waveforms" / f"step_{step_index:04d}.npz"
        if not npz_path.exists():
            raise FileNotFoundError("Waveform not found")

        try:
            with np.load(npz_path) as data:
                def get_arr(key: str) -> np.ndarray:
                    if key not in data:
                        return np.array([], dtype=float)
                    arr = np.asarray(data[key], dtype=float)
                    if arr.size == 0:
                        return np.array([], dtype=float)
                    arr = arr.copy()
                    arr[~np.isfinite(arr)] = 0.0
                    return arr

                def get_window(key: str) -> Optional[Tuple[float, float]]:
                    if key not in data:
                        return None
                    raw = np.asarray(data[key], dtype=float).tolist()
                    if len(raw) != 2:
                        return None
                    if raw[0] == -1 or raw[1] == -1:
                        return None
                    return float(raw[0]), float(raw[1])

                return {
                    "time_axis": get_arr("time_axis"),
                    "raw_up": get_arr("raw_up"),
                    "raw_dw": get_arr("raw_dw"),
                    "fit_up": get_arr("fit_up"),
                    "fit_dw": get_arr("fit_dw"),
                    "window_up": get_window("window_up"),
                    "window_dw": get_window("window_dw"),
                }
        except Exception as exc:
            raise RuntimeError(f"Failed to load waveform: {str(exc)}")

    def load_waveform(self, year: str, month: str, day: str, run_id: str, step_index: int) -> Dict[str, Any]:
        run_dir = self._get_run_dir(year, month, day, run_id)
        waveform = self._load_waveform_arrays(run_dir, step_index)
        return {
            "time_axis": waveform["time_axis"].tolist(),
            "raw_up": waveform["raw_up"].tolist(),
            "raw_dw": waveform["raw_dw"].tolist(),
            "fit_up": waveform["fit_up"].tolist(),
            "fit_dw": waveform["fit_dw"].tolist(),
            "window_up": list(waveform["window_up"]) if waveform["window_up"] is not None else [-1, -1],
            "window_dw": list(waveform["window_dw"]) if waveform["window_dw"] is not None else [-1, -1],
        }

    def _normalize_archive_settings(self, new_settings: Dict[str, Any], fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        settings = dict(config.DEFAULT_ANALYSIS_SETTINGS)
        settings.update(
            {
                "voltage_limit": 0.015,
                "intf_alpha": 0.35,
                "intf_beta": 0.07636,
                "intf_gamma": 0.25,
                "fit_model_key": "gaussian",
                "fit_models": fitting.get_default_fit_models(),
            }
        )

        if isinstance(fallback, dict):
            settings.update(self._sanitize_structure(fallback))
        if isinstance(new_settings, dict):
            settings.update(self._sanitize_structure(new_settings))

        fit_models = fitting.normalize_fit_model_list(settings.get("fit_models") or fitting.get_default_fit_models())
        selected_model = fitting.get_fit_model_by_key(fit_models, settings.get("fit_model_key", "gaussian"))
        settings["fit_models"] = fit_models
        settings["fit_model_key"] = selected_model["key"]

        method = str(settings.get("atom_area_method") or "legacy").strip().lower()
        settings["atom_area_method"] = "edge_line" if method == "edge_line" else "legacy"
        try:
            baseline_points = int(settings.get("atom_area_baseline_points", 2))
        except (TypeError, ValueError):
            baseline_points = 2
        settings["atom_area_baseline_points"] = max(1, baseline_points)

        return settings

    def _safe_scalar(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed

    def _safe_gain(self, settings: Dict[str, Any], key: str) -> float:
        gain = self._safe_scalar(settings.get(key), 1.0)
        return 1.0 if abs(gain) < 1e-12 else gain

    def _safe_window(self, window: Optional[Tuple[float, float]]) -> List[float]:
        if window is None:
            return [-1, -1]
        return [float(window[0]), float(window[1])]

    def _select_fit_data(
        self,
        time_axis: np.ndarray,
        signal: np.ndarray,
        window: Optional[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float]]]:
        if window is None:
            return time_axis, signal, None

        start, end = float(window[0]), float(window[1])
        mask = (time_axis >= start) & (time_axis <= end)
        if np.any(mask):
            return time_axis[mask], signal[mask], (start, end)
        return time_axis, signal, None

    def _prepare_waveform_context(
        self,
        waveform: Dict[str, Any],
        settings: Dict[str, Any],
        original_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        time_axis = np.asarray(waveform.get("time_axis", np.array([], dtype=float)), dtype=float)
        raw_up_stored = np.asarray(waveform.get("raw_up", np.array([], dtype=float)), dtype=float)
        raw_dw_stored = np.asarray(waveform.get("raw_dw", np.array([], dtype=float)), dtype=float)

        min_len = min(len(time_axis), len(raw_up_stored), len(raw_dw_stored))
        if min_len == 0:
            raise ValueError("Archived waveform is empty")

        time_axis = time_axis[:min_len]
        raw_up_stored = raw_up_stored[:min_len]
        raw_dw_stored = raw_dw_stored[:min_len]

        old_gain_up = self._safe_gain(original_settings, "gain_up")
        old_gain_dw = self._safe_gain(original_settings, "gain_dw")
        new_gain_up = self._safe_gain(settings, "gain_up")
        new_gain_dw = self._safe_gain(settings, "gain_dw")

        clean_up = raw_up_stored * old_gain_up
        clean_dw = raw_dw_stored * old_gain_dw
        volt_up = clean_up / new_gain_up
        volt_dw = clean_dw / new_gain_dw

        fit_t_up, fit_v_up, win_up = self._select_fit_data(time_axis, volt_up, waveform.get("window_up"))
        fit_t_dw, fit_v_dw, win_dw = self._select_fit_data(time_axis, volt_dw, waveform.get("window_dw"))

        fit_models = settings.get("fit_models") or fitting.get_default_fit_models()
        fit_model = fitting.get_fit_model_by_key(fit_models, settings.get("fit_model_key", "gaussian"))
        fit_result_up = fitting.perform_configured_fit(fit_model, fit_t_up, fit_v_up, eval_x=time_axis)
        fit_result_dw = fitting.perform_configured_fit(fit_model, fit_t_dw, fit_v_dw, eval_x=time_axis)

        return {
            "time_axis": time_axis,
            "clean_up": clean_up,
            "clean_dw": clean_dw,
            "volt_up": volt_up,
            "volt_dw": volt_dw,
            "fit_t_up": fit_t_up,
            "fit_t_dw": fit_t_dw,
            "fit_v_up": fit_v_up,
            "fit_v_dw": fit_v_dw,
            "win_up": win_up,
            "win_dw": win_dw,
            "fit_result_up": fit_result_up,
            "fit_result_dw": fit_result_dw,
        }

    def _recalculate_point(
        self,
        point: Dict[str, Any],
        waveform: Dict[str, Any],
        settings: Dict[str, Any],
        original_settings: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        context = self._prepare_waveform_context(waveform, settings, original_settings)
        clean_up = context["clean_up"]
        clean_dw = context["clean_dw"]
        fit_t_up = context["fit_t_up"]
        fit_t_dw = context["fit_t_dw"]
        fit_v_up = context["fit_v_up"]
        fit_v_dw = context["fit_v_dw"]
        fit_result_up = context["fit_result_up"]
        fit_result_dw = context["fit_result_dw"]

        v_limit = self._safe_scalar(settings.get("voltage_limit"), 0.015)
        min_limit = self._safe_scalar(settings.get("max_low"), 0.0001)

        max_clean_up = float(np.max(np.abs(clean_up))) if clean_up.size else 0.0
        max_clean_dw = float(np.max(np.abs(clean_dw))) if clean_dw.size else 0.0
        if max_clean_up > v_limit or max_clean_dw > v_limit:
            return None

        amp_up_nf = float(np.max(fit_v_up)) if fit_v_up.size else 0.0
        amp_dw_nf = float(np.max(fit_v_dw)) if fit_v_dw.size else 0.0
        if abs(amp_up_nf) < min_limit and abs(amp_dw_nf) < min_limit:
            return None

        amp_up = fit_result_up.amplitude if fit_result_up is not None else 0.0
        sig_up = fit_result_up.width if fit_result_up is not None else 0.0
        cen_up = fit_result_up.center if fit_result_up is not None else 0.0
        amp_dw = fit_result_dw.amplitude if fit_result_dw is not None else 0.0
        sig_dw = fit_result_dw.width if fit_result_dw is not None else 0.0
        cen_dw = fit_result_dw.center if fit_result_dw is not None else 0.0

        sig_up_nf = fitting.calc_sigma(fit_v_up, fit_t_up) or 0.0
        sig_dw_nf = fitting.calc_sigma(fit_v_dw, fit_t_dw) or 0.0
        cen_up_nf = float(fit_t_up[int(np.argmax(fit_v_up))]) if fit_v_up.size else 0.0
        cen_dw_nf = float(fit_t_dw[int(np.argmax(fit_v_dw))]) if fit_v_dw.size else 0.0

        if settings["atom_area_method"] == "edge_line":
            baseline_points = settings["atom_area_baseline_points"]
            area_up_nf = fitting.calculate_area_with_edge_baseline(fit_t_up, fit_v_up, baseline_points)
            area_dw_nf = fitting.calculate_area_with_edge_baseline(fit_t_dw, fit_v_dw, baseline_points)
            area_up = (
                fitting.calculate_area_with_edge_baseline(fit_t_up, fit_result_up.fit_window_curve, baseline_points)
                if fit_result_up is not None
                else 0.0
            )
            area_dw = (
                fitting.calculate_area_with_edge_baseline(fit_t_dw, fit_result_dw.fit_window_curve, baseline_points)
                if fit_result_dw is not None
                else 0.0
            )
        else:
            area_up_nf = float(abs(np.trapz(fit_v_up, fit_t_up))) if fit_t_up.size > 1 else 0.0
            area_dw_nf = float(abs(np.trapz(fit_v_dw, fit_t_dw))) if fit_t_dw.size > 1 else 0.0
            area_up = fit_result_up.area if fit_result_up is not None else 0.0
            area_dw = fit_result_dw.area if fit_result_dw is not None else 0.0

        n_f2, n_f1 = physics.calculate_atom_numbers(
            area_up,
            area_dw,
            max_vol_up=amp_up,
            max_vol_dw=amp_dw,
            alpha=settings["alpha"],
            beta=settings["beta"],
            R=settings["R"],
            K=settings["K"],
            max_low=settings["max_low"],
        )
        n_f2_nf, n_f1_nf = physics.calculate_atom_numbers(
            area_up_nf,
            area_dw_nf,
            max_vol_up=amp_up_nf,
            max_vol_dw=amp_dw_nf,
            alpha=settings["alpha"],
            beta=settings["beta"],
            R=settings["R"],
            K=settings["K"],
            max_low=settings["max_low"],
        )

        temp_up = physics.calculate_temperature(sig_up, cen_up, settings["launch_velocity"], is_sigma_in_ms=False)
        temp_dw = physics.calculate_temperature(sig_dw, cen_dw, settings["launch_velocity"], is_sigma_in_ms=False)
        temp_up_nf = physics.calculate_temperature(sig_up_nf, cen_up_nf, settings["launch_velocity"], is_sigma_in_ms=False)
        temp_dw_nf = physics.calculate_temperature(sig_dw_nf, cen_dw_nf, settings["launch_velocity"], is_sigma_in_ms=False)

        prob_up, prob_dw = physics.calculate_probabilities(n_f2, n_f1)
        prob_up_nf, prob_dw_nf = physics.calculate_probabilities(n_f2_nf, n_f1_nf)

        intf_n1, intf_n2, intf_p1, intf_p2 = physics.calculate_interferometer_output(
            n_f1,
            n_f2,
            settings.get("intf_alpha", 0.35),
            settings.get("intf_beta", 0.07636),
            settings.get("intf_gamma", 0.25),
        )
        intf_n1_nf, intf_n2_nf, intf_p1_nf, intf_p2_nf = physics.calculate_interferometer_output(
            n_f1_nf,
            n_f2_nf,
            settings.get("intf_alpha", 0.35),
            settings.get("intf_beta", 0.07636),
            settings.get("intf_gamma", 0.25),
        )

        return {
            **point,
            "atom_number_up": n_f2,
            "atom_number_dw": n_f1,
            "amplitude_up": amp_up,
            "amplitude_dw": amp_dw,
            "sigma_up": sig_up * 1000.0,
            "sigma_dw": sig_dw * 1000.0,
            "temperature_up": temp_up,
            "temperature_dw": temp_dw,
            "arrival_time_up": cen_up,
            "arrival_time_dw": cen_dw,
            "transition_probability_up": prob_up,
            "transition_probability_dw": prob_dw,
            "intf_n1": intf_n1,
            "intf_n2": intf_n2,
            "intf_p1": intf_p1,
            "intf_p2": intf_p2,
            "atom_number_up_nofit": n_f2_nf,
            "atom_number_dw_nofit": n_f1_nf,
            "amplitude_up_nofit": amp_up_nf,
            "amplitude_dw_nofit": amp_dw_nf,
            "sigma_up_nofit": sig_up_nf * 1000.0,
            "sigma_dw_nofit": sig_dw_nf * 1000.0,
            "temperature_up_nofit": temp_up_nf,
            "temperature_dw_nofit": temp_dw_nf,
            "arrival_time_up_nofit": cen_up_nf,
            "arrival_time_dw_nofit": cen_dw_nf,
            "transition_probability_up_nofit": prob_up_nf,
            "transition_probability_dw_nofit": prob_dw_nf,
            "intf_n1_nofit": intf_n1_nf,
            "intf_n2_nofit": intf_n2_nf,
            "intf_p1_nofit": intf_p1_nf,
            "intf_p2_nofit": intf_p2_nf,
        }

    def recalculate_run(
        self,
        year: str,
        month: str,
        day: str,
        run_id: str,
        new_settings: Dict[str, Any],
        max_points: Optional[int] = MAX_DISPLAY_POINTS,
    ) -> Dict[str, Any]:
        run_dir = self._get_run_dir(year, month, day, run_id)
        config_data = self._load_config_data(run_dir)
        scan_dimensions = self._resolve_scan_dimensions(config_data)
        original_settings = (
            config_data.get("_system_settings_snapshot")
            or config_data.get("_analysis_snapshot")
            or {}
        )
        settings = self._normalize_archive_settings(new_settings, fallback=original_settings)
        points = self._read_results_csv(run_dir, max_points=None)

        recalculated_points: List[Dict[str, Any]] = []
        for point in points:
            waveform = self._load_waveform_arrays(run_dir, int(point["step"]))
            result = self._recalculate_point(point, waveform, settings, original_settings)
            if result is not None:
                recalculated_points.append(result)

        sampled_points = self._sample_sequence(recalculated_points, max_points)
        is_lock_in = str(config_data.get("mode") or "").strip().lower() == "lock_in"
        expected_lock_in_blocks = self._parse_int(config_data.get("averages"), 0) if is_lock_in else 0
        lock_in_analysis = build_lock_in_analysis(recalculated_points, expected_blocks=expected_lock_in_blocks) if is_lock_in else {}

        return {
            "config": config_data,
            "scan_dimensions": scan_dimensions,
            "settings": settings,
            "data": sampled_points,
            "stats": self._build_stats_array(recalculated_points, scan_dimensions=scan_dimensions),
            "ac_stark_summary": self._build_ac_stark_summary(recalculated_points),
            "lock_in_analysis": lock_in_analysis,
            "preview_map": self._build_preview_map(recalculated_points, scan_dimensions=scan_dimensions),
            "total_points": len(recalculated_points),
        }

    def recalculate_waveforms(
        self,
        year: str,
        month: str,
        day: str,
        run_id: str,
        step_indices: List[int],
        new_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        run_dir = self._get_run_dir(year, month, day, run_id)
        config_data = self._load_config_data(run_dir)
        original_settings = (
            config_data.get("_system_settings_snapshot")
            or config_data.get("_analysis_snapshot")
            or {}
        )
        settings = self._normalize_archive_settings(new_settings, fallback=original_settings)

        waveforms: List[Dict[str, Any]] = []
        for step_index in step_indices:
            waveform = self._load_waveform_arrays(run_dir, int(step_index))
            context = self._prepare_waveform_context(waveform, settings, original_settings)
            fit_curve_up = context["fit_result_up"].fit_curve if context["fit_result_up"] is not None else np.zeros_like(context["time_axis"])
            fit_curve_dw = context["fit_result_dw"].fit_curve if context["fit_result_dw"] is not None else np.zeros_like(context["time_axis"])
            waveforms.append(
                {
                    "step": int(step_index),
                    "time_axis": context["time_axis"].tolist(),
                    "raw_up": context["volt_up"].tolist(),
                    "raw_dw": context["volt_dw"].tolist(),
                    "fit_up": fit_curve_up.tolist(),
                    "fit_dw": fit_curve_dw.tolist(),
                    "window_up": self._safe_window(context["win_up"]),
                    "window_dw": self._safe_window(context["win_dw"]),
                }
            )

        return {"settings": settings, "data": waveforms}
