from __future__ import annotations

import copy
import csv
import json
import math
import random
import re
import threading
import time
import traceback
import zipfile
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

import config
from app.core.data_manager import DataManager
from app.core.experiment_manager import ExperimentManager
from app.core.optimization_manager import OBJECTIVE_METRICS
from app.core.sequence_markers import (
    decode_mot_bytes,
    encode_mot_text,
    inspect_sequence_markers,
    normalize_marker_id,
    render_auto_marker_sequence,
    resolve_marker_definitions,
    render_digital_marker_states,
    sequence_marker_profile_key,
)


MARKER_OBJECTIVES = {
    "spectral_center": "Spectral center (Gaussian fit)",
    "rabi_pi": "First pi pulse (damped Rabi fit)",
    "maximize": "Measured maximum",
    "minimize": "Measured minimum",
}


def _finite_float(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if total <= np.finfo(float).eps:
        return 1.0 if residual <= np.finfo(float).eps else 0.0
    return 1.0 - residual / total


def _nearest_scanned_point(values: Sequence[float], target: float) -> float:
    if not values:
        raise ValueError("No scanned values are available")
    return min((float(value) for value in values), key=lambda value: (abs(value - target), value))


def _gaussian(x: np.ndarray, offset: float, amplitude: float, center: float, sigma: float) -> np.ndarray:
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _damped_rabi(x: np.ndarray, offset: float, amplitude: float, t_pi: float, tau: float) -> np.ndarray:
    return offset + amplitude * np.exp(-x / tau) * np.sin(np.pi * x / (2.0 * t_pi)) ** 2


def analyze_marker_scan(
    points: Sequence[Dict[str, Any]],
    objective: str,
    *,
    minimum_r_squared: float = 0.75,
    marker_kind: str = "",
    marker_decimals: int = 9,
) -> Dict[str, Any]:
    """Analyze aggregated scan points and choose a safe value to apply.

    Fit objectives may apply an in-range, hardware-representable value that was
    not sampled. Measured maximum/minimum objectives still apply a measured point.
    """
    ordered = sorted(
        (
            {
                "value": _finite_float(point.get("value"), "Scan value"),
                "metric_mean": _finite_float(point.get("metric_mean"), "Metric mean"),
                "metric_std": _finite_float(point.get("metric_std", 0.0), "Metric standard deviation"),
                "metric_sem": _finite_float(point.get("metric_sem", 0.0), "Metric SEM"),
                "repeats": list(point.get("repeats") or []),
            }
            for point in points
        ),
        key=lambda item: item["value"],
    )
    if len(ordered) < 2:
        raise ValueError("At least two distinct scan points are required")
    x = np.asarray([item["value"] for item in ordered], dtype=float)
    y = np.asarray([item["metric_mean"] for item in ordered], dtype=float)
    if len(np.unique(x)) != len(x):
        raise ValueError("Scan values must be unique")
    objective_key = str(objective or "maximize").strip().lower()
    if objective_key not in MARKER_OBJECTIVES:
        raise ValueError(f"Unsupported marker objective: {objective_key}")

    fit_parameters: Dict[str, float] = {}
    fit_curve: np.ndarray
    dense_x = np.asarray([], dtype=float)
    dense_curve = np.asarray([], dtype=float)
    covariance: Optional[np.ndarray] = None
    fitted_objective = objective_key in {"spectral_center", "rabi_pi"}
    if objective_key == "maximize":
        best_y = float(np.max(y))
        candidates = x[np.isclose(y, best_y, rtol=1e-12, atol=1e-12)]
        continuous_optimum = float(np.min(candidates))
        fit_curve = np.full_like(y, np.nan)
        r_squared = None
        model = "none"
    elif objective_key == "minimize":
        best_y = float(np.min(y))
        candidates = x[np.isclose(y, best_y, rtol=1e-12, atol=1e-12)]
        continuous_optimum = float(np.min(candidates))
        fit_curve = np.full_like(y, np.nan)
        r_squared = None
        model = "none"
    elif objective_key == "spectral_center":
        if len(x) < 5:
            raise ValueError("Spectral-center fitting requires at least five scan points")
        span = float(x[-1] - x[0])
        spacing = float(np.min(np.diff(x)))
        amplitude_guess = max(float(np.max(y) - np.min(y)), np.finfo(float).eps)
        center_guess = float(x[int(np.argmax(y))])
        sigma_guess = max(span / 6.0, spacing)
        lower = [float(np.min(y) - 2 * amplitude_guess), 0.0, float(x[0]), max(spacing / 10.0, 1e-12)]
        upper = [float(np.max(y) + 2 * amplitude_guess), amplitude_guess * 10.0 + 1e-12, float(x[-1]), max(span * 2.0, spacing)]
        popt, covariance = curve_fit(
            _gaussian,
            x,
            y,
            p0=[float(np.min(y)), amplitude_guess, center_guess, sigma_guess],
            bounds=(lower, upper),
            maxfev=50000,
        )
        fit_curve = _gaussian(x, *popt)
        dense_x = np.linspace(float(x[0]), float(x[-1]), max(500, len(x)))
        dense_curve = _gaussian(dense_x, *popt)
        continuous_optimum = float(popt[2])
        fit_parameters = {
            "offset": float(popt[0]),
            "amplitude": float(popt[1]),
            "center": continuous_optimum,
            "sigma": float(popt[3]),
        }
        r_squared = _r_squared(y, fit_curve)
        model = "Gaussian"
    else:
        if len(x) < 6:
            raise ValueError("Damped-Rabi fitting requires at least six scan points")
        if np.any(x <= 0):
            raise ValueError("Damped-Rabi duration values must be greater than zero")
        span = float(x[-1] - x[0])
        spacing = float(np.min(np.diff(x)))
        amplitude_guess = max(float(np.max(y) - np.min(y)), np.finfo(float).eps)
        t_pi_guess = float(x[int(np.argmax(y))])
        tau_guess = max(float(x[-1]) * 2.0, span, spacing)
        lower_t_pi = max(spacing / 4.0, 1e-12)
        upper_t_pi = max(float(x[-1]) * 2.0, lower_t_pi * 2.0)
        lower = [float(np.min(y) - 2 * amplitude_guess), 0.0, lower_t_pi, max(spacing, 1e-12)]
        upper = [float(np.max(y) + 2 * amplitude_guess), amplitude_guess * 10.0 + 1e-12, upper_t_pi, max(float(x[-1]) * 100.0, spacing * 10.0)]
        popt, covariance = curve_fit(
            _damped_rabi,
            x,
            y,
            p0=[float(np.min(y)), amplitude_guess, t_pi_guess, tau_guess],
            bounds=(lower, upper),
            maxfev=100000,
        )
        fit_curve = _damped_rabi(x, *popt)
        dense_x = np.linspace(float(x[0]), float(x[-1]), max(500, len(x)))
        dense_curve = _damped_rabi(dense_x, *popt)
        continuous_optimum = float(popt[2])
        fit_parameters = {
            "offset": float(popt[0]),
            "amplitude": float(popt[1]),
            "t_pi": continuous_optimum,
            "decay_tau": float(popt[3]),
        }
        r_squared = _r_squared(y, fit_curve)
        model = "Exponentially damped sin-squared Rabi"

    quality_threshold = float(minimum_r_squared)
    if r_squared is not None and (not math.isfinite(r_squared) or r_squared < quality_threshold):
        raise ValueError(f"Fit quality R²={r_squared:.4f} is below the required {quality_threshold:.4f}")

    if fitted_objective:
        tolerance = max(abs(float(x[-1] - x[0])) * 1e-12, 1e-12)
        if continuous_optimum < float(x[0]) - tolerance or continuous_optimum > float(x[-1]) + tolerance:
            raise ValueError(
                f"Fitted optimum {continuous_optimum:g} is outside scan range {float(x[0]):g} to {float(x[-1]):g}"
            )
        kind = str(marker_kind or "").strip().lower()
        if kind in {"dds_element", "duration"}:
            selected_value = int(round(continuous_optimum))
        else:
            decimals = max(0, min(9, int(marker_decimals)))
            selected_value = round(continuous_optimum, decimals)
        if selected_value < float(x[0]) - tolerance or selected_value > float(x[-1]) + tolerance:
            raise ValueError(
                f"Representable optimum {selected_value:g} is outside scan range {float(x[0]):g} to {float(x[-1]):g}"
            )
    else:
        selected_value = _nearest_scanned_point(x.tolist(), continuous_optimum)

    sampled_matches = np.where(np.isclose(x, selected_value, rtol=0.0, atol=1e-12))[0]
    selected_was_sampled = len(sampled_matches) > 0
    selected_index = int(sampled_matches[0]) if selected_was_sampled else None
    selected_metric_mean = float(y[selected_index]) if selected_index is not None else None
    if not fitted_objective and selected_index in {0, len(x) - 1}:
        raise ValueError(
            f"Selected optimum {selected_value:g} is on the scan boundary; expand or shift the scan range"
        )

    parameter_uncertainty = None
    if covariance is not None:
        variance = float(covariance[2, 2])
        if math.isfinite(variance) and variance >= 0:
            parameter_uncertainty = math.sqrt(variance)
    residuals = y - fit_curve if np.all(np.isfinite(fit_curve)) else np.full_like(y, np.nan)
    return {
        "objective": objective_key,
        "model": model,
        "continuous_optimum": continuous_optimum,
        "selected_value": selected_value,
        "selected_index": selected_index,
        "selected_was_sampled": selected_was_sampled,
        "selected_metric_mean": selected_metric_mean,
        "r_squared": r_squared,
        "minimum_r_squared": quality_threshold,
        "optimum_standard_error": parameter_uncertainty,
        "fit_parameters": fit_parameters,
        "fit_curve": [None if not math.isfinite(float(value)) else float(value) for value in fit_curve],
        "fit_x_dense": [float(value) for value in dense_x],
        "fit_curve_dense": [float(value) for value in dense_curve],
        "residuals": [None if not math.isfinite(float(value)) else float(value) for value in residuals],
        "points": ordered,
    }


class MarkerOptimizationManager:
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._artifact_paths: Dict[str, Path] = {}
        self._status = self._idle_status()

    @staticmethod
    def _idle_status() -> Dict[str, Any]:
        return {
            "is_running": False,
            "phase": "idle",
            "message": "IDLE",
            "run_id": None,
            "run_label": "",
            "started_at_ms": None,
            "ended_at_ms": None,
            "current_step": 0,
            "total_steps": 0,
            "current_point": 0,
            "total_points": 0,
            "steps": [],
            "applied_values": {},
            "stop_reason": None,
            "error": None,
            "export_urls": {},
        }

    def _snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._status)

    def get_status(self) -> Dict[str, Any]:
        return self._snapshot()

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._status.get("is_running"))

    def _set_status(self, **updates: Any) -> None:
        with self._lock:
            self._status.update(updates)

    def _emit(self, payload: Dict[str, Any]) -> None:
        callback = self.experiment_manager.on_data_ready
        if callback:
            callback(payload)

    @staticmethod
    def _template_path(settings: Dict[str, Any]) -> Path:
        return Path(
            config.SEQUENCE_TEMPLATE_PATH_WIN
            if config.USE_SIMULATION
            else str(settings.get("template_path") or config.SEQUENCE_TEMPLATE_PATH_LINUX)
        )

    @staticmethod
    def _metric_value(result: Any, metric_key: str, source: str) -> float:
        field = f"{metric_key}_nofit" if source == "nofit" else metric_key
        value = getattr(result, field, None)
        if value is None:
            raise ValueError(f"Objective metric unavailable: {field}")
        return _finite_float(value, field)

    @staticmethod
    def _scan_values(step: Dict[str, Any], definition: Dict[str, Any]) -> List[float]:
        start = _finite_float(step.get("start"), "Scan start")
        stop = _finite_float(step.get("stop"), "Scan stop")
        increment = _finite_float(step.get("step"), "Scan step/count")
        scan_method = str(step.get("scan_method") or "step_size").strip().lower()
        if scan_method not in {"step_size", "n_points"}:
            raise ValueError("Scan method must be step_size or n_points")
        if increment <= 0:
            raise ValueError("Scan step must be greater than zero")
        if start > stop:
            raise ValueError("Scan start must not exceed scan stop")
        hard_min = float(definition["hard_min"])
        hard_max = float(definition["hard_max"])
        if start < hard_min or stop > hard_max:
            raise ValueError(
                f"Scan {start:g} to {stop:g} exceeds marker hard limits {hard_min:g} to {hard_max:g}"
            )
        if scan_method == "n_points":
            if not math.isclose(increment, round(increment), abs_tol=1e-9) or int(round(increment)) < 2:
                raise ValueError("Point-count scan requires an integer count of at least 2")
            count = int(round(increment))
            raw_values = np.linspace(start, stop, count).tolist()
            if definition["kind"] in {"dds_element", "duration"} and any(
                not math.isclose(value, round(value), abs_tol=1e-9) for value in raw_values
            ):
                raise ValueError("Integer Marker point-count scan must produce integer grid values")
            values = [float(int(round(value))) if definition["kind"] in {"dds_element", "duration"} else round(float(value), 9) for value in raw_values]
            if len(set(values)) != len(values):
                raise ValueError("Point-count scan produces duplicate Marker values")
            return values
        integer_type = definition["kind"] in {"dds_element", "duration"}
        if integer_type:
            for label, value in (("start", start), ("stop", stop), ("step", increment)):
                if not math.isclose(value, round(value), abs_tol=1e-9):
                    raise ValueError(f"{definition['kind']} scan {label} must be an integer")
        values: List[float] = []
        current = start
        tolerance = increment * 1e-9 + 1e-12
        while current <= stop + tolerance:
            values.append(float(int(round(current))) if integer_type else round(current, 9))
            current += increment
            if len(values) > 5000:
                raise ValueError("Marker scan exceeds 5000 points")
        return list(dict.fromkeys(values))

    def _validate_payload(self, payload: Dict[str, Any]) -> Tuple[Path, str, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        sequence_name = str(payload.get("sequence_name") or "sequence.mot").strip()
        template_path = self._template_path(self.experiment_manager.settings)
        if not template_path.is_file():
            raise ValueError(f"Sequence template not found: {template_path}")
        content, encoding = decode_mot_bytes(template_path.read_bytes())
        definition_resolution = resolve_marker_definitions(
            content,
            self.experiment_manager.settings,
            sequence_name,
        )
        if definition_resolution["errors"]:
            first_error = definition_resolution["errors"][0]
            raise ValueError(
                f"Invalid embedded Marker definition at line {first_error['line_number']}: {first_error['message']}"
            )
        definitions = definition_resolution["definitions"]
        definition_map = {item["id"]: item for item in definitions}
        inspection = inspect_sequence_markers(content, definitions)
        scan_markers = {
            marker["id"]: marker
            for marker in inspection["markers"]
            if marker["role"] == "scan"
        }
        all_state_markers = {
            marker["id"]: marker
            for marker in inspection["markers"]
            if marker["role"] == "state"
        }
        state_markers = {
            marker_id: marker
            for marker_id, marker in all_state_markers.items()
            if marker.get("status") == "defined"
            and marker.get("definition", {}).get("kind") == "digital_state"
        }
        steps = list(payload.get("steps") or [])
        if not steps:
            raise ValueError("Add at least one Marker optimization step")
        normalized_steps: List[Dict[str, Any]] = []
        for index, raw in enumerate(steps, start=1):
            marker_id = normalize_marker_id(raw.get("marker_id"))
            definition = definition_map.get(marker_id)
            marker = scan_markers.get(marker_id)
            if not definition or not marker:
                raise ValueError(f"Step {index}: marker {marker_id} has no embedded or Settings definition")
            if marker.get("status") != "defined":
                raise ValueError(f"Step {index}: marker {marker_id} is {marker.get('status')}: {marker.get('message')}")
            objective = str(raw.get("objective") or "maximize").strip().lower()
            if objective not in MARKER_OBJECTIVES:
                raise ValueError(f"Step {index}: unsupported objective {objective}")
            if objective == "rabi_pi" and definition["kind"] != "duration":
                raise ValueError(f"Step {index}: first pi-pulse fitting requires a duration marker")
            metric_key = str(raw.get("metric_key") or "transition_probability_up").strip()
            if metric_key not in OBJECTIVE_METRICS:
                raise ValueError(f"Step {index}: unsupported metric {metric_key}")
            source = str(raw.get("metric_source") or "fit").strip().lower()
            if source not in {"fit", "nofit"}:
                raise ValueError(f"Step {index}: metric source must be fit or nofit")
            average_count = int(raw.get("average_count", 1) or 1)
            randomize = bool(raw.get("randomize", False))
            if average_count < 1 or average_count > 1000:
                raise ValueError(f"Step {index}: average count must be between 1 and 1000")
            values = self._scan_values(raw, definition)
            required = 6 if objective == "rabi_pi" else (5 if objective == "spectral_center" else 3)
            if len(values) < required:
                raise ValueError(f"Step {index}: {objective} requires at least {required} scan points")

            raw_state_choices = raw.get("digital_states") or {}
            if not isinstance(raw_state_choices, dict):
                raise ValueError(f"Step {index}: digital states must be an object")
            requested_choices: Dict[str, str] = {}
            for raw_state_id, raw_choice in raw_state_choices.items():
                state_id = normalize_marker_id(raw_state_id)
                state_marker = all_state_markers.get(state_id)
                if state_marker is None:
                    raise ValueError(f"Step {index}: state marker {state_id} was not found")
                if state_marker.get("status") != "defined":
                    raise ValueError(
                        f"Step {index}: state marker {state_id} is {state_marker.get('status')}: "
                        f"{state_marker.get('message')}"
                    )
                choice = str(raw_choice or "current").strip().lower()
                if choice not in {"current", "on", "off"}:
                    raise ValueError(
                        f"Step {index}: digital state for {state_id} must be current, on, or off"
                    )
                requested_choices[state_id] = choice

            effective_states: Dict[str, str] = {}
            digital_conditions: List[Dict[str, Any]] = []
            for state_id, state_marker in state_markers.items():
                current_state = str((state_marker.get("candidate") or {}).get("value") or "").upper()
                if current_state not in {"ON", "OFF"}:
                    raise ValueError(f"Step {index}: state marker {state_id} has no valid current state")
                choice = requested_choices.get(state_id, "current")
                effective_state = current_state if choice == "current" else choice.upper()
                effective_states[state_id] = effective_state
                state_definition = state_marker["definition"]
                digital_conditions.append({
                    "id": state_id,
                    "display_name": state_definition.get("display_name") or state_id,
                    "current_state": current_state,
                    "selection": choice,
                    "effective_state": effective_state,
                    "command": (state_marker.get("candidate") or {}).get("command") or "",
                    "channel": (state_marker.get("candidate") or {}).get("channel") or "",
                    "target_line_number": state_marker.get("target_line_number"),
                })

            normalized_steps.append({
                "index": index,
                "marker_id": marker_id,
                "marker_name": definition.get("display_name") or marker_id,
                "marker_kind": definition["kind"],
                "marker_decimals": int(definition.get("decimals", 0)),
                "objective": objective,
                "metric_key": metric_key,
                "metric_label": OBJECTIVE_METRICS[metric_key],
                "metric_source": source,
                "average_count": average_count,
                "randomize": randomize,
                "minimum_r_squared": _finite_float(raw.get("minimum_r_squared", 0.75), "Minimum R²"),
                "start": values[0],
                "stop": values[-1],
                "scan_method": str(raw.get("scan_method") or "step_size").strip().lower(),
                "step": _finite_float(raw.get("step"), "Scan step"),
                "values": values,
                "digital_state_choices": {
                    state_id: requested_choices.get(state_id, "current")
                    for state_id in state_markers
                },
                "digital_states": effective_states,
                "digital_conditions": digital_conditions,
                "status": "pending",
                "points": [],
                "analysis": None,
                "error": None,
            })
        return template_path, content, encoding, definitions, normalized_steps

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_running():
            return {"status": "error", "message": "Marker optimization workflow already running"}
        acquired, busy_message = self.experiment_manager.acquire_run_slot("marker_optimization")
        if not acquired:
            return {"status": "error", "message": busy_message}
        try:
            self.experiment_manager.refresh_runtime_settings_from_disk()
            validated = self._validate_payload(payload)
            fit_config = self.experiment_manager.build_fit_config(payload)
        except Exception as exc:
            self.experiment_manager.release_run_slot("marker_optimization")
            return {"status": "error", "message": str(exc)}
        self._stop_requested = False
        self._artifact_paths = {}
        status = self._idle_status()
        status.update({
            "is_running": True,
            "phase": "running",
            "message": "Marker optimization workflow queued",
            "started_at_ms": int(time.time() * 1000),
            "total_steps": len(validated[4]),
            "steps": copy.deepcopy(validated[4]),
        })
        with self._lock:
            self._status = status
        self._thread = threading.Thread(
            target=self._run,
            args=(copy.deepcopy(payload), fit_config, validated),
            daemon=True,
        )
        self._thread.start()
        return {"status": "success", "message": "Marker optimization workflow started", "data": self.get_status()}

    def stop(self) -> Dict[str, Any]:
        if not self.is_running():
            return {"status": "warning", "message": "No Marker optimization workflow is running"}
        self._stop_requested = True
        self._set_status(phase="stopping", message="Stopping after the current hardware operation...")
        return {"status": "success", "message": "Marker optimization stop requested"}

    def get_export_file(self, kind: str) -> Tuple[Path, str]:
        path = self._artifact_paths.get(str(kind))
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Marker optimization export is not available: {kind}")
        return path, path.name

    def _update_step(self, step_index: int, **updates: Any) -> None:
        with self._lock:
            steps = self._status.get("steps") or []
            if 1 <= step_index <= len(steps):
                steps[step_index - 1].update(copy.deepcopy(updates))

    def _evaluate_step(
        self,
        step: Dict[str, Any],
        payload: Dict[str, Any],
        fit_config: Dict[str, Any],
        working_path: Path,
        data_manager: DataManager,
        global_shot: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        values = list(step["values"])
        average_count = int(step["average_count"])
        total_shots = sum(
            len(item.get("values") or []) * int(item.get("average_count", 1))
            for item in self._snapshot()["steps"]
        )
        execution_config = {
            **payload,
            "mode": "standard",
            "parameter_source": "markers",
            "marker_axes": [step["marker_id"]],
            "_template_path_override": str(working_path),
        }
        shot_plan = [
            {"point_index": point_index, "value": value, "repeat_index": repeat_index}
            for repeat_index in range(1, average_count + 1)
            for point_index, value in enumerate(values, start=1)
        ]
        if step.get("randomize"):
            random.shuffle(shot_plan)
        repeats_by_value: Dict[float, List[float]] = {float(value): [] for value in values}

        def aggregate_points() -> List[Dict[str, Any]]:
            aggregated: List[Dict[str, Any]] = []
            for value in sorted(repeats_by_value):
                repeats = repeats_by_value[value]
                if not repeats:
                    continue
                deviation = float(pstdev(repeats)) if len(repeats) > 1 else 0.0
                aggregated.append({
                    "value": value,
                    "metric_mean": float(mean(repeats)),
                    "metric_std": deviation,
                    "metric_sem": deviation / math.sqrt(len(repeats)) if len(repeats) > 1 else 0.0,
                    "repeats": [float(item) for item in repeats],
                })
            return aggregated

        points: List[Dict[str, Any]] = []
        for shot_number, shot in enumerate(shot_plan, start=1):
            if self._stop_requested:
                raise InterruptedError("Marker optimization stopped by user")
            point_index = int(shot["point_index"])
            repeat_index = int(shot["repeat_index"])
            value = float(shot["value"])
            metadata = {
                "workflow_step": step["index"],
                "workflow_marker": step["marker_id"],
                "workflow_point": point_index,
                "workflow_repeat": repeat_index,
                "workflow_shot": shot_number,
                "workflow_randomized": bool(step.get("randomize")),
                "digital_states": copy.deepcopy(step.get("digital_states") or {}),
            }
            job = self.experiment_manager.execute_single_measurement(
                [value], execution_config, idx=global_shot, total_steps=total_shots,
                scan_dimensions=1, metadata=metadata,
            )
            result, shot_payload = self.experiment_manager.process_measurement_job(
                job,
                fit_config,
                data_manager=data_manager,
                save_step_index=global_shot + 1,
                stream_type="marker_optimization_shot",
                extra_payload={
                    "workflow_step": step["index"],
                    "marker_id": step["marker_id"],
                    "scan_value": value,
                    "point_index": point_index,
                    "repeat_index": repeat_index,
                    "shot_number": shot_number,
                    "shot_count": len(shot_plan),
                    "average_count": average_count,
                    "randomized": bool(step.get("randomize")),
                    "digital_states": copy.deepcopy(step.get("digital_states") or {}),
                },
            )
            self._emit(shot_payload)
            if result is None:
                raise RuntimeError(shot_payload.get("error") or "Marker optimization shot failed")
            repeats_by_value[value].append(
                self._metric_value(result, step["metric_key"], step["metric_source"])
            )
            global_shot += 1
            points = aggregate_points()
            self._update_step(step["index"], points=points)
            self._set_status(
                current_point=shot_number,
                total_points=len(shot_plan),
                current_scan_value=value,
                message=(
                    f"Step {step['index']}: {step['marker_name']} | shot {shot_number}/{len(shot_plan)} "
                    f"| value {value:g} | repeat {repeat_index}/{average_count}"
                ),
            )
            current_point = next(item for item in points if item["value"] == value)
            self._emit({
                "stream_type": "marker_optimization_point",
                "workflow_step": step["index"],
                "marker_id": step["marker_id"],
                "point": current_point,
                "shot_number": shot_number,
                "shot_count": len(shot_plan),
                "randomized": bool(step.get("randomize")),
            })
        return points, global_shot

    @staticmethod
    def _safe_stem(name: str) -> str:
        stem = Path(str(name or "sequence.mot")).stem
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.") or "sequence"

    @staticmethod
    def _write_step_csv(run_dir: Path, step: Dict[str, Any]) -> Tuple[Path, Path]:
        prefix = f"step_{int(step['index']):02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', step['marker_id'])}"
        analysis = step.get("analysis") or {}
        raw_path = run_dir / f"{prefix}_scan.csv"
        with open(raw_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "scan_value", "metric_mean", "metric_std", "metric_sem", "repeat_values",
                "randomized", "digital_states_json",
            ])
            for point in step.get("points") or []:
                writer.writerow([
                    point["value"], point["metric_mean"], point["metric_std"], point["metric_sem"],
                    ";".join(str(value) for value in point.get("repeats") or []),
                    bool(step.get("randomize")),
                    json.dumps(step.get("digital_states") or {}, sort_keys=True),
                ])
        fit_path = run_dir / f"{prefix}_fit_residuals.csv"
        fit_curve = analysis.get("fit_curve") or []
        residuals = analysis.get("residuals") or []
        with open(fit_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "scan_value", "observed", "fitted", "residual", "continuous_optimum",
                "applied_value", "applied_was_sampled",
            ])
            for index, point in enumerate(step.get("points") or []):
                writer.writerow([
                    point["value"], point["metric_mean"],
                    fit_curve[index] if index < len(fit_curve) else "",
                    residuals[index] if index < len(residuals) else "",
                    analysis.get("continuous_optimum", ""),
                    analysis.get("selected_value", ""),
                    analysis.get("selected_was_sampled", ""),
                ])
        return raw_path, fit_path

    def _write_pdf(self, path: Path, report: Dict[str, Any]) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("MIGA Sequential Marker Optimization Report", fontsize=17, fontweight="bold", y=0.97)
            summary_lines = [
                f"Workflow: {report.get('workflow_name') or 'Untitled workflow'}",
                f"Run label: {report.get('run_label') or 'Unlabeled'}",
                f"Sequence: {report.get('sequence_name')}",
                f"Run ID: {report.get('run_id')}",
                f"Outcome: {str(report.get('phase') or '').upper()}",
                f"Stop reason: {report.get('stop_reason')}",
                f"Completed steps: {report.get('completed_steps')} / {report.get('total_steps')}",
                "",
                "Applied values:",
            ]
            applied = report.get("applied_values") or {}
            summary_lines.extend([f"  {key}: {value}" for key, value in applied.items()] or ["  None"])
            if report.get("error"):
                summary_lines.extend(["", "Failure detail:", str(report["error"])])
            summary_lines.extend([
                "",
                "Decision policy:",
                "  Fit objectives apply the nearest hardware-representable value inside the scan range.",
                "  The applied fitted value does not need to be an actually sampled point.",
                "  Raw extrema remain sampled-point decisions; failed or insufficient fits stop the workflow.",
            ])
            fig.text(0.08, 0.90, "\n".join(summary_lines), va="top", family="monospace", fontsize=10, wrap=True)
            fig.text(0.08, 0.04, "Generated automatically. Inspect raw CSV, residuals, and sequence snapshots in the report bundle.", fontsize=8, color="#555555")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            for step in report.get("steps") or []:
                fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(8.27, 9.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
                points = step.get("points") or []
                x = np.asarray([point["value"] for point in points], dtype=float)
                y = np.asarray([point["metric_mean"] for point in points], dtype=float)
                sem = np.asarray([point.get("metric_sem", 0.0) for point in points], dtype=float)
                if len(x):
                    ax.errorbar(x, y, yerr=sem, fmt="o", capsize=3, label="Measured mean ± SEM")
                analysis = step.get("analysis") or {}
                fit_x_dense = analysis.get("fit_x_dense") or []
                fit_curve_dense = analysis.get("fit_curve_dense") or []
                if len(fit_x_dense) == len(fit_curve_dense) and len(fit_x_dense) > 1:
                    ax.plot(fit_x_dense, fit_curve_dense, "-", linewidth=1.8, label=analysis.get("model") or "Fit")
                else:
                    fit_curve = analysis.get("fit_curve") or []
                    if len(fit_curve) == len(x) and any(value is not None for value in fit_curve):
                        ax.plot(x, np.asarray([np.nan if value is None else value for value in fit_curve]), "-", label=analysis.get("model") or "Fit")
                selected = analysis.get("selected_value")
                if selected is not None:
                    ax.axvline(float(selected), color="#c62828", linestyle="--", label=f"Applied: {selected:g}")
                ax.set_title(f"Step {step.get('index')}: {step.get('marker_name')} [{step.get('status')}]", loc="left", fontweight="bold")
                ax.set_ylabel(step.get("metric_label") or "Metric")
                ax.grid(alpha=0.25)
                ax.legend(loc="best")
                residuals = analysis.get("residuals") or []
                if len(residuals) == len(x) and any(value is not None for value in residuals):
                    residual_ax.axhline(0, color="black", linewidth=0.8)
                    residual_ax.plot(x, [np.nan if value is None else value for value in residuals], "o-")
                residual_ax.set_xlabel(f"{step.get('marker_name')} ({step.get('marker_kind')})")
                residual_ax.set_ylabel("Residual")
                residual_ax.grid(alpha=0.25)
                r2 = analysis.get("r_squared")
                details = f"Objective: {step.get('objective')}    Model: {analysis.get('model') or 'n/a'}"
                details += (
                    f"    Range: {step.get('start')} to {step.get('stop')}"
                    f"    Average: {step.get('average_count', 1)}"
                    f"    Randomized: {'yes' if step.get('randomize') else 'no'}"
                )
                if analysis.get("continuous_optimum") is not None:
                    details += f"\nFit optimum: {analysis.get('continuous_optimum'):.6g}"
                if selected is not None:
                    sampled_label = "sampled" if analysis.get("selected_was_sampled") else "not directly sampled"
                    details += f"    Applied value: {selected:g} ({sampled_label})"
                if r2 is not None:
                    details += f"    R²: {r2:.5f}"
                conditions = step.get("digital_conditions") or []
                if conditions:
                    state_text = ", ".join(
                        f"{item.get('display_name')}: {item.get('effective_state')} "
                        f"({item.get('selection')})"
                        for item in conditions
                    )
                    details += f"\nDigital conditions: {state_text}"
                if step.get("error"):
                    details += f"\nFailure: {step.get('error')}"
                fig.text(0.12, 0.025, details, fontsize=8.5, wrap=True)
                fig.tight_layout(rect=(0, 0.12, 1, 0.98))
                pdf.savefig(fig)
                plt.close(fig)

    def _finalize_artifacts(
        self,
        run_dir: Path,
        payload: Dict[str, Any],
        original_content: str,
        working_content: str,
        encoding: str,
    ) -> Dict[str, Path]:
        status = self._snapshot()
        sequence_name = str(payload.get("sequence_name") or "sequence.mot")
        stem = self._safe_stem(sequence_name)
        original_path = run_dir / f"{stem}_original.mot"
        final_path = run_dir / f"{stem}_optimized.mot"
        original_path.write_bytes(encode_mot_text(original_content, encoding))
        final_path.write_bytes(encode_mot_text(working_content, encoding))
        step_csv_paths: List[Path] = []
        for step in status.get("steps") or []:
            raw_path, fit_path = self._write_step_csv(run_dir, step)
            step_csv_paths.extend([raw_path, fit_path])
        preset = {
            "name": str(payload.get("workflow_name") or "Marker workflow"),
            "sequence_profile": sequence_marker_profile_key(sequence_name),
            "sequence_name": sequence_name,
            "fit_center_up": payload.get("fit_center_up", 0),
            "fit_width_up": payload.get("fit_width_up", 0),
            "fit_center_dw": payload.get("fit_center_dw", 0),
            "fit_width_dw": payload.get("fit_width_dw", 0),
            "ext_trigger": bool(payload.get("ext_trigger", False)),
            "steps": [
                {
                    **{key: step.get(key) for key in (
                        "marker_id", "objective", "metric_key", "metric_source", "average_count", "randomize",
                        "minimum_r_squared", "start", "stop", "step", "scan_method"
                    )},
                    "digital_states": copy.deepcopy(step.get("digital_state_choices") or {}),
                }
                for step in status.get("steps") or []
            ],
        }
        preset_path = run_dir / "workflow_preset.json"
        preset_path.write_text(json.dumps(preset, ensure_ascii=False, indent=2), encoding="utf-8")
        report = {
            "report_version": 3,
            "generated_at_ms": int(time.time() * 1000),
            "workflow_name": payload.get("workflow_name"),
            "run_label": payload.get("run_label") or payload.get("workflow_name") or sequence_name,
            "sequence_name": sequence_name,
            "sequence_encoding": encoding,
            "sequence_profile": sequence_marker_profile_key(sequence_name),
            "run_id": status.get("run_id"),
            "phase": status.get("phase"),
            "stop_reason": status.get("stop_reason"),
            "error": status.get("error"),
            "started_at_ms": status.get("started_at_ms"),
            "ended_at_ms": status.get("ended_at_ms"),
            "completed_steps": len([step for step in status.get("steps") or [] if step.get("status") == "completed"]),
            "total_steps": status.get("total_steps"),
            "applied_values": status.get("applied_values") or {},
            "steps": status.get("steps") or [],
            "configuration": payload,
        }
        report_json_path = run_dir / "marker_optimization_report.json"
        report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        pdf_path = run_dir / "marker_optimization_report.pdf"
        self._write_pdf(pdf_path, report)
        zip_path = run_dir / f"{stem}_marker_optimization_report.zip"
        bundle_paths = [original_path, final_path, preset_path, report_json_path, pdf_path, *step_csv_paths]
        results_csv = run_dir / "results.csv"
        if results_csv.is_file():
            bundle_paths.append(results_csv)
        for execution_sequence in sorted(run_dir.glob("step_*_execution_conditions.mot")):
            bundle_paths.append(execution_sequence)
        for snapshot in sorted(run_dir.glob("step_*_applied_*.mot")):
            bundle_paths.append(snapshot)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in bundle_paths:
                if item.is_file():
                    archive.write(item, arcname=item.name)
        return {
            "report_bundle": zip_path,
            "report_pdf": pdf_path,
            "report_json": report_json_path,
            "original_sequence": original_path,
            "optimized_sequence": final_path,
            "workflow_preset": preset_path,
        }

    def _run(
        self,
        payload: Dict[str, Any],
        fit_config: Dict[str, Any],
        validated: Tuple[Path, str, str, List[Dict[str, Any]], List[Dict[str, Any]]],
    ) -> None:
        _, original_content, encoding, definitions, steps = validated
        working_content = original_content
        data_manager = DataManager()
        run_dir: Optional[Path] = None
        global_shot = 0
        phase = "completed"
        stop_reason = "all_steps_completed"
        error_message = None
        try:
            run_config = copy.deepcopy(payload)
            run_config["_marker_optimization_mode"] = True
            run_config["mode"] = "marker_optimization"
            run_config["scan_dimensions"] = 1
            run_config["run_label"] = (
                str(payload.get("run_label") or "").strip()
                or str(payload.get("workflow_name") or "").strip()
                or str(payload.get("sequence_name") or "Marker optimization").strip()
            )
            run_config["randomize"] = any(bool(step.get("randomize")) for step in steps)
            run_config["_system_settings_snapshot"] = copy.deepcopy(self.experiment_manager.settings)
            data_manager.init_run(run_config)
            run_dir = data_manager.current_run_dir
            working_path = run_dir / "working_sequence.mot"
            working_path.write_text(working_content, encoding="utf-8")
            self._set_status(run_id=data_manager.current_run_id_str, run_label=run_config["run_label"], message="Workflow run initialized")
            for step in steps:
                if self._stop_requested:
                    raise InterruptedError("Marker optimization stopped by user")
                step_index = int(step["index"])
                self._set_status(current_step=step_index, current_point=0, total_points=len(step["values"]))
                execution_content = render_digital_marker_states(
                    working_content,
                    step.get("digital_states") or {},
                    definitions,
                )
                execution_name = f"step_{step_index:02d}_execution_conditions.mot"
                execution_path = run_dir / execution_name
                execution_path.write_text(execution_content, encoding="utf-8")
                self._update_step(
                    step_index,
                    status="running",
                    started_at_ms=int(time.time() * 1000),
                    execution_sequence=execution_name,
                )
                points, global_shot = self._evaluate_step(
                    step, payload, fit_config, execution_path, data_manager, global_shot
                )
                try:
                    analysis = analyze_marker_scan(
                        points,
                        step["objective"],
                        minimum_r_squared=step["minimum_r_squared"],
                        marker_kind=step["marker_kind"],
                        marker_decimals=step.get("marker_decimals", 9),
                    )
                except Exception as exc:
                    message = str(exc)
                    self._update_step(
                        step_index,
                        status="failed",
                        points=points,
                        error=message,
                        ended_at_ms=int(time.time() * 1000),
                    )
                    raise RuntimeError(f"Step {step_index} ({step['marker_name']}) failed: {message}") from exc
                selected = analysis["selected_value"]
                working_content = render_auto_marker_sequence(
                    working_content,
                    [step["marker_id"]],
                    [selected],
                    definitions,
                )
                working_path.write_text(working_content, encoding="utf-8")
                snapshot_name = f"step_{step_index:02d}_applied_{self._safe_stem(step['marker_id'])}_{selected}.mot"
                (run_dir / snapshot_name).write_text(working_content, encoding="utf-8")
                self._update_step(
                    step_index,
                    status="completed",
                    points=points,
                    analysis=analysis,
                    applied_value=selected,
                    ended_at_ms=int(time.time() * 1000),
                )
                applied = self._snapshot().get("applied_values") or {}
                applied[step["marker_id"]] = selected
                self._set_status(applied_values=applied, message=f"Applied {step['marker_name']} = {selected:g}")
                self._emit({
                    "stream_type": "marker_optimization_step_complete",
                    "workflow_step": step_index,
                    "marker_id": step["marker_id"],
                    "analysis": analysis,
                    "applied_values": applied,
                })
        except InterruptedError as exc:
            phase = "stopped"
            stop_reason = "stopped_by_user"
            error_message = str(exc)
            current = int(self._snapshot().get("current_step") or 0)
            if current:
                active = self._snapshot().get("steps", [])[current - 1]
                if active.get("status") == "running":
                    self._update_step(current, status="stopped", error=error_message, ended_at_ms=int(time.time() * 1000))
        except Exception as exc:
            phase = "failed"
            stop_reason = "step_failed"
            error_message = str(exc)
            print(f"[Marker Optimization Error] {traceback.format_exc()}")
        finally:
            data_manager.close_run()
            ended_at = int(time.time() * 1000)
            self._set_status(
                is_running=False,
                phase=phase,
                message=("Marker optimization completed" if phase == "completed" else ("Marker optimization stopped" if phase == "stopped" else "Marker optimization failed")),
                ended_at_ms=ended_at,
                stop_reason=stop_reason,
                error=error_message,
            )
            if run_dir is not None:
                try:
                    artifacts = self._finalize_artifacts(
                        run_dir, payload, original_content, working_content, encoding
                    )
                    self._artifact_paths = artifacts
                    urls = {key: f"/marker-optimization/download/{key}" for key in artifacts}
                    self._set_status(export_urls=urls)
                except Exception as report_exc:
                    report_error = f"Scientific report generation failed: {report_exc}"
                    print(f"[Marker Optimization Report Error] {traceback.format_exc()}")
                    existing_error = self._snapshot().get("error")
                    self._set_status(error=f"{existing_error}; {report_error}" if existing_error else report_error)
            self._emit({"stream_type": "marker_optimization_complete", "status": self.get_status()})
            self.experiment_manager.release_run_slot("marker_optimization")
            self._thread = None
            self._stop_requested = False

    @staticmethod
    def _load_presets() -> List[Dict[str, Any]]:
        path = config.MARKER_OPTIMIZATION_PRESETS_PATH
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
        except Exception:
            return []

    def list_presets(self, sequence_name: str) -> List[Dict[str, Any]]:
        profile = sequence_marker_profile_key(sequence_name)
        return [item for item in self._load_presets() if item.get("sequence_profile") == profile]

    def save_preset(self, sequence_name: str, name: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
        preset_name = str(name or "").strip()
        if not preset_name:
            raise ValueError("Preset name is required")
        profile = sequence_marker_profile_key(sequence_name)
        record = {
            "name": preset_name,
            "sequence_profile": profile,
            "sequence_name": str(sequence_name or "sequence.mot"),
            "updated_at_ms": int(time.time() * 1000),
            "workflow": copy.deepcopy(workflow),
        }
        presets = [
            item for item in self._load_presets()
            if not (item.get("sequence_profile") == profile and str(item.get("name") or "").casefold() == preset_name.casefold())
        ]
        presets.append(record)
        config.MARKER_OPTIMIZATION_PRESETS_PATH.write_text(
            json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return record

    def delete_preset(self, sequence_name: str, name: str) -> bool:
        profile = sequence_marker_profile_key(sequence_name)
        preset_name = str(name or "").casefold()
        presets = self._load_presets()
        kept = [
            item for item in presets
            if not (item.get("sequence_profile") == profile and str(item.get("name") or "").casefold() == preset_name)
        ]
        removed = len(kept) != len(presets)
        if removed:
            config.MARKER_OPTIMIZATION_PRESETS_PATH.write_text(
                json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return removed
