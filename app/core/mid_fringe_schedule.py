from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List
from uuid import uuid4

from app.analysis import interferometer_phase
from app.core.link_export import format_link_parameter_value, render_link_mot
from app.models.schemas import ScanConfig


def calibration_at_reference(calibration: Dict[str, Any], reference_t2_us2: float) -> Dict[str, Any]:
    reference = float(reference_t2_us2)
    if not math.isfinite(reference) or reference < 0:
        raise ValueError("Mid-fringe reference must be a non-negative finite value")
    result = deepcopy(calibration)
    result.update({
        "reference_input_mode": "t2",
        "reference_t_unit": "us",
        "reference_value": reference,
        "reference_t2_us2": reference,
        "phase_conversion_mode": "monotonic_half_fringe",
    })
    result.pop("monotonic_slope", None)
    result["monotonic_slope"] = interferometer_phase.monotonic_slope(result)
    return result


def virtual_shot_count(start: float, stop: float, step: float) -> int:
    start_value = float(start)
    stop_value = float(stop)
    step_value = float(step)
    if not all(math.isfinite(value) for value in (start_value, stop_value, step_value)):
        raise ValueError("Virtual P0 shot range must be finite")
    if step_value <= 0:
        raise ValueError("Virtual P0 step must be positive")
    if stop_value < start_value:
        raise ValueError("Virtual P0 stop must be greater than or equal to start")
    count = int(math.floor((stop_value - start_value) / step_value + 1e-12)) + 1
    if count < 1 or count > 100000:
        raise ValueError("Virtual P0 range must contain between 1 and 100000 shots")
    return count


def build_virtual_scan_config(
    source_config: Dict[str, Any],
    reference: float,
    phase_calibration: Dict[str, Any],
    shot_start: float,
    shot_stop: float,
    shot_step: float,
    sequence_name: str,
) -> Dict[str, Any]:
    payload = ScanConfig(**(source_config or {})).dict()
    payload.update({
        "scan_dimensions": 1,
        "dim1_type": "range",
        "dim1_method": "step_size",
        "param_type": "float",
        "start": float(shot_start),
        "stop": float(shot_stop),
        "step": float(shot_step),
        "custom_list": "",
        "dim2_enabled": False,
        "dim3_enabled": False,
        "averages": 1,
        "randomize": False,
        "mode": "standard",
        "parameter_source": "classic",
        "marker_axes": [],
        "sequence_name": sequence_name,
        "run_label": f"Mid fringe P0={format_link_parameter_value(reference)}",
        "interferometer_phase_calibration_override": deepcopy(phase_calibration),
    })
    return payload


def build_mid_fringe_task(
    *,
    batch_id: str,
    index: int,
    reference: float,
    sequence_name: str,
    sequence_snapshot: str,
    config: Dict[str, Any],
    shot_count: int,
    execution_mode: str,
    sync: Dict[str, Any] | None,
) -> Dict[str, Any]:
    reference_label = format_link_parameter_value(reference)
    task = {
        "id": f"{batch_id}_{index + 1:03d}",
        "batch_id": batch_id,
        "name": f"Mid fringe P0={reference_label}",
        "note": f"Master mid-fringe reference P0={reference_label} (μs)^2",
        "mid_fringe_p0_us2": float(reference),
        "execution_mode": execution_mode,
        "config": config,
        "sequence_name": sequence_name,
        "sequence_file_name": sequence_name,
        "sequence_source": "archive_link_mid_fringe",
        "sequence_snapshot": sequence_snapshot,
        "temporary_sequence": True,
        "scheduled_at": "",
        "estimated_points": int(shot_count),
        "estimated_duration_sec": 0,
        "validation_error": "",
    }
    if execution_mode == "sync":
        task["sync"] = deepcopy(sync or {})
    return task


def validate_selected_mid_fringes(fit_result: Dict[str, Any], selected: Iterable[float]) -> List[float]:
    available = [float(value) for value in (fit_result.get("bragg") or {}).get("mid_fringe_x") or []]
    available = [value for value in available if math.isfinite(value) and value >= 0]
    result: List[float] = []
    for raw_value in selected:
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Selected mid-fringe values must be non-negative and finite")
        tolerance = 1e-9 * max(1.0, abs(value))
        if not any(abs(value - candidate) <= tolerance for candidate in available):
            raise ValueError(f"Selected P0={value:g} is not part of the current Bragg fit")
        if not any(abs(value - existing) <= tolerance for existing in result):
            result.append(value)
    if not result:
        raise ValueError("Select at least one mid fringe")
    return sorted(result)


def save_prepared_queue(run_dir: Path, payload: Dict[str, Any]) -> Path:
    target_dir = Path(run_dir) / "mid_fringe_schedules"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{payload['batch_id']}.json"
    temp = target.with_name(f".{target.name}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)
    return target


def load_prepared_queue(run_dir: Path, batch_id: str) -> Dict[str, Any]:
    normalized = str(batch_id or "").strip()
    if not normalized or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in normalized):
        raise ValueError("Invalid prepared queue id")
    path = Path(run_dir) / "mid_fringe_schedules" / f"{normalized}.json"
    if not path.is_file():
        raise FileNotFoundError("Prepared mid-fringe queue was not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("batch_id") != normalized:
        raise ValueError("Prepared mid-fringe queue is invalid")
    return payload


def new_batch_id() -> str:
    return f"midfringe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
