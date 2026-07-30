import json
import os
import shutil
import time
from copy import deepcopy

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List
from app.analysis import fitting
from app.core.experiment_manager import ExperimentManager
from app.core.data_loader import DataLoader
from app.core.data_manager import DataManager
from app.core.optimization_manager import OBJECTIVE_METRICS, OptimizationManager
from app.models.schemas import (
    AnalysisSettings,
    ArchiveAllanRequest,
    ArchiveRunReference,
    ArchiveScanFitRequest,
    ArchiveWaveformRequest,
    ScanFitModelSaveRequest,
    ExperimentResponse,
    FitModelDefinition,
    IndexUiStateRequest,
    OptimizationConfig,
    ReAnalysisRequest,
    ScanConfig,
    SystemSettings,
    SystemUpdateRequest,
)
import config


router = APIRouter()
manager = ExperimentManager()
optimization_manager = OptimizationManager(manager)
data_loader = DataLoader()


def _default_index_ui_state() -> Dict[str, Any]:
    return {
        "runMode": "live",
        "currentSequenceName": "Default (seq0.mot)",
        "config": ScanConfig().dict(),
        "scheduleSettings": {
            "singlePointDurationSec": 1.615,
            "timingMode": "sequential",
            "sequentialGapSec": 60,
        },
        "scheduledTasks": [],
        "scheduleRuntime": {
            "active": False,
            "stopRequested": False,
            "waiting": False,
            "activeTaskIndex": -1,
            "activeTaskId": None,
            "currentTaskStep": 0,
            "currentTaskTotalSteps": 0,
            "currentTaskStartedAtMs": None,
            "waitUntilMs": None,
            "scheduleStartedAtMs": None,
            "completedTaskIds": [],
        },
        "statusMessage": "IDLE",
    }


def _sanitize_index_ui_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _default_index_ui_state()
    sanitized = deepcopy(defaults)
    if not isinstance(payload, dict):
        return sanitized

    run_mode = str(payload.get("runMode") or defaults["runMode"]).strip().lower()
    sanitized["runMode"] = run_mode if run_mode in {"live", "scheduled"} else defaults["runMode"]

    current_sequence_name = str(payload.get("currentSequenceName") or defaults["currentSequenceName"]).strip()
    sanitized["currentSequenceName"] = current_sequence_name or defaults["currentSequenceName"]

    config_payload = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    allowed_config_keys = set(defaults["config"].keys())
    sanitized["config"].update({key: value for key, value in config_payload.items() if key in allowed_config_keys})

    schedule_settings_payload = payload.get("scheduleSettings") if isinstance(payload.get("scheduleSettings"), dict) else {}
    timing_mode = str(schedule_settings_payload.get("timingMode") or defaults["scheduleSettings"]["timingMode"]).strip().lower()
    single_point_duration = schedule_settings_payload.get("singlePointDurationSec", defaults["scheduleSettings"]["singlePointDurationSec"])
    sequential_gap = schedule_settings_payload.get("sequentialGapSec", defaults["scheduleSettings"]["sequentialGapSec"])
    try:
        single_point_duration = float(single_point_duration)
    except (TypeError, ValueError):
        single_point_duration = defaults["scheduleSettings"]["singlePointDurationSec"]
    try:
        sequential_gap = float(sequential_gap)
    except (TypeError, ValueError):
        sequential_gap = defaults["scheduleSettings"]["sequentialGapSec"]
    sanitized["scheduleSettings"] = {
        "singlePointDurationSec": max(0.001, single_point_duration),
        "timingMode": timing_mode if timing_mode in {"sequential", "specific"} else defaults["scheduleSettings"]["timingMode"],
        "sequentialGapSec": max(0.0, sequential_gap),
    }

    scheduled_tasks_payload = payload.get("scheduledTasks")
    if isinstance(scheduled_tasks_payload, list):
        sanitized["scheduledTasks"] = [task for task in scheduled_tasks_payload if isinstance(task, dict)]

    schedule_runtime_payload = payload.get("scheduleRuntime") if isinstance(payload.get("scheduleRuntime"), dict) else {}
    allowed_runtime_keys = set(defaults["scheduleRuntime"].keys())
    sanitized_runtime = deepcopy(defaults["scheduleRuntime"])
    for key, value in schedule_runtime_payload.items():
        if key in allowed_runtime_keys:
            sanitized_runtime[key] = value
    if not isinstance(sanitized_runtime.get("completedTaskIds"), list):
        sanitized_runtime["completedTaskIds"] = []
    sanitized["scheduleRuntime"] = sanitized_runtime

    status_message = str(payload.get("statusMessage") or defaults["statusMessage"]).strip()
    sanitized["statusMessage"] = status_message or defaults["statusMessage"]
    return sanitized


def _load_index_ui_state_record() -> Dict[str, Any]:
    path = config.INDEX_UI_STATE_PATH
    defaults = {
        "state": _default_index_ui_state(),
        "updated_at_ms": 0,
        "source_id": None,
    }
    if not path.exists():
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return defaults
    return {
        "state": _sanitize_index_ui_state(raw.get("state") if isinstance(raw, dict) else {}),
        "updated_at_ms": int(raw.get("updated_at_ms") or 0) if isinstance(raw, dict) else 0,
        "source_id": str(raw.get("source_id") or "").strip() or None if isinstance(raw, dict) else None,
    }


def _save_index_ui_state_record(state: Dict[str, Any], source_id: str | None = None) -> Dict[str, Any]:
    record = {
        "state": _sanitize_index_ui_state(state),
        "updated_at_ms": int(time.time() * 1000),
        "source_id": str(source_id or "").strip() or None,
    }
    path = config.INDEX_UI_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    return record

# --- 1. Experiment Control ---
@router.post("/experiment/start", response_model=ExperimentResponse)
async def start_experiment(config: ScanConfig):
    result = manager.start_scan(config.dict())
    if result["status"] == "error": raise HTTPException(400, result["message"])
    return ExperimentResponse(status=result["status"], message=result["message"])

@router.post("/experiment/stop", response_model=ExperimentResponse)
async def stop_experiment():
    result = manager.stop_scan()
    return ExperimentResponse(status=result["status"], message=result["message"])


@router.get("/optimization/objective-metrics", response_model=ExperimentResponse)
async def get_optimization_objective_metrics():
    return ExperimentResponse(status="success", message="Objective metrics loaded", data={"metrics": OBJECTIVE_METRICS})


@router.post("/optimization/start", response_model=ExperimentResponse)
async def start_optimization(config: OptimizationConfig):
    result = optimization_manager.start_optimization(config.dict())
    if result["status"] == "error":
        raise HTTPException(400, result["message"])
    return ExperimentResponse(status=result["status"], message=result["message"], data=result.get("data"))


@router.post("/optimization/stop", response_model=ExperimentResponse)
async def stop_optimization():
    result = optimization_manager.stop_optimization()
    return ExperimentResponse(status=result["status"], message=result["message"], data=result.get("data"))


@router.get("/optimization/status", response_model=ExperimentResponse)
async def get_optimization_status():
    return ExperimentResponse(status="success", message="Optimization status loaded", data=optimization_manager.get_status())


@router.get("/optimization/download/{kind}")
async def download_optimization_artifact(kind: str):
    try:
        artifact_path, download_name = optimization_manager.get_export_file(kind)
        media_type = "application/json" if kind == "optimization_report" else "text/plain"
        if kind == "optimization_history":
            media_type = "text/csv"
        return FileResponse(path=artifact_path, media_type=media_type, filename=download_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/index/state", response_model=ExperimentResponse)
async def get_index_state():
    record = _load_index_ui_state_record()
    return ExperimentResponse(status="success", message="Index state loaded", data=record)


@router.post("/index/state", response_model=ExperimentResponse)
async def update_index_state(req: IndexUiStateRequest):
    record = _save_index_ui_state_record(req.state, req.source_id)
    return ExperimentResponse(status="success", message="Index state updated", data=record)


@router.get("/experiment/status", response_model=ExperimentResponse)
async def get_status():
    s = manager.status

    # [NEW] Logic to determine what Run ID to display
    # If running, show current. If idle, predict next.
    if s.is_running:
        run_label = manager.data_manager.current_run_id_str
    else:
        run_label = manager.data_manager.get_next_run_id_str()

    return ExperimentResponse(
        status="success",
        message=s.message,
        data={
            "is_running": s.is_running,
            "current_step": s.current_step,
            "run_id": run_label # <--- This is what index.html needs
        }
    )

# [NEW] Sequence Upload Endpoint
@router.post("/experiment/sequence")
async def upload_sequence(file: UploadFile = File(...)):
    try:
        target_path = config.SEQUENCE_TEMPLATE_PATH_WIN if config.IS_WINDOWS else config.SEQUENCE_TEMPLATE_PATH_LINUX
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "success", "message": f"Sequence loaded: {file.filename}", "filename": file.filename}
    except Exception as e:
        raise HTTPException(500, f"Failed to upload sequence: {str(e)}")

@router.get("/experiment/sequence/template")
async def get_sequence_template_snapshot():
    target_path = config.SEQUENCE_TEMPLATE_PATH_WIN if config.IS_WINDOWS else config.SEQUENCE_TEMPLATE_PATH_LINUX
    if not os.path.exists(target_path):
        raise HTTPException(404, "Current sequence template not found")

    encodings = ("utf-8", "latin-1")
    last_error = None
    for encoding in encodings:
        try:
            with open(target_path, "r", encoding=encoding) as handle:
                return {"status": "success", "content": handle.read()}
        except UnicodeDecodeError as exc:
            last_error = exc

    raise HTTPException(500, f"Failed to read current sequence template: {last_error}")

@router.post("/experiment/load-run-preset", response_model=ExperimentResponse)
async def load_run_preset(req: ArchiveRunReference):
    try:
        data = manager.load_run_preset(req.year, req.month, req.day, req.run_id)
        return ExperimentResponse(status="success", message="Run preset loaded", data=data)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

# --- 2. System Config ---
class SystemMode(BaseModel): simulation: bool

@router.post("/system/mode", response_model=ExperimentResponse)
async def set_system_mode(mode: SystemMode):
    manager.set_simulation_mode(mode.simulation)
    return ExperimentResponse(status="success", message=f"Switched to {'SIMULATION' if mode.simulation else 'REAL'} mode")

@router.get("/system/mode")
async def get_system_mode():
    return {"simulation": config.USE_SIMULATION, "os": "Windows" if config.IS_WINDOWS else "Linux"}

# --- 3. Settings ---
@router.get("/settings/all", response_model=SystemSettings)
async def get_all_settings(): return manager.get_settings()

@router.post("/settings/all", response_model=ExperimentResponse)
async def update_all_settings(settings: SystemSettings):
    try:
        manager.update_settings(settings.dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return ExperimentResponse(status="success", message="Updated")

@router.get("/settings/analysis", response_model=AnalysisSettings)
async def get_analysis_settings(): return manager.get_analysis_config()

@router.post("/settings/analysis", response_model=ExperimentResponse)
async def update_analysis_settings(settings: AnalysisSettings):
    manager.update_analysis_config(settings.dict())
    return ExperimentResponse(status="success", message="Updated")

@router.get("/system/update/status", response_model=ExperimentResponse)
async def get_system_update_status():
    try:
        return ExperimentResponse(status="success", message="Update status loaded", data=manager.get_update_status())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.post("/system/update/fetch", response_model=ExperimentResponse)
async def fetch_system_update_status(req: SystemUpdateRequest):
    try:
        data = manager.fetch_update_metadata(repo_url=req.repo_url, branch=req.branch)
        return ExperimentResponse(status="success", message="Remote metadata refreshed", data=data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.post("/system/update/apply", response_model=ExperimentResponse)
async def apply_system_update(req: SystemUpdateRequest):
    try:
        data = manager.apply_system_update(repo_url=req.repo_url, branch=req.branch)
        return ExperimentResponse(status="success", message="System update completed", data=data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.get("/fitting/models/defaults", response_model=List[FitModelDefinition])
async def get_default_fitting_models():
    return fitting.get_default_fit_models()


@router.get("/fitting/models/scan-defaults", response_model=List[FitModelDefinition])
async def get_default_scan_fitting_models():
    return manager.get_scan_fit_models()


@router.post("/fitting/models/scan-custom")
async def save_custom_scan_fitting_model(req: ScanFitModelSaveRequest):
    try:
        saved_model = manager.save_custom_scan_fit_model(req.model.dict(), req.name)
        return {
            "saved_model": saved_model,
            "models": manager.get_scan_fit_models(),
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

# --- 4. Archive ---
@router.get("/archive/tree")
async def get_archive_tree(): return data_loader.get_archive_tree()

@router.get("/archive/load/{year}/{month}/{day}/{run_id}")
async def load_archived_run(year: str, month: str, day: str, run_id: str):
    try: return data_loader.load_run(year, month, day, run_id)
    except FileNotFoundError: raise HTTPException(404, "Run not found")
    except Exception as e: raise HTTPException(500, str(e))

@router.get("/archive/sequence/{year}/{month}/{day}/{run_id}")
async def download_archived_sequence(year: str, month: str, day: str, run_id: str):
    try:
        sequence_path, download_name = data_loader.get_archived_sequence_file(year, month, day, run_id)
        return FileResponse(path=sequence_path, media_type="text/plain", filename=download_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.get("/archive/waveform/{year}/{month}/{day}/{run_id}/{step_index}")
async def load_archived_waveform(year: str, month: str, day: str, run_id: str, step_index: int):
    try: return data_loader.load_waveform(year, month, day, run_id, step_index)
    except FileNotFoundError: raise HTTPException(404, "Waveform not found")
    except Exception as e: raise HTTPException(500, str(e))

@router.post("/archive/recalculate")
async def recalculate_archived_run(req: ReAnalysisRequest):
    try:
        return data_loader.recalculate_run(
            req.year,
            req.month,
            req.day,
            req.run_id,
            req.new_settings.dict(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.post("/archive/allan")
async def calculate_archived_allan(req: ArchiveAllanRequest):
    try:
        return data_loader.calculate_allan_run(
            req.year,
            req.month,
            req.day,
            req.run_id,
            req.order,
            req.display_mode,
            req.new_settings.dict(),
            p0_min=req.p0_min,
            p0_max=req.p0_max,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.post("/archive/waveforms/recalculate")
async def recalculate_archived_waveforms(req: ArchiveWaveformRequest):
    try:
        return data_loader.recalculate_waveforms(
            req.year,
            req.month,
            req.day,
            req.run_id,
            req.step_indices,
            req.new_settings.dict(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@router.post("/archive/scan-fit")
async def fit_archive_scan(req: ArchiveScanFitRequest):
    if len(req.x_values) != len(req.y_values):
        raise HTTPException(400, "x_values and y_values must have the same length")

    model_definition = req.model.dict()
    model_error = fitting.validate_fit_model_definition(model_definition)
    if model_error:
        raise HTTPException(400, f"Invalid fit model: {model_error}")

    fit_min = float(req.fit_min) if req.fit_min is not None else None
    fit_max = float(req.fit_max) if req.fit_max is not None else None
    if fit_min is not None and fit_max is not None and fit_min > fit_max:
        fit_min, fit_max = fit_max, fit_min

    filtered_pairs = []
    for raw_x, raw_y in zip(req.x_values, req.y_values):
        x_val = float(raw_x)
        y_val = float(raw_y)
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue
        if fit_min is not None and x_val < fit_min:
            continue
        if fit_max is not None and x_val > fit_max:
            continue
        filtered_pairs.append((x_val, y_val))

    if len(filtered_pairs) < 2:
        raise HTTPException(400, "Need at least 2 valid points inside the selected scan range")

    filtered_pairs.sort(key=lambda item: item[0])
    x_data = np.asarray([item[0] for item in filtered_pairs], dtype=float)
    y_data = np.asarray([item[1] for item in filtered_pairs], dtype=float)

    eval_points = max(32, min(int(req.eval_points or 400), 4000))
    if np.isclose(x_data[0], x_data[-1]):
        eval_x = x_data.copy()
    else:
        eval_x = np.linspace(float(x_data[0]), float(x_data[-1]), eval_points)

    fit_result = fitting.perform_configured_fit(model_definition, x_data, y_data, eval_x=eval_x)
    if fit_result is None:
        raise HTTPException(400, "Fit failed. Try a different model, scan range, or initial guesses.")

    return {
        "model_key": fit_result.model_key,
        "model_label": fit_result.model_label,
        "point_count": len(filtered_pairs),
        "fit_min": float(x_data[0]),
        "fit_max": float(x_data[-1]),
        "fit_x": eval_x.tolist(),
        "fit_y": fit_result.fit_curve.tolist(),
        "parameter_values": fit_result.parameter_values,
        "residual_variance": fit_result.residual_variance,
        "amplitude": fit_result.amplitude,
        "width": fit_result.width,
        "center": fit_result.center,
        "offset": fit_result.offset,
    }


@router.post("/archive/overwrite", response_model=ExperimentResponse)
async def overwrite_archived_run(req: ReAnalysisRequest):
    try:
        # Use a fresh DataManager instance for overwrite ops to avoid state conflict
        dm = DataManager()
        recalculated = data_loader.recalculate_run(
            req.year,
            req.month,
            req.day,
            req.run_id,
            req.new_settings.dict(),
            max_points=None,
        )
        dm.overwrite_run(
            req.year, req.month, req.day, req.run_id,
            req.new_settings.dict(),
            recalculated["data"],
        )
        return ExperimentResponse(status="success", message="Run overwritten successfully")
    except Exception as e:
        raise HTTPException(500, f"Overwrite failed: {str(e)}")
