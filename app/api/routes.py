import os
import shutil

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List
from app.analysis import fitting
from app.core.experiment_manager import ExperimentManager
from app.core.data_loader import DataLoader
from app.core.data_manager import DataManager
from app.models.schemas import (
    AnalysisSettings,
    ArchiveAllanRequest,
    ArchiveRunReference,
    ArchiveScanFitRequest,
    ArchiveWaveformRequest,
    ExperimentResponse,
    FitModelDefinition,
    ReAnalysisRequest,
    ScanConfig,
    SystemSettings,
    SystemUpdateRequest,
)
import config

router = APIRouter()
manager = ExperimentManager()
data_loader = DataLoader()

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
    return fitting.get_default_scan_fit_models()

# --- 4. Archive ---
@router.get("/archive/tree")
async def get_archive_tree(): return data_loader.get_archive_tree()

@router.get("/archive/load/{year}/{month}/{day}/{run_id}")
async def load_archived_run(year: str, month: str, day: str, run_id: str):
    try: return data_loader.load_run(year, month, day, run_id)
    except FileNotFoundError: raise HTTPException(404, "Run not found")
    except Exception as e: raise HTTPException(500, str(e))

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
