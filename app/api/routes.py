import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List
from app.core.experiment_manager import ExperimentManager
from app.core.data_loader import DataLoader
from app.core.data_manager import DataManager
from app.models.schemas import ScanConfig, ExperimentResponse, SystemSettings, AnalysisSettings, ReAnalysisRequest
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
    manager.update_settings(settings.dict())
    return ExperimentResponse(status="success", message="Updated")

@router.get("/settings/analysis", response_model=AnalysisSettings)
async def get_analysis_settings(): return manager.get_analysis_config()

@router.post("/settings/analysis", response_model=ExperimentResponse)
async def update_analysis_settings(settings: AnalysisSettings):
    manager.update_analysis_config(settings.dict())
    return ExperimentResponse(status="success", message="Updated")

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

@router.post("/archive/overwrite", response_model=ExperimentResponse)
async def overwrite_archived_run(req: ReAnalysisRequest):
    try:
        # Use a fresh DataManager instance for overwrite ops to avoid state conflict
        dm = DataManager() 
        dm.overwrite_run(
            req.year, req.month, req.day, req.run_id,
            req.new_settings.dict(),
            req.updated_data
        )
        return ExperimentResponse(status="success", message="Run overwritten successfully")
    except Exception as e:
        raise HTTPException(500, f"Overwrite failed: {str(e)}")