"""Archive UI compatibility endpoints; no controller or hardware initialization.

Science handlers follow the controller API and use request-scoped dependencies.
"""
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import quote
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool
from app.models.schemas import *
from app.analysis import fitting, interferometer_phase, phase_calibration_optimization, phase_noise
from app.analysis.transfer_function import optimize_slave_normalization_scale
from app.core.archive_labplot import build_archive_project
from app.core.mid_fringe_storage import load_prepared_queue
from app.archive.ui_context import scoped, DeviceRoute, versions
router = APIRouter(route_class=DeviceRoute)
data_loader = scoped('loader')
manager = scoped('settings')
archive_collection_store = scoped('collections')
archive_audit_service = scoped('audit')
sync_manager = scoped('offline_sync')
archive_allan_lock = asyncio.Lock()

async def _synchronize_archive_phase_metadata(year, month, day, run_id, request):
    from app.archive.ui_context import _scope
    if _scope.get()['writable']:
        await run_in_threadpool(data_loader.update_archive_phase_analysis_sync_state,
            year, month, day, run_id, sync_status='local',
            sync_message='Server-only analysis; no reverse synchronization', coordinator_url='')
    return {"status": "local", "message": "Archive Server only; no reverse synchronization"}


@router.post('/archive/overwrite')
async def save_server_analysis_version(req: ReAnalysisRequest):
    from app.core.data_manager import DataManager
    settings = req.new_settings.model_dump()
    settings['_interferometer_phase_calibration'] = manager.get_active_bragg_phase_calibration()
    result = await run_in_threadpool(data_loader.recalculate_run, req.year, req.month, req.day, req.run_id,
                                    settings, max_points=None, node_id=req.node_id)
    root = data_loader.get_run_dir(req.year, req.month, req.day, req.run_id)
    node = data_loader._resolve_archive_node_dir(root, req.node_id)
    await run_in_threadpool(DataManager().overwrite_run, req.year, req.month, req.day, req.run_id,
                            settings, result['data'], result.get('transfer_function_summary'), target_directory=node)
    return {'status': 'success', 'message': 'New server analysis version saved; original backup unchanged'}


@router.get('/archive/versions/{year}/{month}/{day}/{run_id}')
async def list_server_analysis_versions(request: Request, year: str, month: str, day: str, run_id: str):
    from app.archive.ui_context import _repository
    return {'versions': versions(_repository, request.path_params['device_id'], [year, month, day, run_id])}


@router.post('/archive/sync/retry/{year}/{month}/{day}/{run_id}')
async def no_reverse_sync():
    raise HTTPException(403, 'SYNC replication is controller-only; Archive Server never writes to controllers')


@router.get('/archive/device')
async def archive_device_info(request: Request):
    from app.archive.ui_context import _repository
    device_id = request.path_params['device_id']
    return {'device_id': device_id, 'name': _repository.configuration.load()['devices'][device_id].get('name') or device_id}

def _attachment_response(payload: bytes, filename: str, media_type: str) -> Response:
    ascii_name = filename.encode("ascii", "ignore").decode("ascii") or "bragg_export"
    headers = {
        "Content-Disposition": (
            f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"
        ),
        "Cache-Control": "no-store",
    }
    return Response(content=payload, media_type=media_type, headers=headers)

def _sync_manifest_for_run(run_dir: Path) -> Dict[str, Any]:
    path = Path(run_dir) / "sync_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}

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

@router.get("/archive/tree")
async def get_archive_tree(): return await run_in_threadpool(data_loader.get_archive_tree)

@router.get("/archive/navigation/years")
async def get_archive_years():
    return {"years": await run_in_threadpool(data_loader.list_archive_years)}

@router.get("/archive/navigation/months/{year}")
async def get_archive_months(year: str):
    try:
        return {"year": year, "months": await run_in_threadpool(data_loader.list_archive_months, year)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.get("/archive/navigation/days/{year}/{month}")
async def get_archive_days(year: str, month: str):
    try:
        return {"year": year, "month": month, "days": await run_in_threadpool(data_loader.list_archive_days, year, month)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.get("/archive/navigation/runs/{year}/{month}/{day}")
async def get_archive_runs(year: str, month: str, day: str):
    try:
        runs = await run_in_threadpool(data_loader.list_archive_runs, year, month, day)
        return {"year": year, "month": month, "day": day, "runs": runs}
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.get("/archive/latest")
async def get_latest_archive_run():
    try:
        return await run_in_threadpool(data_loader.get_latest_archive_run)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))

@router.post("/archive/audit/jobs")
async def start_archive_audit():
    return archive_audit_service.start()

@router.get("/archive/audit/jobs/{job_id}")
async def get_archive_audit_job(job_id: str):
    try:
        return archive_audit_service.status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))

@router.get("/archive/audit/reports")
async def list_archive_audit_reports():
    return {"reports": await run_in_threadpool(archive_audit_service.list_reports)}

@router.get("/archive/audit/reports/{report_id}.{format_name}")
async def download_archive_audit_report(report_id: str, format_name: str):
    if format_name not in {"json", "html"}:
        raise HTTPException(400, "Archive audit report format must be json or html")
    try:
        path = archive_audit_service.report_path(report_id, format_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    return FileResponse(
        path=path,
        media_type="text/html" if format_name == "html" else "application/json",
        filename=path.name,
    )

@router.get("/archive/collections")
async def get_archive_collections():
    return archive_collection_store.snapshot()

@router.post("/archive/collections/folders")
async def create_archive_collection_folder(req: ArchiveCollectionFolderCreate):
    try:
        return archive_collection_store.create_folder(req.name, req.parent_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))

@router.patch("/archive/collections/folders/{folder_id}")
async def update_archive_collection_folder(folder_id: int, req: ArchiveCollectionFolderUpdate):
    try:
        return archive_collection_store.update_folder(folder_id, req.name, req.parent_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))

@router.delete("/archive/collections/folders/{folder_id}")
async def delete_archive_collection_folder(folder_id: int):
    try:
        archive_collection_store.delete_folder(folder_id)
        return {"status": "ok"}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))

@router.post("/archive/collections/favorites")
async def create_archive_favorite(req: ArchiveFavoriteCreate):
    try:
        entry = data_loader.get_run_entry(req.year, req.month, req.day, req.run_id)
        preview = data_loader.build_collection_preview(
            req.year, req.month, req.day, req.run_id, req.preview_metric, req.preview_step
        )
        return archive_collection_store.create_favorite(
            req.folder_id,
            req.dict(include={"year", "month", "day", "run_id"}),
            {
                "source_type": "optimization" if entry.get("has_marker_optimization") else "scan",
                "original_label": entry.get("run_label") or "",
                "sequence_name": entry.get("sequence_name") or "",
                "summary": entry.get("summary") or "",
            },
            preview,
            req.alias,
            req.note,
            req.preview_metric,
            req.preview_step,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))

@router.patch("/archive/collections/favorites/{favorite_id}")
async def update_archive_favorite(favorite_id: int, req: ArchiveFavoriteUpdate):
    try:
        return archive_collection_store.update_favorite(
            favorite_id, **req.dict(exclude_unset=True)
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))

@router.delete("/archive/collections/favorites/{favorite_id}")
async def delete_archive_favorite(favorite_id: int):
    try:
        archive_collection_store.delete_favorite(favorite_id)
        return {"status": "ok"}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))

@router.post("/archive/collections/favorites/batch")
async def batch_archive_favorites(req: ArchiveFavoriteBatchRequest):
    try:
        return archive_collection_store.batch_favorites(req.action, req.favorite_ids, req.folder_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.get("/archive/load/{year}/{month}/{day}/{run_id}")
async def load_archived_run(
    year: str, month: str, day: str, run_id: str, node_id: str = "", analysis_copy_id: str = ""
):
    try: return await run_in_threadpool(data_loader.load_run,
        year, month, day, run_id, node_id=node_id or None,
        current_phase_calibration=manager.get_active_bragg_phase_calibration(),
        analysis_copy_id=analysis_copy_id or None,
    )
    except FileNotFoundError as exc: raise HTTPException(404, str(exc) or "Run not found")
    except ValueError as exc: raise HTTPException(400, str(exc))
    except Exception as e: raise HTTPException(500, str(e))

@router.get("/archive/phase-calibrations/{year}/{month}/{day}/{run_id}")
async def get_archive_phase_calibrations(
    year: str, month: str, day: str, run_id: str, node_id: str = ""
):
    try:
        run_dir = data_loader.get_run_dir(year, month, day, run_id)
        manifest = _sync_manifest_for_run(run_dir)
        if not manifest:
            return {
                "node_id": "archive",
                "source": "local_settings",
                "calibrations": manager.get_bragg_phase_calibrations(),
                "active": manager.get_active_bragg_phase_calibration(),
            }
        metadata = data_loader.archive_phase_analysis_metadata(year, month, day, run_id)
        target = str(node_id or sync_manager.archive_local_node_id(run_dir) or "master")
        return await run_in_threadpool(
            sync_manager.get_archive_node_phase_calibrations,
            run_dir,
            target,
            metadata.get("coordinator_url") or "",
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Unable to load phase calibrations from the selected SYNC node: {exc}")

@router.post("/archive/phase-reference-override")
async def save_archive_phase_reference_override(req: ArchivePhaseReferenceOverrideRequest, request: Request):
    try:
        selected_calibration = None
        calibration_id = str(req.calibration_id or "").strip()
        if calibration_id:
            run_dir = data_loader.get_run_dir(req.year, req.month, req.day, req.run_id)
            manifest = _sync_manifest_for_run(run_dir)
            if manifest:
                metadata = data_loader.archive_phase_analysis_metadata(req.year, req.month, req.day, req.run_id)
                available = await run_in_threadpool(
                    sync_manager.get_archive_node_phase_calibrations,
                    run_dir,
                    str(req.node_id or sync_manager.archive_local_node_id(run_dir) or "master"),
                    metadata.get("coordinator_url") or "",
                )
                calibrations = available.get("calibrations") or []
            else:
                calibrations = manager.get_bragg_phase_calibrations()
            selected_calibration = next(
                (item for item in calibrations if str(item.get("id") or "") == calibration_id),
                None,
            )
            if selected_calibration is None:
                raise ValueError("Selected Settings fringe calibration was not found")
        context = data_loader.save_archive_phase_reference_override(
            req.year,
            req.month,
            req.day,
            req.run_id,
            req.node_id,
            req.reference_input_mode,
            req.reference_value,
            req.reference_t_unit,
            req.monotonic_slope,
            current_phase_calibration=manager.get_active_bragg_phase_calibration(),
            selected_phase_calibration=selected_calibration,
        )
        sync_result = await _synchronize_archive_phase_metadata(
            req.year, req.month, req.day, req.run_id, request
        )
        return {"status": "success", "context": context, "sync": sync_result}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.delete("/archive/phase-reference-override/{year}/{month}/{day}/{run_id}")
async def delete_archive_phase_reference_override(
    year: str, month: str, day: str, run_id: str, request: Request, node_id: str = ""
):
    try:
        deleted = data_loader.delete_archive_phase_reference_override(
            year, month, day, run_id, node_id or None
        )
        sync_result = await _synchronize_archive_phase_metadata(year, month, day, run_id, request)
        return {"status": "success", "deleted": deleted, "sync": sync_result}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.post("/archive/phase-analysis/sync/{year}/{month}/{day}/{run_id}")
async def retry_archive_phase_analysis_sync(
    year: str, month: str, day: str, run_id: str, request: Request
):
    try:
        result = await _synchronize_archive_phase_metadata(year, month, day, run_id, request)
        return {"status": "success", "sync": result}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, str(exc))

@router.get("/archive/sequence/{year}/{month}/{day}/{run_id}")
async def download_archived_sequence(year: str, month: str, day: str, run_id: str, node_id: str = ""):
    try:
        sequence_path, download_name = data_loader.get_archived_sequence_file(year, month, day, run_id, node_id=node_id or None)
        return FileResponse(path=sequence_path, media_type="text/plain", filename=download_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.get("/archive/run-log/{year}/{month}/{day}/{run_id}")
async def download_archived_run_log(year: str, month: str, day: str, run_id: str, node_id: str = ""):
    try:
        log_path, download_name = data_loader.get_archived_run_log(year, month, day, run_id, node_id=node_id or None)
        return FileResponse(path=log_path, media_type="application/x-ndjson", filename=download_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.get("/archive/run-log/{year}/{month}/{day}/{run_id}/merged")
async def download_merged_sync_run_log(year: str, month: str, day: str, run_id: str):
    try:
        payload, download_name = data_loader.build_merged_sync_run_log(year, month, day, run_id)
        return Response(
            content=payload, media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.get("/archive/run-log/{year}/{month}/{day}/{run_id}/view")
async def view_archived_run_log(
    year: str, month: str, day: str, run_id: str,
    node_id: str = "", merged: bool = False,
):
    try:
        return data_loader.get_archived_run_log_view(
            year, month, day, run_id, node_id=node_id or None, merged=merged,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.get("/archive/bragg-fringe-calibration/{year}/{month}/{day}/{run_id}/mot")
async def download_bragg_fringe_calibration_mot(year: str, month: str, day: str, run_id: str):
    try:
        path, filename = data_loader.get_bragg_fringe_calibration_mot(year, month, day, run_id)
        return FileResponse(path=path, media_type="text/plain", filename=filename)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except (OSError, ValueError) as exc:
        raise HTTPException(400, str(exc))

@router.get("/archive/mid-fringe-schedule/{year}/{month}/{day}/{run_id}/{batch_id}")
async def get_archive_mid_fringe_schedule(
    year: str, month: str, day: str, run_id: str, batch_id: str
):
    try:
        run_dir = data_loader.get_run_dir(year, month, day, run_id)
        return load_prepared_queue(run_dir, batch_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.get("/archive/marker-optimization/artifact/{year}/{month}/{day}/{run_id}/{kind}")
async def download_archived_marker_optimization_artifact(
    year: str, month: str, day: str, run_id: str, kind: str
):
    try:
        artifact_path, download_name = data_loader.get_marker_optimization_artifact(
            year, month, day, run_id, kind
        )
        return FileResponse(path=artifact_path, media_type="application/octet-stream", filename=download_name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.get("/archive/waveform/{year}/{month}/{day}/{run_id}/{step_index}")

async def load_archived_waveform(year: str, month: str, day: str, run_id: str, step_index: int, node_id: str = ""):
    try: return await run_in_threadpool(data_loader.load_waveform, year, month, day, run_id, step_index, node_id=node_id or None)
    except FileNotFoundError: raise HTTPException(404, "Waveform not found")
    except Exception as e: raise HTTPException(500, str(e))

@router.post("/archive/recalculate")
async def recalculate_archived_run(req: ReAnalysisRequest):
    try:
        settings = req.new_settings.dict()
        settings["_interferometer_phase_calibration"] = manager.get_active_bragg_phase_calibration()
        return await run_in_threadpool(data_loader.recalculate_run,
            req.year,
            req.month,
            req.day,
            req.run_id,
            settings,
            node_id=req.node_id,
            phase_noise_allan_orders=req.phase_noise_allan_orders,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.post("/archive/intf-alpha/reanalyze")
async def reanalyze_archive_intf_alpha(req: ArchiveIntfAlphaReanalysisRequest):
    try:
        root = data_loader._get_run_dir(req.year, req.month, req.day, req.run_id)
        run_dir = data_loader._resolve_archive_node_dir(root, req.node_id or None)
        result = await run_in_threadpool(data_loader.load_run,
            req.year, req.month, req.day, req.run_id,
            node_id=req.node_id or None,
            current_phase_calibration=manager.get_active_bragg_phase_calibration(),
            intf_alpha_accepted_ids=req.accepted_calibration_ids,
            intf_alpha_interpolation_method=req.interpolation_method,
        )
        saved = None
        if req.save:
            saved = data_loader.save_intf_alpha_analysis_copy(
                run_dir, req.name, req.accepted_calibration_ids, req.interpolation_method,
                req.accepted_min, req.accepted_max,
            )
            result["intf_alpha_analysis_copies"] = data_loader.load_intf_alpha_analysis_copies(run_dir)
        result["saved_intf_alpha_analysis_copy"] = saved
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.post("/archive/allan")
async def calculate_archived_allan(req: ArchiveAllanRequest):
    try:
        settings = req.new_settings.dict()
        settings["_interferometer_phase_calibration"] = manager.get_active_bragg_phase_calibration()
        # Allan reanalysis can be CPU- and I/O-intensive.  Keep it off the
        # event loop so live WebSocket updates continue, and serialize these
        # jobs to avoid competing with active acquisition for resources.
        async with archive_allan_lock:
            return await run_in_threadpool(
                data_loader.calculate_allan_run,
                req.year,
                req.month,
                req.day,
                req.run_id,
                req.order,
                req.display_mode,
                settings,
                req.p0_min,
                req.p0_max,
                req.node_id,
                manager.get_active_bragg_phase_calibration(),
                req.metric,
                req.source,
                req.intf_alpha_selection.model_dump() if req.intf_alpha_selection else None,
            )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.post("/archive/phase-noise/allan")
async def calculate_archived_phase_noise_allan(req: ArchivePhaseNoiseAllanRequest):
    settings = req.new_settings.model_dump()
    settings["_interferometer_phase_calibration"] = manager.get_active_bragg_phase_calibration()
    try:
        if req.intf_alpha_selection is not None and req.display_mode == "recalculated":
            payload = await run_in_threadpool(
                data_loader.load_run,
                req.year, req.month, req.day, req.run_id, req.node_id,
                manager.get_active_bragg_phase_calibration(), req.orders,
                intf_alpha_accepted_ids=req.intf_alpha_selection.accepted_calibration_ids,
                intf_alpha_interpolation_method=req.intf_alpha_selection.interpolation_method,
            )
        elif str(req.display_mode or "saved").strip().lower() == "recalculated":
            payload = await run_in_threadpool(
                data_loader.recalculate_run,
                req.year, req.month, req.day, req.run_id, settings,
                None, req.node_id, req.orders,
            )
        else:
            payload = await run_in_threadpool(
                data_loader.load_run,
                req.year, req.month, req.day, req.run_id, req.node_id,
                manager.get_active_bragg_phase_calibration(), req.orders,
            )
        if str((payload.get("config") or {}).get("mode") or "").strip().lower() != "phase_noise":
            raise ValueError("Selected archive is not a Phase Noise Analyze run")
        return {"orders": req.orders, "phase_noise_summary": payload.get("phase_noise_summary") or []}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.post("/archive/interferometer-beta/optimize")
@router.post("/archive/mean-parameter/optimize")
async def optimize_archived_interferometer_beta(req: ArchiveInterferometerBetaOptimizeRequest):
    try:
        settings = req.new_settings.dict()
        settings["_interferometer_phase_calibration"] = manager.get_active_bragg_phase_calibration()
        return await run_in_threadpool(data_loader.optimize_archive_interferometer_beta,
            req.year,
            req.month,
            req.day,
            req.run_id,
            settings,
            p0_min=req.p0_min,
            p0_max=req.p0_max,
            parameter=req.parameter,
            metric=req.metric,
            statistic=req.statistic,
            allan_order=req.allan_order,
            alpha_min=req.alpha_min,
            alpha_max=req.alpha_max,
            beta_min=req.beta_min,
            beta_max=req.beta_max,
            probability_mean_tolerance=req.probability_mean_tolerance,
            parameter_prior_weight=req.parameter_prior_weight,
            source=req.source,
            channel=req.channel,
            target_mean=req.target_mean,
            node_id=req.node_id,
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
        return await run_in_threadpool(data_loader.recalculate_waveforms,
            req.year,
            req.month,
            req.day,
            req.run_id,
            req.step_indices,
            req.new_settings.dict(),
            node_id=req.node_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@router.post("/archive/scan-fit")
async def fit_archive_scan(req: ArchiveScanFitRequest):
    x_values = req.x_values
    y_values = req.y_values
    y_std_values = req.y_std_values
    point_counts_values = req.point_counts
    if req.bragg_fit_method == "shot_level" and req.shot_source:
        source = req.shot_source
        try:
            shots = data_loader.load_archive_scan_shots(
                source["year"], source["month"], source["day"], source["run_id"],
                source["metric_tab"], source.get("channel", "up"), source.get("source_key", "fit"),
                source.get("node_id") or None,
            )
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, f"Could not load all archive shots: {exc}")
        x_values = [point[0] for point in shots]
        y_values = [point[1] for point in shots]
        y_std_values = None
        point_counts_values = None
    if len(x_values) != len(y_values):
        raise HTTPException(400, "x_values and y_values must have the same length")
    if y_std_values is not None and len(y_std_values) != len(x_values):
        raise HTTPException(400, "y_std_values must have the same length as x_values")
    if point_counts_values is not None and len(point_counts_values) != len(x_values):
        raise HTTPException(400, "point_counts must have the same length as x_values")

    model_definition = req.model.dict()
    model_error = fitting.validate_fit_model_definition(model_definition)
    if model_error:
        raise HTTPException(400, f"Invalid fit model: {model_error}")

    fit_min = float(req.fit_min) if req.fit_min is not None else None
    fit_max = float(req.fit_max) if req.fit_max is not None else None
    if fit_min is not None and fit_max is not None and fit_min > fit_max:
        fit_min, fit_max = fit_max, fit_min

    filtered_pairs = []
    for index, (raw_x, raw_y) in enumerate(zip(x_values, y_values)):
        x_val = float(raw_x)
        y_val = float(raw_y)
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue
        if fit_min is not None and x_val < fit_min:
            continue
        if fit_max is not None and x_val > fit_max:
            continue
        raw_std = y_std_values[index] if y_std_values is not None else None
        raw_count = point_counts_values[index] if point_counts_values is not None else None
        filtered_pairs.append((x_val, y_val, raw_std, raw_count))

    if len(filtered_pairs) < 2:
        raise HTTPException(400, "Need at least 2 valid points inside the selected scan range")

    filtered_pairs.sort(key=lambda item: item[0])
    x_data = np.asarray([item[0] for item in filtered_pairs], dtype=float)
    y_data = np.asarray([item[1] for item in filtered_pairs], dtype=float)
    y_std_data = None if y_std_values is None else np.asarray([item[2] for item in filtered_pairs], dtype=float)
    point_counts = None if point_counts_values is None else np.asarray([item[3] for item in filtered_pairs], dtype=float)

    eval_points = max(32, min(int(req.eval_points or 400), 4000))
    if np.isclose(x_data[0], x_data[-1]):
        eval_x = x_data.copy()
    else:
        eval_x = np.linspace(float(x_data[0]), float(x_data[-1]), eval_points)

    if model_definition.get("key") == "bragg_fringes":
        fringe_result = fitting.perform_bragg_fringe_fit(
            x_data,
            y_data,
            wavelength_nm=req.bragg_wavelength_nm,
            bragg_order=req.bragg_order,
            eval_x=eval_x,
            fit_method=req.bragg_fit_method,
            y_std=y_std_data,
            counts=point_counts,
        )
        if fringe_result is None:
            raise HTTPException(400, "Bragg fringe fit failed. At least 4 distinct P0 points are required.")
        return {
            "model_key": "bragg_fringes",
            "model_label": "Bragg Fringes",
            "point_count": len(filtered_pairs),
            "fit_min": float(x_data[0]),
            "fit_max": float(x_data[-1]),
            "fit_x": eval_x.tolist(),
            "fit_y": fringe_result.fit_curve.tolist(),
            "parameter_values": fringe_result.parameter_values,
            "residual_variance": fringe_result.residual_variance,
            "amplitude": fringe_result.parameter_values["A"],
            "width": None,
            "center": None,
            "offset": fringe_result.parameter_values["C"],
            "area": None,
            "bragg": {
                "wavelength_nm": float(req.bragg_wavelength_nm),
                "order": int(req.bragg_order),
                "effective_wavevector_rad_m": fringe_result.effective_wavevector_rad_m,
                "angular_frequency_rad_per_us2": fringe_result.angular_frequency_rad_per_us2,
                "gravity_m_s2": 9.80665,
                "symbolic_formula": "y = C + A cos[k_eff a (P0 x 10^-12) + phi0]",
                "mid_fringe_x": fringe_result.mid_fringe_x,
                "mid_fringe_spacing_us2": fringe_result.mid_fringe_spacing_us2,
                "fit_method": fringe_result.fit_method,
                "weighted_point_count": fringe_result.weighted_point_count,
            },
        }

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
        "area": fit_result.area,
    }

@router.post("/archive/sync-differential-fit")
async def fit_archive_sync_differential(req: ArchiveSyncDifferentialFitRequest):
    if len(req.x_values) != len(req.y_values):
        raise HTTPException(400, "x_values and y_values must have the same length")
    fit_result = await run_in_threadpool(
        fitting.perform_sync_differential_ellipse_fit,
        np.asarray(req.x_values, dtype=float),
        np.asarray(req.y_values, dtype=float),
        req.eval_points,
    )
    if fit_result is None:
        raise HTTPException(
            400,
            "Ellipse fit failed. At least 8 valid points spanning both X and Y are required.",
        )
    payload = {
        "model_key": "sync_differential_ellipse",
        "model_label": "SYNC Differential Ellipse",
        "point_count": len(req.x_values),
        "source": req.source,
        "parameter_values": fit_result.parameter_values,
        "parameter_uncertainties": fit_result.parameter_uncertainties,
        "fit_x": fit_result.fit_x.tolist(),
        "fit_y": fit_result.fit_y.tolist(),
        "point_fit_x": fit_result.point_fit_x.tolist(),
        "point_fit_y": fit_result.point_fit_y.tolist(),
        "normalized_rms_residual": fit_result.normalized_rms_residual,
        "residual_variance": fit_result.residual_variance,
        "r_squared": fit_result.r_squared,
        "phase_coverage_rad": fit_result.phase_coverage_rad,
        "condition_number": fit_result.condition_number,
        "warnings": fit_result.warnings,
    }
    try:
        return data_loader.save_sync_differential_fit(
            req.year, req.month, req.day, req.run_id, payload
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.post("/archive/sync-phase-calibration-optimize")
async def optimize_archive_sync_phase_calibrations(req: ArchiveSyncPhaseCalibrationOptimizeRequest):
    if req.reference_node_id == req.target_node_id:
        raise HTTPException(400, "Reference and target nodes must be different")
    try:
        loaded = await run_in_threadpool(
            data_loader.load_run,
            req.year,
            req.month,
            req.day,
            req.run_id,
            None,
            manager.get_active_bragg_phase_calibration(),
        )
        manifest = loaded.get("sync_manifest") or {}
        if not manifest:
            raise ValueError("The selected archive is not a SYNC run")
        contexts = loaded.get("archive_phase_reference_contexts") or {}

        def calibration_for(node_id: str) -> Dict[str, Any]:
            context = contexts.get(node_id) or {}
            calibration = context.get("effective_calibration")
            if not isinstance(calibration, dict):
                raise ValueError(f"Node {node_id} has no effective interferometer phase calibration")
            return calibration

        reference_calibration = calibration_for(req.reference_node_id)
        target_calibration = calibration_for(req.target_node_id)
        if req.phase_noise_t2_us2 is not None:
            reference_calibration = phase_noise.calibration_at_mid_fringe(
                reference_calibration, req.phase_noise_t2_us2
            )
            target_calibration = phase_noise.calibration_at_mid_fringe(
                target_calibration, req.phase_noise_t2_us2
            )
        node_results = manifest.get("node_results") or {}

        # SYNC manifest rows capture the live signal.  Periodic I_alpha analysis
        # is finalized per node in its own archive, so overlay those recalculated
        # science rows before optimizing A/C.
        reanalyzed_rows: Dict[str, List[Dict[str, Any]]] = {}
        for selected_node_id in {req.reference_node_id, req.target_node_id}:
            node_loaded = await run_in_threadpool(
                data_loader.load_run,
                req.year, req.month, req.day, req.run_id,
                selected_node_id,
                manager.get_active_bragg_phase_calibration(),
                None, None, None,
                None,
                req.intf_alpha_interpolation_method,
            )
            node_rows = node_loaded.get("archive_node_data") or node_loaded.get("data") or []
            reanalyzed_rows[selected_node_id] = [row for row in node_rows if isinstance(row, dict)]

        def rows_for(node_id: str) -> List[Dict[str, Any]]:
            if node_id == "master":
                rows = node_results.get("master")
            else:
                rows = (node_results.get("slaves") or {}).get(node_id)
            if isinstance(rows, list) and rows:
                merged = [dict(row) for row in rows if isinstance(row, dict)]
                recalculated = reanalyzed_rows.get(node_id) or []
                for index, recalculated_row in enumerate(recalculated[:len(merged)]):
                    merged[index].update({
                        key: recalculated_row.get(key)
                        for key in (
                            "intf_p1", "intf_p2", "intf_p1_nofit", "intf_p2_nofit",
                            "intf_alpha_applied", "interferometer_phase",
                            "interferometer_phase_valid", "interferometer_phase_source_value",
                        )
                    })
                return merged
            pairs = manifest.get("pairs") or []
            if node_id == "master":
                candidates = [item.get("master") for item in pairs]
            else:
                candidates = [
                    item.get("slave") for item in pairs
                    if str(item.get("slave_node_id") or "") == str(node_id)
                ]
            unique: Dict[int, Dict[str, Any]] = {}
            for index, row in enumerate(candidates):
                if not isinstance(row, dict):
                    continue
                shot = int(row.get("sync_shot_index", row.get("step", index)))
                unique[shot] = row
            return [unique[key] for key in sorted(unique)]

        def shot_number(row: Dict[str, Any], fallback: int) -> int:
            try:
                return int(row.get("sync_shot_index", row.get("step", fallback)))
            except (TypeError, ValueError):
                return fallback

        def p0_value(row: Dict[str, Any]) -> Optional[float]:
            value = row.get("sync_p0", row.get("parameter"))
            if value is None and isinstance(row.get("all_parameters"), list) and row["all_parameters"]:
                value = row["all_parameters"][0]
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result if np.isfinite(result) else None

        def transfer_value(row: Dict[str, Any], field: str) -> Optional[float]:
            try:
                value = float(row.get(field))
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        reference_rows = rows_for(req.reference_node_id)
        target_rows = rows_for(req.target_node_id)
        reference_by_shot = {shot_number(row, index): row for index, row in enumerate(reference_rows)}
        reference_field = interferometer_phase.source_field(reference_calibration)
        target_field = interferometer_phase.source_field(target_calibration)
        pairs = []
        for index, target_row in enumerate(target_rows):
            shot = shot_number(target_row, index)
            reference_row = reference_by_shot.get(shot)
            if not isinstance(reference_row, dict):
                continue
            reference_p0 = p0_value(reference_row)
            target_p0 = p0_value(target_row)
            transfer_selection = req.transfer_frequency_hz is not None
            if not transfer_selection:
                if reference_p0 is None or target_p0 is None:
                    continue
                if abs(reference_p0 - target_p0) > 1e-9 * max(1.0, abs(reference_p0), abs(target_p0)):
                    continue
                if req.p0_min is not None and target_p0 < req.p0_min:
                    continue
                if req.p0_max is not None and target_p0 > req.p0_max:
                    continue
            if req.shot_index_min is not None and shot < req.shot_index_min:
                continue
            if req.shot_index_max is not None and shot > req.shot_index_max:
                continue
            if req.phase_noise_t2_us2 is not None:
                if target_p0 is None or not math.isclose(
                    target_p0, req.phase_noise_t2_us2,
                    rel_tol=1e-9, abs_tol=1e-9,
                ):
                    continue
            if req.transfer_frequency_hz is not None:
                reference_frequency = transfer_value(reference_row, "transfer_frequency_hz")
                target_frequency = transfer_value(target_row, "transfer_frequency_hz")
                if reference_frequency is None or target_frequency is None:
                    continue
                if not (
                    math.isclose(reference_frequency, target_frequency, abs_tol=1e-9)
                    and math.isclose(target_frequency, req.transfer_frequency_hz, abs_tol=1e-9)
                ):
                    continue
            if req.transfer_phase_deg is not None:
                reference_phase = transfer_value(reference_row, "transfer_phase_deg")
                target_phase = transfer_value(target_row, "transfer_phase_deg")
                if reference_phase is None or target_phase is None:
                    continue
                if not (
                    math.isclose(reference_phase, target_phase, abs_tol=1e-9)
                    and math.isclose(target_phase, req.transfer_phase_deg, abs_tol=1e-9)
                ):
                    continue
            try:
                reference_signal = float(reference_row.get(reference_field))
                target_signal = float(target_row.get(target_field))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(reference_signal) or not np.isfinite(target_signal):
                continue
            local_phase_noise_shot = target_row.get("phase_noise_repeat")
            pairs.append({
                # Phase Noise excludes calibration shots and intentionally
                # treats the remaining repeats as one contiguous sequence.
                "shot": int(local_phase_noise_shot) if req.phase_noise_t2_us2 is not None and local_phase_noise_shot is not None else shot,
                "sync_shot_index": shot,
                "p0": target_p0,
                "transfer_frequency_hz": transfer_value(target_row, "transfer_frequency_hz"),
                "transfer_phase_deg": transfer_value(target_row, "transfer_phase_deg"),
                "reference_signal": reference_signal,
                "target_signal": target_signal,
            })

        result = await run_in_threadpool(
            phase_calibration_optimization.optimize_sync_phase_calibrations,
            pairs,
            reference_calibration,
            target_calibration,
            req.objective,
            req.combined_allan_weight,
            req.parameter_bound_fraction,
            req.fringe_weight,
            req.prior_weight,
        )
        result["reference_node_id"] = req.reference_node_id
        result["target_node_id"] = req.target_node_id
        result["reference_node_label"] = "Master" if req.reference_node_id == "master" else req.reference_node_id
        result["target_node_label"] = "Master" if req.target_node_id == "master" else req.target_node_id
        result["source_fields"] = {"reference": reference_field, "target": target_field}
        result["settings"] = {
            "objective": req.objective,
            "combined_allan_weight": req.combined_allan_weight,
            "parameter_bound_fraction": req.parameter_bound_fraction,
            "fringe_weight": req.fringe_weight,
            "prior_weight": req.prior_weight,
            "p0_min": req.p0_min,
            "p0_max": req.p0_max,
            "shot_index_min": req.shot_index_min,
            "shot_index_max": req.shot_index_max,
            "transfer_frequency_hz": req.transfer_frequency_hz,
            "transfer_phase_deg": req.transfer_phase_deg,
            "phase_noise_t2_us2": req.phase_noise_t2_us2,
            "intf_alpha_interpolation_method": req.intf_alpha_interpolation_method,
        }
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.post("/archive/sync-phase-calibration-optimization/save")
async def save_archive_sync_phase_calibration_optimization(
    req: ArchiveSyncPhaseCalibrationSaveRequest,
):
    try:
        return await run_in_threadpool(
            data_loader.save_sync_phase_calibration_optimization,
            req.year,
            req.month,
            req.day,
            req.run_id,
            req.result,
            req.name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.post("/archive/sync-phase-calibration-optimization/apply")
async def apply_archive_sync_phase_calibration_optimization(
    req: ArchiveSyncPhaseCalibrationApplyRequest,
    request: Request,
):
    try:
        contexts = await run_in_threadpool(
            data_loader.apply_sync_phase_calibration_optimization,
            req.year,
            req.month,
            req.day,
            req.run_id,
            req.result,
            manager.get_active_bragg_phase_calibration(),
        )
        sync_result = await _synchronize_archive_phase_metadata(
            req.year, req.month, req.day, req.run_id, request
        )
        return {"status": "success", "contexts": contexts, "sync": sync_result}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.post("/archive/sync-transfer-normalization-optimize")
async def optimize_archive_sync_transfer_normalization(req: ArchiveSyncTransferNormalizationRequest):
    try:
        analysis_override = {"phase_calibration_result": req.phase_calibration_result} if isinstance(req.phase_calibration_result, dict) else None
        loaded = await run_in_threadpool(
            data_loader.load_run, req.year, req.month, req.day, req.run_id, None,
            manager.get_active_bragg_phase_calibration(), None, req.analysis_copy_id, analysis_override,
        )
        pairs = []
        for pair in (loaded.get("sync_manifest") or {}).get("pairs", []):
            try:
                frequency = float((pair.get("master") or {}).get("transfer_frequency_hz"))
            except (TypeError, ValueError):
                continue
            if math.isclose(frequency, req.transfer_frequency_hz, abs_tol=1e-9):
                pairs.append(pair)
        result = optimize_slave_normalization_scale(pairs, req.slave_node_id, req.bound_fraction)
        result["transfer_frequency_hz"] = req.transfer_frequency_hz
        return result
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc))

@router.post("/archive/sync-analysis-copies/save")
async def save_archive_sync_analysis_copy(req: ArchiveSyncAnalysisCopySaveRequest):
    try:
        return await run_in_threadpool(
            data_loader.save_sync_analysis_copy, req.year, req.month, req.day, req.run_id, req.name,
            req.phase_calibration_result, req.transfer_normalization_result,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc))

@router.delete(
    "/archive/sync-phase-calibration-optimization/{year}/{month}/{day}/{run_id}/{optimization_id}"
)
async def delete_archive_sync_phase_calibration_optimization(
    year: str, month: str, day: str, run_id: str, optimization_id: str
):
    try:
        deleted = await run_in_threadpool(
            data_loader.delete_sync_phase_calibration_optimization,
            year,
            month,
            day,
            run_id,
            optimization_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    if not deleted:
        raise HTTPException(404, "Saved phase calibration optimization was not found")
    return {"deleted": True, "id": optimization_id}

@router.delete("/archive/sync-differential-fit/{year}/{month}/{day}/{run_id}/{fit_id}")
async def delete_archive_sync_differential_fit(
    year: str, month: str, day: str, run_id: str, fit_id: str
):
    try:
        deleted = data_loader.delete_sync_differential_fit(year, month, day, run_id, fit_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    if not deleted:
        raise HTTPException(404, "Saved differential ellipse fit was not found")
    return {"deleted": True, "id": fit_id}

@router.post("/archive/labplot-export")
async def export_archive_labplot(req: ArchiveLabPlotExportRequest):
    source = str(req.source or "fit").strip().lower()
    if source not in {"fit", "nofit"}:
        raise HTTPException(400, "LabPlot source must be fit or nofit")
    try:
        content = await run_in_threadpool(
            build_archive_project,
            data_loader,
            req.year,
            req.month,
            req.day,
            req.run_id,
            req.metrics,
            source,
            req.include_fits,
            req.include_differential,
            manager.get_active_bragg_phase_calibration(),
            req.current_fit,
            req.transfer_function_summary,
            req.phase_noise_summary,
            req.phase_noise_x_axis,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    safe_run = "".join(char if char.isalnum() or char in "-_" else "_" for char in req.run_id) or "archive"
    return Response(
        content=content,
        media_type="application/x-labplot",
        headers={"Content-Disposition": f'attachment; filename="miga_{safe_run}.lml"'},
    )

@router.get("/archive/bragg-phase-calibrations")
async def list_archive_bragg_phase_calibrations():
    return manager.get_bragg_phase_calibrations()

@router.get("/settings/interferometer-phase-calibrations")
async def list_interferometer_phase_calibrations():
    return {
        "calibrations": manager.get_bragg_phase_calibrations(),
        "active": manager.get_active_bragg_phase_calibration(),
    }

@router.post("/archive/bragg-phase-calibrations")
async def save_archive_bragg_phase_calibration(req: BraggPhaseCalibrationSaveRequest):
    try:
        return manager.save_bragg_phase_calibration(req.name, req.fit_result, req.source)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.patch("/settings/interferometer-phase-calibrations/{calibration_id}")
async def update_interferometer_phase_calibration(
    calibration_id: str, req: InterferometerPhaseCalibrationUpdateRequest
):
    try:
        return manager.update_bragg_phase_calibration(calibration_id, req.updates)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.post("/settings/interferometer-phase-calibrations/active")
async def activate_interferometer_phase_calibration(req: InterferometerPhaseCalibrationActivateRequest):
    try:
        return {"active": manager.set_active_bragg_phase_calibration(req.calibration_id)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))

@router.delete("/archive/bragg-phase-calibrations/{calibration_id}")
async def delete_archive_bragg_phase_calibration(calibration_id: str):
    if not manager.delete_bragg_phase_calibration(calibration_id):
        raise HTTPException(404, "Bragg phase calibration not found")
    return {"status": "deleted", "id": calibration_id}
