"""Shared result management API for Controller and Archive Server."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
import json
import math
from app.core.analysis_store import store_for, fingerprint, capture_inputs, capture_result, stamp_digest
from app.models.schemas import ArchiveAnalysisSaveRequest, ArchiveSyncPhaseAllanSaveRequest


def build_analysis_router(loader_provider, route_class):
    router = APIRouter(route_class=route_class)

    @router.get('/archive/analysis-results/{year}/{month}/{day}/{run_id}')
    async def list_results(year: str, month: str, day: str, run_id: str):
        try:
            return {'analyses': await run_in_threadpool(store_for(loader_provider(), [year, month, day, run_id]).list)}
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400, str(exc))

    @router.post('/archive/analysis-results/save')
    async def save_result(req: ArchiveAnalysisSaveRequest):
        try:
            loader = loader_provider()
            current_hash = await run_in_threadpool(fingerprint, loader.get_run_dir(req.year, req.month, req.day, req.run_id)) if 'plot_snapshot' in req.view else None
            return await run_in_threadpool(store_for(loader, [req.year, req.month, req.day, req.run_id]).save,
                req.candidate_id, req.name, req.note, req.view, current_hash)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400, str(exc))

    @router.post('/archive/analysis-results/sync-phase/save')
    async def save_sync_phase_result(req: ArchiveSyncPhaseAllanSaveRequest):
        def save_snapshot():
            loader = loader_provider()
            root = loader.get_run_dir(req.year, req.month, req.day, req.run_id)
            if not (root / 'sync_manifest.json').is_file():
                raise ValueError('SYNC phase Allan requires a synchronized archive')
            source = capture_inputs(loader, req)
            if stamp_digest(source['_stamp']) != req.input_stamp:
                raise ValueError('Archive input changed since loading; reload and recompute before saving')
            curves = req.result.get('curves')
            if not isinstance(curves, list) or not curves:
                raise ValueError('Compute SYNC phase Allan before saving')
            for item in curves:
                curve = item.get('curve') if isinstance(item, dict) else None
                if not isinstance(curve, dict) or not curve.get('orders'):
                    raise ValueError('Invalid SYNC Allan curve')
                orders = curve['orders']
                if not isinstance(orders, list) or orders != list(range(1, len(orders) + 1)):
                    raise ValueError('Invalid SYNC Allan orders')
                for key in ('deviations', 'validWindowCounts', 'edfWhite', 'ciLower', 'ciUpper', 'errorMinus', 'errorPlus'):
                    values = curve.get(key)
                    if not isinstance(values, list) or len(values) != len(orders) or any(
                        value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)) for value in values):
                        raise ValueError('Invalid SYNC Allan numerical values')
            result = {**req.result, 'origin': 'browser'}
            captured = capture_result(loader, req, result, req.new_settings.model_dump(), 'sync_phase_allan', source)
            return store_for(loader, [req.year, req.month, req.day, req.run_id]).save(
                captured['analysis_candidate_id'], req.name, req.note, req.view)
        try:
            return await run_in_threadpool(save_snapshot)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400, str(exc))

    @router.get('/archive/analysis-results/{year}/{month}/{day}/{run_id}/{analysis_id}')
    async def load_result(year: str, month: str, day: str, run_id: str, analysis_id: str, download: bool = False):
        try:
            loader = loader_provider()
            record = await run_in_threadpool(store_for(loader, [year, month, day, run_id]).read, analysis_id)
            if download:
                return Response(json.dumps(record, ensure_ascii=False, allow_nan=False), media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename="analysis_{analysis_id}.json"'})
            record['input_matches_current'] = record['source']['input_sha256'] == await run_in_threadpool(
                fingerprint, loader.get_run_dir(year, month, day, run_id))
            return record
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400, str(exc))

    return router
