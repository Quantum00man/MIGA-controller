"""Shared result management API for Controller and Archive Server."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
import json
from app.core.analysis_store import store_for, fingerprint
from app.models.schemas import ArchiveAnalysisSaveRequest


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
