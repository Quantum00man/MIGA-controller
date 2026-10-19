"""Identical amplitude optimization endpoints for Controller and Archive Server."""
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.analysis.transfer_amplitude import AmplitudeProblem


class AmplitudeRequest(BaseModel):
    year: str
    month: str
    day: str
    run_id: str
    slave_node_id: str
    transfer_frequency_hz: float = Field(..., allow_inf_nan=False)
    master_bound_fraction: float = Field(.05, gt=0, le=.5, allow_inf_nan=False)
    slave_bound_fraction: float = Field(.05, gt=0, le=.5, allow_inf_nan=False)
    shot_index_min: Optional[int] = None
    shot_index_max: Optional[int] = None


class AmplitudeApplyRequest(AmplitudeRequest):
    result: dict


def build_transfer_amplitude_router(loader_provider, route_class, synchronize):
    router = APIRouter(route_class=route_class)

    def problem(req):
        return AmplitudeProblem(loader_provider(), [req.year, req.month, req.day, req.run_id],
            req.slave_node_id, req.transfer_frequency_hz, req.shot_index_min, req.shot_index_max)

    @router.post('/archive/sync-transfer-amplitude-optimize')
    async def preview(req: AmplitudeRequest):
        def calculate():
            return problem(req).optimize(req.master_bound_fraction, req.slave_bound_fraction,
                req.shot_index_min, req.shot_index_max)
        try:
            return await run_in_threadpool(calculate)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.post('/archive/sync-transfer-amplitude-optimization/apply')
    async def apply(req: AmplitudeApplyRequest, request: Request):
        try:
            result = req.result
            expected = {'master_bound_fraction': req.master_bound_fraction,
                'slave_bound_fraction': req.slave_bound_fraction,
                'shot_index_min': req.shot_index_min, 'shot_index_max': req.shot_index_max}
            if result.get('settings') != expected or result.get('slave_node_id') != req.slave_node_id or result.get('transfer_frequency_hz') != req.transfer_frequency_hz:
                raise ValueError('Optimization selection changed; recompute the preview')
            contexts = await run_in_threadpool(lambda: problem(req).apply(result))
            sync = await synchronize(req.year, req.month, req.day, req.run_id, request)
            return {'status': 'success', 'contexts': contexts, 'sync': sync}
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc

    return router
