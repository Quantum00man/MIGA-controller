"""Shared Bayesian fringe endpoints for controller and archive server."""
from typing import Optional, Dict, List, Literal
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from app.analysis import bayesian_fringe
from app.core.analysis_store import input_stamp, stamp_digest

class FringeRequest(BaseModel):
    year: str
    month: str
    day: str
    run_id: str
    node_id: Optional[str] = None
    channel: Literal[1, 2] = 1
    source: Literal["fit", "nofit"] = "fit"
    t2: Optional[float] = Field(None, allow_inf_nan=False)
    shot_min: Optional[int] = None
    shot_max: Optional[int] = None
    bounds: Optional[Dict[str, List[float]]] = None
    grid: int = Field(25, ge=9, le=51)
    phase_points: int = Field(96, ge=32, le=256)
    input_stamp: Optional[str] = None


def build_bayesian_fringe_router(loader_provider, route_class):
    router = APIRouter(route_class=route_class)
    def calculate(req):
        if req.shot_min is not None and req.shot_max is not None and req.shot_min > req.shot_max:
            raise ValueError("Shot start must not exceed shot end")
        loader=loader_provider()
        root=loader.get_run_dir(req.year, req.month, req.day, req.run_id)
        before=stamp_digest(input_stamp(root))
        if req.input_stamp and req.input_stamp != before:
            raise ValueError("Archive changed since analysis; recalculate")
        result=bayesian_fringe.compute(loader,req)
        if stamp_digest(input_stamp(root)) != before:
            raise ValueError("Archive changed during analysis; recalculate")
        result["input_stamp"]=before
        return result
    @router.post("/archive/bayesian-fringe/shots")
    async def shots(req: FringeRequest):
        try:
            records, identity, mode = await run_in_threadpool(bayesian_fringe.select_shots, loader_provider(), req, False)
            return {"records": records, "node": identity.get("resolved_node") or "local", "mode": mode}
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            raise HTTPException(400,str(exc)) from exc
    @router.post("/archive/bayesian-fringe")
    async def preview(req: FringeRequest):
        try:
            return await run_in_threadpool(calculate,req)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            raise HTTPException(400,str(exc)) from exc
    @router.post("/archive/bayesian-fringe/report")
    async def report(req: FringeRequest):
        try:
            if not req.input_stamp:
                raise ValueError("Run analysis before exporting")
            def generate():
                result=calculate(req)
                return bayesian_fringe.package(req,result)
            content=await run_in_threadpool(generate)
            return Response(content,media_type="application/zip",headers={"Content-Disposition": "attachment; filename=bayesian_fringe_report.zip"})
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            raise HTTPException(400,str(exc)) from exc
    return router
