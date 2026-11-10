import io
import json
import zipfile
import numpy as np
import pytest
from app.analysis.bayesian_fringe import compute, infer, select_shots, package
from app.api.bayesian_fringe_routes import FringeRequest

class Loader:
    def __init__(self, points): self.points = points
    def get_run_dir(self, *args): return None
    def _resolve_archive_node_identity(self, root, node): return None, {"resolved_node": node or ""}
    def _load_config_data(self, node): return {"mode": "phase_noise"}
    def _read_results_csv(self, node, max_points=None): return self.points

def test_selects_exact_t_and_source_and_keeps_exclusion():
    points = [{"step":i, "phase_noise_t2_us2":100 if i < 22 else 200,
               "phase_noise_science_shot":i+1, "intf_p2_nofit":float(i)} for i in range(24)]
    points[3]["intf_p2_nofit"] = None
    req=FringeRequest(year="2026",month="01",day="01",run_id="test",channel=2,source="nofit",t2=100,node_id="slave")
    records, identity, mode=select_shots(Loader(points),req)
    assert len(records)==22 and sum(r["included"] for r in records)==21
    assert records[3]["exclusion_reason"] and identity["resolved_node"]=="slave"
    assert all(r["t2_us2"]==100 for r in records)

def test_link_shots_respect_range_channel_source_and_node():
    points = [{"step": i, "parameter": i / 10, "intf_p2_nofit": float(i)} for i in range(30)]
    loader = Loader(points)
    loader._load_config_data = lambda node: {"mode": "link"}
    req = FringeRequest(year="2026", month="01", day="01", run_id="run01", node_id="slave",
                        channel=2, source="nofit", shot_min=3, shot_max=25)
    records, identity, mode = select_shots(loader, req)
    assert mode == "link" and identity["resolved_node"] == "slave"
    assert [r["step"] for r in records] == list(range(3, 26))
    assert records[0]["probability_percent"] == 3.0
    assert records[0]["scan_parameter_p0"] == .3
    assert all(r["t2_us2"] is None for r in records)
    req.grid = 9
    req.phase_points = 32
    result = compute(loader, req)
    assert any("different P0" in warning for warning in result["warnings"])
    with zipfile.ZipFile(io.BytesIO(package(req, result))) as archive:
        assert json.loads(archive.read("configuration.json"))["scan_mode"] == "link"
        assert "scan_parameter_p0" in archive.read("shots.csv").decode("utf-8")

def test_simulated_parameters_and_report():
    rng=np.random.default_rng(1234)
    values=45+22*np.cos(rng.uniform(0,np.pi,300))+rng.normal(0,.5,300)
    result=infer(values,{"A":[20,24],"B":[43,47],"sigma":[.2,1.2]},15,128)
    assert abs(result["parameters"]["A"]["mean"]-22)<.5
    assert abs(result["parameters"]["B"]["mean"]-45)<.5
    assert abs(result["parameters"]["sigma"]["mean"]-.5)<.2
    assert result["parameters"]["contrast_2A"]["mean"]==2*result["parameters"]["A"]["mean"]
    result.update(records=[{"step":i,"timestamp":None,"t2_us2":100,"probability_percent":float(v),"included":True,"exclusion_reason":""} for i,v in enumerate(values)],count=len(values),excluded=0,node="master",mode="phase_noise")
    req=FringeRequest(year="2026",month="01",day="01",run_id="test",t2=100,grid=15,phase_points=128)
    with zipfile.ZipFile(io.BytesIO(package(req,result))) as z:
        assert set(z.namelist())=={"report.pdf","configuration.json","shots.csv"}
        assert z.read("report.pdf").startswith(b"%PDF")
        config=json.loads(z.read("configuration.json"))
        assert config["resolved_bounds"]==result["bounds"]
        assert config["scan_mode"] == "phase_noise"
        import hashlib
        assert config["shots_csv_sha256"]==hashlib.sha256(z.read("shots.csv")).hexdigest()

def test_rejects_invalid_noise_bounds():
    with pytest.raises(ValueError): infer(np.arange(25),{"A":[1,2],"B":[1,2],"sigma":[0,2]},9,32)


def test_shared_api_preview_and_export(tmp_path):
    import asyncio
    from fastapi import HTTPException
    from fastapi.routing import APIRoute
    from app.api.bayesian_fringe_routes import build_bayesian_fringe_router
    rng=np.random.default_rng(42)
    points=[{"step":i,"phase_noise_t2_us2":100,"phase_noise_science_shot":i+1,
             "intf_p1":float(45+22*np.cos(p)+e)}
            for i,(p,e) in enumerate(zip(rng.uniform(0,np.pi,40),rng.normal(0,1,40)))]
    loader=Loader(points); loader.get_run_dir=lambda *args: tmp_path
    routes={route.path:route.endpoint for route in build_bayesian_fringe_router(lambda:loader,APIRoute).routes}
    req=FringeRequest(year="2026",month="01",day="01",run_id="run01",t2=100,grid=9,phase_points=32)
    assert len(asyncio.run(routes['/archive/bayesian-fringe/shots'](req))['records'])==40
    result=asyncio.run(routes['/archive/bayesian-fringe'](req))
    assert len(result['model_x'])==400
    req.input_stamp=result['input_stamp']
    response=asyncio.run(routes['/archive/bayesian-fringe/report'](req))
    assert response.body[:2]==b'PK'
    (tmp_path/'changed').write_text('new')
    with pytest.raises(HTTPException,match='changed'):
        asyncio.run(routes['/archive/bayesian-fringe/report'](req))
