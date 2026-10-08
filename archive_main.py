"""Hardware-free entry point for the long-running MIGA Archive Server."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydantic import Field
from starlette.concurrency import run_in_threadpool
import threading
import re
from app.archive.backup import BackupService
from app.archive.repository import ArchiveRepository

from app.archive.configuration import ArchiveServerConfiguration


app = FastAPI(title="MIGA Archive Server")
configuration = ArchiveServerConfiguration()
backup = BackupService(configuration)
repository = ArchiveRepository(configuration)
STATIC_DIR = Path(__file__).resolve().parent / "static"
from app.archive.ui_context import configure as configure_archive_ui
from app.archive.ui_routes import router as archive_ui_router
configure_archive_ui(repository)
app.include_router(archive_ui_router, prefix='/archive-server/view/{device_id}')


@app.get('/archive-server/view/{device_id}/archive.html')
async def device_archive_page(device_id: str):
    repository.device_root(device_id)
    return FileResponse(STATIC_DIR / 'archive.html')


@app.get('/archive-server-ui.js')
async def archive_ui_script():
    return FileResponse(STATIC_DIR / 'archive-server-ui.js', media_type='application/javascript')


@app.get('/plot-publication.js')
async def publication_plot_script():
    return FileResponse(STATIC_DIR / 'plot-publication.js', media_type='application/javascript')


class ArchiveRootRequest(BaseModel):
    path: str


class ArchiveDeviceRequest(BaseModel):
    device_id: str
    name: str = ""
    role: str = "standalone"
    transport: str = "ssh"
    host: str = ""
    ssh_user: str = ""
    source_path: str
    collection_path: str = ""
    enabled: bool = True
    identity_file: str = "~/.ssh/miga_archive_ed25519"
    controller_url: str = "http://127.0.0.1:8000"
    automatic_backup: bool = False


@app.get("/archive-server/status")
async def archive_server_status():
    current = configuration.load()
    root = current.get("archive_root")
    return {
        "mode": "archive", "configured": bool(root) and not current.get("configuration_error"),
        "configuration": current, "storage": configuration.inspect_root(root) if root else None,
    }


@app.post("/archive-server/setup/inspect")
async def inspect_archive_root(request: ArchiveRootRequest):
    if not request.path.startswith('/'):
        raise ValueError('Enter a local absolute mount path, for example /mnt/miga-archive')
    return await run_in_threadpool(configuration.inspect_root, request.path)


@app.post("/archive-server/setup/initialize")
async def initialize_archive_root(request: ArchiveRootRequest):
    try:
        return await run_in_threadpool(configuration.initialize, request.path)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/archive-server/devices")
async def list_archive_devices():
    return {"devices": list((configuration.load().get("devices") or {}).values())}


@app.post("/archive-server/devices")
async def register_archive_device(request: ArchiveDeviceRequest):
    try:
        with configuration.lock:
            configuration.checked_root()
            if request.device_id in configuration.load().get("devices", {}):
                raise ValueError("Device already exists. Use Edit to update it")
            return configuration.register_device(request.dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/setup")
async def archive_setup_page():
    return FileResponse(STATIC_DIR / "archive-setup.html")


@app.on_event('startup')
def start_scheduler():
    threading.Thread(target=backup.schedule_loop, daemon=True).start()


@app.on_event('shutdown')
def stop_scheduler():
    backup.stop.set()
    backup.executor.shutdown(wait=True)


@app.exception_handler(ValueError)
async def invalid_request(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content={'detail': str(exc)})


@app.exception_handler(FileNotFoundError)
async def missing_resource(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content={'detail': str(exc)})


class EnabledRequest(BaseModel):
    enabled: bool


class AnalysisRequest(BaseModel):
    year: str
    month: str
    day: str
    run_id: str
    node_id: str = ''
    settings: dict = Field(default_factory=dict)
    name: str = ''
    save: bool = False


def ensure_idle(device_id):
    if any(j['device_id'] == device_id and j['status'] in {'queued', 'running'} for j in backup.jobs()):
        raise ValueError('Wait for this device backup to finish')


@app.put('/archive-server/devices/{device_id}')
async def edit_device(device_id: str, request: ArchiveDeviceRequest):
    previous = backup.device(device_id)
    ensure_idle(device_id)
    if device_id != request.device_id:
        raise ValueError('Device ID cannot change')
    if any(previous.get(k) != getattr(request, k) for k in ('host', 'source_path')):
        runs = repository.device_root(device_id) / 'runs'
        if runs.exists() and any(runs.iterdir()):
            raise ValueError('Register a new device ID to change the source of a backed-up device')
    with configuration.lock:
        configuration.checked_root()
        return configuration.register_device(request.dict())


@app.patch('/archive-server/devices/{device_id}')
async def enable_device(device_id: str, request: EnabledRequest):
    return configuration.set_enabled(device_id, request.enabled)


@app.delete('/archive-server/devices/{device_id}')
async def remove_device(device_id: str):
    ensure_idle(device_id)
    configuration.remove_device(device_id)
    return {'removed': device_id, 'archive_data_preserved': True}


@app.post('/archive-server/devices/{device_id}/test')
async def test_device(device_id: str):
    return await run_in_threadpool(backup.test, device_id)


@app.post('/archive-server/devices/{device_id}/backup')
async def backup_device(device_id: str):
    return backup.start(device_id)


@app.get('/archive-server/jobs')
async def backup_jobs():
    return {'jobs': backup.jobs()}


@app.post('/archive-server/devices/{device_id}/verify')
async def verify_device(device_id: str):
    ensure_idle(device_id)
    return await run_in_threadpool(backup.verify, device_id)


@app.get('/archive-server/archive/{device_id}/tree')
async def tree(device_id: str):
    return await run_in_threadpool(repository.loader(device_id).get_archive_tree)


@app.get('/archive-server/archive/{device_id}/collections')
async def collections(device_id: str):
    return await run_in_threadpool(repository.collections, device_id)


@app.get('/archive-server/archive/{device_id}/load/{year}/{month}/{day}/{run_id}')
async def load(device_id: str, year: str, month: str, day: str, run_id: str, node_id: str = '', revision_id: str = ''):
    repository.reference(device_id, year, month, day, run_id)
    return await run_in_threadpool(repository.loader(device_id, [year, month, day, run_id], revision_id).load_run, year, month, day, run_id, node_id or None)


@app.get('/archive-server/archive/{device_id}/revisions/{year}/{month}/{day}/{run_id}')
async def raw_revisions(device_id: str, year: str, month: str, day: str, run_id: str):
    return {'revisions': repository.revisions(device_id, [year, month, day, run_id])}


@app.get('/archive-server/archive/{device_id}/waveform/{year}/{month}/{day}/{run_id}/{step}')
async def waveform(device_id: str, year: str, month: str, day: str, run_id: str, step: int, node_id: str = ''):
    repository.reference(device_id, year, month, day, run_id)
    return await run_in_threadpool(repository.loader(device_id).load_waveform, year, month, day, run_id, step, node_id or None)


@app.get('/archive-server/archive/{device_id}/analyses/{year}/{month}/{day}/{run_id}')
async def analyses(device_id: str, year: str, month: str, day: str, run_id: str):
    return {'analyses': repository.analyses(device_id, [year, month, day, run_id])}


@app.post('/archive-server/archive/{device_id}/recalculate')
async def recalculate(device_id: str, request: AnalysisRequest):
    return await run_in_threadpool(repository.recalculate, device_id, [request.year, request.month, request.day, request.run_id], request.settings, request.node_id, request.name, request.save)


@app.get('/archive-server/archive/{device_id}/analysis/{year}/{month}/{day}/{run_id}/{analysis_id}')
async def analysis_result(device_id: str, year: str, month: str, day: str, run_id: str, analysis_id: str):
    if not re.fullmatch(r'[a-f0-9]{32}', analysis_id):
        raise ValueError('Invalid analysis ID')
    repository.reference(device_id, year, month, day, run_id)
    path = repository.analysis_dir(device_id, [year, month, day, run_id]) / analysis_id / 'result.json'
    if not path.is_file():
        raise FileNotFoundError('Analysis not found')
    return FileResponse(path, media_type='application/json', filename='analysis_' + analysis_id + '.json')


@app.get('/')
async def dashboard():
    return FileResponse(STATIC_DIR / 'archive-server.html')
