"""Hardware-free entry point for the long-running MIGA Archive Server."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.archive.configuration import ArchiveServerConfiguration


app = FastAPI(title="MIGA Archive Server")
configuration = ArchiveServerConfiguration()
STATIC_DIR = Path(__file__).resolve().parent / "static"


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
    return configuration.inspect_root(request.path)


@app.post("/archive-server/setup/initialize")
async def initialize_archive_root(request: ArchiveRootRequest):
    try:
        return configuration.initialize(request.path)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/archive-server/devices")
async def list_archive_devices():
    return {"devices": list((configuration.load().get("devices") or {}).values())}


@app.post("/archive-server/devices")
async def register_archive_device(request: ArchiveDeviceRequest):
    try:
        return configuration.register_device(request.dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/")
@app.get("/setup")
async def archive_setup_page():
    return FileResponse(STATIC_DIR / "archive-setup.html")
