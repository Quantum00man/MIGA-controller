"""Request-scoped archive UI adapters and copy-on-write analysis metadata.

Only immutable waveform directories are shared with the original archive.
All editable metadata is copied into a new version before a handler can write.
"""
import asyncio
from contextvars import ContextVar
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool
from app.archive.backup import atomic_json, now
from app.archive.ui_settings import ServerSettings
from app.core.archive_collection_store import ArchiveCollectionStore
from app.core.archive_audit import ArchiveAuditService

_scope = ContextVar('archive_ui_scope')
_repository = None
_locks = {}
_audits = {}


class DeviceRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()
        async def handler(request):
            # Commit the new immutable version before sending the HTTP response.
            async with asynccontextmanager(request_context)(request):
                return await original(request)
        return handler


def configure(repository):
    global _repository
    _repository = repository


class scoped:
    def __init__(self, key):
        self.key = key

    def __getattr__(self, name):
        return getattr(_scope.get()[self.key], name)


class Collections:
    def __init__(self, repository, device_id):
        self.repository, self.device_id = repository, device_id
        # SQLite must stay on local storage, not CIFS/NFS.
        local = repository.configuration.path.parent / 'ui' / device_id
        local.mkdir(parents=True, exist_ok=True)
        self.store = ArchiveCollectionStore(repository.device_root(device_id) / 'runs', local / 'collections.sqlite3')
        self.snapshots = repository.configuration.checked_root() / 'derived' / device_id / 'collections'
        pointer = self.snapshots / 'latest.json'
        if not self.store.snapshot()['folders'] and pointer.exists():
            filename = json.loads(pointer.read_text())['file']
            if not re.fullmatch(r'[a-f0-9]{32}\.sqlite3', filename):
                raise ValueError('Invalid server Collection snapshot')
            shutil.copyfile(self.snapshots / filename, self.store.database_path)

    def snapshot(self):
        own = self.store.snapshot()
        source = self.repository.collections(self.device_id)
        if source['folders']:
            own['folders'].append({'id': -1, 'parent_id': 0, 'name': 'Source Collections (read-only)', 'item_count': 0})
        for folder in source['folders']:
            row = dict(folder)
            row.update(id=-int(folder['id'])-1, parent_id=-int(folder['parent_id'])-1 if folder['parent_id'] else -1,
                       item_count=sum(f['folder_id'] == folder['id'] for f in source['favorites']), read_only=True)
            own['folders'].append(row)
        for favorite in source['favorites']:
            row = dict(favorite)
            row.update(id=-int(favorite['id']), folder_id=-int(favorite['folder_id'])-1, read_only=True,
                       preview=json.loads(favorite.get('preview_json') or '{}'),
                       display_name=favorite.get('alias') or favorite.get('original_label') or favorite['run_id'],
                       source_ref='/'.join(str(favorite[key]) for key in ('year', 'month', 'day', 'run_id')))
            own['favorites'].append(row)
        return own

    def __getattr__(self, name):
        operation = getattr(self.store, name)
        def mutate(*args, **kwargs):
            ids = []
            if name in {'update_folder', 'delete_folder', 'update_favorite', 'delete_favorite', 'create_favorite'}:
                ids.extend(args[:1])
            if name == 'create_folder':
                ids.extend(args[1:2])
            if name == 'update_folder':
                ids.extend(args[2:3])
            if name == 'batch_favorites':
                ids.extend(args[1]); ids.extend(args[2:3])
            ids.extend(kwargs.get(k) for k in ('folder_id', 'parent_id') if kwargs.get(k) is not None)
            if any(value is not None and int(value) < 0 for value in ids):
                raise HTTPException(403, 'Imported Collections are read-only; use a server Collection')
            result = operation(*args, **kwargs)
            self.snapshots.mkdir(parents=True, exist_ok=True)
            filename = uuid.uuid4().hex + '.sqlite3'
            temporary = self.store.database_path.parent / filename
            import sqlite3
            with self.store._connect() as source, sqlite3.connect(temporary) as destination:
                source.backup(destination)
            try:
                shutil.copyfile(temporary, self.snapshots / filename)
                atomic_json(self.snapshots / 'latest.json', {'file': filename, 'created_at': now()})
            finally:
                temporary.unlink(missing_ok=True)
            return result
        return mutate


class OfflineSync:
    def __init__(self, loader, settings):
        self.loader, self.settings = loader, settings

    def archive_local_node_id(self, root):
        return self.loader._resolve_archive_node_identity(root, None)[1].get('resolved_node') or 'master'

    def get_archive_node_phase_calibrations(self, root, node_id, *unused):
        manifest = self.loader._load_sync_manifest_payload(root) or {}
        store = self.loader._read_archive_phase_reference_store(root)
        override = (store.get('overrides') or {}).get(node_id) or {}
        calibration = override.get('calibration') or (store.get('original_calibrations') or {}).get(node_id) or self.loader._archive_phase_node_calibration(root, manifest, node_id, self.settings.get_active_bragg_phase_calibration())
        return {'node_id': node_id, 'source': 'backup_snapshot',
                'calibrations': ([calibration] if calibration else []) + self.settings.get_bragg_phase_calibrations(),
                'active': calibration}


def version_root(repository, device_id, reference):
    return repository.analysis_dir(device_id, reference) / 'ui-versions'


def versions(repository, device_id, reference):
    repository.reference(device_id, *reference)
    root = version_root(repository, device_id, reference)
    rows = [json.loads(p.read_text()) for p in root.glob('*/provenance.json')]
    return sorted(rows, key=lambda r: r['created_at'], reverse=True)


def resolve(repository, device_id, reference, selection='latest'):
    original = repository.reference(device_id, *reference)
    if selection == 'original':
        return original
    root = version_root(repository, device_id, reference)
    if selection == 'latest':
        pointer = root / 'latest.json'
        if not pointer.exists():
            return original
        selection = json.loads(pointer.read_text())['id']
    if not re.fullmatch(r'[a-f0-9]{32}', selection):
        raise ValueError('Invalid analysis version')
    path = root / selection / 'run'
    if not path.is_dir() or not (path.parent / 'provenance.json').is_file():
        raise FileNotFoundError('Analysis version not found')
    return path


def clone_metadata(source, destination, original):
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        target = destination / item.name
        if item.name == 'waveforms' and item.is_dir():
            # The loader reads waveforms; none of the UI mutation APIs writes them.
            raw = original / 'waveforms'
            if not raw.is_dir() or raw.is_symlink():
                raise ValueError('Invalid original waveform directory')
            target.symlink_to(raw.resolve(), target_is_directory=True)
        elif item.is_symlink():
            raise ValueError('Unexpected link in archive metadata')
        elif item.is_dir():
            clone_metadata(item, target, original / item.name)
        elif item.is_file():
            shutil.copyfile(item, target)


def writable_operation(request, body):
    path = request.url.path
    if request.method == 'GET':
        return False
    return any(key in path for key in (
        '/archive/overwrite', '/archive/phase-reference-override', '/archive/phase-analysis/sync/', '/archive/sync-differential-fit',
        '/archive/sync-phase-calibration-optimization/', '/archive/sync-analysis-copies/save',
    )) or ('/archive/intf-alpha/reanalyze' in path and body.get('save', False))


async def request_context(request: Request):
    repository = _repository
    device_id = request.path_params['device_id']
    repository.device_root(device_id)
    selection = request.headers.get('x-archive-version') or request.query_params.get('server_version', 'latest')
    body = await request.json() if request.method in {'POST', 'PATCH', 'PUT'} and await request.body() else {}
    if not isinstance(body, dict):
        raise ValueError('Request body must be an object')
    mutable = writable_operation(request, body)
    lock = _locks.setdefault(device_id, asyncio.Lock())
    if request.method != 'GET':
        await lock.acquire()
    token = None
    directory = None
    try:
        loader = repository.loader(device_id)
        loader._get_run_dir = lambda *ref: resolve(repository, device_id, list(ref), selection)
        settings = ServerSettings(repository.configuration.checked_root() / 'derived' / device_id / 'settings')
        collections = await run_in_threadpool(Collections, repository, device_id)
        if device_id not in _audits:
            _audits[device_id] = ArchiveAuditService(repository.device_root(device_id) / 'runs', collections,
                repository.configuration.checked_root() / 'derived' / device_id / 'audit-reports')
        context = {'loader': loader, 'settings': settings, 'collections': collections,
                   'audit': _audits[device_id], 'offline_sync': OfflineSync(loader, settings), 'writable': mutable}
        if mutable:
            ref = [str(body.get(k) or request.path_params.get(k) or '') for k in ('year', 'month', 'day', 'run_id')]
            original = repository.reference(device_id, *ref)
            source = resolve(repository, device_id, ref, selection)
            version_id = uuid.uuid4().hex
            directory = version_root(repository, device_id, ref) / version_id
            await run_in_threadpool(clone_metadata, source, directory / 'run', original)
            loader._get_run_dir = lambda *requested: directory / 'run' if list(requested) == ref else resolve(repository, device_id, list(requested), selection)
        token = _scope.set(context)
        yield
        if mutable:
            software = await run_in_threadpool(subprocess.run, ['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            atomic_json(directory / 'provenance.json', {'id': version_id, 'created_at': now(), 'device_id': device_id,
                'reference': ref, 'operation': request.url.path.split('/archive/', 1)[-1],
                'parent': source.parent.name if source != original else 'original', 'schema_version': 1,
                'archive_uuid': repository.configuration.load().get('archive_uuid'),
                'node_id': body.get('node_id'), 'request': body,
                'software_commit': software.stdout.strip() if software.returncode == 0 else 'unknown'})
            atomic_json(directory.parent / 'latest.json', {'id': version_id})
    finally:
        if directory is not None and not (directory / 'provenance.json').exists():
            await run_in_threadpool(shutil.rmtree, directory, True)
        if token is not None:
            _scope.reset(token)
        if request.method != 'GET':
            lock.release()
