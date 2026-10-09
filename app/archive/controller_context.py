"""Controller Archive copy-on-write isolation, without changing acquisition loaders."""
import asyncio
from contextvars import ContextVar
import hashlib
from pathlib import Path
import re
import shutil
import uuid
from fastapi import HTTPException
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool
from app.archive.backup import atomic_json, now
from app.archive.ui_context import clone_metadata, resolve, version_root, versions, writable_operation
from app.core.analysis_store import software_commit
import config

run_resolver = ContextVar('controller_archive_run_resolver', default=None)
_lock = asyncio.Lock()


class ControllerRepository:
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir if base_dir is not None else config.DATA_BASE_DIR)

    def reference(self, device_id, year, month, day, run_id):
        if not (re.fullmatch(r'\d{4}', year) and re.fullmatch(r'\d{2}', month) and re.fullmatch(r'\d{2}', day)
                and re.fullmatch(r'run[A-Za-z0-9_.-]+', run_id) and '..' not in run_id):
            raise ValueError('Invalid run reference')
        path = self.base_dir.joinpath(year, month, day, run_id)
        if not path.is_dir():
            raise FileNotFoundError('Run not found')
        return path

    def analysis_dir(self, device_id, reference):
        self.reference(device_id, *reference)
        identity = hashlib.sha256('/'.join(reference).encode()).hexdigest()
        return self.base_dir.parent / (self.base_dir.name + '_analysis') / identity


class ControllerArchiveRoute(APIRoute):
    def get_route_handler(self):
        original_handler = super().get_route_handler()
        async def handler(request):
            if not request.url.path.startswith('/archive/'):
                return await original_handler(request)
            repository = ControllerRepository()
            selection = request.headers.get('x-archive-version') or request.query_params.get('server_version', 'original')
            body = await request.json() if request.method in {'POST', 'PATCH', 'PUT'} and await request.body() else {}
            mutable = writable_operation(request, body)
            locked = request.method != 'GET'
            if locked:
                await _lock.acquire()
            directory = None
            token = None
            try:
                resolver = lambda *ref: resolve(repository, 'controller', list(ref), selection)
                if mutable:
                    ref = [str(body.get(key) or request.path_params.get(key) or '') for key in ('year', 'month', 'day', 'run_id')]
                    raw = repository.reference('controller', *ref)
                    source = resolver(*ref)
                    identifier = uuid.uuid4().hex
                    directory = version_root(repository, 'controller', ref) / identifier
                    await run_in_threadpool(clone_metadata, source, directory / 'run', raw)
                    resolver = lambda *requested: directory / 'run' if list(requested) == ref else resolve(repository, 'controller', list(requested), selection)
                token = run_resolver.set((repository.base_dir.resolve(), resolver))
                response = await original_handler(request)
                if mutable and response.status_code < 400:
                    atomic_json(directory / 'provenance.json', {
                        'id': identifier, 'created_at': now(), 'device_id': 'controller', 'reference': ref,
                        'operation': request.url.path.split('/archive/', 1)[-1],
                        'parent': source.parent.name if source != raw else 'original',
                        'schema_version': 1, 'node_id': body.get('node_id'), 'request': body,
                        'software_commit': await run_in_threadpool(software_commit)})
                    atomic_json(directory.parent / 'latest.json', {'id': identifier})
                elif directory:
                    await run_in_threadpool(shutil.rmtree, directory)
                return response
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(400 if isinstance(exc, ValueError) else 404, str(exc))
            finally:
                if token is not None:
                    run_resolver.reset(token)
                if directory and not (directory / 'provenance.json').exists():
                    await run_in_threadpool(shutil.rmtree, directory, True)
                if locked:
                    _lock.release()
        return handler
