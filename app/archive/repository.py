"""Device-isolated immutable archives and append-only analysis versions."""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid
import subprocess
from app.archive.backup import atomic_json, now


class ArchiveRepository:
    def __init__(self, configuration):
        self.configuration = configuration

    def device_root(self, device_id):
        if device_id not in self.configuration.load().get('devices', {}):
            raise FileNotFoundError('Device is not registered')
        return self.configuration.checked_root() / 'devices' / device_id

    def loader(self, device_id, reference=None, revision_id=''):
        from app.core.data_loader import DataLoader
        loader = DataLoader()
        loader.base_dir = self.device_root(device_id) / 'runs'
        loader.analysis_device_id = device_id
        loader.analysis_directory = lambda reference: self.analysis_dir(device_id, reference)
        loader.original_directory = lambda reference: self.reference(device_id, *reference)
        if revision_id:
            if not reference or not re.fullmatch(r'[a-f0-9]{64}', revision_id):
                raise ValueError('Invalid raw revision')
            self.reference(device_id, *reference)
            revision = self.device_root(device_id) / 'revisions' / Path(*reference) / revision_id
            if not revision.is_dir():
                raise FileNotFoundError('Raw revision not found')
            original_resolver = loader._get_run_dir
            loader._get_run_dir = lambda year, month, day, run_id: revision if [year, month, day, run_id] == list(reference) else original_resolver(year, month, day, run_id)
        return loader

    def revisions(self, device_id, reference):
        self.reference(device_id, *reference)
        directory = self.device_root(device_id) / 'revisions' / Path(*reference)
        return [p.name for p in sorted(directory.glob('*')) if p.is_dir() and re.fullmatch(r'[a-f0-9]{64}', p.name)]

    def reference(self, device_id, year, month, day, run_id):
        if not re.fullmatch(r'\d{4}', year) or not re.fullmatch(r'\d{2}', month) or not re.fullmatch(r'\d{2}', day):
            raise ValueError('Invalid archive date')
        if not re.fullmatch(r'run[A-Za-z0-9_.-]+', run_id) or '..' in run_id:
            raise ValueError('Invalid run ID')
        path = self.device_root(device_id) / 'runs' / year / month / day / run_id
        if not path.is_dir():
            raise FileNotFoundError('Run is not backed up')
        return path

    def collections(self, device_id):
        imports = self.device_root(device_id) / 'collection-imports'
        latest = imports / 'latest.json'
        if not latest.exists():
            return {'folders': [], 'favorites': [], 'read_only': True}
        filename = json.loads(latest.read_text())['file']
        if Path(filename).name != filename:
            raise ValueError('Invalid Collection snapshot')
        with sqlite3.connect((imports / filename).as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            folders = [dict(row) for row in db.execute('SELECT * FROM collection_folders')]
            favorites = [dict(row) for row in db.execute('SELECT * FROM archive_favorites')]
        for item in favorites:
            try:
                self.reference(device_id, item['year'], item['month'], item['day'], item['run_id'])
                item['integrity'] = 'ok'
            except (ValueError, FileNotFoundError):
                item['integrity'] = 'missing'
        return {'folders': folders, 'favorites': favorites, 'read_only': True}

    def analysis_dir(self, device_id, reference):
        identity = hashlib.sha256((device_id + '/' + '/'.join(reference)).encode()).hexdigest()
        return self.configuration.checked_root() / 'derived' / device_id / identity

    def analyses(self, device_id, reference):
        self.reference(device_id, *reference)
        return sorted([json.loads(path.read_text()) for path in self.analysis_dir(device_id, reference).glob('*/provenance.json')], key=lambda row: row['created_at'], reverse=True)

    def recalculate(self, device_id, reference, settings, node_id='', name='', save=False):
        self.reference(device_id, *reference)
        result = self.loader(device_id).recalculate_run(*reference, settings, max_points=None, node_id=node_id or None)
        if not save:
            return result
        raw_root = self.reference(device_id, *reference)
        digest = hashlib.sha256()
        for path in sorted(raw_root.rglob('*')):
            if path.is_file():
                digest.update(path.relative_to(raw_root).as_posix().encode())
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
        analysis_id = uuid.uuid4().hex
        directory = self.analysis_dir(device_id, reference) / analysis_id
        directory.mkdir(parents=True, exist_ok=False)
        provenance = {'id': analysis_id, 'name': name.strip() or 'Reanalysis', 'created_at': now(), 'device_id': device_id, 'reference': reference, 'node_id': node_id, 'raw_sha256': digest.hexdigest(), 'settings': settings, 'schema_version': 1}
        version = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        provenance['software_commit'] = version.stdout.strip() if version.returncode == 0 else 'unknown'
        atomic_json(directory / 'result.json', result)
        atomic_json(directory / 'provenance.json', provenance)
        return {'analysis': provenance, 'result': result}
