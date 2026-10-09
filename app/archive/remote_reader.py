"""Executed on a controller with python3 via SSH; no third-party dependencies."""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import urllib.request


def entries(root, hashes=False):
    result = []
    def fail(error):
        raise error
    for folder, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        if any((Path(folder) / d).is_symlink() for d in dirs):
            raise ValueError('Symbolic link directory in archive')
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (Path(folder) / d).is_symlink())
        for name in sorted(names):
            path = Path(folder) / name
            if path.is_symlink():
                raise ValueError('Symbolic links cannot be archived: ' + str(path))
            if name.endswith(('.tmp', '.part')):
                continue
            stat = path.stat()
            record = {'path': path.relative_to(root).as_posix(), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if hashes:
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(chunk)
                record['sha256'] = digest.hexdigest()
            result.append(record)
    return sorted(result, key=lambda row: row['path'])


def main():
    request = json.loads(sys.argv[1])
    root = Path(request['source_path'])
    operation = request['operation']
    if operation == 'collection':
        source = Path(request['collection_path'])
        if not source.exists():
            return {'exists': False}
        with tempfile.TemporaryDirectory(prefix='miga_collection_') as temp:
            target = Path(temp) / 'snapshot.sqlite3'
            with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as reader:
                with sqlite3.connect(target) as writer:
                    reader.backup(writer)
            return {'exists': True, 'data': base64.b64encode(target.read_bytes()).decode('ascii')}
    if not root.is_dir():
        raise ValueError('Data_log does not exist')
    if operation == 'manifest':
        relative = Path(request['run'])
        if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 4:
            raise ValueError('Invalid run path')
        run = root / relative
        before = entries(run)
        hashed = entries(run, hashes=True)
        if before != entries(run):
            raise ValueError('Run changed during checksum calculation; retry later')
        return {'files': hashed}
    running = None
    try:
        with urllib.request.urlopen(request['controller_url'].rstrip('/') + '/experiment/status', timeout=3) as response:
            status = json.load(response)
            running = status.get('is_running')
            if not isinstance(running, bool):
                running = None
    except Exception:
        pass
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    runs, deferred = [], []
    for run in sorted(root.glob('*/*/*/run*')):
        if not run.is_dir():
            continue
        relative = run.relative_to(root).as_posix()
        parts = relative.split('/')
        if len(parts) != 4 or not all(p.isdigit() for p in parts[:3]):
            continue
        files = entries(run)
        latest = max((item['mtime_ns'] for item in files), default=0) / 1e9
        closed = (run / 'archive_complete.json').is_file()
        # Old controllers have no completion marker. Avoid today's archives
        # unless their status endpoint confirms the experiment is idle.
        if time.time() - latest < max(60,int(request.get('quiet_seconds',60))) or (not closed and relative.startswith(today) and running is not False):
            deferred.append(relative)
            continue
        if not (run / 'config.json').is_file() or not (run / 'results.csv').is_file():
            deferred.append(relative)
            continue
        sync_status = None
        manifest_path = run / 'sync_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            sync_status = (manifest.get('archive_replication') or {}).get('status', 'unknown')
        runs.append({'path': relative, 'files': files, 'sync_status': sync_status})
    return {'runs': runs, 'deferred': deferred, 'controller_running': running}


if __name__ == '__main__':
    print(json.dumps(main()))
