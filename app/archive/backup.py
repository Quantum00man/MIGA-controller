"""Persistent pull jobs. Published archives are immutable; changes become revisions."""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def fingerprint(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


class BackupService:
    def __init__(self, configuration):
        self.configuration = configuration
        self.state_dir = configuration.path.parent / 'jobs'
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.maintenance = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='archive-pull')
        self.stop = threading.Event()
        self.control_dir = configuration.path.parent / 'ssh-control'
        self.control_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for path in self.state_dir.glob('*.json'):
            job = json.loads(path.read_text())
            if job['status'] in {'queued', 'running'}:
                job.update(status='interrupted', error='Server restarted; run backup again to resume', finished_at=now())
                atomic_json(path, job)

    def device(self, device_id):
        record = self.configuration.load().get('devices', {}).get(device_id)
        if not record:
            raise FileNotFoundError('Device is not registered')
        return record

    def ssh(self, device):
        return ['ssh', '-i', str(Path(device.get('identity_file') or '~/.ssh/miga_archive_ed25519').expanduser()),
                '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
                '-o', 'ControlMaster=auto', '-o', 'ControlPersist=60',
                '-o', 'ControlPath=' + str(self.control_dir / '%C'),
                device['ssh_user'] + '@' + device['host']]

    def read_remote(self, device, operation, **extra):
        request = {**device, 'operation': operation, **extra}
        reader = Path(__file__).with_name('remote_reader.py').read_text()
        command = self.ssh(device) + ['python3 - ' + shlex.quote(json.dumps(request))]
        if device.get('transport') == 'local':
            command = ['python3', '-', json.dumps(request)]
        result = subprocess.run(command, input=reader, text=True, capture_output=True, timeout=1800)
        if result.returncode:
            raise ValueError(result.stderr[-3000:] or 'Controller read failed')
        return json.loads(result.stdout)

    def test(self, device_id):
        device = self.device(device_id)
        result = self.read_remote(device, 'inventory')
        return {'ok': True, 'eligible_runs': len(result['runs']), 'deferred_runs': len(result['deferred']),
                'controller_running': result['controller_running']}

    def jobs(self):
        return sorted((json.loads(path.read_text()) for path in self.state_dir.glob('*.json')),
                      key=lambda row: row['started_at'], reverse=True)

    def start(self, device_id):
        with self.lock:
            if self.maintenance:
                raise ValueError('Server is reloading; wait for it to reconnect before starting backups')
            device = self.device(device_id)
            if not device.get('enabled', True):
                raise ValueError('Enable this device before starting a backup')
            if any(j['device_id'] == device_id and j['status'] in {'queued', 'running'} for j in self.jobs()):
                raise ValueError('A backup for this device is already queued or running')
            root = self.configuration.checked_root()
            for other_id, other in self.configuration.load().get('devices', {}).items():
                if other_id != device_id and other.get('enabled', True) and all(other.get(k) == device.get(k) for k in ('host', 'ssh_user', 'source_path')):
                    raise ValueError('Duplicate enabled source: disable ' + other_id + ' first')
            job = {'id': uuid.uuid4().hex, 'device_id': device_id, 'status': 'queued', 'started_at': now(),
                   'finished_at': None, 'copied': 0, 'skipped': 0, 'failed': [], 'deferred': [], 'sync_warnings': [], 'total': 0, 'current_run': '', 'phase': 'queued', 'phase_started_at': now(), 'verified_files': 0, 'run_files': 0, 'run_bytes': 0, 'error': ''}
            atomic_json(self.state_dir / (job['id'] + '.json'), job)
            self.executor.submit(self.run, job, device, root)
            return job

    def save_job(self, job):
        with self.lock:
            atomic_json(self.state_dir / (job['id'] + '.json'), job)

    def run(self, job, device, root):
        try:
            job['status'] = 'running'
            job.update(phase='scanning', phase_started_at=now())
            self.save_job(job)
            inventory = self.read_remote(device, 'inventory')
            job.update(total=len(inventory['runs']), deferred=inventory['deferred'])
            job['sync_warnings'] = [r['path'] for r in inventory['runs'] if r.get('sync_status') not in {None, 'complete'}]
            device_root = root / 'devices' / device['device_id']
            for row in inventory['runs']:
                if self.stop.is_set():
                    raise ValueError('Server is stopping; retry to resume')
                self.configuration.checked_root()
                relative = row['path']
                job['current_run'] = relative
                job.update(run_files=len(row['files']), run_bytes=sum(f['size'] for f in row['files']), verified_files=0)
                self.save_job(job)
                try:
                    def progress(phase, verified=None):
                        if job.get('phase') != phase:
                            job.update(phase=phase, phase_started_at=now())
                        if verified is not None:
                            job['verified_files'] = verified
                        self.save_job(job)
                    self.copy_run(device, device_root, root, row, progress=progress)
                    job['copied'] += 1
                except AlreadyArchived:
                    job['skipped'] += 1
                except Exception as exc:
                    job['failed'].append({'run': relative, 'error': str(exc)})
                self.save_job(job)
            self.configuration.checked_root()
            job.update(phase='collections', phase_started_at=now())
            self.save_job(job)
            snapshot = self.read_remote(device, 'collection')
            if snapshot.get('exists'):
                content = base64.b64decode(snapshot['data'], validate=True)
                digest = hashlib.sha256(content).hexdigest()
                path = device_root / 'collection-imports' / (digest + '.sqlite3')
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp = path.with_suffix('.part')
                    temp.write_bytes(content)
                    with sqlite3.connect(temp.as_uri() + '?mode=ro', uri=True) as db:
                        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                            raise ValueError('Collection snapshot integrity check failed')
                    temp.replace(path)
                atomic_json(device_root / 'collection-imports' / 'latest.json', {'file': path.name, 'received_at': now()})
            job['status'] = 'incomplete' if job['failed'] or job['deferred'] or job['sync_warnings'] else 'complete'
        except Exception as exc:
            job.update(status='failed', error=str(exc))
        finally:
            job.update(finished_at=now(), current_run='', phase=job['status'])
            self.save_job(job)

    def verify_files(self, directory, manifest, progress=None):
        def check(item):
            path = directory / item['path']
            if path.is_symlink():
                raise ValueError('Symlink in staging')
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != item['sha256']:
                raise ValueError('Checksum mismatch: ' + item['path'])
        # A few outstanding SMB reads amortize network latency without launching
        # simultaneous run transfers or overwhelming the experiment hosts.
        completed = 0
        last_report = time.monotonic()
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix='archive-checksum') as workers:
            futures = [workers.submit(check, item) for item in manifest]
            for future in as_completed(futures):
                future.result()
                completed += 1
                if progress and (time.monotonic() - last_report >= 2 or completed == len(manifest)):
                    progress('verifying', completed)
                    last_report = time.monotonic()

    def copy_run(self, device, device_root, root, row, progress=None):
        def phase(name):
            if progress:
                progress(name)
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 4:
            raise ValueError('Invalid source run path')
        source_fingerprint = fingerprint(row['files'])
        receipt = device_root / 'state' / relative / 'receipt.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('source_fingerprint') == source_fingerprint:
            raise AlreadyArchived()
        storage = self.configuration.inspect_root(str(root))
        cifs = storage.get('filesystem') in {'cifs', 'smb3'}
        original = device_root / 'runs' / relative
        # Older versions renamed the verified data before chmod and writing the
        # receipt. Recover such a run by verifying it, without copying it again.
        if original.exists() and not receipt.exists():
            phase('source_checksums')
            recovered = self.read_remote(device, 'manifest', run=relative.as_posix())['files']
            current = [{k: item[k] for k in ('path', 'size', 'mtime_ns')} for item in recovered]
            if current != row['files']:
                raise ValueError('Source changed during recovery; retry later')
            actual = {p.relative_to(original).as_posix() for p in original.rglob('*') if p.is_file()}
            if actual != {item['path'] for item in recovered}:
                raise ValueError('Published run without receipt has a different file list; inspect integrity')
            phase('verifying')
            self.verify_files(original, recovered, progress)
            self.configuration.checked_root()
            atomic_json(receipt, {'source_fingerprint': source_fingerprint, 'received_at': now(),
                                  'path': str(original.relative_to(root)), 'files': recovered})
            return
        staging = root / 'incoming' / device['device_id'] / relative
        if shutil.disk_usage(root).free < sum(item['size'] for item in row['files']) + 1024 * 1024 * 100:
            raise ValueError('Insufficient NAS free space for this run')
        staging.mkdir(parents=True, exist_ok=True)
        phase('transferring')
        if device.get('transport') == 'local':
            shutil.copytree(Path(device['source_path']) / relative, staging, dirs_exist_ok=True)
        else:
            source = device['ssh_user'] + '@' + device['host'] + ':' + device['source_path'].rstrip('/') + '/' + relative.as_posix() + '/'
            command = ['rsync', '-rlt', '--protect-args', '--partial', '--partial-dir=.rsync-partial',
                       '--exclude=*.tmp', '--exclude=*.part',
                       '-e', shlex.join(self.ssh(device)[:-1]), '--', source, str(staging) + '/']
            if not cifs:
                command.insert(2, '--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=')
            result = subprocess.run(command, text=True, capture_output=True, timeout=7200)
            if result.returncode:
                raise ValueError(result.stderr[-2000:])
        phase('source_checksums')
        manifest = self.read_remote(device, 'manifest', run=relative.as_posix())['files']
        # Source metadata must still agree with the inventory observed before transfer.
        current = [{k: item[k] for k in ('path', 'size', 'mtime_ns')} for item in manifest]
        if current != row['files']:
            raise ValueError('Source changed during backup; waiting for next scan')
        expected = {item['path'] for item in manifest}
        actual = {p.relative_to(staging).as_posix() for p in staging.rglob('*') if p.is_file() and '.rsync-partial' not in p.parts}
        if actual != expected:
            quarantine = root / 'quarantine' / (device['device_id'] + '_' + uuid.uuid4().hex)
            staging.rename(quarantine)
            raise ValueError('Source file list changed. Staging preserved in quarantine; retry will start fresh')
        phase('verifying')
        self.verify_files(staging, manifest, progress)
        self.configuration.checked_root()
        target = device_root / 'runs' / relative
        if target.exists():
            target = device_root / 'revisions' / relative / source_fingerprint
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # Publication may have succeeded just before a process crash, with
            # the local receipt still unwritten. Recover without replacing files.
            self.verify_files(target, manifest, progress)
            atomic_json(receipt, {'source_fingerprint': source_fingerprint, 'received_at': now(),
                                  'path': str(target.relative_to(root)), 'files': manifest})
            return
        phase('publishing')
        staging.rename(target)
        # Files cannot be modified by analysis code; only new revision directories are published.
        # CIFS without Unix extensions obtains permissions from the mount's
        # file_mode/dir_mode. Per-file chmod cannot enforce raw immutability
        # there and adds an SMB round trip for every archived waveform.
        if not cifs:
            phase('permissions')
            for path in target.rglob('*'):
                path.chmod(0o550 if path.is_dir() else 0o440)
            target.chmod(0o550)
        atomic_json(receipt, {'source_fingerprint': source_fingerprint, 'received_at': now(),
                              'path': str(target.relative_to(root)), 'files': manifest})

    def schedule_loop(self):
        while not self.stop.wait(600):
            for device_id, device in self.configuration.load().get('devices', {}).items():
                if device.get('enabled', True) and device.get('automatic_backup', False):
                    try:
                        self.start(device_id)
                    except (ValueError, FileNotFoundError):
                        pass

    def verify(self, device_id):
        device_root = self.configuration.checked_root() / 'devices' / self.device(device_id)['device_id']
        issues, checked = [], 0
        for receipt in (device_root / 'state').glob('*/*/*/*/receipt.json'):
            record = json.loads(receipt.read_text())
            target = self.configuration.checked_root() / record['path']
            if device_root not in target.resolve().parents:
                raise ValueError('Invalid receipt path')
            for item in record['files']:
                path = target / item['path']
                digest = hashlib.sha256()
                try:
                    with path.open('rb') as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                            digest.update(chunk)
                    if digest.hexdigest() != item['sha256']:
                        issues.append(str(path.relative_to(device_root)))
                except OSError:
                    issues.append(str(path.relative_to(device_root)))
                checked += 1
        result = {'checked_files': checked, 'issues': issues, 'verified_at': now(), 'ok': not issues}
        atomic_json(device_root / 'state' / 'integrity.json', result)
        return result


class AlreadyArchived(Exception):
    pass
