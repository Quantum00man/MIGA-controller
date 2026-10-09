"""Read-only dashboard projections; does not change backup scheduling or publication."""
import json
from datetime import datetime
from pathlib import Path
import threading
import time

from app.archive.backup import atomic_json, now


def successful(job):
    return job.get('status') in {'complete', 'incomplete'} and not job.get('error') and not job.get('failed') and job.get('copied', 0) + job.get('skipped', 0) == job.get('total', 0)


class ArchiveDashboard:
    def __init__(self, configuration, backup, guide):
        self.configuration, self.backup, self.guide = configuration, backup, guide
        self.lock = threading.RLock()
        self.cache_path = configuration.path.parent / 'dashboard/connections.json'
        self.storage_cache = None
        self.storage_deadline = 0

    @staticmethod
    def read(path):
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def identity(device):
        return {key: device.get(key) for key in ('host', 'ssh_user', 'source_path', 'collection_path', 'identity_file', 'transport')}

    def probe(self, device_id):
        device = dict(self.backup.device(device_id))
        result = self.guide.probe(device_id)
        with self.lock:
            checks = self.read(self.cache_path)
            checks[device_id] = {'identity': self.identity(device), 'result': result}
            atomic_json(self.cache_path, checks)
        return result

    def monitor(self):
        # No full inventory scan, file checksums or source writes. Avoid probes during backup.
        while not self.backup.stop.wait(600):
            for device_id, device in self.configuration.load().get('devices', {}).items():
                if self.backup.stop.is_set(): return
                if not device.get('enabled', True): continue
                if any(job['status'] in {'queued', 'running'} for job in self.backup.jobs()): break
                try: self.probe(device_id)
                except (OSError, ValueError, KeyError): pass

    def storage(self, config):
        with self.lock:
            if self.storage_cache is not None and time.monotonic() < self.storage_deadline:
                return self.storage_cache
            result = {'ok': False, 'checked_at': now(), 'path': config.get('archive_root'), 'error': ''}
            try:
                if not config.get('archive_root'): raise ValueError('Storage is not initialized')
                result.update(self.configuration.inspect_root(config['archive_root']))
                self.configuration.checked_root()
                result['ok'] = bool(result.get('readable') and result.get('writable'))
                if not result['ok']: result['error'] = 'Storage is not readable/writable'
            except (OSError, ValueError) as exc:
                result['error'] = str(exc)
            self.storage_cache, self.storage_deadline = result, time.monotonic() + 15
            return result

    def snapshot(self):
        config = self.configuration.load()
        storage = self.storage(config)
        jobs = self.backup.jobs()
        checks = self.read(self.cache_path)
        previous_checks = self.read(self.guide.progress).get('checks', {})
        rows, alerts = [], []
        if not storage['ok']: alerts.append({'kind':'storage', 'severity':'danger', 'message': storage['error']})
        for device_id, device in config.get('devices', {}).items():
            history = [job for job in jobs if job['device_id'] == device_id]
            latest = history[0] if history else None
            success = next((job for job in history if successful(job)), None)
            published = next((job for job in history if job.get('copied', 0) > 0), None)
            cached = checks.get(device_id, {})
            connection = cached.get('result') or previous_checks.get(device_id)
            stale = cached.get('identity') != self.identity(device)
            if connection and not stale:
                try: stale = time.time() - datetime.fromisoformat(connection['checked_at']).timestamp() > 1200
                except (KeyError, ValueError): stale = True
            root = Path(config.get('archive_root') or '.') / 'devices' / device_id
            archived = None
            integrity, collection = {}, {}
            if storage['ok']:
                archived = next((root / 'runs').glob('*/*/*/run*'), None) is not None
                integrity = self.read(root / 'state/integrity.json')
                collection = self.read(root / 'collection-imports/latest.json')
            if latest and (latest['status'] in {'failed', 'interrupted'} or latest.get('failed')):
                alerts.append({'kind':'backup','severity':'danger','device_id':device_id,'message':latest.get('error') or 'Backup has failed runs; retry after reviewing the details'})
            if latest and latest.get('sync_warnings'):
                alerts.append({'kind':'sync','severity':'warning','device_id':device_id,'message':f"{len(latest['sync_warnings'])} source SYNC archives report incomplete replication (not a NAS transfer failure)"})
            if integrity.get('issues'):
                alerts.append({'kind':'integrity','severity':'danger','device_id':device_id,'message':f"{len(integrity['issues'])} checksum issues in last verification"})
            if connection and not connection.get('ok') and not stale and device.get('enabled', True):
                alerts.append({'kind':'connection','severity':'warning','device_id':device_id,'message':connection.get('message','Source check failed; existing NAS archives remain accessible')})
            rows.append({**device, 'connection':connection, 'connection_stale':stale, 'archive_available':archived,
                'latest_job':latest, 'last_successful_check':success, 'last_publication_job':published,
                'collection_received_at':collection.get('received_at'), 'integrity':integrity,
                'archive_url':f'/archive-server/view/{device_id}/archive.html'})
        return {'generated_at':now(), 'storage':storage, 'devices':rows, 'alerts':alerts,
                'jobs':jobs[:30], 'active_jobs':[job for job in jobs if job['status'] in {'queued','running'}],
                'monitor_interval_seconds':600}
