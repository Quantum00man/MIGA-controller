"""Read-only setup probes and resumable, advisory wizard progress."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from app.archive.backup import atomic_json, now


PROBE = r'''
import json, os, pathlib, shutil, sys
request = json.loads(sys.argv[1])
home = pathlib.Path.home()
candidates = [pathlib.Path(request['source_path']), home/'MIGA-controller/Data_log', home/'miga_lastversion/MIGA-controller/Data_log']
paths = []
for path in candidates:
    if path.is_dir() and str(path) not in paths: paths.append(str(path))
source = pathlib.Path(request['source_path'])
collection = pathlib.Path(request.get('collection_path') or source/'archive_collections.sqlite3')
print(json.dumps({'source_path':str(source), 'source_exists':source.is_dir(), 'source_readable':os.access(source, os.R_OK | os.X_OK),
    'candidates':paths, 'collection_exists':collection.is_file(), 'collection_readable':os.access(collection, os.R_OK),
    'rsync_available':bool(shutil.which('rsync')), 'python_available':True,
    'permissions_scope':'Top-level check only; Storage Audit or backup verifies nested content'}))
'''


class SetupGuide:
    def __init__(self, configuration, backup):
        self.configuration, self.backup = configuration, backup
        self.progress = configuration.path.parent / 'setup/progress.json'

    def state(self):
        config = self.configuration.load()
        try:
            saved = json.loads(self.progress.read_text()) if self.progress.exists() else {}
        except (ValueError, OSError):
            saved = {}
        mounts = []
        try:
            for line in Path('/proc/mounts').read_text().splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[2] in {'cifs', 'smb3', 'nfs', 'nfs4', 'fuse.afpfs', 'fuse.gvfsd-fuse'}:
                    mounts.append({'path':parts[1].replace('\\040', ' '), 'filesystem':parts[2]})
        except OSError:
            pass
        tools = {name:bool(shutil.which(name)) for name in ['ssh', 'ssh-keygen', 'ssh-copy-id', 'rsync']}
        return {'configuration': config, 'mounts':mounts, 'tools':tools, 'checks':saved.get('checks', {}),
                'steps':['Storage', 'Devices', 'SSH & folders', 'Backup verification', 'Automatic startup'],
                'password_policy':'Passwords and privileged actions are handled only by the local LaunchUI terminal, never this web page'}

    def probe(self, device_id):
        device = self.backup.device(device_id)
        if device.get('transport') == 'local':
            command = ['python3', '-', json.dumps(device)]
        else:
            command = self.backup.ssh(device) + ['python3 - ' + shlex.quote(json.dumps(device))]
        try:
            result = subprocess.run(command, input=PROBE, capture_output=True, text=True, timeout=25)
        except subprocess.TimeoutExpired:
            result = None
        if result is not None and result.returncode == 0:
            payload = json.loads(result.stdout)
            payload['ok'] = payload['source_exists'] and payload['source_readable'] and payload['rsync_available'] and (not payload['collection_exists'] or payload['collection_readable'])
            payload['message'] = 'Connection and top-level checks passed' if payload['ok'] else 'SSH connected; review folder/rsync checks'
        else:
            detail = result.stderr[-2000:] if result else 'Connection timed out'
            if 'Host key verification failed' in detail or 'REMOTE HOST IDENTIFICATION HAS CHANGED' in detail:
                message = 'Verify the source host fingerprint in local LaunchUI. Existing known_hosts is never replaced automatically.'
            elif 'Permission denied' in detail:
                message = 'Use local LaunchUI → Pair / reuse existing SSH, or authorize its public key on the source.'
            elif 'refused' in detail.lower():
                message = 'Enable SSH using LaunchUI on the SOURCE computer, and check its address/firewall.'
            else:
                message = 'Check that the source is online and its IP/SSH service is reachable.'
            payload = {'ok':False, 'message':message, 'detail':detail}
        payload.update(checked_at=now(), device_id=device_id, host=device.get('host'), source_path=device['source_path'])
        with self.configuration.lock:
            try:
                saved = json.loads(self.progress.read_text()) if self.progress.exists() else {'checks':{}}
            except (OSError, ValueError):
                saved = {'checks':{}}
            saved.setdefault('checks', {})[device_id] = payload
            atomic_json(self.progress, saved)
        return payload
