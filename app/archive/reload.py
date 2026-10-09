"""Explicit reload coordination. No filesystem watching or forced termination."""
import os
from pathlib import Path
import subprocess
import uuid

from app.archive.backup import atomic_json, now
from app.archive import launcher_runtime as runtime


class ArchiveReload:
    def __init__(self, configuration, backup):
        self.configuration, self.backup = configuration, backup
        self.path = configuration.path.parent / 'launcher/reload.json'

    def state(self):
        try:
            result = runtime.read_json(self.path)
        except (OSError, ValueError):
            result = {}
        if result.get('status') == 'failed':
            self.backup.maintenance = False
        return result

    def available(self):
        service = runtime.service_info()
        if service.get('state') == 'active' and service.get('main_pid') == os.getpid():
            return {'mode':'service', 'port':service['port']}
        if service.get('state') in {'active','activating','deactivating'}:
            raise ValueError('Service/process identity is ambiguous; inspect LaunchUI before reloading')
        state = runtime.read_json(self.configuration.path.parent / 'launcher/runtime.json')
        if state.get('pid') == os.getpid() and runtime.managed_process(state) and not state.get('stop_requested'):
            return {'mode':'managed', 'port':state['port']}
        raise ValueError('Reload requires a server started by LaunchUI or its matching systemd user service. No unknown process will be stopped.')

    def schedule(self):
        with self.backup.lock:
            self.state()
            if self.backup.maintenance:
                raise ValueError('A reload is already pending')
            if any(job['status'] in {'running','queued'} for job in self.backup.jobs()):
                raise ValueError('Wait for all queued/running backups before reloading')
            launch = self.available()
            request_id = uuid.uuid4().hex
            record = {'id':request_id, 'status':'scheduled', 'requested_at':now(),
                      'old_pid':os.getpid(), **launch}
            command = [runtime.python(), '-m', 'app.archive.launcher_runtime', 'reload-worker',
                       '--port',str(launch['port']), '--expected-pid',str(os.getpid()),
                       '--request-id',request_id]
            env = {**os.environ, 'MIGA_ARCHIVE_CONFIG':str(self.configuration.path)}
            self.backup.maintenance = True
            atomic_json(self.path,record)
            try:
                if launch['mode'] == 'service':
                    # Escape the server unit's cgroup, so its stop cannot kill the helper.
                    subprocess.run(['systemd-run','--user','--collect',
                        '--unit=miga-archive-reload-'+request_id,
                        '--property=WorkingDirectory='+str(runtime.ROOT),
                        '--setenv=MIGA_ARCHIVE_CONFIG='+str(self.configuration.path),
                        *command],check=True,capture_output=True,text=True,timeout=10)
                else:
                    with (self.path.parent/'reload.log').open('ab') as output:
                        subprocess.Popen(command,cwd=runtime.ROOT,env=env,
                            stdin=subprocess.DEVNULL,stdout=output,stderr=output,start_new_session=True)
            except (OSError, subprocess.SubprocessError) as exc:
                self.backup.maintenance = False
                atomic_json(self.path,{**record,'status':'failed','error':str(exc)})
                raise ValueError('Cannot start reload helper; server stays running. Restart via LaunchUI.') from exc
            return record
