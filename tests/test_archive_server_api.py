import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request


class ArchiveServerApiTests(unittest.TestCase):
    def test_device_management_backup_and_hardware_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            storage = base / 'nas'
            storage.mkdir()
            source = base / 'source'
            run = source / '2025/01/02/run00_20250102'
            run.mkdir(parents=True)
            (run / 'config.json').write_text('{"mode":"standard"}')
            (run / 'results.csv').write_text('Step,Parameter_P0\n0,1\n')
            for path in run.iterdir():
                os.utime(path, (1, 1))
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = {**os.environ, 'MIGA_ARCHIVE_CONFIG': str(base / 'config.json')}
            process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'archive_main:app', '--host', '127.0.0.1', '--port', str(port)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            def request(path, method='GET', body=None):
                data = json.dumps(body).encode() if body is not None else None
                req = urllib.request.Request('http://127.0.0.1:' + str(port) + path, data=data, method=method, headers={'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(req, timeout=10) as response:
                        return response.status, json.load(response)
                except urllib.error.HTTPError as error:
                    return error.code, json.load(error)
            try:
                deadline = time.monotonic() + 15
                while True:
                    try:
                        request('/archive-server/status')
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            self.fail('Server did not start')
                        time.sleep(.1)
                self.assertEqual(request('/archive-server/setup/initialize', 'POST', {'path': str(storage)})[0], 200)
                device = {'device_id': 'master', 'role': 'master', 'transport': 'local', 'source_path': str(source)}
                self.assertEqual(request('/archive-server/devices', 'POST', device)[0], 200)
                self.assertEqual(request('/archive-server/devices', 'POST', device)[0], 400)
                self.assertEqual(request('/archive-server/devices/master/test', 'POST')[1]['eligible_runs'], 1)
                self.assertEqual(request('/archive-server/devices/master/backup', 'POST')[0], 200)
                while True:
                    jobs = request('/archive-server/jobs')[1]['jobs']
                    if jobs[0]['status'] not in {'queued', 'running'}:
                        break
                    if time.monotonic() > deadline:
                        self.fail('Backup did not finish')
                    time.sleep(.1)
                self.assertEqual(jobs[0]['status'], 'complete', jobs)
                self.assertEqual(request('/archive-server/devices/master/verify', 'POST')[1]['checked_files'], 2)
                tree = request('/archive-server/archive/master/tree')[1]
                self.assertIn('2025', tree)
                self.assertEqual(request('/archive-server/archive/master/load/2025/01/02/run00_20250102')[0], 200)
                self.assertEqual(request('/experiment/start', 'POST', {})[0], 404)
                self.assertEqual(request('/archive-server/devices/master', 'PATCH', {'enabled': False})[0], 200)
                self.assertEqual(request('/archive-server/devices/master/backup', 'POST')[0], 400)
                self.assertEqual(request('/archive-server/devices/master', 'DELETE')[0], 200)
                self.assertTrue((storage / 'devices/master/runs/2025/01/02/run00_20250102').is_dir())
            finally:
                process.terminate()
                process.wait(timeout=10)
                for path in base.rglob('*'):
                    path.chmod(0o700 if path.is_dir() else 0o600)


if __name__ == '__main__':
    unittest.main()
