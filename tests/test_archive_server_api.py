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
            (run / 'results.csv').write_text('Step,Parameter_P0,Atom_UP,Atom_DW\n0,1,100,110\n1,2,150,140\n')
            from app.core.archive_collection_store import ArchiveCollectionStore
            source_collections = ArchiveCollectionStore(source)
            imported_folder = source_collections.create_folder('Controller collection')
            source_collections.create_favorite(imported_folder['id'], {'year': '2025', 'month': '01', 'day': '02', 'run_id': 'run00_20250102'}, {}, {}, alias='Source figure', note='Keep source note')
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
                        return response.status, json.load(response) if 'application/json' in response.headers.get('Content-Type', '') else response.read()
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
                original_config = (base / 'config.json').read_bytes()
                guide = request('/archive-server/setup/progress')[1]
                self.assertEqual(guide['configuration']['devices']['master']['device_id'], 'master')
                self.assertEqual(request('/archive-server/setup/devices/master/probe', 'POST')[0], 200)
                probe = request('/archive-server/setup/progress')[1]['checks']['master']
                self.assertTrue(probe['source_exists'])
                self.assertTrue(probe['source_readable'])
                self.assertIn(str(source),probe['candidates'])
                self.assertEqual((base / 'config.json').read_bytes(),original_config)
                self.assertIn(b'LaunchUI',request('/setup')[1])
                self.assertIn(b'displayStep',request('/archive-setup.js')[1])
                dashboard = request('/archive-server/dashboard')[1]
                self.assertTrue(dashboard['storage']['ok'])
                self.assertFalse(dashboard['devices'][0]['connection_stale'])
                self.assertIn(b'Device management',request('/devices')[1])
                self.assertIn(b'renderCards',request('/archive-server-dashboard.js')[1])
                self.assertIn(b'Your device archives',request('/')[1])
                self.assertIn(b'DEVICE ARCHIVES',request('/')[1])
                self.assertIn(b'Tabler',request('/archive-dashboard.css')[1])
                self.assertEqual(request('/archive-assets/tabler/tabler.min.css')[0],200)
                self.assertEqual(request('/archive-server/update/status')[0],200)
                unsafe = urllib.request.Request('http://127.0.0.1:' + str(port) + '/archive-server/update/apply',
                    data=json.dumps({'branch':'main'}).encode(),method='POST',
                    headers={'Content-Type':'application/json','Origin':'https://untrusted.example'})
                with self.assertRaises(urllib.error.HTTPError) as rejected:
                    urllib.request.urlopen(unsafe,timeout=10)
                self.assertEqual(rejected.exception.code,403)
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
                dashboard = request('/archive-server/dashboard')[1]
                self.assertTrue(dashboard['devices'][0]['archive_available'])
                self.assertIsNotNone(dashboard['devices'][0]['last_successful_check'])
                self.assertEqual(request('/archive-server/devices/master/verify', 'POST')[1]['checked_files'], 2)
                tree = request('/archive-server/archive/master/tree')[1]
                self.assertIn('2025', tree)
                self.assertEqual(request('/archive-server/archive/master/load/2025/01/02/run00_20250102')[0], 200)
                ui = '/archive-server/view/master'
                self.assertIn(b'Data Archive', request(ui + '/archive.html')[1])
                self.assertIn(b'Vue.createApp', request('/archive-server-ui.js')[1])
                self.assertEqual(request(ui + '/archive/navigation/years')[1]['years'], ['2025'])
                self.assertEqual(request(ui + '/archive/load/2025/01/02/run00_20250102')[0], 200)
                self.assertTrue(request(ui + '/fitting/models/defaults')[1])
                self.assertTrue(request(ui + '/fitting/models/scan-defaults')[1])
                source_snapshot = request(ui + '/archive/collections')[1]
                imported = next(f for f in source_snapshot['folders'] if f['name'] == 'Controller collection')
                self.assertLess(imported['id'], 0)
                self.assertEqual(source_snapshot['favorites'][0]['display_name'], 'Source figure')
                self.assertEqual(source_snapshot['favorites'][0]['note'], 'Keep source note')
                self.assertEqual(request(ui + '/archive/collections/folders/' + str(imported['id']), 'DELETE')[0], 403)
                own = request(ui + '/archive/collections/folders', 'POST', {'name': 'Server collection'})[1]
                self.assertGreater(own['id'], 0)
                self.assertIn(request(ui + '/archive/collections/folders', 'POST', {'name': 'Unsafe', 'parent_id': imported['id']})[0], (403, 422))
                self.assertEqual(request(ui + '/archive/sync/retry/2025/01/02/run00_20250102', 'POST')[0], 403)
                reference = {'year': '2025', 'month': '01', 'day': '02', 'run_id': 'run00_20250102'}
                saved = request(ui + '/archive/intf-alpha/reanalyze', 'POST', {**reference, 'save': True, 'name': 'Version one'})
                self.assertEqual(saved[0], 200, saved)
                version_list = request(ui + '/archive/versions/2025/01/02/run00_20250102')[1]['versions']
                self.assertEqual(len(version_list), 1)
                original = storage / 'devices/master/runs/2025/01/02/run00_20250102'
                self.assertFalse((original / 'intf_alpha_analysis_copies.json').exists())
                self.assertEqual(request(ui + '/archive/intf-alpha/reanalyze', 'POST', {**reference, 'save': True, 'name': 'Version two'})[0], 200)
                self.assertEqual(len(request(ui + '/archive/versions/2025/01/02/run00_20250102')[1]['versions']), 2)
                self.assertEqual(request(ui + '/archive/load/2025/01/02/run00_20250102?server_version=original')[0], 200)
                self.assertEqual(request(ui + '/archive/load/2025/01/02/run00_20250102?server_version=' + version_list[0]['id'])[0], 200)
                failed_save = request(ui + '/archive/intf-alpha/reanalyze', 'POST', {**reference, 'save': True, 'name': ''})
                self.assertEqual(failed_save[0], 400)
                self.assertEqual(len(request(ui + '/archive/versions/2025/01/02/run00_20250102')[1]['versions']), 2)
                export = request(ui + '/archive/labplot-export', 'POST', {**reference, 'metrics': ['atoms']})
                self.assertEqual(export[0], 200, export)
                self.assertTrue(export[1])
                self.assertEqual(request('/archive-server/devices', 'POST', {'device_id': 'slave', 'role': 'slave', 'transport': 'local', 'source_path': str(base / 'slave-source')})[0], 200)
                self.assertEqual(request('/archive-server/view/slave/archive/collections')[1], {'folders': [], 'favorites': []})
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [(node, pool.submit(request, '/archive-server/view/' + node + '/archive/navigation/years')) for node in ['master', 'slave'] * 4]
                    for node, future in futures:
                        self.assertEqual(future.result()[1]['years'], ['2025'] if node == 'master' else [])
                self.assertEqual(request('/archive-server/view/unknown/archive/navigation/years')[0], 404)
                self.assertEqual(request(ui + '/archive/load/2025/01/02/run00_20250102?server_version=../../')[0], 400)
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
