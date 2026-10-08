import json
from pathlib import Path
import subprocess
import tempfile
import socket
import time
import unittest
from unittest.mock import patch

from app.archive import launcher_runtime as runtime


class ArchiveLauncherTests(unittest.TestCase):
    def test_service_detection_is_project_and_config_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'with spaces/config.json'
            output='ActiveState=active\nLoadState=loaded\nWorkingDirectory='+str(runtime.ROOT)+'\nEnvironment="MIGA_ARCHIVE_CONFIG='+str(config)+'"\nExecStart=/usr/bin/python -m uvicorn archive_main:app --port 8768\n'
            with patch.object(runtime,'CONFIG',config), patch.object(runtime.shutil,'which',return_value='/usr/bin/systemctl'), patch.object(runtime.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout=output)):
                self.assertTrue(runtime.service_active())
                self.assertEqual(runtime.service_info()['port'],8768)
                with patch.object(runtime,'CONFIG',Path(directory)/'other.json'):
                    self.assertFalse(runtime.service_active())

    def test_installed_inactive_user_service_is_started_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.json'
            with patch.object(runtime,'CONFIG',config), patch.object(runtime.os,'geteuid',return_value=1000), patch.object(runtime,'status',side_effect=[{'running':False,'service':False},{'responding':True}]), patch.object(runtime,'check'), patch.object(runtime,'service_info',return_value={'state':'inactive','port':8768}), patch.object(runtime.subprocess,'run',return_value=subprocess.CompletedProcess([],0)) as run, patch.object(runtime.subprocess,'Popen') as spawn:
                runtime.start(8765)
                run.assert_called_once_with(['systemctl','--user','start','--no-block','miga-archive.service'],check=True)
                spawn.assert_not_called()

    @unittest.skipIf(runtime.os.geteuid() == 0, 'Archive Server intentionally rejects root startup')
    def test_real_detached_start_reuses_process_and_stops_gracefully(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.json'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            with patch.object(runtime,'CONFIG',config), patch.object(runtime,'service_active',return_value=False):
                try:
                    runtime.start(port);first=runtime.status(port)
                    self.assertTrue(first['managed'])
                    runtime.start(port)
                    self.assertEqual(runtime.status(port)['pid'],first['pid'])
                    runtime.stop(port)
                    deadline=time.monotonic()+15
                    while runtime.managed_process(runtime.read_json(config.parent/'launcher/runtime.json')):
                        if time.monotonic()>deadline:self.fail('Managed server did not exit gracefully')
                        time.sleep(.1)
                finally:
                    runtime.stop(port)

    def test_working_ssh_is_reused_without_key_or_host_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text(json.dumps({'devices':{'master':{'transport':'ssh','host':'10.0.0.2','ssh_user':'miga','identity_file':'/existing/key'}}}))
            before = config.read_bytes()
            with patch.object(runtime, 'CONFIG', config), patch.object(runtime.subprocess, 'run', return_value=subprocess.CompletedProcess([],0)), patch.object(runtime, 'ensure_key') as generate:
                runtime.pair('master')
                generate.assert_not_called()
            self.assertEqual(config.read_bytes(), before)

    def test_stop_sends_only_one_graceful_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            runtime.save_json(config.parent / 'launcher/runtime.json', {'pid':12345,'port':8765})
            with patch.object(runtime,'CONFIG',config), patch.object(runtime,'service_active',return_value=False), patch.object(runtime,'managed_process',return_value=True), patch.object(runtime.os,'kill') as kill:
                runtime.stop();runtime.stop()
                kill.assert_called_once_with(12345,runtime.signal.SIGINT)
            self.assertTrue(runtime.read_json(config.parent / 'launcher/runtime.json')['stop_requested'])

    def test_unknown_process_is_never_killed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runtime,'CONFIG',Path(directory)/'config.json'), patch.object(runtime,'service_active',return_value=False), patch.object(runtime,'managed_process',return_value=False), patch.object(runtime,'status',return_value={'running':True}), patch.object(runtime.os,'kill') as kill:
                with self.assertRaises(ValueError): runtime.stop()
                kill.assert_not_called()

    def test_existing_dedicated_identity_is_not_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory) / 'miga_archive_ed25519'
            public = runtime.ensure_key(private)
            original = private.read_bytes(), public.read_bytes()
            self.assertEqual(runtime.ensure_key(private), public)
            self.assertEqual((private.read_bytes(),public.read_bytes()),original)
            public.write_text('ssh-ed25519 invalid\n')
            with self.assertRaises(ValueError): runtime.ensure_key(private)
            self.assertEqual(private.read_bytes(),original[0])
            self.assertEqual(public.read_text(),'ssh-ed25519 invalid\n')

    def test_host_trust_requires_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);config=home/'config.json'
            config.write_text(json.dumps({'devices':{'master':{'host':'10.0.0.2','ssh_user':'miga'}}}))
            replies=[subprocess.CompletedProcess([],255),subprocess.CompletedProcess([],1,stdout=''),subprocess.CompletedProcess([],0,stdout='10.0.0.2 ssh-ed25519 key\n'),subprocess.CompletedProcess([],0,stdout='256 SHA256:expected source (ED25519)\n')]
            with patch.object(runtime,'CONFIG',config), patch.object(runtime.Path,'home',return_value=home), patch.object(runtime,'ensure_key',return_value=home/'key.pub'), patch.object(runtime.subprocess,'run',side_effect=replies), patch('builtins.input',return_value='SHA256:wrong'):
                with self.assertRaises(ValueError):runtime.pair('master')
            self.assertFalse((home/'.ssh/known_hosts').exists())

    def test_local_authorization_preserves_existing_keys_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);public=runtime.ensure_key(home/'key').read_text()
            (home/'.ssh').mkdir();authorized=home/'.ssh/authorized_keys';authorized.write_text('# preserved\nssh-ed25519 existing-key comment\n')
            with patch.object(runtime.Path,'home',return_value=home), patch.object(runtime.os,'geteuid',return_value=1000), patch('builtins.input',return_value='AUTHORIZE'):
                runtime.authorize(public);first=authorized.read_bytes();runtime.authorize(public)
            self.assertEqual(authorized.read_bytes(),first)
            self.assertIn(b'ssh-ed25519 existing-key comment',first)
            self.assertIn(b'restrict ssh-ed25519',first)

    def test_source_path_cannot_be_broad(self):
        with self.assertRaises(ValueError):runtime.validate_source('/')
        with self.assertRaises(ValueError):runtime.validate_source(str(Path.home()))
        with self.assertRaises(ValueError):runtime.validate_source(str(runtime.ROOT))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):runtime.validate_source(directory)
            source=Path(directory)/'Data_log';source.mkdir()
            self.assertEqual(runtime.validate_source(str(source)),source)

    def test_existing_source_ssh_needs_no_install_or_service_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'Data_log';source.mkdir()
            with patch.object(runtime.os,'geteuid',return_value=1000), patch.object(runtime.shutil,'which',return_value='/usr/bin/tool'), patch.object(runtime.Path,'is_file',return_value=True), patch.object(runtime.subprocess,'run',return_value=subprocess.CompletedProcess([],0)) as run:
                runtime.source_prepare(str(source))
                self.assertEqual([call.args[0][0] for call in run.call_args_list],['systemctl','ssh-keygen'])

    def test_read_acl_repair_does_not_remove_existing_write_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'Data_log';source.mkdir()
            replies=[subprocess.CompletedProcess([],0),subprocess.CompletedProcess([],0,stdout='user:'+runtime.login_user()+':rwx\n')]
            with patch.object(runtime.os,'geteuid',return_value=1000), patch.object(runtime.shutil,'which',return_value='/usr/bin/tool'), patch.object(runtime.Path,'is_file',return_value=True), patch.object(runtime.subprocess,'run',side_effect=replies) as run, patch('builtins.input',return_value='GRANT READ'):
                with self.assertRaises(ValueError):runtime.source_prepare(str(source),True)
                self.assertFalse(any('setfacl' in call.args[0] for call in run.call_args_list))


if __name__ == '__main__':unittest.main()
