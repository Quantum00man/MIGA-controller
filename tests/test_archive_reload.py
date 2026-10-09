import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock, MagicMock

from app.archive.configuration import ArchiveServerConfiguration
from app.archive.reload import ArchiveReload
from app.archive import launcher_runtime as runtime
from app.archive.backup import BackupService


class ArchiveReloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.config=ArchiveServerConfiguration(Path(self.temp.name)/'config.json')
        self.backup=SimpleNamespace(lock=threading.RLock(),maintenance=False,jobs=Mock(return_value=[]))
        self.reload=ArchiveReload(self.config,self.backup)

    def test_running_backups_block_reload(self):
        self.backup.jobs.return_value=[{'status':'running'}]
        with patch.object(self.reload,'available') as available:
            with self.assertRaisesRegex(ValueError,'backups'):self.reload.schedule()
            available.assert_not_called()
        self.assertFalse(self.backup.maintenance)

    def test_unknown_server_never_restarted(self):
        with patch.object(runtime,'service_info',return_value={}),patch.object(runtime,'managed_process',return_value=False):
            with self.assertRaisesRegex(ValueError,'LaunchUI'):self.reload.available()

    def test_detached_helper_and_duplicate_guard(self):
        with patch.object(self.reload,'available',return_value={'mode':'managed','port':8765}),patch('app.archive.reload.subprocess.Popen') as spawn:
            result=self.reload.schedule()
            self.assertTrue(self.backup.maintenance)
            self.assertEqual(result['old_pid'],os.getpid())
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            with self.assertRaisesRegex(ValueError,'pending'):self.reload.schedule()
            self.assertEqual(spawn.call_count,1)

    def test_service_helper_escapes_server_cgroup(self):
        with patch.object(self.reload,'available',return_value={'mode':'service','port':8765}),patch('app.archive.reload.subprocess.run') as run:
            self.reload.schedule()
            self.assertEqual(run.call_args.args[0][0],'systemd-run')
            self.assertIn('--user',run.call_args.args[0])

    def test_spawn_failure_unblocks_existing_server(self):
        with patch.object(self.reload,'available',return_value={'mode':'managed','port':8765}),patch('app.archive.reload.subprocess.Popen',side_effect=OSError('failed')):
            with self.assertRaisesRegex(ValueError,'helper'):self.reload.schedule()
        self.assertFalse(self.backup.maintenance)
        self.assertEqual(self.reload.state()['status'],'failed')

    def test_backup_start_fenced_during_reload(self):
        fake=SimpleNamespace(lock=threading.RLock(),maintenance=True)
        with self.assertRaisesRegex(ValueError,'reloading'):BackupService.start(fake,'master')

    def test_stale_worker_request_never_touches_process(self):
        runtime.save_json(self.reload.path,{'id':'actual','old_pid':123})
        with patch.object(runtime,'CONFIG',self.config.path),patch.object(runtime.os,'kill') as kill:
            with self.assertRaisesRegex(ValueError,'identity'):runtime.reload_worker(8765,123,'wrong')
            kill.assert_not_called()

    def test_service_worker_uses_restart_and_verifies_new_process(self):
        runtime.save_json(self.reload.path,{'id':'request','old_pid':123,'mode':'service'})
        response=MagicMock()
        response.__enter__.return_value.read.return_value=b'{"mode":"archive","process_id":456}'
        with patch.object(runtime,'CONFIG',self.config.path),patch.object(runtime.time,'sleep'),patch.object(runtime,'service_info',return_value={'state':'active','main_pid':123}),patch.object(runtime.subprocess,'run') as run,patch.object(runtime.urllib.request,'urlopen',return_value=response),patch.object(runtime.os,'kill') as kill:
            runtime.reload_worker(8765,123,'request')
            run.assert_called_once_with(['systemctl','--user','restart','--no-block','miga-archive.service'],check=True)
            kill.assert_not_called()
        self.assertEqual(self.reload.state()['status'],'complete')
        self.assertEqual(self.reload.state()['new_pid'],456)
