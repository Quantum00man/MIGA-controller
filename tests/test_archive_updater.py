import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from app.archive.configuration import ArchiveServerConfiguration
from app.archive.updater import ArchiveUpdater


class ArchiveUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.remote = self.base / 'remote'
        self.remote.mkdir()
        self.git(self.remote, 'init', '-b', 'main')
        self.git(self.remote, 'config', 'user.name', 'Test')
        self.git(self.remote, 'config', 'user.email', 'test@example.invalid')
        (self.remote / 'file').write_text('initial')
        self.git(self.remote, 'add', 'file')
        self.git(self.remote, 'commit', '-m', 'initial')
        self.checkout = self.base / 'checkout'
        self.git(self.base, 'clone', str(self.remote), str(self.checkout))
        self.config = ArchiveServerConfiguration(self.base / 'config.json')
        self.jobs = []
        self.updater = ArchiveUpdater(self.config, SimpleNamespace(lock=threading.RLock(), jobs=lambda:self.jobs), self.checkout)

    def git(self, root, *args):
        return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()

    def advance(self):
        (self.remote / 'file').write_text('updated')
        self.git(self.remote, 'commit', '-am', 'updated')

    def test_fast_forward_and_saved_branch(self):
        self.advance()
        result = self.updater.run('main', True)
        self.assertTrue(result['restart_required'])
        self.assertEqual((self.checkout / 'file').read_text(), 'updated')
        self.assertEqual(self.config.load()['update_branch'], 'main')

    def test_dirty_checkout_preserved(self):
        (self.checkout / 'local').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'Local changes'):
            self.updater.run('main', True)
        self.assertEqual((self.checkout / 'local').read_text(), 'keep')

    def test_backup_blocks_update(self):
        self.jobs.append({'status':'queued'})
        with self.assertRaisesRegex(ValueError, 'backups'):
            self.updater.run('main', True)

    def test_branch_switch(self):
        self.git(self.remote, 'switch', '-c', 'feature')
        self.advance()
        result = self.updater.run('feature', True)
        self.assertEqual(result['current_branch'], 'feature')

    def test_diverged_branch_not_overwritten(self):
        self.git(self.checkout, 'config', 'user.name', 'Test')
        self.git(self.checkout, 'config', 'user.email', 'test@example.invalid')
        (self.checkout / 'other').write_text('local commit')
        self.git(self.checkout, 'add', 'other')
        self.git(self.checkout, 'commit', '-m', 'local')
        before = self.git(self.checkout, 'rev-parse', 'HEAD')
        self.advance()
        with self.assertRaises(ValueError):
            self.updater.run('main', True)
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), before)

    def test_fetch_does_not_change_checkout(self):
        before = self.git(self.checkout, 'rev-parse', 'HEAD')
        self.advance()
        self.updater.run('main')
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), before)

    def test_invalid_branch(self):
        with self.assertRaises(ValueError):
            self.updater.run('--bad', True)

    def test_auto_reload_is_scheduled_after_success(self):
        self.config._save({'auto_reload_after_update':True})
        self.updater.reload_controller=Mock()
        self.updater.reload_controller.schedule.return_value={'status':'scheduled','old_pid':123}
        self.advance()
        result=self.updater.run('main',True)
        self.assertFalse(result['restart_required'])
        self.assertEqual(result['reload']['status'],'scheduled')
        self.updater.reload_controller.available.assert_called_once()
        self.assertEqual((self.checkout/'file').read_text(),'updated')

    def test_unmanaged_auto_reload_rejects_before_checkout(self):
        self.config._save({'auto_reload_after_update':True})
        self.updater.reload_controller=Mock()
        self.updater.reload_controller.available.side_effect=ValueError('unknown server')
        self.advance()
        with self.assertRaisesRegex(ValueError,'unknown server'):self.updater.run('main',True)
        self.assertEqual((self.checkout/'file').read_text(),'initial')

    def test_failed_reload_preserves_successful_code_update(self):
        self.config._save({'auto_reload_after_update':True})
        self.updater.reload_controller=Mock()
        self.updater.reload_controller.schedule.side_effect=ValueError('helper failed')
        self.advance()
        result=self.updater.run('main',True)
        self.assertTrue(result['restart_required'])
        self.assertIn('Restart manually',result['message'])
        self.assertEqual((self.checkout/'file').read_text(),'updated')
