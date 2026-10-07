import json
import os
from pathlib import Path
import tempfile
import unittest

from app.archive.configuration import ArchiveServerConfiguration
from app.archive.backup import BackupService
from app.archive.repository import ArchiveRepository
from app.core.archive_collection_store import ArchiveCollectionStore


class ArchiveBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / 'source'
        self.source.mkdir()
        self.storage = self.base / 'nas'
        self.storage.mkdir()
        self.config = ArchiveServerConfiguration(self.base / 'config/config.json')
        self.config.initialize(str(self.storage))
        self.device = self.config.register_device({'device_id': 'master', 'role': 'master', 'transport': 'local', 'source_path': str(self.source)})
        self.service = BackupService(self.config)
        self.relative = '2025/01/02/run00_20250102'
        self.run = self.source / self.relative
        self.run.mkdir(parents=True)
        (self.run / 'config.json').write_text('{"mode":"standard"}')
        (self.run / 'results.csv').write_text('Step,Parameter_P0\n0,1\n')
        for path in self.run.iterdir():
            os.utime(path, (1, 1))
        self.collections = ArchiveCollectionStore(self.source)
        self.folder = self.collections.create_folder('Paper')
        self.collections.create_favorite(self.folder['id'], {'year': '2025', 'month': '01', 'day': '02', 'run_id': 'run00_20250102'}, {}, {}, alias='Figure 1', note='Keep this')

    def tearDown(self):
        self.service.executor.shutdown(wait=True)
        for path in self.base.rglob('*'):
            path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    def test_verified_copy_and_idempotent_retry(self):
        row = self.service.read_remote(self.device, 'inventory')['runs'][0]
        from app.archive.backup import AlreadyArchived
        self.service.copy_run(self.device, self.storage / 'devices/master', self.storage, row)
        target = self.storage / 'devices/master/runs' / self.relative
        self.assertEqual((target / 'results.csv').read_bytes(), (self.run / 'results.csv').read_bytes())
        with self.assertRaises(AlreadyArchived):
            self.service.copy_run(self.device, self.storage / 'devices/master', self.storage, row)

    def test_changed_source_preserves_original_and_publishes_revision(self):
        row = self.service.read_remote(self.device, 'inventory')['runs'][0]
        self.service.copy_run(self.device, self.storage / 'devices/master', self.storage, row)
        old = (self.storage / 'devices/master/runs' / self.relative / 'results.csv').read_bytes()
        (self.run / 'results.csv').write_text('Step,Parameter_P0\n0,2\n')
        os.utime(self.run / 'results.csv', (2, 2))
        row = self.service.read_remote(self.device, 'inventory')['runs'][0]
        self.service.copy_run(self.device, self.storage / 'devices/master', self.storage, row)
        self.assertEqual((self.storage / 'devices/master/runs' / self.relative / 'results.csv').read_bytes(), old)
        self.assertEqual(len(list((self.storage / 'devices/master/revisions' / self.relative).glob('*/results.csv'))), 1)

    def test_changed_during_transfer_is_not_published(self):
        row = self.service.read_remote(self.device, 'inventory')['runs'][0]
        (self.run / 'results.csv').write_text('Step,Parameter_P0\n1,9\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.service.copy_run(self.device, self.storage / 'devices/master', self.storage, row)
        self.assertFalse((self.storage / 'devices/master/runs' / self.relative).exists())

    def test_collection_snapshot_keeps_alias_note_and_device_context(self):
        job = {'id': 'fixture', 'device_id': 'master', 'status': 'queued', 'started_at': '', 'copied': 0, 'skipped': 0, 'failed': [], 'deferred': []}
        self.service.run(job, self.device, self.storage)
        self.assertEqual(job['status'], 'complete', job)
        result = ArchiveRepository(self.config).collections('master')
        self.assertEqual(result['favorites'][0]['note'], 'Keep this')
        self.assertEqual(result['favorites'][0]['alias'], 'Figure 1')
        self.assertEqual(result['favorites'][0]['integrity'], 'ok')

    def test_missing_mount_identity_blocks_copy(self):
        marker = self.storage / 'archive-root.json'
        marker.rename(self.storage / 'archive-root.saved')
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            self.service.start('master')

    def test_duplicate_enabled_sources_are_rejected(self):
        self.config.register_device({**self.device, 'device_id': 'duplicate'})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.service.start('master')
        self.config.set_enabled('duplicate', False)
        self.assertFalse(self.config.load()['devices']['duplicate']['enabled'])
        self.config.remove_device('duplicate')
        self.assertTrue((self.storage / 'devices/duplicate').is_dir())

    def test_archive_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            ArchiveRepository(self.config).reference('master', '2025', '01', '02', 'run../../escape')

    def test_cifs_publication_skips_per_file_chmod_but_checks_contents(self):
        from unittest.mock import patch
        row = self.service.read_remote(self.device, 'inventory')['runs'][0]
        real_inspect = self.config.inspect_root
        def inspect(value):
            info = real_inspect(value)
            info['filesystem'] = 'cifs'
            return info
        # Keep checked_root independent of our fake filesystem for this fixture.
        with patch.object(self.config, 'inspect_root', side_effect=inspect), patch.object(self.config, 'checked_root', return_value=self.storage), patch.object(Path, 'chmod', side_effect=AssertionError('CIFS must not chmod each file')):
            self.service.copy_run(self.device, self.storage / 'devices/master', self.storage, row)
        self.assertEqual((self.storage / 'devices/master/runs' / self.relative / 'results.csv').read_bytes(), (self.run / 'results.csv').read_bytes())

    def test_published_run_without_receipt_recovers_without_duplicate(self):
        from unittest.mock import patch
        row = self.service.read_remote(self.device, 'inventory')['runs'][0]
        self.service.copy_run(self.device, self.storage / 'devices/master', self.storage, row)
        receipt = self.storage / 'devices/master/state' / self.relative / 'receipt.json'
        receipt.unlink()
        with patch('app.archive.backup.shutil.copytree', side_effect=AssertionError('Recovery must not retransmit')):
            self.service.copy_run(self.device, self.storage / 'devices/master', self.storage, row)
        self.assertTrue(receipt.exists())
        self.assertFalse((self.storage / 'devices/master/revisions').exists())

    def test_parallel_checksum_verification_rejects_corruption(self):
        row = self.service.read_remote(self.device, 'inventory')['runs'][0]
        manifest = self.service.read_remote(self.device, 'manifest', run=self.relative)['files']
        (self.run / 'results.csv').write_text('corrupted')
        with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
            self.service.verify_files(self.run, manifest)

    def test_analysis_versions_never_modify_raw_and_remain_independent(self):
        from unittest.mock import patch, Mock
        row = self.service.read_remote(self.device, 'inventory')['runs'][0]
        self.service.copy_run(self.device, self.storage / 'devices/master', self.storage, row)
        repository = ArchiveRepository(self.config)
        reference = ['2025', '01', '02', 'run00_20250102']
        target = repository.reference('master', *reference)
        before = {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob('*') if p.is_file()}
        loader = Mock()
        loader.recalculate_run.return_value = {'data': [{'parameter': 1, 'atom_number_up': 20}], 'settings': {'alpha': 0.1}}
        with patch.object(repository, 'loader', return_value=loader):
            first = repository.recalculate('master', reference, {'alpha': 0.1}, save=True)
            second = repository.recalculate('master', reference, {'alpha': 0.2}, save=True)
        self.assertNotEqual(first['analysis']['id'], second['analysis']['id'])
        self.assertEqual(len(repository.analyses('master', reference)), 2)
        self.assertEqual(before, {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
