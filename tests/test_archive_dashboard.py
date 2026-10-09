from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from app.archive.configuration import ArchiveServerConfiguration
from app.archive.dashboard import ArchiveDashboard, successful
from app.archive.backup import atomic_json, now


class ArchiveDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.nas = self.base / 'nas'; self.nas.mkdir()
        self.config = ArchiveServerConfiguration(self.base / 'config.json')
        self.config.initialize(str(self.nas))
        self.device = self.config.register_device({'device_id':'master','name':'Master','host':'10.0.0.2','ssh_user':'miga','source_path':'/home/miga/Data_log'})
        self.run = self.nas / 'devices/master/runs/2025/01/02/run00_20250102'; self.run.mkdir(parents=True)
        self.backup = SimpleNamespace(jobs=Mock(return_value=[]),device=lambda key:self.config.load()['devices'][key])
        self.guide = SimpleNamespace(progress=self.base/'setup/progress.json',probe=Mock(return_value={'ok':False,'message':'Source is offline','host':'10.0.0.2','source_path':'/home/miga/Data_log','checked_at':now()}))
        self.dashboard = ArchiveDashboard(self.config,self.backup,self.guide)

    def tearDown(self): self.temp.cleanup()

    def job(self, **kwargs):
        return {'id':'test','device_id':'master','status':'complete','copied':0,'skipped':2,'total':2,'failed':[],'deferred':[],'sync_warnings':[],'error':'','started_at':'2025-01-02T00:00:00+00:00','finished_at':'2025-01-02T00:01:00+00:00',**kwargs}

    def test_sync_warning_is_not_a_failed_pull(self):
        job=self.job(status='incomplete',sync_warnings=['run00'])
        self.assertTrue(successful(job));self.backup.jobs.return_value=[job]
        result=self.dashboard.snapshot()
        self.assertEqual(result['devices'][0]['last_successful_check']['id'],'test')
        self.assertEqual([alert['kind'] for alert in result['alerts']],['sync'])

    def test_failed_attempt_does_not_replace_last_success(self):
        old=self.job(id='old');latest=self.job(id='latest',status='failed',error='SSH offline')
        self.backup.jobs.return_value=[latest,old]
        row=self.dashboard.snapshot()['devices'][0]
        self.assertEqual(row['latest_job']['id'],'latest');self.assertEqual(row['last_successful_check']['id'],'old')

    def test_offline_source_keeps_archive_navigation(self):
        self.dashboard.probe('master')
        result=self.dashboard.snapshot();row=result['devices'][0]
        self.assertTrue(row['archive_available']);self.assertFalse(row['connection']['ok'])
        self.assertEqual(row['archive_url'],'/archive-server/view/master/archive.html')
        self.assertTrue(any(alert['kind']=='connection' for alert in result['alerts']))

    def test_snapshot_does_not_connect_to_sources_or_change_configuration(self):
        before=self.config.path.read_bytes();self.dashboard.snapshot()
        self.guide.probe.assert_not_called();self.assertEqual(before,self.config.path.read_bytes())

    def test_changed_source_invalidates_cached_connection(self):
        self.dashboard.probe('master')
        self.config.register_device({**self.device,'host':'10.0.0.3'})
        self.assertTrue(self.dashboard.snapshot()['devices'][0]['connection_stale'])

    def test_missing_nas_identity_is_not_reported_as_empty_archive(self):
        (self.nas/'archive-root.json').unlink()
        result=self.dashboard.snapshot()
        self.assertFalse(result['storage']['ok']);self.assertIsNone(result['devices'][0]['archive_available'])
        self.assertEqual(result['alerts'][0]['kind'],'storage')

    def test_checksum_issues_are_separate_from_sync_warnings(self):
        atomic_json(self.nas/'devices/master/state/integrity.json',{'ok':False,'issues':['bad-file'],'verified_at':now()})
        self.assertTrue(any(alert['kind']=='integrity' for alert in self.dashboard.snapshot()['alerts']))


if __name__ == '__main__': unittest.main()
