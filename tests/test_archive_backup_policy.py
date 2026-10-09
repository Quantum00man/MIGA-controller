import threading
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock
from app.archive.backup_policy import DEFAULTS,effective,next_pull,validate
from app.archive.backup import BackupService


class BackupPolicyTests(unittest.TestCase):
    def setUp(self):
        self.device={'device_id':'master','enabled':True,'automatic_backup':True}
        self.boot=1000

    def timestamp(self,value):return datetime.fromisoformat(value).timestamp()

    def test_legacy_defaults(self):
        self.assertEqual(effective({},self.device),DEFAULTS)
        self.assertEqual(self.timestamp(next_pull({},self.device,[],self.boot)),1600)

    def test_override_and_startup(self):
        config={'backup_settings':{'interval_minutes':20}}
        self.device['backup_overrides']={'interval_minutes':2,'pull_on_startup':True}
        self.assertEqual(effective(config,self.device)['interval_minutes'],2)
        self.assertEqual(self.timestamp(next_pull(config,self.device,[],self.boot)),1000)

    def test_backoff_and_reset(self):
        policy={'backup_settings':{'retry_minutes':2,'retry_max_minutes':5}}
        jobs=[{'device_id':'master','started_at':'1970-01-01T00:20:00+00:00','status':'failed'}]*3
        self.assertEqual(self.timestamp(next_pull(policy,self.device,jobs,self.boot)),1500)
        jobs[0]={**jobs[0],'status':'complete'}
        self.assertEqual(self.timestamp(next_pull(policy,self.device,jobs,self.boot)),1800)

    def test_no_duplicate_or_disabled_schedule(self):
        self.assertIsNone(next_pull({},self.device,[{'device_id':'master','status':'queued'}],self.boot))
        self.device['enabled']=False
        self.assertIsNone(next_pull({},self.device,[],self.boot))

    def test_validation_keeps_safety(self):
        for bad in [{'quiet_seconds':0},{'quiet_seconds':59},{'interval_minutes':True},{'collections_first':'yes'},{'unknown':1}]:
            with self.assertRaises(ValueError):validate(bad)
        with self.assertRaises(ValueError):effective({'backup_settings':{'retry_minutes':10,'retry_max_minutes':1}},self.device)

    def test_all_sources_independent_results(self):
        devices={'master':self.device,'disabled':{'enabled':False},'slave':{'enabled':True},'offline':{'enabled':True}}
        config=SimpleNamespace(checked_root=Mock(),load=lambda:{'devices':devices})
        def start(device):
            if device=='offline':raise ValueError('source error')
            return {'id':'new'}
        fake=SimpleNamespace(lock=threading.RLock(),maintenance=False,configuration=config,
            jobs=lambda:[{'device_id':'master','status':'running'}],start=start)
        result=BackupService.start_all(fake)
        self.assertEqual([r['status'] for r in result['devices']],['already_queued','disabled','queued','failed'])

    def test_reload_blocks_all_sources(self):
        fake=SimpleNamespace(lock=threading.RLock(),maintenance=True)
        with self.assertRaises(ValueError):BackupService.start_all(fake)

    def test_scheduled_immediate_pull(self):
        device={**self.device,'backup_overrides':{'pull_on_startup':True}}
        fake=SimpleNamespace(stop=Mock(),lock=threading.RLock(),boot_time=1000,schedule_blocked_until={},
            configuration=SimpleNamespace(load=lambda:{'devices':{'master':device}}),jobs=lambda:[],start=Mock())
        fake.stop.wait.side_effect=[False,True]
        BackupService.schedule_loop(fake)
        fake.start.assert_called_once_with('master')
