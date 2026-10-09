import threading
import time
import unittest
from unittest.mock import patch

from app.core.schedule_manager import TaskStartError
from test_schedule_sync import make_scheduler, sync_task


def task(task_id, mode='scan'):
    value = sync_task()
    value.update(id=task_id, name=task_id, execution_mode=mode, temporary_sequence=True)
    if mode == 'scan':
        value.pop('sync')
    return value


class ScheduleQueueEditingTests(unittest.TestCase):
    def scheduler(self):
        value = make_scheduler()
        value._log_event = lambda *args, **kwargs: None
        return value

    def edit(self, scheduler, tasks, snapshot=None):
        state = snapshot or scheduler.editing_snapshot()
        return scheduler.update_pending({'scheduleId': state['scheduleId'],
                                         'revision': state['revision'], 'tasks': tasks})

    def worker(self, scheduler):
        errors = []
        def run():
            try:
                scheduler._run_once()
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, errors

    def test_cancel_and_stop_armed_queue_do_not_abort_external_runs(self):
        for mode in ('scan', 'sync'):
            for operation in ('cancel', 'stop'):
                with self.subTest(mode=mode, operation=operation):
                    scheduler = self.scheduler()
                    if mode == 'scan':
                        scheduler.manager.active_mode = 'scan'
                        scheduler.manager.status.is_running = True
                    else:
                        scheduler.sync_manager.runtime['active'] = True
                    scheduler.start({'tasks': [task('next')], 'startAfterCurrentRun': True})
                    getattr(scheduler, operation)()
                    scheduler._run_once()
                    self.assertFalse(scheduler._state['active'])
                    self.assertFalse(scheduler.manager.stopped)
                    self.assertEqual(scheduler.sync_manager.stopped, [])
                    self.assertEqual(scheduler.manager.started, [])

    def test_cancel_during_scan_or_sync_finishes_current_without_next_dispatch(self):
        for mode in ('scan', 'sync'):
            with self.subTest(mode=mode):
                scheduler = self.scheduler()
                started = threading.Event()
                if mode == 'scan':
                    def start_scan(config):
                        scheduler.manager.status.is_running = True
                        scheduler.manager.active_mode = 'scan'
                        started.set()
                        return {'status': 'success'}
                    scheduler.manager.start_scan = start_scan
                else:
                    def start_sync(payload):
                        scheduler.sync_manager.runtime.update(active=True, status='running')
                        started.set()
                    scheduler.sync_manager.start_master = start_sync
                scheduler.start({'tasks': [task('current', mode), task('next')]})
                thread, errors = self.worker(scheduler)
                try:
                    self.assertTrue(started.wait(2))
                    scheduler.cancel()
                    self.assertTrue(thread.is_alive())
                    self.assertFalse(scheduler.manager.stopped)
                    self.assertEqual(scheduler.sync_manager.stopped, [])
                finally:
                    scheduler.manager.status.is_running = False
                    scheduler.manager.active_mode = None
                    scheduler.sync_manager.runtime.update(active=False, status='done')
                    thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(scheduler._state['statusMessage'], 'CANCELLED')
                self.assertIn('current', scheduler._state['completedTaskIds'])
                self.assertNotIn('next', scheduler._state['completedTaskIds'])

    def test_running_queue_executes_latest_pending_order_and_settings(self):
        scheduler = self.scheduler()
        started, release = threading.Event(), threading.Event()
        executed = []
        def execute(current, *_):
            executed.append((current['id'], current['config']['start']))
            if current['id'] == 'first':
                started.set()
                if not release.wait(2):
                    raise RuntimeError('Test release timed out')
            return True
        scheduler._execute_task_with_start_retries = execute
        scheduler.start({'tasks': [task('first'), task('second'), task('third')]})
        thread, errors = self.worker(scheduler)
        try:
            self.assertTrue(started.wait(2))
            changed = task('third', 'sync')
            changed['config']['start'] = 42
            self.edit(scheduler, [changed, task('new')])
        finally:
            release.set()
            thread.join(2)
        self.assertEqual(errors, [])
        scheduler._run_once()
        scheduler._run_once()
        scheduler._run_once()
        self.assertEqual([item[0] for item in executed], ['first', 'third', 'new'])
        self.assertEqual(executed[1][1], 42)
        self.assertFalse(scheduler._state['active'])

    def test_claimed_tasks_and_stale_editor_versions_are_rejected(self):
        scheduler = self.scheduler()
        scheduler.start({'tasks': [task('first'), task('second')]})
        old = scheduler.editing_snapshot()
        scheduler._state.update(activeTaskId='first', taskPhase='running')
        scheduler._state['revision'] += 1
        with self.assertRaisesRegex(ValueError, 'Queue changed'):
            self.edit(scheduler, [task('second')], old)
        with self.assertRaisesRegex(ValueError, 'cannot be edited'):
            self.edit(scheduler, [task('first')])
        current = scheduler.editing_snapshot()
        self.edit(scheduler, [task('second')], current)
        with self.assertRaisesRegex(ValueError, 'Queue changed'):
            self.edit(scheduler, [task('second')], current)
        current = scheduler.editing_snapshot()
        current['scheduleId'] = 'other-queue'
        with self.assertRaisesRegex(ValueError, 'Queue changed'):
            self.edit(scheduler, [], current)

    def test_appointment_can_be_edited_while_waiting(self):
        scheduler = self.scheduler()
        scheduled = task('old')
        scheduled['scheduledAtMs'] = int(time.time() * 1000 + 60000)
        scheduler.start({'tasks': [scheduled], 'timingMode': 'specific'})
        entered = threading.Event()
        original_wait = scheduler._wait_until
        def wait(*args):
            entered.set()
            return original_wait(*args)
        scheduler._wait_until = wait
        executed = []
        scheduler._execute_task_with_start_retries = lambda value, *_: executed.append(value['id']) or True
        thread, errors = self.worker(scheduler)
        try:
            self.assertTrue(entered.wait(2))
            replacement = task('new')
            replacement['scheduledAtMs'] = int(time.time() * 1000 - 1000)
            self.edit(scheduler, [replacement])
        finally:
            scheduler._wake.set()
            thread.join(2)
        self.assertEqual(errors, [])
        self.assertFalse(thread.is_alive())
        scheduler._run_once()
        self.assertEqual(executed, ['new'])

    def test_empty_pending_queue_and_pause_keep_current_task(self):
        scheduler = self.scheduler()
        scheduler.start({'tasks': [task('first'), task('second')]})
        scheduler._state.update(activeTaskId='first', taskPhase='running')
        scheduler.pause()
        self.edit(scheduler, [])
        self.assertEqual([value['id'] for value in scheduler._state['tasks']], ['first'])
        self.assertTrue(scheduler._state['paused'])
        self.assertFalse(scheduler.manager.stopped)
        self.assertEqual(scheduler.sync_manager.stopped, [])

    def test_pause_allows_current_completion_but_blocks_next_task_until_resume(self):
        scheduler = self.scheduler()
        scheduler.start({'tasks': [task('first'), task('second')]})
        executed = []
        def execute(current, *_):
            executed.append(current['id'])
            if current['id'] == 'first':
                scheduler.pause()
            return True
        scheduler._execute_task_with_start_retries = execute
        scheduler._run_once()
        self.assertEqual(executed, ['first'])
        paused = threading.Event()
        original_wait = scheduler._wait_while_paused
        def wait():
            paused.set()
            return original_wait()
        scheduler._wait_while_paused = wait
        thread, errors = self.worker(scheduler)
        try:
            self.assertTrue(paused.wait(2))
            self.assertEqual(executed, ['first'])
            scheduler.resume()
        finally:
            scheduler._wake.set()
            thread.join(2)
        self.assertEqual(errors, [])
        self.assertEqual(executed, ['first', 'second'])

    def test_cancel_during_start_retry_prevents_another_attempt(self):
        scheduler = self.scheduler()
        scheduler.start({'tasks': [task('first')]})
        attempts = []
        def fail(current):
            attempts.append(current['id'])
            raise TaskStartError('Not accepted')
        scheduler._execute_task = fail
        def retry_wait(*_):
            scheduler.cancel()
            return False
        scheduler._wait_until = retry_wait
        scheduler._run_once()
        self.assertEqual(attempts, ['first'])
        self.assertEqual(scheduler._state['statusMessage'], 'CANCELLED')

    def test_edit_snapshot_keeps_sequences_separate_from_status(self):
        scheduler = self.scheduler()
        scheduler.start({'tasks': [task('sync', 'sync')]})
        full = scheduler.editing_snapshot()['tasks'][0]
        public = scheduler.get_status()['tasks'][0]
        self.assertIn('sequence_snapshot', full)
        self.assertNotIn('sequence_snapshot', public)
        self.assertIn('sequence_content_base64', full['sync']['slaves'][0])
        self.assertNotIn('sequence_content_base64', public['sync']['slaves'][0])
