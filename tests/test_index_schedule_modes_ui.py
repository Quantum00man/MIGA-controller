"""Exercise Scheduled Scan/SYNC task editing using the actual page methods."""
from pathlib import Path
import shutil
import subprocess
import unittest


class IndexScheduleModesUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for UI regression tests')
    def test_mixed_queue_and_runtime_are_independent_of_editor_type(self):
        script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const page = fs.readFileSync('static/index.html', 'utf8');
let options;
global.window = {};
global.Vue = {createApp(value) {options = value; return {mount() {return {}}}}};
const scripts = [...page.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
const context = vm.createContext({Vue: global.Vue, window: global.window, File});
scripts.forEach(script => vm.runInContext(script[1], context));
const app = Object.assign(options.data(), options.methods, {
    runMode: 'scheduled', scheduledExecutionMode: 'scan', isRunning: false,
    config: {mode: 'standard', scan_dimensions: 1, parameter_source: 'classic'},
    syncConfig: {role: 'master'}, syncMasterDelayMs: 125,
    syncSlaves: [{id: 'slave-a', name: 'Slave A', base_url: 'http://slave:8000',
        sequence_name: 'slave.mot', sequence_content_base64: 'c2xhdmU='}],
    prepareScanPayload(config) {return this.cloneConfig(config)},
    async captureCurrentSequenceForTask() {return {sequence_name: 'master.mot',
        sequence_file_name: 'master.mot', sequence_snapshot: 'master sequence'}},
    buildScheduledTaskName() {return 'Task'}, refreshScheduledTaskEstimates() {},
    confirmIndependentP0Placeholders() {return true}, addLog() {}, saveState() {},
    resetHistory() {}, renderMainPlot() {}, initPlots() {}, fetchSequenceMarkers() {},
    $nextTick(callback) {if (callback) callback(); return Promise.resolve()}, getConfiguredScanDimensions() {return 1},
});
(async () => {
    assert(app.isScheduledMode()); assert.equal(app.isSyncMode(), false);
    await app.addScheduledTaskFromCurrentSetup();
    app.scheduledExecutionMode = 'sync';
    assert(app.isScheduledMode()); assert(app.isSyncMode());
    await app.addScheduledTaskFromCurrentSetup();
    assert.equal(JSON.stringify(app.scheduledTasks.map(task => task.execution_mode)), '["scan","sync"]');
    const [scan, sync] = app.scheduledTasks;
    assert.equal(sync.sync.slaves[0].sequence_content_base64, 'c2xhdmU=');
    app.syncSlaves[0].sequence_content_base64 = 'changed';
    assert.equal(sync.sync.slaves[0].sequence_content_base64, 'c2xhdmU=');
    app.loadScheduledTaskIntoEditor(scan.id);
    assert.equal(app.runMode, 'scheduled'); assert.equal(app.scheduledExecutionMode, 'scan');
    app.loadScheduledTaskIntoEditor(sync.id);
    assert.equal(app.runMode, 'scheduled'); assert.equal(app.scheduledExecutionMode, 'sync');
    assert.equal(app.syncSlaves[0].sequence_content_base64, 'c2xhdmU=');
    app.scheduleRuntime.active = true;
    app.scheduleRuntime.activeTaskId = sync.id;
    app.scheduledExecutionMode = 'scan';
    app.isRunning = true;
    assert.equal(app.isSyncAcquisition(), true);
    app.scheduleRuntime.activeTaskId = scan.id;
    app.scheduledExecutionMode = 'sync';
    assert.equal(app.isSyncAcquisition(), false);
    app.scheduleRuntime.waitingForCurrentRun = true;
    app.syncRuntime.active = true;
    assert.equal(app.isSyncAcquisition(), true);
    app.syncRuntime.active = false;
    assert.equal(app.isSyncAcquisition(), false);
    // Bragg calibration must remain inside Scheduled for both task types.
    for (const type of ['scan', 'sync']) {
        app.scheduledExecutionMode = type;
        app.config.mode = 'bragg_fringe_calibration';
        options.watch['config.mode'].call(app, app.config.mode);
        assert.equal(app.runMode, 'scheduled');
    }
    // Changing the next task type must not send stop to the wrong controller.
    app.scheduleRuntime.active = false;
    const requests = [];
    context.axios = {post: async path => {requests.push(path); return {data: {}}}};
    app.scheduledExecutionMode = 'sync';
    app.isRunning = true;
    app.syncRuntime.active = false;
    await app.stopScan();
    assert.equal(requests.at(-1), '/experiment/stop');
    app.scheduledExecutionMode = 'scan';
    app.isRunning = true;
    app.syncRuntime.active = true;
    await app.stopScan();
    assert.equal(requests.at(-1), '/sync/stop');
    // Only unstarted tasks are editable; reordering cannot cross a locked task.
    app.scheduleRuntime = {active: true, activeTaskId: scan.id, taskPhase: 'running',
        completedTaskIds: [], scheduleId: 'queue-1', revision: 3};
    assert.equal(app.isScheduledTaskEditable(scan), false);
    assert.equal(app.isScheduledTaskEditable(sync), true);
    assert.equal(app.canMoveScheduledTask(1, -1), false);
    app.removeScheduledTask(scan.id);
    assert.equal(app.scheduledTasks.length, 2);
    app.scheduleRuntime.taskPhase = 'waiting';
    assert.equal(app.isScheduledTaskEditable(scan), true);
    app.scheduleRuntime.taskPhase = 'running';
    app.scheduledQueueId = 'queue-1'; app.scheduledQueueRevision = 3;
    app.scheduledQueueDirty = true;
    const snapshot = {...app.scheduleRuntime, tasks: JSON.parse(JSON.stringify(app.scheduledTasks)),
        timingMode: 'sequential', sequentialGapSec: 0};
    let submitted;
    context.axios = {
        get: async () => ({data: {data: snapshot}}),
        put: async (path, body) => {
            assert.equal(path, '/schedule/queue'); submitted = body;
            snapshot.revision = 4;
            return {data: {data: snapshot}};
        },
        post: async path => {
            requests.push(path);
            return {data: {data: {...snapshot, cancelRequested: true}}};
        },
    };
    app.scheduledTasks.forEach(task => task.estimated_points = 1);
    snapshot.tasks.forEach(task => task.estimated_points = 1);
    await app.applyScheduledQueueChanges();
    assert.equal(app.scheduledQueueError, false, app.scheduledQueueMessage);
    assert.equal(JSON.stringify(submitted.tasks.map(task => task.id)), JSON.stringify([sync.id]));
    assert.equal(submitted.revision, 3);
    assert.equal(app.scheduledQueueDirty, false);
    app.scheduledQueueDirty = true;
    context.axios.put = async () => {throw {response: {data: {detail: 'Queue changed'}}}};
    app.formatRequestError = error => error.response?.data?.detail || error.message;
    await app.applyScheduledQueueChanges();
    assert.equal(app.scheduledQueueDirty, true);
    assert.equal(app.scheduledQueueError, true);
    assert.equal(app.scheduledQueueMessage, 'Queue changed');
    const countBeforeCancel = requests.length;
    await app.cancelScheduledQueue();
    assert.equal(requests.length, countBeforeCancel + 1);
    assert.equal(requests.at(-1), '/schedule/cancel');
    assert.equal(app.scheduledQueueBusy, false);
})().catch(error => {console.error(error); process.exitCode = 1});
"""
        result = subprocess.run(['node', '-e', script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
