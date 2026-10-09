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
    $nextTick(callback) {callback()}, getConfiguredScanDimensions() {return 1},
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
})().catch(error => {console.error(error); process.exitCode = 1});
"""
        result = subprocess.run(['node', '-e', script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
