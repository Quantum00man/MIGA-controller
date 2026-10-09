"""Run the index preset restoration and mode watchers in Node.js."""
from pathlib import Path
import shutil
import subprocess
import unittest


class IndexSyncPresetUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for UI regression tests')
    def test_supported_sync_modes_stay_sync_after_preset_restoration(self):
        script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const page = fs.readFileSync('static/index.html', 'utf8');
const watchStart = page.indexOf('            watch: {');
const watchEnd = page.indexOf('                mounted()', watchStart);
const watch = vm.runInThisContext('({' + page.slice(watchStart, watchEnd) + '})').watch;
const applyStart = page.indexOf('                applyLoadedRunPreset(payload = {}) {');
const applyEnd = page.indexOf('                async loadRunPreset()', applyStart);
const apply = vm.runInThisContext('({' + page.slice(applyStart, applyEnd) + '})').applyLoadedRunPreset;
const supported = ['standard', 'timing', 'rabi', 'half', 'link', 'bragg_rabi',
    'bragg_fringe_calibration', 'phase_noise', 'transfer_function', 'transfer_burst_time_scan'];
for (const initial of supported) {
    for (const target of supported) {
        const app = {
            runMode: 'sync', isRunning: false, currentTab: 'atoms', plotRevision: 0,
            config: {mode: initial, parameter_source: 'classic', scan_dimensions: 1,
                averages: 1, randomize: false, sequence_name: 'old.mot',
                dim1_type: 'range', dim2_type: 'range', dim3_type: 'range'},
            syncSlaves: [{id: 'slave-a', name: 'Slave A'}],
            isSyncMode() {return this.runMode === 'sync'},
            normalizeRunEntry(entry) {return entry}, getSelectedRepeatRunEntry() {return null},
            getConfiguredScanDimensions() {return this.config.scan_dimensions},
            renderMainPlot() {}, resetHistory() {}, fetchPhaseNoiseCalibrations() {},
            renderPhaseNoiseSelectionPlot() {}, $nextTick(callback) {callback()},
        };
        const payload = {
            config: {mode: target, parameter_source: 'classic', sequence_name: 'master.mot'},
            sequence_name: 'master.mot', sequence_loaded: true, run_id: 'run01',
            sync_preset: {master_delay_ms: 37.5, independent_p0_enabled: false,
                slaves: [{node_id: 'slave-a', sequence_name: 'slave.mot',
                    sequence_content_base64: 'c2xhdmU=', p0_scan_config: {start: 1}}]},
        };
        apply.call(app, payload);
        // Vue queues watchers until the synchronous preset assignment is complete.
        if (initial !== target) watch['config.mode'].call(app, app.config.mode);
        assert.equal(app.runMode, 'sync', `${initial} -> ${target}`);
        assert.equal(app.config.mode, target);
        assert.equal(app.currentSequenceName, 'master.mot');
        assert.equal(app.syncSlaves[0].sequence_content_base64, 'c2xhdmU=');
        assert.equal(app.syncMasterDelayMs, 37.5);
    }
}
// Modes unsupported by SYNC keep the existing Standard fallback.
for (const mode of ['lock_in', 'ac_stark', 'ramsey_interferometer']) {
    const app = {runMode: 'sync', config: {mode, parameter_source: 'classic'}};
    watch['config.mode'].call(app, mode);
    assert.equal(app.runMode, 'sync');
    assert.equal(app.config.mode, 'standard');
}
// Single-node Bragg calibration still uses Live rather than Scheduled.
for (const runMode of ['live', 'scheduled']) {
    const app = {runMode, isRunning: true, currentTab: 'atoms',
        config: {mode: 'bragg_fringe_calibration', parameter_source: 'classic'}};
    watch['config.mode'].call(app, app.config.mode);
    assert.equal(app.runMode, 'live');
}
"""
        root = Path(__file__).resolve().parents[1]
        subprocess.run(['node', '-e', script], cwd=root, check=True, capture_output=True, text=True)
