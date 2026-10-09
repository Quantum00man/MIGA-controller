"""Exercise shared UI restore logic with a calculator that must never be invoked."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_restore_allan_and_phase_noise_without_recalculation():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required')
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('static/archive.html', 'utf8');
const begin = html.indexOf('                analysisContextKey(kind) {');
const end = html.indexOf('                getAllanCacheMode() {', begin);
const methods = vm.runInThisContext('({' + html.slice(begin, end) + '})');
let calls = [];
let record = {
    id: 'saved-id', candidate_id: 'draft-id', name: 'Stored Allan', note: 'Original input',
    kind: 'allan', source: {version: 'original'}, input_matches_current: true,
    parameters: {node_id: null, metric: 'atoms', source: 'fit', order: 3,
        display_mode: 'saved', new_settings: {alpha: .2}, p0_min: null, p0_max: null},
    result: {orders: [1, 2, 3], used_max_order: 3, metrics: {atoms: {fit: {up: {values: [1, 2, 3]}}}}},
    view: {allanScaleMode: 'relative', injectedField: 'untrusted'}
};
global.axios = {get: async url => {calls.push(url); return {data: record}},
    post: () => {throw Error('Restore must not call any calculation API')}};
const bucket = {};
const app = Object.assign({
    selectedAnalysisResultId: record.id, loadedRunRef: {year: '2026', month: '10', day: '09', run_id: 'run01'},
    loadedSyncNode: '', serverVersion: 'original', normalizeAnalysis: value => value,
    normalizeAllanP0RangeValues: () => null,
    syncAllanP0RangeFromPayload() {}, getAllanCacheKey: () => 'current',
    getAllanCacheBucket: () => bucket,
    renderMainPlot() {this.rendered = this.currentAllanData},
    renderPhaseNoiseArchivePlot() {this.rendered = this.phaseNoiseSummary},
    $nextTick: async () => {},
}, methods);
(async () => {
    await app.restoreAnalysisResult();
    assert.equal(calls.length, 1);
    assert.deepEqual(app.rendered.metrics, record.result.metrics);
    assert.equal(app.analysisCandidateId, 'saved-id');
    assert.equal(app.allanScaleMode, 'relative');
    assert.equal(app.injectedField, undefined);
    assert.match(app.analysisResultMessage, /without recalculation/);
    assert.equal(app.analysisResultContext, app.analysisContextKey('allan'));
    app.analysis = {alpha: .5};
    assert.notEqual(app.analysisResultContext, app.analysisContextKey('allan'));
    record = {...record, kind: 'phase_noise_allan', input_matches_current: false,
        result: {orders: [1, 3], phase_noise_summary: [{allan_deviations: [{order: 1, measured_phase_noise_rad: .1}]}], phase_noise_series: [1, 2, 3]}};
    await app.restoreAnalysisResult();
    assert.equal(calls.length, 2);
    assert.deepEqual(app.rendered, record.result.phase_noise_summary);
    assert.deepEqual(app.phaseNoiseSeries, [1, 2, 3]);
    assert.match(app.analysisResultMessage, /Current input differs/);
    assert.equal(app.analysisResultBusy, false);
    // Use the actual renderer to verify a restored local Allan curve is replayed,
    // without invoking its browser-side calculator or rebuilding traces.
    const plotStart = html.indexOf('                renderPhaseNoiseArchivePlot() {');
    const plotEnd = html.indexOf('                renderMainPlot() {', plotStart);
    const renderer = vm.runInThisContext('({' + html.slice(plotStart, plotEnd) + '})').renderPhaseNoiseArchivePlot;
    const plots = [];
    global.Plotly = {react: (...args) => plots.push(args)};
    app.analysisSnapshotRestored = true;
    app.phaseNoiseAllanPlot = {traces: [{x: [1, 2], y: [.1, .08]}], layout: {title: 'Saved curve'}};
    app.phaseNoiseStatistic = 'allan';
    app.analysisResultContext = app.analysisContextKey('phase_noise_allan');
    app.renderPhaseNoiseSingleAllanPlot = () => {throw Error('Must not recalculate local Allan')};
    app.isLegacyMultiTPhaseNoiseArchive = () => false;
    renderer.call(app);
    assert.deepEqual(plots[0][1], app.phaseNoiseAllanPlot.traces);
    assert.equal(plots.length, 3);
})().catch(error => {console.error(error); process.exitCode = 1});
"""
    subprocess.run([node, '-e', script], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)


def test_sync_allan_save_readiness_cached_statistics_and_restored_curves():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required')
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('static/archive.html', 'utf8');
const computedStart = html.indexOf('            computed: {');
const computedEnd = html.indexOf('            watch:', computedStart);
const computed = vm.runInThisContext('({' + html.slice(computedStart, computedEnd) + '})').computed;
const methodStart = html.indexOf('                analysisContextKey(kind) {');
const methodEnd = html.indexOf('                getAllanCacheMode() {', methodStart);
const management = vm.runInThisContext('({' + html.slice(methodStart, methodEnd) + '})');
const curveStart = html.indexOf('                syncPhaseAllanCurve(rows) {');
const curveEnd = html.indexOf('                formatSyncPhaseStatistic(value)', curveStart);
const scientific = vm.runInThisContext('({' + html.slice(curveStart, curveEnd) + '})');
global.Vue = {markRaw: value => value};
const stored = [{id:'master', label:'Master', curve:{orders:[1], deviations:[.1], mean:.2, rms:.3, standardDeviation:.1, count:300}}];
const app = Object.assign({
    syncManifest:{}, syncPhasePlotMode:'allan', analysisResultBusy:false, isLoadingAllan:false,
    analysisResultName:'', analysisCandidateId:'', analysisResultKind:'sync_phase_allan',
    analysisContextKey: () => 'context', syncPhaseAnalysisParameters: () => ({order:1}),
    loadedArchiveInputStamp:'receipt',
    syncPhaseAllanDefinitions: () => {throw Error('Saved result must not rebuild input rows')},
    syncPhaseAllanCurve: () => {throw Error('Saved result must not calculate curves')},
    restoredSyncPhaseAllan:{context:'context', id:'saved', name:'op allan', result:{curves:stored}}
}, management);
// Reuse a stored curve and record readiness with no backend candidate id.
app.analysisContextKey = () => 'context';
app.syncPhaseAnalysisParameters = () => ({order:1});
app.syncPhaseAllanDefinitions = () => {throw Error('Unexpected definitions calculation')};
app.syncPhaseAllanCurve = () => {throw Error('Unexpected Allan calculation')};
app.syncPhaseAllanResults = computed.syncPhaseAllanResults.call(app);
assert.equal(app.syncPhaseAllanResults, stored);
app.restoredSyncPhaseAllan = null;
app.rememberSyncPhaseAllan(stored, [{x:[1], y:[.1]}], {});
assert.equal(app.analysisCandidateId, '');
app.hasAnalysisResult = computed.hasAnalysisResult.call(app);
app.analysisResultIsStale = false;
assert.equal(computed.canSaveAnalysisResult.call(app), false);
app.analysisResultName = 'op allan';
assert.equal(computed.canSaveAnalysisResult.call(app), true);
for (const letter of 'op allan') {
    app.analysisResultName += letter;
    const rows = scientific.syncPhaseAllanStatisticsRows.call(app);
    assert.equal(rows[0].sampleCount, 300);
    assert.equal(rows[0].mean, .2);
}
app.analysisResultIsStale = true;
assert.equal(computed.canSaveAnalysisResult.call(app), false);
// Verify the contiguous-window EDF shortcut against the original pairwise formula.
function originalEdf(starts,n) {
    const overlap = d => Math.max(0,n-Math.abs(d));
    let squares=starts.length/(n*n);
    for(let i=0;i<starts.length;i++) for(let j=i+1;j<starts.length;j++) {
        const d=starts[j]-starts[i]; if(d>=2*n) break;
        const dot=(2*overlap(d)-overlap(d+n)-overlap(d-n))/(2*n*n);
        squares+=2*dot*dot;
    }
    return (starts.length/n)**2/squares;
}
for (const starts of [[0], [0,1,2,3,4,5,6], [3,4,5,6], [0,2,3,6,8]]) {
    for(const order of [1,2,3,5]) {
        const actual = scientific.allanWhiteNoiseEdf(starts, order);
        assert.ok(Math.abs(actual-originalEdf(starts,order))<1e-10);
    }
}
"""
    subprocess.run([node, '-e', script], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
