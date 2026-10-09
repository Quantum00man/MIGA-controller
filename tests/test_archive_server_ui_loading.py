"""Exercise the server adapter with the shared page's actual node-change handler."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_sync_node_switch_preserves_load_result():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required for the archive UI regression test')
    root = Path(__file__).resolve().parents[1]
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
global.location = {pathname: '/archive-server/view/master/archive.html', search: ''};
global.axios = {defaults: {}, interceptors: {
    request: {use() {}}, response: {use() {}}
}};
global.Vue = {createApp: options => options};
vm.runInThisContext(fs.readFileSync('static/archive-server-ui.js', 'utf8'));
const page = fs.readFileSync('static/archive.html', 'utf8');
const start = page.indexOf('async handleSyncArchiveNodeSourceChange() {');
const end = page.indexOf('                isLegacySyncArchive()', start);
const handler = vm.runInThisContext('({' + page.slice(start, end) + '})')
    .handleSyncArchiveNodeSourceChange;
const calls = [];
let failedNode = '';
const methods = {
    loadedRunShareUrl() {}, collectionFolderOptions() {return []},
    async loadRun(nodeId, copyId) {
        calls.push([nodeId, copyId]);
        if (nodeId === failedNode) { this.errorMsg = 'Replica unavailable'; return false; }
        this.loadedSyncNode = nodeId;
        this.syncArchiveNodeSource = nodeId;
        this.displayedData = nodeId === 'master' ? [22] : [21];
        return true;
    }
};
const options = Vue.createApp({data() {return {}}, methods});
const app = Object.assign(options.data(), methods, {
    selectedYear: '2026', selectedMonth: '09', selectedDay: '23', selectedRun: 'run01',
    loadedSyncNode: 'master', syncArchiveNodeSource: 'slave-a',
    syncManifest: {archive_nodes: {master: {}, 'slave-a': {}}},
    isLegacySyncArchive() {return false},
    async refreshServerVersions() {this.versionRefreshes = (this.versionRefreshes || 0) + 1}
});
(async () => {
    await handler.call(app);
    assert.equal(app.loadedSyncNode, 'slave-a');
    assert.deepEqual(app.displayedData, [21]);
    assert.deepEqual(calls, [['slave-a', undefined]]);
    app.syncArchiveNodeSource = 'master';
    await handler.call(app);
    assert.equal(app.loadedSyncNode, 'master');
    assert.deepEqual(app.displayedData, [22]);
    assert.equal(calls.length, 2);
    failedNode = 'slave-a';
    app.syncArchiveNodeSource = 'slave-a';
    await handler.call(app);
    assert.equal(app.loadedSyncNode, 'master');
    assert.equal(app.errorMsg, 'Replica unavailable');
    assert.deepEqual(calls.slice(2), [['slave-a', undefined], ['master', undefined]]);
    assert.equal(await app.loadRun('master', 'copy-1'), true);
    assert.deepEqual(calls.at(-1), ['master', 'copy-1']);
    assert.equal(app.versionRefreshes, 5);
})().catch(error => {console.error(error); process.exitCode = 1});
"""
    subprocess.run([node, '-e', script], cwd=root, check=True, capture_output=True, text=True)
