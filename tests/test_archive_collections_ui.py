"""Collection browsing preserves analysis state and opens runs only on request."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_collection_navigation_and_workspace_transitions():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required')
    script = r"""
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const html=fs.readFileSync('static/archive.html','utf8');
const start=html.indexOf('                async fetchCollections() {');
const end=html.indexOf('                openBatchDialog(action)',start);
const methods=vm.runInThisContext('({'+html.slice(start,end)+'})');
global.localStorage={setItem(){}};
const folders=[{id:-1,parent_id:0,name:'Source Collections'},{id:-2,parent_id:-1,name:'Bragg'},
    {id:-3,parent_id:-2,name:'Fringes'},{id:1,parent_id:0,name:'Paper'}, {id:2,parent_id:1,name:'Allan'}];
let loadCalls=0, resized=0;
const app=Object.assign({
    archiveSourceMode:'collections', collectionWorkspaceMode:'browse', collectionFolders:folders,
    collapsedCollectionFolderIds:[], selectedCollectionFolderId:-3,
    selectedCollectionFavorite:{id:4}, selectedCollectionFavoriteIds:[4],
    collectionSearch:'noise', collectionTypeFilter:'all', collectionIntegrityFilter:'all',
    loadedRunRef:{run_id:'old'}, collectionRunLoading:false,
    $refs:{collectionResults:{scrollTop:700}}, async $nextTick(){},
    resizeCollectionAnalysisPlots(){resized++}, updateCollectionWorkspaceHeight(){},
    async fetchArchiveMonths(){}, async fetchArchiveDays(){}, async fetchArchiveRuns(){},
    async loadRun(){loadCalls++; this.loadedRunRef={run_id:this.selectedRun}; return true},
},methods);
// Replace DOM-specific size helpers, keeping actual transition and navigation methods.
app.resizeCollectionAnalysisPlots=()=>{resized++}; app.updateCollectionWorkspaceHeight=()=>{};
(async()=>{
    assert.deepEqual(app.visibleCollectionFolders().map(x=>x.id),[1,2,-1,-2,-3]);
    app.toggleCollectionFolder(-2);
    assert.equal(app.visibleCollectionFolders().some(x=>x.id===-3),false);
    assert.equal(app.selectedCollectionFolderId,-3); // Expanding is not selecting.
    app.toggleCollectionFolder(-2);
    app.selectCollectionFolder(2);
    assert.equal(app.collectionWorkspaceMode,'browse'); assert.equal(loadCalls,0);
    assert.equal(app.$refs.collectionResults.scrollTop,0);
    assert.equal(app.collectionSearch,'noise');
    const favorite={id:5,integrity:'ok',year:'2026',month:'10',day:'09',run_id:'run05'};
    await app.loadCollectionFavorite(favorite);
    assert.equal(loadCalls,1); assert.equal(app.collectionWorkspaceMode,'analyze');
    assert.equal(app.selectedCollectionFavorite.id,5); assert.equal(app.collectionRunLoading,false);
    app.$refs.collectionResults.scrollTop=700;
    await app.setCollectionWorkspaceMode('browse'); await app.setCollectionWorkspaceMode('analyze');
    assert.equal(loadCalls,1); assert.equal(app.$refs.collectionResults.scrollTop,700);
    assert.equal(app.loadedRunRef.run_id,'run05'); assert.ok(resized>=2);
    // Missing or failed loads must not switch away from browsing.
    await app.setCollectionWorkspaceMode('browse');
    await app.loadCollectionFavorite({...favorite,integrity:'missing'});
    assert.equal(loadCalls,1); assert.equal(app.collectionWorkspaceMode,'browse');
    app.loadRun=async()=>false;
    await app.loadCollectionFavorite(favorite);
    assert.equal(app.collectionWorkspaceMode,'browse'); assert.equal(app.collectionRunLoading,false);
    app.fetchArchiveMonths=async()=>{throw new Error('Unavailable archive')};
    await app.loadCollectionFavorite(favorite);
    assert.equal(app.collectionError,'Unavailable archive'); assert.equal(app.collectionRunLoading,false);
    app.collectionSidebarWidth=320;
    app.adjustCollectionSidebar(1000); assert.equal(app.collectionSidebarWidth,520);
    app.adjustCollectionSidebar(-1000); assert.equal(app.collectionSidebarWidth,240);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    subprocess.run([node, '-e', script], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
