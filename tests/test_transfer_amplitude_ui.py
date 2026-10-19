from pathlib import Path
import shutil
import subprocess

import pytest


def test_amplitude_preview_apply_and_stale_selection():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    root = Path(__file__).resolve().parents[1]
    script = r"""
const assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm');
const html=fs.readFileSync('static/archive.html','utf8');
const start=html.indexOf('                transferAmplitudePayload() {');
const end=html.indexOf('                syncArchiveHasPhaseOverrides() {',start);
const methods=vm.runInThisContext('({' + html.slice(start,end) + '})');
let writes=[], renders=0, loads=0;
global.axios={post:async(url,body)=>{writes.push({url,body}); return {data:url.endsWith('/apply')?{sync:{status:'local'}}:{parameters_after:{master:21,'slave-a':20},preview_rows:[]}}}};
const app={...methods, transferAmplitude:{frequency:10,masterPercent:5,slavePercent:5,shotMin:'',shotMax:'',busy:false,result:null},
 selectedYear:'2026',selectedMonth:'10',selectedDay:'19',selectedRun:'run01',syncArchiveSlaveId:'slave-a',
 loadedSyncNode:'master',loadedArchiveInputStamp:'stamp',serverVersion:'original',
 archivePhaseReferenceContexts:{master:{effective_calibration:{parameter_values:{A:20}}}},syncTransferNormalization:{},
 syncAnalysisCopy:{selectedId:''},hasSyncAnalysisPreview:()=>false,currentSyncAnalysisPhaseResult:()=>null,currentSyncAnalysisNormalizationResult:()=>null,
 renderSyncArchiveNodePlots(){renders++},async loadRun(){loads++;return true},$nextTick(fn){fn()}};
(async()=>{
 assert.equal(app.transferAmplitudePayload().shot_index_min,null);
 await app.previewTransferAmplitude();
 assert.equal(writes.length,1); assert.equal(writes[0].url,'/archive/sync-transfer-amplitude-optimize');
 assert.equal(app.transferAmplitudePreviewCurrent(),true); assert.equal(loads,0);
 app.transferAmplitude.slavePercent=4;
 assert.equal(app.transferAmplitudePreviewCurrent(),false);
 await app.applyTransferAmplitude(); assert.equal(writes.length,1);
 app.transferAmplitude.slavePercent=5;
 app.loadedArchiveInputStamp='different'; assert.equal(app.transferAmplitudePreviewCurrent(),false);
 app.loadedArchiveInputStamp='stamp';
 await app.applyTransferAmplitude(); assert.equal(writes.length,2);assert.equal(loads,1);
 assert.equal(app.transferAmplitude.result,null); assert.equal(app.syncArchiveViewMode,'differential');
 await app.previewTransferAmplitude(); const count=writes.length;
 app.discardTransferAmplitude(); assert.equal(writes.length,count);assert.equal(app.transferAmplitude.result,null);
 const panel=html.indexOf('Optimize A — selected frequency');
 assert.ok(panel>html.indexOf('id="syncArchivePlot"'));
 assert.ok(panel<html.indexOf('id="syncArchiveTransferPhaseDifferencePlot"'));
})().catch(e=>{console.error(e);process.exitCode=1});
"""
    subprocess.run([node, '-e', script], cwd=root, check=True, capture_output=True, text=True)
