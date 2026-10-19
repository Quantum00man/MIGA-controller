from pathlib import Path
import shutil
import subprocess

import pytest


def test_transfer_tof_selects_phases_nodes_physical_steps_and_handles_missing_data():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    root = Path(__file__).resolve().parents[1]
    script = r"""
const assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm');
const html=fs.readFileSync('static/archive.html','utf8');
const start=html.indexOf('                transferTofNodes() {');
const end=html.indexOf('                calcStats(arr)',start);
const methods=vm.runInThisContext('({' + html.slice(start,end) + '})');
const gets=[], posts=[], plotted=[];
let failStep=null;
const waveform={time_axis:[0,.001],raw_up:[1,2],raw_dw:[2,1],fit_up:[1,2],fit_dw:[2,1],window_up:null,window_dw:null};
const row=(step,phase,frequency=101,extra={})=>({step,sync_shot_index:step-100,transfer_frequency_hz:frequency,transfer_phase_deg:phase,...extra});
const master=[row(10,0),row(11,90),row(12,0,101,{transfer_zero_phase_baseline:true}),row(13,0,200),row(14,0,101,{intf_alpha_calibration_block_id:1})];
const slave=[row(120,0),row(121,90),row(122,0,101,{error:'bad shot'})];
global.document={getElementById:()=>({scrollIntoView(){}})};
global.Plotly={react:(id,traces)=>plotted.push({id,traces}),relayout(){},Plots:{resize(){}}};
global.axios={get:async url=>{
 gets.push(url);
 if(url.includes('/archive/load/')) return {data:{archive_node_data:slave}};
 if(failStep!==null && url.includes('/'+failStep+'?')) throw {response:{data:{detail:'Waveform not archived'}}};
 return {data:waveform};
},post:async(url,body)=>{posts.push({url,body});return {data:{data:[{step:10,...waveform}]}}}};
const app={...methods,transferTof:{open:false,frequency:null,node:'',phase:'all',shot:'',rows:[],error:'',request:0},waveformRequestId:0,
 selectedYear:'2026',selectedMonth:'09',selectedDay:'30',selectedRun:'run48',loadedSyncNode:'master',
 loadedPoints:master,loadedSyncNodeData:master,syncManifest:{},displayMode:'recalculated',
 showTofFitCurve:true,showTofRawCurve:true,currentWaveforms:[],historyMap:{},
 layoutOscUp:{shapes:[]},layoutOscDw:{shapes:[]},$nextTick:fn=>fn(),
 isTransferFunctionArchive:()=>true,syncArchiveSlaveOptions:()=>['slave-a'],
 syncArchiveNodeLabel:node=>node,getArchiveXAxisLabel:()=> 'Frequency (Hz)',
 normalizeWaveformResponse:(data,step)=>({...data,step}),buildArchivePayload:extra=>extra,
 selectedPointKeys:['old'],selectedPointKey:'old',maximizedView:'analysis'};
(async()=>{
 await app.openTransferTof({x:101,data:{meta:{transferPhase:90}}});
 assert.equal(app.maximizedView,'scope');assert.equal(app.transferTof.phase,90);assert.equal(app.transferTof.shot,11);
 assert.match(gets.at(-1),/\/11\?node_id=master$/);assert.equal(posts.length,0);
 assert.equal(app.currentWaveforms.length,1);
 app.transferTof.node='slave-a';await app.changeTransferTofNode();
 assert.match(gets.at(-1),/\/121\?node_id=slave-a$/);
 assert.equal(app.loadedSyncNode,'master');assert.equal(app.transferTof.rows.length,2);
 app.transferTof.phase='all';await app.changeTransferTofPhase();assert.equal(app.transferTof.shot,120);
 await app.stepTransferTof(1);assert.equal(app.transferTof.shot,121);
 await app.stepTransferTof(1);assert.equal(app.transferTof.shot,121);
 failStep=120;app.transferTof.shot=120;await app.showTransferTofShot();
 assert.match(app.transferTof.error,/Waveform not archived/);assert.equal(app.currentWaveforms.length,0);
 const before=gets.length;await app.openTransferTof({x:101,data:{name:'SEM² noise floor'}});assert.equal(gets.length,before);
 failStep=null;await app.loadWaveforms([10]);assert.equal(posts.length,1); // Existing recalculated previews still work.
 app.closeTransferTof();assert.equal(app.maximizedView,'');assert.equal(app.transferTof.open,false);assert.equal(app.currentWaveforms.length,0);
 // Standalone TF uses its own node and retains phase-invalid shots for diagnosis.
 app.syncManifest=null;app.loadedSyncNode='';app.loadedPoints=[row(1,0,101,{interferometer_phase_valid:false})];
 await app.openTransferTof({x:101,data:{}});assert.match(gets.at(-1),/\/1$/);assert.equal(app.transferTof.rows.length,1);
 // A slow previous response cannot replace the newly selected TOF.
 let release;
 axios.get=url=>url.endsWith('/10')?new Promise(resolve=>{release=resolve}):Promise.resolve({data:waveform});
 const slow=app.loadWaveforms([10],{savedOnly:true});
 await app.loadWaveforms([11],{savedOnly:true});
 release({data:waveform});await slow;
 assert.equal(app.currentWaveforms[0].step,11);assert.equal(app.isLoadingWaveforms,false);
 assert.ok(html.includes('!isSquareSummaryArchive() || transferTof.open'));
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run([node, '-e', script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
