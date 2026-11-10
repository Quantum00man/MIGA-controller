from pathlib import Path
import shutil
import subprocess
import pytest


def test_bayesian_plots_are_independent_and_plot_errors_keep_result():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required")
    script = r"""
const assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm');
const html=fs.readFileSync('static/archive.html','utf8');
const start=html.indexOf('                bayesPayload() {');
const end=html.indexOf('                phaseNoiseDetailRows() {',start);
const methods=vm.runInThisContext('({' + html.slice(start,end) + '})');
const result={input_stamp:'stamp',records:[{step:1,included:true,probability_percent:42}],
 model_x:[1,2],model_density:[.1,.2],axes:{A:[1,2],B:[1,2],sigma:[1,2]},marginals:{A:[.2,.8],B:[.4,.6],sigma:[.5,.5]}};
const plots=[]; let fail=false;
global.axios={post:async()=>({data:result})};
global.Plotly={react:async(id,traces,layout)=>{plots.push({id,layout}); if(fail) throw Error('plot failure');}};
const app={...methods,bayes:{node:'master',channel:1,source:'fit',shotMin:null,shotMax:null,bounds:{A:['',''],B:['',''],sigma:['','']},grid:25,phasePoints:96,shots:[],result:null,stale:true},
 loadedRunRef:{year:'2026',month:'01',day:'01',run_id:'run01'},syncManifest:null,isPhaseNoiseArchive:()=>false,$nextTick:async()=>{}};
(async()=>{
 await app.bayesAnalyze();
 assert.equal(app.bayes.error,'');
 assert.deepEqual(plots.map(p=>p.id),['bayesShotPlot','bayesDistributionPlot','bayesPosteriorA','bayesPosteriorB','bayesPosteriorSigma']);
 assert.ok(plots.every(p=>!p.layout.grid));
 plots.length=0; fail=true;
 await app.bayesAnalyze();
 assert.match(app.bayes.error,/Analysis completed/);
 assert.equal(app.bayes.result,result);
 assert.equal(app.bayes.stale,false);
 assert.equal(app.bayes.snapshot.input_stamp,'stamp');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    subprocess.run([node, "-e", script], cwd=Path(__file__).resolve().parents[1], check=True)
