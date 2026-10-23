import asyncio
import base64
import importlib
import json
import lzma
import math
from pathlib import Path
import shutil
import struct
import subprocess
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from app.analysis.cmrr import cmrr_rows
from app.analysis.transfer_function import build_differential_transfer_function_summary
from app.core.archive_labplot import build_archive_project
from app.models.schemas import ArchiveLabPlotExportRequest


def pairs(proportional=False):
    result=[]
    for phase in (0,90):
        for m,s in zip((.9,1,1.1), (.81,.9,.99) if proportional else (.85,.88,.97)):
            # Construct known dimensionless normalized responses from physical phases.
            def point(value, distance):
                return dict(transfer_frequency_hz=10,transfer_phase_deg=phase,transfer_frequency_modulation_mhz=1,
                            transfer_atom_mirror_distance_m=distance,interferometer_phase_valid=True,
                            interferometer_phase=value*4*math.pi*1e6*distance/299792458)
            result.append(dict(slave_node_id='s',master=point(m,2.23),slave=point(s,3.77)))
    return result


def calculate(samples=None, scales=None):
    samples=pairs() if samples is None else samples
    responses=build_differential_transfer_function_summary(samples, scales)
    return cmrr_rows(responses,samples,scales)


def node(script):
    return json.loads(subprocess.run(['node','-e',script],check=True,text=True,capture_output=True).stdout)


def test_negative_convention_and_covariance_of_proportional_channels():
    row=calculate(pairs(True))[0]
    assert row['cmrr_master_s2']==pytest.approx(2)
    assert row['cmrr_slave_s2']==pytest.approx(2*.9**2)
    assert row['cmrr_reference_s2']==pytest.approx(1+.9**2)
    assert row['cmrr_db']==pytest.approx(10*math.log10(.02/1.81))
    # Perfectly correlated multiplicative fluctuations cancel in the ratio.
    assert row['cmrr_sem_db']<1e-12


def test_sem_matches_joint_covariance_and_numerical_derivatives():
    row=calculate()[0]
    samples=np.array([[.9,.85],[1,.88],[1.1,.97]])
    mean=samples.mean(axis=0)
    def db(v):
        m,s=v
        return 10*math.log10((m-s)**2/((m*m+s*s)/2))
    gradient=np.array([(db(mean+np.eye(2)[i]*1e-6)-db(mean-np.eye(2)[i]*1e-6))/2e-6 for i in range(2)])
    # Same means at both independent quadratures: each gradient is halved.
    expected=math.sqrt(2*((gradient/2) @ (np.cov(samples.T,ddof=1)/len(samples)) @ (gradient/2)))
    assert row['cmrr_sem_db']==pytest.approx(expected,rel=1e-7)
    assert row['cmrr_sem_db']>0


@pytest.mark.skipif(not shutil.which('node'),reason='Node required')
def test_browser_server_parity_scales_and_singularities():
    for scales in ({},{'s':1.2}):
        samples=pairs()
        response=build_differential_transfer_function_summary(samples,scales)
        response += [{**response[0],'differential_s2':0}, {**response[0],'differential_s2':None},
                     {**response[0],'differential_s2':1}, {**response[0],'slave_node_id':'missing'}]
        expected=cmrr_rows(response,samples,scales)
        actual=node("const C=require('./static/cmrr.js');console.log(JSON.stringify(C.rows("+','.join(json.dumps(x) for x in (response,samples,scales))+')));')
        for a,b in zip(actual,expected):
            for key in ('cmrr_master_s2','cmrr_slave_s2','cmrr_reference_s2','cmrr_ratio','cmrr_db','cmrr_sem_db'):
                assert a[key]==pytest.approx(b[key]) if b[key] is not None else a[key] is None
            assert a['cmrr_status']==b['cmrr_status']
        assert expected[1]['cmrr_status']=='zero_residual_minus_infinity'
        assert expected[1]['cmrr_db'] is None
        assert expected[3]['cmrr_status']=='response_mismatch'
    single=[pairs()[0],pairs()[3]]
    assert calculate(single)[0]['cmrr_status']=='missing_uncertainty'
    assert calculate(pairs()[:3])[0]['cmrr_status']=='missing_quadratures'
    zero=pairs()
    for pair in zero:
        pair['master']['interferometer_phase']=pair['slave']['interferometer_phase']=0
    assert calculate(zero)[0]['cmrr_status']=='reference_null'


@pytest.mark.skipif(not shutil.which('node'),reason='Node required')
def test_ui_renderer_always_linear_and_csv_includes_cmrr():
    html=Path('static/archive.html').read_text()
    renderer=html[html.index('                renderSyncArchiveCMRRPlot() {'):html.index('                renderSyncArchivePlot() {')]
    script="const CMRR=require('./static/cmrr.js'); const captures=[]; const document={getElementById:()=>({})}; const Plotly={react:(e,d,l)=>captures.push(l.yaxis.type)}; const app={"+renderer+"syncArchiveCMRRRows:()=>[],getArchiveXAxisLabel:()=> 'Frequency (Hz)'}; app.transferFunctionYAxisScale='linear';app.renderSyncArchiveCMRRPlot();app.transferFunctionYAxisScale='log';app.renderSyncArchiveCMRRPlot();console.log(JSON.stringify(captures));"
    assert node(script)==['linear','linear']
    assert html.index('id="syncArchiveStrainSensitivityPlot"')<html.index('id="syncArchiveCMRRPlot"')
    methods=html[html.index('                withTransferUncertainties(input) {'):html.index('                renderSyncArchiveTransferFunctionPlot(element) {')]
    script="const CMRR=require('./static/cmrr.js');const StrainSensitivity=require('./static/strain-sensitivity.js');const app={"+methods+'''
      syncManifest:null,syncArchiveSlaveId:'s',syncTransferNormalizationOptimization:{},syncTransferNormalization:{},
      syncArchivePairs(){return PAIRS;},syncArchiveNodeLabel:id=>id,sanitizeFilenamePart:v=>v,csvCell:v=>String(v),
      async downloadCsvContent(csv,name){console.log(JSON.stringify(csv));}
    };app.downloadSyncTransferFunctionDifferentialCSV();'''.replace('PAIRS',json.dumps(pairs()))
    csv=node(script)
    import csv as csvlib,io
    record=next(csvlib.DictReader(io.StringIO(csv.lstrip('\ufeff'))))
    expected=calculate()[0]
    assert float(record['CMRR_dB'])==pytest.approx(expected['cmrr_db'])
    assert float(record['CMRR_SEM_dB'])==pytest.approx(expected['cmrr_sem_db'])
    assert float(record['CMRR_Reference_S2'])==pytest.approx(expected['cmrr_reference_s2'])


class Loader:
    def load_run(self,*args,**kwargs):
        return dict(config={'mode':'transfer_function'},sync_manifest={'pairs':pairs()})


def cmrr_columns(payload):
    root=ET.fromstring(lzma.decompress(payload))
    sheet=next(s for s in root.iter('spreadsheet') if s.get('name')=='Data - SYNC CMRR - s')
    values={}
    for c in sheet.iter('column'):
        data=base64.b64decode(c.find('output_filter').tail)
        values[c.get('name')]=struct.unpack(f'={len(data)//8}d',data)
    return root,values


def test_labplot_native_cmrr_errors_and_linear_axis():
    root,columns=cmrr_columns(build_archive_project(Loader(),'2026','10','23','run00',['phase']))
    expected=calculate()[0]
    assert columns['P1_CMRR']==pytest.approx([expected['cmrr_db']])
    assert columns['P1_CMRR_ErrorPlus']==pytest.approx([expected['cmrr_sem_db']])
    assert columns['P1_CMRR_ErrorMinus']==columns['P1_CMRR_ErrorPlus']
    sheets=list(root.iter('worksheet'))
    names=[s.get('name') for s in sheets]
    assert names.index('SYNC CMRR - s')==names.index('SYNC Strain Sensitivity - s')+1
    worksheet=next(s for s in sheets if s.get('name')=='SYNC CMRR - s')
    assert next(worksheet.iter('errorBars')).get('yErrorType')=='2'
    axes=list(worksheet.iter('axis'))
    assert axes and all(a.find('general').get('scale')=='0' for a in axes)
    assert all(r.get('scale')=='0' for r in worksheet.iter('yRange'))
    assert 'negative is better' in root.find('comment').text


@pytest.mark.parametrize('module_name',['app.api.routes','app.archive.ui_routes'])
def test_both_labplot_routes_apply_current_normalization(module_name,monkeypatch):
    module=importlib.import_module(module_name)
    monkeypatch.setattr(module,'data_loader',Loader())
    monkeypatch.setattr(module,'manager',SimpleNamespace(get_active_bragg_phase_calibration=lambda:None))
    async def threadpool(fn,*args): return fn(*args)
    monkeypatch.setattr(module,'run_in_threadpool',threadpool)
    req=ArchiveLabPlotExportRequest(year='2026',month='10',day='23',run_id='run00',metrics=['phase'],cmrr_slave_normalization_scales={'s':1.2})
    response=asyncio.run(module.export_archive_labplot(req))
    _,columns=cmrr_columns(response.body)
    assert columns['P1_CMRR']==pytest.approx([calculate(scales={'s':1.2})[0]['cmrr_db']])
