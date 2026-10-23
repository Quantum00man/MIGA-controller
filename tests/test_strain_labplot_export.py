"""Verify LabPlot's embedded strain data/error columns and both export endpoints."""
import asyncio
import base64
import importlib
import json
import lzma
import math
import shutil
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from app.analysis.strain_sensitivity import strain_sensitivity_rows
from app.analysis.transfer_function import build_differential_transfer_function_summary
from app.core.archive_labplot import build_archive_project
from app.models.schemas import ArchiveLabPlotExportRequest, StrainSensitivitySettings


def pairs():
    result = []
    for phase in (0, 90):
        for value in (.10, .12, .14):
            master = dict(transfer_frequency_hz=10, transfer_phase_deg=phase,
                          interferometer_phase=value, interferometer_phase_valid=True,
                          transfer_frequency_modulation_mhz=1, transfer_atom_mirror_distance_m=2.23,
                          transfer_phase_noise_sigma_mrad=100)
            slave = {**master, 'interferometer_phase': .02, 'transfer_atom_mirror_distance_m': 3.77}
            result.append(dict(slave_node_id='slave', master=master, slave=slave))
    return result


class Loader:
    def __init__(self, config=None):
        self.config = {'mode': 'transfer_function', **(config or {})}

    def load_run(self, *args, **kwargs):
        return dict(config=self.config, sync_manifest={'pairs': pairs()})


def parse(payload):
    return ET.fromstring(lzma.decompress(payload))


def column_values(root, name):
    sheet = next(item for item in root.iter('spreadsheet') if item.get('name') == 'Data - SYNC Strain Sensitivity - slave')
    column = next(item for item in sheet.iter('column') if item.get('name') == name)
    data = base64.b64decode(column.find('output_filter').tail)
    return struct.unpack(f'={len(data)//8}d', data)


def test_labplot_strain_curve_errors_and_archived_defaults():
    root = parse(build_archive_project(Loader(), '2026', '10', '23', 'run00', ['phase']))
    expected = strain_sensitivity_rows(build_differential_transfer_function_summary(pairs()), pairs(), StrainSensitivitySettings().model_dump())[0]
    assert column_values(root, 'P1_X1') == (10,)
    assert column_values(root, 'P1_Strain ASD') == pytest.approx([expected['strain_asd_sqrt_hz']])
    for suffix in ('ErrorPlus', 'ErrorMinus'):
        assert column_values(root, f'P1_Strain ASD_{suffix}') == pytest.approx([expected['strain_asd_sem_sqrt_hz']])
    sheet = next(item for item in root.iter('worksheet') if item.get('name') == 'SYNC Strain Sensitivity - slave')
    errors = next(sheet.iter('errorBars'))
    assert errors.get('yErrorType') == '2'
    assert errors.get('yErrorPlusColumn').endswith('P1_Strain ASD_ErrorPlus')
    assert 'transfer_strain_sigma_mrad=74' in root.find('comment').text
    assert 'Nyquist=' in root.find('comment').text
    names = [item.get('name') for item in root.iter('worksheet')]
    assert names.index('SYNC Strain Sensitivity - slave') == names.index('SYNC Path S2 - slave') + 1


def test_current_noise_inputs_override_archived_settings_and_nulls_are_omitted():
    config = {'transfer_strain_sigma_mrad': 90, 'transfer_strain_wavelength_nm': 800}
    params = {**StrainSensitivitySettings().model_dump(), 'transfer_strain_noise_model': 'fixed_asd',
              'transfer_strain_phase_asd_rad_sqrt_hz': .3, 'transfer_strain_wavelength_nm': 790}
    rows = build_differential_transfer_function_summary(pairs())
    rows.append({**rows[0], 'frequency_hz':20, 'differential_path_s2':0})
    root = parse(build_archive_project(Loader(config), '2026','10','23','run00',['phase'],
                 transfer_function_differential_summary=rows, strain_sensitivity_settings=params))
    expected = strain_sensitivity_rows(rows, pairs(), params)[0]
    assert column_values(root, 'P1_Strain ASD') == pytest.approx([expected['strain_asd_sqrt_hz']])
    assert 'transfer_strain_noise_model=fixed_asd' in root.find('comment').text
    assert 'transfer_strain_wavelength_nm=790' in root.find('comment').text
    archived = parse(build_archive_project(Loader(config), '2026','10','23','run00',['phase']))
    archived_params = StrainSensitivitySettings(**config).model_dump()
    expected_archived = strain_sensitivity_rows(rows, pairs(), archived_params)[0]
    assert column_values(archived,'P1_Strain ASD') == pytest.approx([expected_archived['strain_asd_sqrt_hz']])
    excluded = parse(build_archive_project(Loader(), '2026','10','23','run00',['phase'], include_differential=False,
                     transfer_function_summary=[{'frequency_hz':10,'interferometer_phase_s2':1}]))
    assert not any('Strain' in item.get('name','') for item in excluded.iter('worksheet'))


@pytest.mark.skipif(not shutil.which('node'), reason='Node required for browser/server parity')
def test_browser_and_labplot_calculation_parity():
    rows = build_differential_transfer_function_summary(pairs())
    rows += [{**rows[0], 'differential_path_s2':0}, {**rows[0], 'differential_path_s2':1, 'differential_path_s2_sem':None}]
    for model in ('white_phase', 'fixed_asd'):
        settings = StrainSensitivitySettings(transfer_strain_noise_model=model).model_dump()
        script = "const S=require('./static/strain-sensitivity.js');console.log(JSON.stringify(S.rows(" + ','.join(json.dumps(item) for item in (rows, pairs(), settings)) + ')));'
        browser = json.loads(subprocess.run(['node','-e',script], text=True, capture_output=True, check=True).stdout)
        server = strain_sensitivity_rows(rows, pairs(), settings)
        for a,b in zip(browser,server):
            for key in ('strain_baseline_m','strain_response_rad','strain_phase_asd_rad_sqrt_hz','strain_asd_sqrt_hz','strain_asd_sem_sqrt_hz'):
                assert a[key] == pytest.approx(b[key]) if b[key] is not None else a[key] is None
            assert a['strain_status'] == b['strain_status']


@pytest.mark.parametrize('module_name', ['app.api.routes','app.archive.ui_routes'])
def test_both_export_routes_use_current_noise_parameters(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module,'data_loader',Loader())
    monkeypatch.setattr(module,'manager',SimpleNamespace(get_active_bragg_phase_calibration=lambda:None))
    async def threadpool(function, *args):
        return function(*args)
    monkeypatch.setattr(module,'run_in_threadpool',threadpool)
    req = ArchiveLabPlotExportRequest(year='2026',month='10',day='23',run_id='run00',metrics=['phase'],
            strain_sensitivity_settings={'transfer_strain_noise_model':'fixed_asd','transfer_strain_phase_asd_rad_sqrt_hz':.5})
    response = asyncio.run(module.export_archive_labplot(req))
    root = parse(response.body)
    params = req.strain_sensitivity_settings.model_dump()
    expected = strain_sensitivity_rows(build_differential_transfer_function_summary(pairs()), pairs(), params)[0]
    assert column_values(root,'P1_Strain ASD') == pytest.approx([expected['strain_asd_sqrt_hz']])
    assert response.media_type == 'application/x-labplot'
    ui = Path('static/archive.html').read_text()
    assert 'strain_sensitivity_settings: this.isTransferFunctionArchive() ? StrainSensitivity.settings(this.analysis) : null' in ui
