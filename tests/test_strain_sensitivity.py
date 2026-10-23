"""Physical calibration, legacy defaults, singularities and log error-bar regression."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
from app.models.schemas import StrainSensitivitySettings, ArchiveAnalysisSettings, SystemSettings, ScanConfig
from app.core.data_loader import DataLoader


def js(expression):
    result = subprocess.run(['node', '-e', "const S = require('./static/strain-sensitivity.js'); console.log(JSON.stringify(" + expression + '));'], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


@pytest.mark.skipif(not shutil.which('node'), reason='Node required for production plotting code')
def test_ideal_peak_noise_models_and_sem():
    # Ideal identical three-pulse interferometers: |HB|=4 at the first peak.
    row = js("S.calculate({differential_path_s2:16, differential_path_s2_sem:2}, 1.54)")
    response = 2 * math.pi / 780e-9 * 1.54 * 4
    assert row['strain_response_rad'] == pytest.approx(response)
    assert row['strain_phase_asd_rad_sqrt_hz'] == pytest.approx(.074 * math.sqrt(5.2))
    assert row['strain_asd_sqrt_hz'] == pytest.approx(.074 * math.sqrt(5.2) / response)
    assert row['strain_asd_sem_sqrt_hz'] == pytest.approx(row['strain_asd_sqrt_hz'] / 16)
    fixed = js("S.calculate({differential_path_s2:16, differential_path_s2_sem:2}, -1.54, {transfer_strain_noise_model:'fixed_asd', transfer_strain_phase_asd_rad_sqrt_hz:0.2})")
    assert fixed['strain_asd_sqrt_hz'] == pytest.approx(.2 / response)


@pytest.mark.skipif(not shutil.which('node'), reason='Node required')
def test_null_missing_uncertainty_and_log_bounds():
    result = js("""(() => {
        const nullRow = S.calculate({frequency_hz:1,differential_path_s2:0},1.54);
        const missing = S.calculate({frequency_hz:2,differential_path_s2:1},1.54);
        const weak = S.calculate({frequency_hz:3,differential_path_s2:1,differential_path_s2_sem:3},1.54);
        return {nullRow,missing,weak,log:S.figure([nullRow,missing,weak],{},'log'),linear:S.figure([weak],{},'linear')};
    })()""")
    assert result['nullRow']['strain_status'] == 'response_null'
    assert result['nullRow']['strain_asd_sqrt_hz'] is None
    assert result['missing']['strain_asd_sem_sqrt_hz'] is None
    trace = result['log']['data'][0]
    assert trace['y'][0] is None
    assert trace['error_y']['array'][1] is None
    assert trace['error_y']['arrayminus'][2] < trace['y'][2]
    assert trace['error_y']['array'][2] == result['weak']['strain_asd_sem_sqrt_hz']
    assert any('clipped' in note['text'] for note in result['log']['layout']['annotations'])
    assert result['linear']['data'][0]['error_y']['arrayminus'][0] == result['weak']['strain_asd_sem_sqrt_hz']


@pytest.mark.skipif(not shutil.which('node'), reason='Node required')
def test_legacy_pair_geometry_and_multiple_slaves():
    result = js("""(() => {
        const point=(d)=>({interferometer_phase_valid:true,interferometer_phase:0.1,transfer_phase_deg:0,transfer_frequency_hz:10,transfer_atom_mirror_distance_m:d});
        const pairs=[{slave_node_id:'a',master:point(2.23),slave:point(3.77)}, {slave_node_id:'b',master:point(2.23),slave:point(4.23)}];
        const rows=['a','b','missing'].map(slave_node_id=>({slave_node_id,frequency_hz:10,differential_path_s2:16,differential_path_s2_sem:1}));
        const valid=S.rows(rows,pairs,{});
        pairs.push({slave_node_id:'a',master:point(2.23),slave:point(3.8)});
        return {valid,mixed:S.rows(rows,pairs,{})};
    })()""")
    assert result['valid'][0]['strain_baseline_m'] == pytest.approx(-1.54)
    assert result['valid'][1]['strain_baseline_m'] == pytest.approx(-2)
    assert result['valid'][2]['strain_asd_sqrt_hz'] is None
    assert result['mixed'][0]['strain_asd_sqrt_hz'] is None


def test_schema_and_legacy_reanalysis_defaults():
    expected = StrainSensitivitySettings().model_dump()
    assert expected['transfer_strain_sigma_mrad'] == 74
    for cls in (ArchiveAnalysisSettings, SystemSettings, ScanConfig):
        assert 'transfer_strain_noise_model' in cls.model_fields
    settings = DataLoader.__new__(DataLoader)._normalize_archive_settings({})
    for key, value in expected.items():
        assert settings[key] == value
    custom = {**expected, 'transfer_strain_noise_model': 'fixed_asd', 'transfer_strain_phase_asd_rad_sqrt_hz': .25}
    settings = DataLoader.__new__(DataLoader)._normalize_archive_settings(custom)
    assert settings['transfer_strain_phase_asd_rad_sqrt_hz'] == .25
    for invalid in ({'transfer_strain_cycle_time_s':0}, {'transfer_strain_sigma_mrad':-1}, {'transfer_strain_noise_model':'unknown'}, {'transfer_strain_wavelength_nm':float('nan')}):
        with pytest.raises(ValidationError):
            StrainSensitivitySettings(**invalid)


def test_both_deployments_and_settings_reanalysis_wiring():
    archive = Path('static/archive.html').read_text()
    assert archive.index('id="syncArchiveTransferPathNormalizedPlot"') < archive.index('id="syncArchiveStrainSensitivityPlot"')
    assert '/strain-sensitivity.js' in Path('archive_main.py').read_text()
    for field in StrainSensitivitySettings.model_fields:
        assert f'v-model.number="analysis.{field}"' in archive or f'v-model="analysis.{field}"' in archive
        settings = Path('static/settings.html').read_text()
        assert f'v-model.number="s.{field}"' in settings or f'v-model="s.{field}"' in settings


def test_reanalysis_copy_persists_noise_inputs(tmp_path, monkeypatch):
    from app.core.data_manager import DataManager, RESULTS_CSV_HEADER
    acquisition = tmp_path / 'raw'
    acquisition.mkdir()
    destination = tmp_path / 'analysis'
    destination.mkdir()
    (destination / 'config.json').write_text(json.dumps({'mode': 'transfer_function'}))
    (destination / 'results.csv').write_text(','.join(RESULTS_CSV_HEADER) + '\n')
    monkeypatch.setattr('app.core.data_manager.config.DATA_BASE_DIR', str(acquisition))
    settings = {**StrainSensitivitySettings().model_dump(), 'transfer_strain_noise_model': 'fixed_asd', 'transfer_strain_phase_asd_rad_sqrt_hz': .31}
    DataManager().overwrite_run('2026','10','23','run00', settings, [], [], target_directory=destination)
    saved = json.loads((destination / 'config.json').read_text())
    for key, value in settings.items():
        assert saved[key] == value
        assert saved['_analysis_snapshot'][key] == value
        assert saved['_system_settings_snapshot'][key] == value


def test_acquisition_config_captures_strain_settings():
    from app.core.experiment_manager import ExperimentManager
    manager = object.__new__(ExperimentManager)
    manager.settings = {**StrainSensitivitySettings().model_dump(), 'transfer_strain_sigma_mrad': 90}
    config = {'mode':'transfer_function', 'scan_dimensions':1, 'parameter_source':'classic',
              'transfer_frequency_source':'list', 'transfer_frequency_list_hz':'10', 'transfer_repeats':2}
    manager._build_transfer_function_execution(config)
    assert config['transfer_strain_sigma_mrad'] == 90
    assert config['transfer_strain_cycle_time_s'] == 2.6
    # ScanConfig is the SYNC boundary: it must retain the new fields.
    assert ScanConfig(**config).model_dump()['transfer_strain_sigma_mrad'] == 90
