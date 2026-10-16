"""Protect recorded TF values when adding uncertainties or loading SYNC archives."""
import asyncio
import base64
from copy import deepcopy
import importlib
import json
import lzma
import math
from pathlib import Path
import shutil
import struct
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree as ET

import pytest

from app.analysis import interferometer_phase
from app.analysis.transfer_function import build_differential_transfer_function_summary
from app.core.archive_labplot import build_archive_project
from app.core.data_loader import DataLoader


def calibration():
    return dict(id='test-fringe', parameter_values=dict(A=20, C=50, phi0=0),
                bragg=dict(angular_frequency_rad_per_us2=1), reference_t2_us2=math.pi / 2,
                monotonic_slope='negative', phase_conversion_mode='monotonic_half_fringe')


def archive_fixture():
    """Recorded science phases are zeroed, but probabilities retain a DC offset."""
    pairs = []
    results = {'master': [], 'slave': []}
    for index, (phase, response) in enumerate([(0, .03), (0, .05), (90, -.02), (90, -.04)]):
        records = {}
        for node, distance, offset in [('master', 2.23, -.3), ('slave', 3.77, .2)]:
            recorded = response * distance / 2.23
            records[node] = dict(
                step=index + 1, sync_shot_index=index + 1, parameter=101,
                transfer_frequency_hz=101, transfer_phase_deg=phase,
                transfer_frequency_modulation_mhz=1.58, transfer_atom_mirror_distance_m=distance,
                transfer_phase_noise_sigma_mrad=61, transfer_zero_phase_block_id=0,
                intf_p1=50 + 20 * math.cos(math.pi / 2 + recorded + offset),
                interferometer_phase=recorded, interferometer_phase_raw=recorded + offset,
                interferometer_phase_valid=True, interferometer_phase_calibration_id='test-fringe',
            )
            results[node].append(deepcopy(records[node]))
        pairs.append(dict(slave_node_id='slave', master=records['master'], slave=records['slave']))
    # Baselines are unpaired and only present in node_results.
    for node, offset in [('master', -.3), ('slave', .2)]:
        results[node].insert(0, dict(
            step=0, sync_shot_index=0, parameter=101,
            transfer_frequency_hz=101, transfer_phase_deg=0,
            transfer_zero_phase_baseline=True, transfer_zero_phase_block_id=0,
            intf_p1=50 + 20 * math.cos(math.pi / 2 + offset),
            interferometer_phase=offset, interferometer_phase_raw=offset,
            interferometer_phase_valid=True,
        ))
    manifest = dict(
        archive_nodes={'master': {'path': '.', 'local': True},
                       'slave': {'path': 'sync_nodes/slave', 'local': False}},
        pairs=pairs, node_results={'master': results['master'], 'slaves': {'slave': results['slave']}},
        transfer_function_differential_formula_version=9,
        transfer_function_differential_summary=build_differential_transfer_function_summary(pairs),
    )
    contexts = {node: {'effective_calibration': calibration(), 'has_override': False}
                for node in ('master', 'slave')}
    return manifest, contexts, results


def browser_rows(manifest, contexts=None, scales=None, preview=None, force_version=None):
    source = Path('static/archive.html').read_text()
    methods = source[source.index('                withTransferUncertainties(input) {'):
                     source.index('                syncArchiveHasPhaseOverrides()')]
    state = dict(syncManifest=manifest, syncArchiveSlaveId='slave',
                 syncTransferNormalizationOptimization={'result': preview},
                 syncTransferNormalization={'slave_scales': scales or {}},
                 syncAnalysisCopy={'selectedId': ''}, contexts=contexts or {})
    if force_version is not None:
        state['syncManifest'] = {**manifest, 'transfer_function_differential_formula_version': force_version}
    script = 'const app={' + methods + '''
        syncArchiveHasPhaseOverrides(){return Object.values(this.contexts).some(c=>c.has_override);},
        syncArchivePairs(){return this.syncManifest.pairs;}
    }; Object.assign(app, ''' + json.dumps(state) + ''');
    console.log(JSON.stringify(app.syncArchiveTransferDifferentialRows()));'''
    return json.loads(subprocess.run(['node'], input=script, capture_output=True, text=True, check=True).stdout)


def test_plain_load_preserves_recorded_zeroed_phases_and_validity():
    manifest, contexts, _ = archive_fixture()
    original = deepcopy(manifest)
    output = DataLoader.__new__(DataLoader)._apply_sync_phase_reference_overrides(manifest, contexts)
    assert manifest == original
    assert output['pairs'] == original['pairs']
    assert output['node_results'] == original['node_results']
    assert build_differential_transfer_function_summary(output['pairs'])[0]['differential_s2'] == pytest.approx(0, abs=1e-25)


@pytest.mark.skipif(not shutil.which('node'), reason='Node required for Archive calculations')
def test_version9_summary_survives_even_if_pairs_were_reconverted():
    manifest, contexts, _ = archive_fixture()
    saved = deepcopy(manifest['transfer_function_differential_summary'][0])
    # Reproduce the old bug's divergent pairs. Version 9 must remain compatible.
    for pair in manifest['pairs']:
        pair['master'] = interferometer_phase.apply_phase(pair['master'], calibration())
    for preview in (None, {'scale_after': None}):
        row = browser_rows(manifest, contexts, preview=preview)[0]
        for field in ('differential_s2', 'delta_s_0deg_mean', 'delta_s_90deg_mean'):
            assert row[field] == saved[field]
        assert row['differential_s2_sem'] is not None


@pytest.mark.parametrize('version', [1, 2])
def test_override_rebases_each_node_before_differencing(tmp_path, version):
    manifest, contexts, results = archive_fixture()
    loader = DataLoader()
    for node in ('master', 'slave'):
        node_dir = tmp_path if node == 'master' else tmp_path / 'sync_nodes' / node
        node_dir.mkdir(parents=True, exist_ok=True)
        config = dict(mode='transfer_function', transfer_calibrate_zero_phase=True,
                      transfer_baseline_calibration_version=version)
        (node_dir / 'config.json').write_text(json.dumps(config))
        contexts[node]['has_override'] = True
    (tmp_path / 'sync_manifest.json').write_text(json.dumps(manifest))
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with patch.object(loader, '_read_results_csv', return_value=[]):
        output = loader._apply_sync_phase_reference_overrides(manifest, contexts, tmp_path)
    for pair in output['pairs']:
        for role, node in [('master', 'master'), ('slave', 'slave')]:
            expected = loader._rebase_converted_transfer_phases(
                [interferometer_phase.apply_phase(record, calibration()) for record in results[node]],
                dict(transfer_baseline_calibration_version=version), calibration())
            shot = pair[role]['sync_shot_index']
            assert pair[role]['interferometer_phase'] == pytest.approx(expected[shot]['interferometer_phase'])
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before
    if version == 1:
        assert build_differential_transfer_function_summary(output['pairs'])[0]['differential_s2'] == pytest.approx(0, abs=1e-25)


def test_override_loads_unpaired_baselines_from_each_nodes_csv(tmp_path):
    manifest, contexts, results = archive_fixture()
    manifest['node_results'] = {'master': [], 'slaves': {}}
    for node in contexts:
        contexts[node]['has_override'] = True
        folder = tmp_path if node == 'master' else tmp_path / 'sync_nodes' / node
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'config.json').write_text(json.dumps({'mode':'transfer_function'}))
    (tmp_path / 'sync_manifest.json').write_text(json.dumps(manifest))
    loader = DataLoader()
    with patch.object(loader, '_read_results_csv', side_effect=lambda folder, **kw: deepcopy(results['master' if folder == tmp_path else 'slave'])):
        output = loader._apply_sync_phase_reference_overrides(manifest, contexts, tmp_path)
    assert build_differential_transfer_function_summary(output['pairs'])[0]['differential_s2'] == pytest.approx(0, abs=1e-25)


def test_missing_baseline_does_not_return_unzeroed_science_phase():
    manifest, contexts, _ = archive_fixture()
    manifest['node_results'] = {}
    contexts['master']['has_override'] = True
    output = DataLoader.__new__(DataLoader)._apply_sync_phase_reference_overrides(manifest, contexts)
    for pair in output['pairs']:
        assert pair['master']['interferometer_phase'] is None
        assert pair['master']['interferometer_phase_valid'] is False
        assert pair['master']['transfer_zero_phase_reference_missing'] is True
        assert pair['slave']['interferometer_phase'] == manifest['pairs'][output['pairs'].index(pair)]['slave']['interferometer_phase']
    assert not build_differential_transfer_function_summary(output['pairs'])


@pytest.mark.skipif(not shutil.which('node'), reason='Node required for Archive calculations')
def test_frontend_and_backend_filter_the_same_non_science_pairs():
    manifest, _, _ = archive_fixture()
    clean = deepcopy(manifest['pairs'])
    for flags in [dict(transfer_zero_phase_baseline=True), dict(intf_alpha_calibration_block_id='alpha'),
                  dict(power_meter_invalid_reason='threshold_exceeded'), dict(error='bad shot'),
                  dict(interferometer_phase=None), dict(transfer_frequency_hz=None)]:
        pair = deepcopy(clean[0])
        pair['master'].update(flags)
        pair['slave'].update(flags)
        manifest['pairs'].append(pair)
    expected = build_differential_transfer_function_summary(clean)[0]
    actual = build_differential_transfer_function_summary(manifest['pairs'])[0]
    browser = browser_rows(manifest, force_version=8)[0]
    for key in ('delta_s_0deg_count', 'delta_s_90deg_count', 'phase_difference_0deg_count', 'differential_s2'):
        assert actual[key] == pytest.approx(expected[key])
        assert browser[key] == pytest.approx(expected[key], abs=1e-25)


@pytest.mark.parametrize('with_override', [False, True])
@pytest.mark.parametrize('module_name', ['app.api.routes', 'app.archive.ui_routes'])
def test_both_archive_load_routes_preserve_tf_science_data(tmp_path, module_name, monkeypatch, with_override):
    manifest, contexts, results = archive_fixture()
    for node in ('master', 'slave'):
        folder = tmp_path if node == 'master' else tmp_path / 'sync_nodes' / node
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'config.json').write_text(json.dumps(dict(
            mode='transfer_function', scan_dimensions=1, transfer_frequency_modulation_mhz=1.58,
            transfer_atom_mirror_distance_m=2.23 if node == 'master' else 3.77,
            _interferometer_phase_calibration_snapshot=calibration())))
    (tmp_path / 'sync_manifest.json').write_text(json.dumps(manifest))
    loader = DataLoader()
    monkeypatch.setattr(loader, '_get_run_dir', lambda *a: tmp_path)
    if with_override:
        for context in contexts.values():
            context['has_override'] = True
            context['effective_calibration']['reference_t2_us2'] += .15
        monkeypatch.setattr(loader, '_archive_phase_reference_context', lambda *a: ('master', contexts['master'], contexts))
    monkeypatch.setattr(loader, '_read_results_csv', lambda folder, **kw: deepcopy(results['master' if folder == tmp_path else 'slave']))
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, 'data_loader', loader)
    monkeypatch.setattr(module, 'manager', SimpleNamespace(get_active_bragg_phase_calibration=lambda: calibration()))
    loaded = asyncio.run(module.load_archived_run('2026','09','30','run48',node_id='master'))
    for actual, original in zip(loaded['sync_manifest']['pairs'], manifest['pairs']):
        assert actual['master']['interferometer_phase'] == pytest.approx(original['master']['interferometer_phase'])
        assert actual['slave']['interferometer_phase'] == pytest.approx(original['slave']['interferometer_phase'])


@pytest.mark.parametrize('scale', [None, 1.1])
@pytest.mark.parametrize('module_name', ['app.api.routes', 'app.archive.ui_routes'])
def test_both_labplot_routes_keep_version9_values(module_name, monkeypatch, scale):
    manifest, contexts, _ = archive_fixture()
    # Make an intentionally distinguishable archived result to check selection.
    manifest['transfer_function_differential_summary'][0]['differential_s2'] = .1073915578647957
    class Loader:
        def load_run(self, *args, **kwargs):
            return dict(config={'mode':'transfer_function'}, sync_manifest=manifest,
                        archive_phase_reference_contexts=contexts,
                        sync_transfer_normalization={'slave_scales': {'slave': scale}} if scale else {})
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, 'data_loader', Loader())
    monkeypatch.setattr(module, 'manager', SimpleNamespace(get_active_bragg_phase_calibration=lambda: calibration()))
    from app.models.schemas import ArchiveLabPlotExportRequest
    request = ArchiveLabPlotExportRequest(year='2026',month='09',day='30',run_id='run48',metrics=['phase'])
    response = asyncio.run(module.export_archive_labplot(request))
    root = ET.fromstring(lzma.decompress(response.body))
    sheet = next(element for element in root.iter('spreadsheet') if element.get('name') == 'Data - SYNC Transfer S2 - slave')
    column = next(element for element in sheet.iter('column') if element.get('name') == 'P1_Quadrature sum')
    values = struct.unpack('=d', base64.b64decode(column.find('output_filter').tail))
    if scale is None:
        assert values[0] == manifest['transfer_function_differential_summary'][0]['differential_s2']
    else:
        expected = build_differential_transfer_function_summary(manifest['pairs'], {'slave': scale})[0]
        assert values[0] == pytest.approx(expected['differential_s2'])


def test_legacy_path_error_backfill_requires_matching_recorded_mean():
    manifest, contexts, _ = archive_fixture()
    saved = manifest['transfer_function_differential_summary'][0]
    for phase in ('0deg', '90deg'):
        for statistic in ('count', 'std', 'sem', 's2_sem'):
            saved.pop(f'phase_path_normalized_{phase}_{statistic}', None)
    saved['phase_path_normalized_0deg_mean'] += .7
    original_s2 = saved['differential_s2']
    output = DataLoader.__new__(DataLoader)._apply_sync_phase_reference_overrides(manifest, contexts)
    row = output['transfer_function_differential_summary'][0]
    assert row['differential_s2'] == original_s2
    assert 'phase_path_normalized_0deg_std' not in row
    assert row['phase_path_normalized_0deg_s2_sem'] is None
    assert row['phase_path_normalized_90deg_std'] > 0
    assert row['phase_path_normalized_90deg_s2_sem'] > 0
