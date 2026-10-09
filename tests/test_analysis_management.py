from pathlib import Path
from unittest.mock import patch
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.analysis_store import AnalysisStore, fingerprint, capture_inputs, capture_result
from app.core.data_loader import DataLoader
from app.core.data_manager import DataManager
from app.models.schemas import ArchiveAllanRequest
from app.archive.controller_context import ControllerRepository
from app.archive.ui_context import versions, resolve


REFERENCE = {'year': '2026', 'month': '10', 'day': '09', 'run_id': 'run01'}
REF = list(REFERENCE.values())
SETTINGS = dict(alpha=0.2, beta=0.1, R=1, K=1, z_up=0, z_dw=0, launch_velocity=1, chan_launch='0', chan_trigger='1', gain_up=1, gain_dw=1)


@pytest.fixture
def controller(tmp_path, monkeypatch):
    import config
    from app.api import routes
    raw = tmp_path / 'Data_log'
    run = raw.joinpath(*REF)
    run.mkdir(parents=True)
    (run / 'config.json').write_text('{"mode":"standard","scan_dimensions":1}')
    (run / 'results.csv').write_text('Step,Timestamp,Parameter_P0,Atom_UP,Atom_DW\n' +
        ''.join(f'{i},{i},1,{100+i},{110-i}\n' for i in range(12)))
    (run / 'waveforms').mkdir()
    (run / 'waveforms' / 'evidence.npz').write_bytes(b'original waveform evidence')
    loader = DataLoader()
    loader.base_dir = raw
    monkeypatch.setattr(config, 'DATA_BASE_DIR', raw)
    monkeypatch.setattr(routes, 'data_loader', loader)
    monkeypatch.setattr(routes.manager, 'get_active_bragg_phase_calibration', lambda: None)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client, loader, run


def test_controller_allan_save_restore_export_and_input_change(controller):
    client, loader, raw = controller
    before = fingerprint(raw)
    computed = client.post('/archive/allan', json={**REFERENCE, 'order': 4, 'metric': 'atoms', 'source': 'fit', 'new_settings': SETTINGS})
    assert computed.status_code == 200, computed.text
    result = computed.json()
    assert result['analysis_candidate_id']
    assert client.get('/archive/analysis-results/' + '/'.join(REF)).json()['analyses'] == []
    # Saving and restoring never invoke the calculator again.
    with patch.object(loader, 'calculate_allan_run', side_effect=AssertionError('Unexpected recalculation')):
        saved = client.post('/archive/analysis-results/save', json={**REFERENCE, 'candidate_id': result['analysis_candidate_id'], 'name': 'Allan baseline', 'note': '12 shots'})
        assert saved.status_code == 200, saved.text
        record = saved.json()
        assert record['result'] == {key: value for key, value in result.items() if key != 'analysis_candidate_id'}
        assert record['parameters']['order'] == 4
        url = '/archive/analysis-results/' + '/'.join(REF) + '/' + record['id']
        restored = client.get(url).json()
        assert restored['input_matches_current'] is True
        assert restored['source']['raw_sha256'] == before
        exported = client.get(url + '?download=true')
        assert 'attachment' in exported.headers['content-disposition']
        assert exported.json()['result'] == record['result']
        second = client.post('/archive/analysis-results/save', json={**REFERENCE, 'candidate_id': result['analysis_candidate_id'], 'name': 'Second version'}).json()
        assert second['id'] != record['id']
        assert client.get(url).json()['name'] == 'Allan baseline'
    assert fingerprint(raw) == before
    # Historical results survive changed inputs and report that mismatch.
    (raw / 'waveforms' / 'evidence.npz').write_bytes(b'externally changed evidence')
    assert client.get(url).json()['input_matches_current'] is False
    assert client.get(url).json()['result'] == record['result']
    assert client.post('/archive/analysis-results/save', json={**REFERENCE, 'candidate_id': '../bad', 'name': 'Bad'}).status_code == 400
    assert client.post('/archive/analysis-results/save', json={**REFERENCE, 'candidate_id': result['analysis_candidate_id'], 'name': '   '}).status_code == 400
    assert len(client.get('/archive/analysis-results/' + '/'.join(REF)).json()['analyses']) == 2


def test_controller_overwrite_and_alpha_saves_are_isolated(controller):
    client, loader, raw = controller
    before = fingerprint(raw)
    with patch.object(loader, 'recalculate_run', return_value={'data': [{'step': 0, 'atom_number_up': 321}], 'config': {}}):
        saved = client.post('/archive/overwrite', json={**REFERENCE, 'new_settings': SETTINGS})
    assert saved.status_code == 200, saved.text
    repository = ControllerRepository(loader.base_dir)
    entries = versions(repository, 'controller', REF)
    assert len(entries) == 1
    first = entries[0]['id']
    selected = resolve(repository, 'controller', REF, first)
    assert (selected / 'waveforms').is_symlink()
    assert '321' in (selected / 'results.csv').read_text()
    assert fingerprint(raw) == before
    # Save a metadata-only operation on a selected historical branch.
    saved_alpha = client.post('/archive/intf-alpha/reanalyze', headers={'X-Archive-Version': first},
        json={**REFERENCE, 'save': True, 'name': 'Alpha copy'})
    assert saved_alpha.status_code == 200, saved_alpha.text
    entries = versions(repository, 'controller', REF)
    assert len(entries) == 2
    assert entries[0]['parent'] == first
    assert not (selected / 'intf_alpha_analysis_copies.json').exists()
    assert fingerprint(raw) == before
    failed = client.post('/archive/intf-alpha/reanalyze', json={**REFERENCE, 'save': True, 'name': ''})
    assert failed.status_code == 400
    assert len(versions(repository, 'controller', REF)) == 2
    assert len(list(repository.analysis_dir('controller', REF).joinpath('ui-versions').glob('*/run'))) == 2
    assert fingerprint(raw) == before
    # HTTP-free legacy overwrite calls are also blocked.
    with pytest.raises(ValueError):
        DataManager().overwrite_run(*REF, {}, [])
    with pytest.raises(ValueError):
        DataManager().overwrite_run(*REF, {}, [], target_directory=raw)


def test_changed_during_calculation_is_not_publishable(tmp_path):
    loader = DataLoader()
    loader.base_dir = tmp_path
    run = tmp_path.joinpath(*REF)
    run.mkdir(parents=True)
    (run / 'results.csv').write_text('original')
    req = ArchiveAllanRequest(**REFERENCE, new_settings=SETTINGS)
    source = capture_inputs(loader, req)
    (run / 'results.csv').write_text('changed acquisition')
    with pytest.raises(ValueError, match='during calculation'):
        capture_result(loader, req, {}, {}, 'allan', source)
    assert not (tmp_path.parent / (tmp_path.name + '_analysis')).exists()


def test_fingerprint_includes_linked_waveforms(tmp_path):
    raw = tmp_path / 'raw'
    (raw / 'waveforms').mkdir(parents=True)
    (raw / 'waveforms' / 'shot.npz').write_bytes(b'waveform')
    clone = tmp_path / 'clone'
    clone.mkdir()
    (clone / 'waveforms').symlink_to(raw / 'waveforms', target_is_directory=True)
    assert fingerprint(clone) == fingerprint(raw)
    (raw / 'waveforms' / 'shot.npz').write_bytes(b'changed')
    assert fingerprint(clone) == fingerprint(raw)


def test_phase_noise_result_can_be_saved_and_restored_after_draft_cleanup(controller):
    client, loader, raw = controller
    (raw / 'config.json').write_text('{"mode":"phase_noise","scan_dimensions":1,"_interferometer_phase_calibration_snapshot":null}')
    (raw / 'results.csv').write_text('Step,Timestamp,Parameter_P0,Interferometer_Phase_Rad,Interferometer_Phase_Valid,Interferometer_Phase_Reference_T2_us2\n' +
        ''.join(f'{i},{i},100,{i%3*.01},1,100\n' for i in range(12)))
    before = fingerprint(raw)
    response = client.post('/archive/phase-noise/allan', json={**REFERENCE, 'orders': [1, 3], 'new_settings': SETTINGS})
    assert response.status_code == 200, response.text
    computed = response.json()
    assert computed['phase_noise_summary'][0]['allan_deviations'][1]['order'] == 3
    saved = client.post('/archive/analysis-results/save', json={**REFERENCE, 'candidate_id': computed['analysis_candidate_id'], 'name': 'Phase noise Allan'}).json()
    store = AnalysisStore(ControllerRepository(loader.base_dir).analysis_dir('controller', REF) / 'results')
    store._path('drafts', computed['analysis_candidate_id']).unlink()
    with patch.object(loader, 'load_run', side_effect=AssertionError('Restore must not load shots')):
        url = '/archive/analysis-results/' + '/'.join(REF) + '/' + saved['id']
        assert client.get(url).json()['result'] == saved['result']
        new_saved = client.post('/archive/analysis-results/save', json={**REFERENCE, 'candidate_id': saved['id'], 'name': 'New snapshot'})
        assert new_saved.status_code == 200, new_saved.text
        assert new_saved.json()['id'] != saved['id']
        assert new_saved.json()['result'] == saved['result']
    assert fingerprint(raw) == before


def test_browser_plot_snapshot_preserves_values_and_rejects_changed_inputs(tmp_path):
    store = AnalysisStore(tmp_path)
    candidate = store.candidate('phase_noise_allan', {'orders': [1]}, {}, {'input_sha256': 'original'})
    plot = {'traces': [{'x': [1, 2], 'y': [.12, .08]}], 'layout': {'title': 'Single T'}, 'origin': 'browser'}
    record = store.save(candidate, 'Single T', view={'plot_snapshot': plot}, input_sha256='original')
    assert record['result']['plot_snapshot'] == plot
    assert 'plot_snapshot' not in record['view']
    with pytest.raises(ValueError, match='Input changed'):
        store.save(candidate, 'Changed', view={'plot_snapshot': plot}, input_sha256='changed')
    # An unchanged historical snapshot can be copied even after its input changed.
    copied = store.save(record['id'], 'Historical copy', view={'plot_snapshot': plot}, input_sha256='changed')
    assert copied['result']['plot_snapshot'] == plot
