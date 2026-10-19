from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from app.analysis.transfer_amplitude import AmplitudeProblem
from app.analysis.transfer_function import build_differential_transfer_function_summary
from app.core.data_loader import DataLoader


def make_problem(tmp_path, monkeypatch, legacy_slave=False):
    root = tmp_path / '2026/10/19/run01'
    slave = root / 'sync_nodes/slave-a'
    slave.mkdir(parents=True)
    cal = {'id': 'cal', 'parameter_values': {'A': 20., 'C': 50., 'phi0': 0.},
        'bragg': {'angular_frequency_rad_per_us2': .01}, 'reference_t2_us2': 100., 'monotonic_slope': 'negative'}
    for directory in (root, slave):
        config = {'mode': 'transfer_function', '_interferometer_phase_calibration_snapshot': cal,
            'transfer_baseline_calibration_version': 1 if directory == slave and legacy_slave else 2}
        (directory / 'config.json').write_text(json.dumps(config))
        (directory / 'results.csv').write_text('original results')
    points = {root: [], slave: []}
    for directory in points:
        points[directory].append({'intf_p1': 51., 'transfer_zero_phase_baseline': True, 'transfer_zero_phase_block_id': 0})
    pairs = []
    for frequency in (10., 20.):
        for phase in (0., 90.):
            for k in range(4):
                shot = len(pairs)
                rows = []
                for index, directory in enumerate((root, slave)):
                    signal = 51 + (2 + .1 * k) * (1 if phase == 0 else .6) * (1.025 if index == 0 else 1)
                    row = {'intf_p1': signal, 'transfer_zero_phase_block_id': 0,
                        'sync_shot_index': shot, 'step': shot, 'parameter': 100.,
                        'transfer_frequency_hz': frequency, 'transfer_phase_deg': phase,
                        'transfer_frequency_modulation_mhz': 1., 'transfer_atom_mirror_distance_m': 2.23}
                    rows.append(row)
                    points[directory].append(row)
                pairs.append({'slave_node_id': 'slave-a', 'master': rows[0], 'slave': rows[1]})
    manifest = {'archive_nodes': {'master': {'path': '.', 'local': True},
        'slave-a': {'path': 'sync_nodes/slave-a', 'local': False}}, 'pairs': pairs}
    (root / 'sync_manifest.json').write_text(json.dumps(manifest))
    loader = DataLoader()
    loader.base_dir = tmp_path
    monkeypatch.setattr(loader, '_read_results_csv', lambda directory, **kwargs: deepcopy(points[directory]))
    return AmplitudeProblem(loader, ['2026', '10', '19', 'run01'], 'slave-a', 10), loader, root, points


def test_joint_optimization_matches_applied_production_summary(tmp_path, monkeypatch):
    problem, loader, root, points = make_problem(tmp_path, monkeypatch)
    original = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = problem.optimize(.05, .05)
    assert result['after']['differential_s2'] < result['before']['differential_s2'] * 1e-5
    assert result['pair_count'] == 8
    assert len(result['preview_rows']) == 2
    for node in problem.nodes:
        assert 19 <= result['parameters_after'][node] <= 21
    assert not (root / 'sync_phase_analysis.json').exists()
    problem.apply(result)
    contexts = loader.archive_phase_reference_contexts(*problem.reference)
    for node in problem.nodes:
        assert contexts[node]['effective_calibration']['parameter_values']['C'] == 50
    applied = loader._apply_sync_phase_reference_overrides(problem.manifest, contexts, root)
    rows = build_differential_transfer_function_summary(applied['pairs'])
    for expected, actual in zip(result['preview_rows'], rows):
        assert actual['differential_s2'] == pytest.approx(expected['differential_s2'], abs=1e-15)
    assert all(p.read_bytes() == content for p, content in original.items())


def test_each_node_keeps_its_own_baseline_mode(tmp_path, monkeypatch):
    problem, loader, root, points = make_problem(tmp_path, monkeypatch, legacy_slave=True)
    pair = problem.convert([20, 20], problem.target_manifest)[0]
    assert pair['master']['interferometer_phase'] == pytest.approx(-math.asin(2.05 / 20))
    assert pair['slave']['interferometer_phase'] == pytest.approx(math.acos(3 / 20) - math.acos(1 / 20))
    assert problem.objective([1, 1]) == pytest.approx(1.)


def test_stale_input_and_out_of_bounds_apply_rejected(tmp_path, monkeypatch):
    problem, loader, root, points = make_problem(tmp_path, monkeypatch)
    result = problem.optimize(.05, .03)
    bad = deepcopy(result)
    bad['parameters_after']['master'] = 25
    with pytest.raises(ValueError, match='allowed ranges'):
        problem.apply(bad)
    points[root][1]['intf_p1'] += .1
    updated = AmplitudeProblem(loader, problem.reference, 'slave-a', 10)
    with pytest.raises(ValueError, match='changed'):
        updated.apply(result)


def test_missing_quadrature_and_empty_shot_range_rejected(tmp_path, monkeypatch):
    problem, loader, root, points = make_problem(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='quadratures'):
        AmplitudeProblem(loader, problem.reference, 'slave-a', 10, 0, 3)
    with pytest.raises(ValueError, match='quadratures'):
        AmplitudeProblem(loader, problem.reference, 'slave-a', 10, 1000, 2000)


def test_candidate_cannot_drop_fixed_pairs(tmp_path, monkeypatch):
    problem, loader, root, points = make_problem(tmp_path, monkeypatch)
    assert problem.objective([.05, .05]) == 1e12
    for bound in (0, -.1, float('nan'), .6):
        with pytest.raises(ValueError, match='ranges'):
            problem.optimize(bound, .05)


def test_server_preview_then_apply_retains_originals_and_matches_preview(tmp_path, monkeypatch):
    import shutil
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.archive.configuration import ArchiveServerConfiguration
    from app.archive.repository import ArchiveRepository
    from app.archive import ui_context
    from app.archive.ui_routes import router
    problem, loader, root, points = make_problem(tmp_path / 'fixture', monkeypatch)
    source_points = {path.name: rows for path, rows in points.items()}
    nas = tmp_path / 'nas'
    nas.mkdir()
    configuration = ArchiveServerConfiguration(tmp_path / 'configuration.json')
    configuration.initialize(str(nas))
    configuration.register_device({'device_id': 'master', 'role': 'master', 'transport': 'local', 'source_path': str(root)})
    repository = ArchiveRepository(configuration)
    target = repository.device_root('master') / 'runs' / Path(*problem.reference)
    shutil.copytree(root, target)
    monkeypatch.setattr(DataLoader, '_read_results_csv', lambda self, directory, **kwargs: deepcopy(source_points[directory.name if directory.name == 'slave-a' else root.name]))
    monkeypatch.setattr(ui_context, '_repository', repository)
    monkeypatch.setattr(ui_context, '_audits', {})
    app = FastAPI()
    app.include_router(router, prefix='/view/{device_id}')
    client = TestClient(app)
    payload = dict(zip(('year', 'month', 'day', 'run_id'), problem.reference))
    payload.update(slave_node_id='slave-a', transfer_frequency_hz=10, master_bound_fraction=.05, slave_bound_fraction=.05)
    response = client.post('/view/master/archive/sync-transfer-amplitude-optimize', json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert not ui_context.versions(repository, 'master', problem.reference)
    assert not (target / 'sync_phase_analysis.json').exists()
    response = client.post('/view/master/archive/sync-transfer-amplitude-optimization/apply', json={**payload, 'result': result})
    assert response.status_code == 200, response.text
    assert not (target / 'sync_phase_analysis.json').exists()
    latest = ui_context.resolve(repository, 'master', problem.reference)
    updated = repository.loader('master')
    updated._get_run_dir = lambda *unused: latest
    contexts = updated.archive_phase_reference_contexts(*problem.reference)
    rebuilt = updated._apply_sync_phase_reference_overrides(updated._load_sync_manifest_payload(latest), contexts, latest)
    row = build_differential_transfer_function_summary(rebuilt['pairs'])[0]
    assert row['differential_s2'] == pytest.approx(result['after']['differential_s2'], abs=1e-15)


def test_saved_normalization_and_shot_selection_are_used(tmp_path, monkeypatch):
    problem, loader, root, points = make_problem(tmp_path, monkeypatch)
    (root / 'sync_transfer_normalization.json').write_text(json.dumps({'slave_scales': {'slave-a': 1.1}}))
    selected = AmplitudeProblem(loader, problem.reference, 'slave-a', 10, 1, 6)
    assert selected.before['delta_s_0deg_count'] == 3
    assert selected.before['delta_s_90deg_count'] == 3
    assert selected.scales['slave-a'] == 1.1
    pairs = selected.convert(selected.initial, selected.target_manifest)
    expected = build_differential_transfer_function_summary(pairs, {'slave-a': 1.1})[0]
    assert selected.before['differential_s2'] == pytest.approx(expected['differential_s2'])


def test_controller_apply_also_creates_an_independent_version(tmp_path, monkeypatch):
    import config
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.transfer_amplitude_routes import build_transfer_amplitude_router
    from app.archive.controller_context import ControllerArchiveRoute, ControllerRepository
    from app.archive.ui_context import resolve
    problem, loader, root, points = make_problem(tmp_path, monkeypatch)
    monkeypatch.setattr(config, 'DATA_BASE_DIR', tmp_path)
    monkeypatch.setattr(loader, '_read_results_csv', lambda directory, **kwargs: deepcopy(points[root / 'sync_nodes/slave-a' if directory.name == 'slave-a' else root]))
    async def synchronize(*args):
        return {'status': 'local'}
    app = FastAPI()
    app.include_router(build_transfer_amplitude_router(lambda: loader, ControllerArchiveRoute, synchronize))
    client = TestClient(app)
    payload = dict(zip(('year', 'month', 'day', 'run_id'), problem.reference))
    payload.update(slave_node_id='slave-a', transfer_frequency_hz=10)
    response = client.post('/archive/sync-transfer-amplitude-optimize', json=payload)
    assert response.status_code == 200, response.text
    response = client.post('/archive/sync-transfer-amplitude-optimization/apply', json={**payload, 'result': response.json()})
    assert response.status_code == 200, response.text
    latest = resolve(ControllerRepository(), 'controller', problem.reference)
    assert latest != root
    assert (latest / 'sync_phase_analysis.json').is_file()
    assert not (root / 'sync_phase_analysis.json').exists()
