import json
from pathlib import Path
from unittest.mock import patch

from app.archive.configuration import ArchiveServerConfiguration
from app.archive.repository import ArchiveRepository
from app.archive.sync_links import attach_sync_links, pin_sync_links, find_links
from app.archive.ui_context import clone_metadata


def setup_archives(tmp_path):
    nas = tmp_path / 'nas'
    nas.mkdir()
    configuration = ArchiveServerConfiguration(tmp_path / 'config.json')
    configuration.initialize(str(nas))
    for device in ('master', 'slave'):
        configuration.register_device({'device_id': device, 'role': device,
            'transport': 'local', 'source_path': str(tmp_path / device)})
    repository = ArchiveRepository(configuration)
    ref = ['2026', '09', '30', 'run48_20260930']
    slave_ref = ['2026', '09', '30', 'run55_20260930']
    master = repository.device_root('master') / 'runs' / Path(*ref)
    slave = repository.device_root('slave') / 'runs' / Path(*slave_ref)
    calibration = {'id': 'fringe', 'name': 'Fringe', 'parameter_values': {'A': 20, 'C': 50, 'phi0': 0},
        'bragg': {'angular_frequency_rad_per_us2': .01}, 'reference_t2_us2': 100}
    for directory, role, identity in ((master, 'master', 'Master'), (slave, 'slave', 'Slave')):
        directory.mkdir(parents=True)
        (directory / 'config.json').write_text(json.dumps({'sync_run_id': 'sync-test',
            'sync_role': role, 'sync_node_id': identity, 'mode': 'transfer_function',
            '_interferometer_phase_calibration_snapshot': calibration}))
        (directory / 'results.csv').write_text('Step,Parameter_P0\n0,100\n')
        (directory / 'waveforms').mkdir()
        (directory / 'waveforms/step.npz').write_bytes(b'waveform')
    manifest = {'archive_nodes': {'master': {'path': '.', 'local': True}},
        'runtime': {'sync_run_id': 'sync-test', 'slaves': [{'node_id': 'slave-a',
            'name': 'Slave', 'run_id': slave_ref[-1]}]},
        'pairs': [{'slave_node_id': 'slave-a', 'master': {'intf_p1': 50}, 'slave': {'intf_p1': 50}}]}
    (master / 'sync_manifest.json').write_text(json.dumps(manifest))
    return repository, ref, master, slave


def test_link_reads_slave_without_changing_originals(tmp_path):
    repository, ref, master, slave = setup_archives(tmp_path)
    original = (master / 'sync_manifest.json').read_bytes()
    loader = repository.loader('master')
    attach_sync_links(loader, repository)
    directory, identity = loader._resolve_archive_node_identity(master, 'slave-a')
    assert directory == slave
    assert identity['resolved_node'] == 'slave-a'
    contexts = loader.archive_phase_reference_contexts(*ref)
    assert contexts['slave-a']['effective_calibration']['parameter_values']['A'] == 20
    assert loader._load_sync_manifest_payload(master)['archive_nodes']['slave-a']['source'] == 'server_archive'
    assert (master / 'sync_manifest.json').read_bytes() == original


def test_pin_keeps_both_amplitudes_and_slave_baseline_independent(tmp_path):
    import math
    repository, ref, master, slave = setup_archives(tmp_path)
    loader = repository.loader('master')
    version = tmp_path / 'version'
    clone_metadata(master, version, master)
    pin_sync_links(loader, repository, version, clone_metadata)
    loader._get_run_dir = lambda *unused: version
    attach_sync_links(loader, repository)
    loader.save_archive_phase_reference_override(*ref, 'master', 't2', 100, 'us', 'negative', amplitude=30)
    loader.save_archive_phase_reference_override(*ref, 'slave-a', 't2', 100, 'us', 'negative', amplitude=40)
    contexts = loader.archive_phase_reference_contexts(*ref)
    assert contexts['master']['effective_calibration']['parameter_values']['A'] == 30
    assert contexts['slave-a']['effective_calibration']['parameter_values']['A'] == 40
    assert not (master / 'sync_phase_analysis.json').exists()
    assert not (slave / 'sync_phase_analysis.json').exists()
    # Pinning preserves source settings rather than borrowing the Master's baseline mode.
    assert 'transfer_baseline_calibration_version' not in loader.get_archive_config(*ref, 'slave-a')
    points = [
        {'transfer_zero_phase_baseline': True, 'transfer_zero_phase_block_id': 0, 'intf_p1': 50},
        {'transfer_zero_phase_block_id': 0, 'intf_p1': 54},
    ]
    settings = {'mode': 'transfer_function'}
    with patch.object(loader, '_load_config_data', return_value=settings), patch.object(loader, '_read_results_csv', return_value=points):
        manifest = loader._load_sync_manifest_payload(version)
        manifest['pairs'][0]['slave'] = points[1]
        rebuilt = loader._apply_sync_phase_reference_overrides(manifest, contexts, version)
    assert math.isclose(rebuilt['pairs'][0]['slave']['interferometer_phase'], -math.asin(.1), abs_tol=1e-12)
    (slave / 'config.json').write_text('{}')
    assert loader.get_archive_config(*ref, 'slave-a')['_interferometer_phase_calibration_snapshot']['parameter_values']['A'] == 20


def test_wrong_sync_id_and_ambiguous_candidates_are_not_linked(tmp_path):
    import shutil
    repository, ref, master, slave = setup_archives(tmp_path)
    manifest = json.loads((master / 'sync_manifest.json').read_text())
    config = json.loads((slave / 'config.json').read_text())
    config['sync_run_id'] = 'other-sync'
    (slave / 'config.json').write_text(json.dumps(config))
    assert not find_links(repository, master, manifest)[0]
    config['sync_run_id'] = 'sync-test'
    (slave / 'config.json').write_text(json.dumps(config))
    repository.configuration.register_device({'device_id': 'duplicate', 'role': 'slave',
        'transport': 'local', 'source_path': str(tmp_path / 'duplicate')})
    duplicate = repository.device_root('duplicate') / 'runs/2026/09/30/run55_20260930'
    shutil.copytree(slave, duplicate)
    links, warnings = find_links(repository, master, manifest)
    assert not links
    assert warnings


def test_server_apply_creates_private_linked_versions_for_both_nodes(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.archive import ui_context
    from app.archive.ui_routes import router
    repository, ref, master, slave = setup_archives(tmp_path)
    app = FastAPI()
    app.include_router(router, prefix='/view/{device_id}')
    with patch.object(ui_context, '_repository', repository), patch.object(ui_context, '_audits', {}):
        client = TestClient(app)
        response = client.get('/view/master/archive/load/' + '/'.join(ref) + '?node_id=slave-a')
        assert response.status_code == 200, response.text
        assert response.json()['archive_node_identity']['resolved_node'] == 'slave-a'
        payload = dict(zip(('year', 'month', 'day', 'run_id'), ref))
        payload.update(reference_value=100, monotonic_slope='negative')
        for node, amplitude in [('master', 30), ('slave-a', 40)]:
            response = client.post('/view/master/archive/phase-reference-override',
                json={**payload, 'node_id': node, 'amplitude': amplitude})
            assert response.status_code == 200, response.text
        from app.archive.ui_context import resolve
        latest = resolve(repository, 'master', ref)
        loader = repository.loader('master')
        loader._get_run_dir = lambda *unused: latest
        contexts = loader.archive_phase_reference_contexts(*ref)
        assert contexts['master']['effective_calibration']['parameter_values']['A'] == 30
        assert contexts['slave-a']['effective_calibration']['parameter_values']['A'] == 40
        assert loader._resolve_archive_node_dir(latest, 'slave-a') == latest / 'sync_nodes/slave-a'
        assert not (master / 'sync_phase_analysis.json').exists()
        assert not (slave / 'sync_phase_analysis.json').exists()


def test_existing_replica_takes_priority_over_separate_backup(tmp_path):
    repository, ref, master, slave = setup_archives(tmp_path)
    replica = master / 'sync_nodes/slave-a'
    import shutil
    shutil.copytree(slave, replica)
    manifest = json.loads((master / 'sync_manifest.json').read_text())
    manifest['archive_nodes']['slave-a'] = {'path': 'sync_nodes/slave-a', 'local': False}
    (master / 'sync_manifest.json').write_text(json.dumps(manifest))
    assert not find_links(repository, master, manifest)[0]
    loader = repository.loader('master')
    attach_sync_links(loader, repository)
    assert loader._resolve_archive_node_dir(master, 'slave-a') == replica


def test_changed_source_during_pin_is_rejected(tmp_path):
    import pytest
    repository, ref, master, slave = setup_archives(tmp_path)
    loader = repository.loader('master')
    version = tmp_path / 'version'
    clone_metadata(master, version, master)

    def changing_clone(source, target, original, **kwargs):
        clone_metadata(source, target, original, **kwargs)
        (source / 'results.csv').write_text('updated by concurrent backup')

    with pytest.raises(ValueError, match='changed while copying'):
        pin_sync_links(loader, repository, version, changing_clone)
