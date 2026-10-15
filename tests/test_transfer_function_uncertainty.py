import csv
import io
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET
import lzma

import pytest

from app.analysis.transfer_function import build_transfer_function_summary, build_differential_transfer_function_summary
from app.core.archive_labplot import build_archive_project
from app.core.data_manager import DataManager


def shots():
    return [dict(transfer_frequency_hz=10, transfer_phase_deg=phase,
                 interferometer_phase=value, interferometer_phase_valid=True,
                 transfer_frequency_modulation_mhz=1, transfer_atom_mirror_distance_m=distance,
                 transfer_phase_noise_sigma_mrad=100)
            for phase in (0, 90) for value in (-0.1, 0.1)
            for distance in (2.23,)]


def pairs():
    return [dict(slave_node_id='slave', master=shot,
                 slave={**shot, 'interferometer_phase': -shot['interferometer_phase'],
                        'transfer_atom_mirror_distance_m': 1}) for shot in shots()]


def test_single_node_sem_and_square_uncertainty_at_null_are_saved():
    summary = build_transfer_function_summary(shots(), 1, [0, 90])
    row = summary[0]
    assert row['interferometer_phase_0deg_std_rad'] == pytest.approx(math.sqrt(.02))
    assert row['interferometer_phase_0deg_sem_rad'] == pytest.approx(.1)
    assert row['interferometer_phase_0deg_phase2_sem_rad2'] == pytest.approx(math.sqrt(2) * .01)
    assert row['interferometer_phase_phase2_sem_rad2'] == pytest.approx(.02)
    assert row['interferometer_phase_s2_sem'] > 0
    with tempfile.TemporaryDirectory() as directory:
        manager = DataManager()
        manager.current_run_dir = Path(directory)
        manager.save_transfer_function_summary(summary)
        saved = next(csv.DictReader(io.StringIO((Path(directory) / 'transfer_function_summary.csv').read_text())))
    for field in ('interferometer_phase_0deg_std_rad', 'interferometer_phase_0deg_sem_rad', 'interferometer_phase_s2_sem'):
        assert float(saved[field]) == pytest.approx(row[field])


def test_paired_sync_uncertainties_and_common_mode_cancellation():
    row = build_differential_transfer_function_summary(pairs())[0]
    assert row['phase_difference_0deg_sem_rad'] == pytest.approx(.2)
    assert row['differential_s2_sem'] > 0
    assert row['differential_path_s2_sem'] > 0
    assert row['phase_path_normalized_0deg_std'] > 0
    common = pairs()
    for pair in common:
        pair['slave']['interferometer_phase'] = pair['master']['interferometer_phase']
    cancelled = build_differential_transfer_function_summary(common)[0]
    assert cancelled['phase_difference_0deg_std_rad'] == 0
    assert cancelled['phase_difference_0deg_sem_rad'] == 0


def test_insufficient_shots_leave_error_missing():
    row = build_transfer_function_summary(shots()[:1], 1, [0, 90])[0]
    assert row['interferometer_phase_0deg_sem_rad'] is None
    assert row['interferometer_phase_s2_sem'] is None


def test_labplot_contains_single_and_sync_sem_error_columns_and_std():
    class Loader:
        def load_run(self, *args, **kwargs):
            return dict(config={'mode': 'transfer_function'},
                        transfer_function_summary=build_transfer_function_summary(shots(), 1, [0, 90]),
                        sync_manifest={'pairs': pairs()})
    payload = build_archive_project(Loader(), '2026', '10', '15', 'test', ['phase'])
    root = ET.fromstring(lzma.decompress(payload))
    names = [element.get('name') for element in root.iter('worksheet')]
    assert 'SYNC Standard Deviation - slave' in names
    assert 'SYNC Phase Std - slave' in names
    assert 'Interferometer Phase - Standard Deviation' in names
    for title in ('Transfer Function S2', 'Interferometer Phase - Mean', 'SYNC Transfer S2 - slave', 'SYNC Path S2 - slave'):
        worksheet = next(element for element in root.iter('worksheet') if element.get('name') == title)
        assert any(element.get('yErrorType') == '2' for element in worksheet.iter('errorBars'))


@pytest.mark.skipif(not shutil.which('node'), reason='Node required for browser calculation parity')
def test_browser_and_backend_uncertainty_parity():
    source = Path('static/archive.html').read_text()
    helper = source[source.index('                withTransferUncertainties(input) {'):source.index('                syncArchiveTransferDifferentialRows() {')]
    records = [build_transfer_function_summary(shots(), 1, [0, 90])[0], build_differential_transfer_function_summary(pairs())[0]]
    result = subprocess.run(['node', '-e', 'const app = {' + helper + '}; console.log(JSON.stringify(' + json.dumps(records) + '.map(row => app.withTransferUncertainties(row))));'], check=True, text=True, capture_output=True)
    browser = json.loads(result.stdout)
    for expected, actual in zip(records, browser):
        for field, value in expected.items():
            if '_sem' in field:
                assert actual[field] == pytest.approx(value) if value is not None else actual[field] is None


@pytest.mark.skipif(not shutil.which('node'), reason='Node required for Archive export checks')
def test_browser_rebuild_and_csv_exports_include_sem_and_std():
    source = Path('static/archive.html').read_text()
    methods = source[source.index('                withTransferUncertainties(input) {'):source.index('                renderSyncArchiveTransferFunctionPlot(element) {')]
    export = source[source.index('                async exportTransferFunctionCSV() {'):source.index('                async exportLockInCSV() {')]
    script = 'const app = {' + methods + export + '''
        syncManifest: null, syncArchiveSlaveId: 'slave',
        syncTransferNormalizationOptimization: {}, syncTransferNormalization: {},
        syncArchivePairs() { return PAIRS; },
        syncArchiveNodeLabel(id) { return id; },
        sanitizeFilenamePart(value) { return value; },
        csvCell(value) { return String(value); },
        downloads: [], async downloadCsvContent(csv, name) { this.downloads.push({csv, name}); },
        transferFunctionSummary: SUMMARY
    };
    (async () => {
        const rows = app.syncArchiveTransferDifferentialRows();
        await app.downloadSyncTransferFunctionDifferentialCSV();
        await app.exportTransferFunctionCSV();
        console.log(JSON.stringify({rows, downloads: app.downloads}));
    })();'''
    script = script.replace('return PAIRS;', 'return ' + json.dumps(pairs()) + ';').replace('SUMMARY', json.dumps(build_transfer_function_summary(shots(), 1, [0, 90])))
    result = subprocess.run(['node', '-e', script], check=True, text=True, capture_output=True)
    output = json.loads(result.stdout)
    expected = build_differential_transfer_function_summary(pairs())[0]
    for field in ('differential_s2_sem', 'differential_path_s2_sem', 'phase_path_normalized_0deg_std'):
        assert output['rows'][0][field] == pytest.approx(expected[field])
    sync = next(csv.DictReader(io.StringIO(output['downloads'][0]['csv'].lstrip('\ufeff'))))
    single = next(csv.DictReader(io.StringIO(output['downloads'][1]['csv'].lstrip('\ufeff'))))
    assert float(sync['differential_s2_sem']) > 0
    assert float(sync['delta_s_0deg_std']) > 0
    assert float(single['interferometer_phase_s2_sem']) > 0
    assert float(single['Phase_0deg_Std_Rad']) > 0
