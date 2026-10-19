from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from app.analysis.transfer_function import (
    add_transfer_function_uncertainties, build_differential_transfer_function_summary,
    build_transfer_function_summary, bragg_phase_modulation_rad,
)


def samples():
    return [dict(transfer_frequency_hz=100, transfer_phase_deg=phase,
                 interferometer_phase=.2, interferometer_phase_valid=True)
            for phase, count in ((0, 4), (90, 9)) for _ in range(count)]


def test_single_node_uses_each_quadrature_valid_count():
    rows = samples() + [dict(transfer_frequency_hz=100, transfer_phase_deg=0,
                            interferometer_phase=99, interferometer_phase_valid=False)]
    row = build_transfer_function_summary(rows, 1, [0, 90])[0]
    expected = .1 ** 2 / 4 + .1 ** 2 / 9
    assert row['interferometer_phase_noise_phase2_rad2'] == pytest.approx(expected)
    assert row['interferometer_phase_noise_s2'] == pytest.approx(expected / bragg_phase_modulation_rad(1) ** 2)
    assert row['interferometer_phase_0deg_count'] == 4
    assert row['interferometer_phase_90deg_count'] == 9
    original = deepcopy(row)
    add_transfer_function_uncertainties(row)
    assert row == original


def test_no_quadrature_metadata_uses_total_valid_count_and_empty_group_is_unknown():
    rows = [{k: v for k, v in item.items() if k != 'transfer_phase_deg'} for item in samples()[:4]]
    row = build_transfer_function_summary(rows, 1)[0]
    assert row['interferometer_phase_noise_phase2_rad2'] == pytest.approx(.1 ** 2 / 4)
    row = build_transfer_function_summary(samples()[:4], 1, [0, 90])[0]
    assert row['interferometer_phase_noise_phase2_rad2'] is None


def paired_samples():
    pairs = []
    for row in samples():
        records = [dict(row, transfer_phase_noise_sigma_mrad=sigma,
                        transfer_frequency_modulation_mhz=1, transfer_atom_mirror_distance_m=distance)
                   for sigma, distance in ((100, 2), (200, 3))]
        pairs.append(dict(slave_node_id='slave', master=records[0], slave=records[1]))
    return pairs


def test_differential_uses_paired_counts_and_preserves_response():
    pairs = paired_samples()
    invalid = deepcopy(pairs[0])
    invalid['slave']['interferometer_phase_valid'] = False
    row = build_differential_transfer_function_summary(pairs + [invalid], {'slave': 1.1})[0]
    variance = (.1 / bragg_phase_modulation_rad(1, 2)) ** 2 + (.2 / (bragg_phase_modulation_rad(1, 3) * 1.1)) ** 2
    assert row['delta_s_0deg_noise_s2'] == pytest.approx(variance / 4)
    assert row['delta_s_90deg_noise_s2'] == pytest.approx(variance / 9)
    assert row['differential_noise_s2'] == pytest.approx(variance * (1 / 4 + 1 / 9))
    path_variance = (.1 ** 2 + .2 ** 2) / bragg_phase_modulation_rad(1, 1) ** 2
    assert row['differential_path_noise_s2'] == pytest.approx(path_variance * (1 / 4 + 1 / 9))
    assert row['delta_s_0deg_count'] == 4
    assert row['delta_s_90deg_count'] == 9
    expected_mean = .2 / bragg_phase_modulation_rad(1, 2) - .2 / (bragg_phase_modulation_rad(1, 3) * 1.1)
    assert row['differential_s2'] == pytest.approx(2 * expected_mean ** 2)


def legacy_differential():
    row = dict(slave_node_id='slave', differential_s2=123, differential_noise_s2=4,
               differential_path_noise_s2=8)
    for phase, count in (('0deg', 4), ('90deg', 9)):
        for base, variance in (('delta_s', 2), ('phase_path_normalized', 4)):
            row[f'{base}_{phase}_count'] = count
            row[f'{base}_{phase}_mean'] = 1
            row[f'{base}_{phase}_noise_s2'] = variance
    return row


def test_legacy_migration_is_idempotent_and_does_not_change_saved_response():
    row = legacy_differential()
    add_transfer_function_uncertainties(row)
    assert row['differential_s2'] == 123
    assert row['differential_noise_s2'] == pytest.approx(2 / 4 + 2 / 9)
    once = deepcopy(row)
    add_transfer_function_uncertainties(row)
    assert row == once
    unknown = add_transfer_function_uncertainties({'interferometer_phase_noise_phase2_rad2': .02})
    assert unknown['interferometer_phase_noise_phase2_rad2'] is None


def test_incomplete_noise_metadata_is_not_treated_as_full_sample_noise():
    pairs = paired_samples()
    pairs[0]['master'].pop('transfer_phase_noise_sigma_mrad')
    row = build_differential_transfer_function_summary(pairs)[0]
    assert row['differential_noise_s2'] is None
    assert row['differential_path_noise_s2'] is None


def test_frontend_legacy_conversion_matches_python():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    page = (Path(__file__).resolve().parents[1] / 'static/archive.html').read_text()
    start = page.index('                withTransferUncertainties(input) {')
    end = page.index('                syncArchiveTransferDifferentialRows() {', start)
    legacy_single = dict(interferometer_phase_noise_phase2_rad2=.02,
        interferometer_phase_noise_s2=2., interferometer_phase_s2_components=[dict(phase_deg=0, count=4), dict(phase_deg=90, count=9)])
    records = [legacy_single, legacy_differential(), build_transfer_function_summary(samples(), 1, [0, 90])[0]]
    script = 'const app = {' + page[start:end] + '}; const rows=' + json.dumps(records) + '; console.log(JSON.stringify(rows.map(row=>app.withTransferUncertainties(app.withTransferUncertainties(row)))));'
    result = subprocess.run([node, '-e', script], check=True, text=True, capture_output=True)
    frontend = json.loads(result.stdout)
    for raw, actual in zip(records, frontend):
        expected = add_transfer_function_uncertainties(deepcopy(raw))
        for key in ('interferometer_phase_noise_phase2_rad2', 'interferometer_phase_noise_s2', 'differential_noise_s2', 'differential_path_noise_s2'):
            assert actual.get(key) == pytest.approx(expected[key]) if expected.get(key) is not None else actual.get(key) is None


def test_live_phase_squared_floor_uses_each_points_actual_counts():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    page = (Path(__file__).resolve().parents[1] / 'static/index.html').read_text()
    start = page.index('                                const noiseFloor = point => {')
    end = page.index('                                traces.push({', start)
    script = "const sigmaRad=.1, phases=[0,90], field='phase';" + page[start:end]
    script += "const point={phaseData:{[Number(0).toPrecision(15)]:{phase:[1,2,3,4]},[Number(90).toPrecision(15)]:{phase:Array(9).fill(1)}}};console.log(JSON.stringify([noiseFloor(point),noiseFloor({phaseData:{}})]));"
    result = subprocess.run([node, '-e', script], check=True, text=True, capture_output=True)
    values = json.loads(result.stdout)
    assert values[0] == pytest.approx(.1 ** 2 / 4 + .1 ** 2 / 9)
    assert values[1] is None
