"""Complete, recalculated Phase Noise exports shared by both Archive servers."""
import csv
import io
import math
import statistics
import zipfile
from collections import defaultdict

from app.core.labplot_export import Curve, Plot, Worksheet, build_project


METRICS = {
    'atom_number_up': ('Atom number UP', 'atoms'), 'atom_number_dw': ('Atom number DOWN', 'atoms'),
    'temperature_up': ('Temperature UP', 'uK'), 'temperature_dw': ('Temperature DOWN', 'uK'),
    'sigma_up': ('Width UP', 'ms'), 'sigma_dw': ('Width DOWN', 'ms'),
    'amplitude_up': ('Amplitude UP', 'V'), 'amplitude_dw': ('Amplitude DOWN', 'V'),
    'arrival_time_up': ('Arrival UP', 's'), 'arrival_time_dw': ('Arrival DOWN', 's'),
    'transition_probability_up': ('Probability UP', '%'), 'transition_probability_dw': ('Probability DOWN', '%'),
    'intf_n1': ('Interferometer N1', 'atoms'), 'intf_n2': ('Interferometer N2', 'atoms'),
    'intf_p1': ('Interferometer P1', '%'), 'intf_p2': ('Interferometer P2', '%'),
}
FIELDS = list(METRICS) + [field + '_nofit' for field in METRICS] + [
    'tail_mean_up_raw', 'tail_mean_dw_raw', 'interferometer_phase', 'intf_alpha_applied',
    'power_meter_power_w',
]
META = ['node_id', 'node_name', 'step', 'timestamp', 'phase_noise_t2_us2',
        'phase_noise_repeat', 'phase_noise_total_repeats', 'phase_noise_t_index', 'phase_noise_t_count',
        'phase_noise_science_shot', 'sync_shot_index', 'shot_type',
        'reanalysis_status', 'interferometer_phase_valid', 'power_meter_valid',
        'power_meter_invalid_reason', 'interferometer_phase_calibration_id',
        'interferometer_phase_calibration_name', 'interferometer_phase_reference_t2_us2',
        'intf_alpha_calibration_block_id', 'intf_alpha_calibration_shot',
        'power_meter_measured_at', 'power_meter_wavelength_nm', 'power_meter_serial_number',
        'power_meter_reference_w', 'power_meter_deviation_percent']


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def collect(loader, reference, settings, calibration):
    root = loader._get_run_dir(*reference)
    manifest = loader._load_sync_manifest_payload(root) or {}
    nodes = manifest.get('archive_nodes') or {}
    runtime = manifest.get('runtime') or {}
    node_ids = list(nodes) if manifest else ['']
    if manifest:
        node_ids = list(dict.fromkeys(['master'] + node_ids + [str(x.get('node_id')) for x in runtime.get('slaves', []) if x.get('node_id')]))
    warnings = list(manifest.get('server_archive_link_warnings') or [])
    rows = []
    for node in node_ids:
        try:
            run_dir, identity = loader._resolve_archive_node_identity(root, node or None)
            config = loader._load_config_data(run_dir)
            if str(config.get('mode') or '').lower() != 'phase_noise':
                raise ValueError('Not a Phase Noise archive')
            if not (run_dir / 'results.csv').is_file():
                raise FileNotFoundError('results.csv is missing')
            original = loader._read_results_csv(run_dir, max_points=None)
            request_settings = dict(settings, _interferometer_phase_calibration=calibration)
            result = loader.recalculate_run(*reference, request_settings, max_points=None, node_id=node or None,
                                            tolerate_missing_waveforms=True)
            by_step = {p['step']: p for p in result['data']}
            missing_steps = set(result.get('missing_waveform_steps') or [])
            if missing_steps:
                warnings.append(f'{node or "local"}: {len(missing_steps)} waveform(s) missing; affected shots have no recalculated values')
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(f'{node or "local"}: {exc}; node unavailable or reanalysis failed')
            continue
        node_results = (manifest.get('node_results') or {})
        sync_data = node_results.get('master', []) if node == 'master' else (node_results.get('slaves') or {}).get(node, [])
        sync_by_shot = defaultdict(set)
        for item in sync_data:
            if item.get('phase_noise_science_shot') is not None and item.get('sync_shot_index') is not None:
                sync_by_shot[item['phase_noise_science_shot']].add(item['sync_shot_index'])
        name = (nodes.get(node) or {}).get('name') or identity.get('recorded_node_id') or node or 'Standalone'
        for point in original:
            calibration_shot = point.get('intf_alpha_calibration_block_id') is not None
            calculated = by_step.get(point['step']) if not calibration_shot else None
            row = {key: point.get(key) for key in META}
            row.update(node_id=node or 'standalone', node_name=name,
                       sync_shot_index=(next(iter(sync_by_shot[point['phase_noise_science_shot']]))
                                        if len(sync_by_shot.get(point.get('phase_noise_science_shot'), ())) == 1 else None),
                       shot_type='calibration' if calibration_shot else 'science',
                       reanalysis_status='ok' if calculated else ('calibration_excluded' if calibration_shot else ('waveform_missing' if point['step'] in missing_steps else 'rejected')))
            row.update({field: calculated.get(field) if calculated else None for field in FIELDS})
            if calculated:
                for key in META[META.index('interferometer_phase_valid'):]:
                    row[key] = calculated.get(key)
                row['intf_alpha_applied'] = (result.get('settings') or {}).get('intf_alpha')
            rows.append(row)
    if not rows:
        raise ValueError('No recalculated node data available. ' + '; '.join(warnings))
    return rows, warnings


def summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['reanalysis_status'] == 'ok':
            groups[(row['node_id'], row['node_name'], row.get('phase_noise_t2_us2'))].append(row)
    output = []
    for (node, name, t2), group in groups.items():
        for field in FIELDS:
            values = [_number(row.get(field)) for row in group
                      if field != 'interferometer_phase' or row.get('interferometer_phase_valid') is True]
            values = [v for v in values if v is not None]
            output.append([node, name, t2, field, len(values), statistics.mean(values) if values else None,
                           statistics.stdev(values) if len(values) > 1 else None])
    return output


def csv_data(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return ('\ufeff' + stream.getvalue()).encode('utf-8')


def csv_zip(rows, warnings, reference):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('shots.csv', csv_data(META + FIELDS, [[row.get(k) for k in META + FIELDS] for row in rows]))
        archive.writestr('statistics.csv', csv_data(['node_id', 'node_name', 'T2_us2', 'metric', 'valid_count', 'mean', 'sample_std'], summaries(rows)))
        lines = ['Phase Noise recalculated export: ' + '/'.join(reference),
                 'Science shot statistics exclude rejected and calibration shots. Phase statistics exclude invalid phase values. Empty values are unavailable; sample standard deviation requires at least two values.',
                 'Calibration and rejected shots retain identifiers only; analysis values are not substituted from saved results.',
                 'Fit fields have no suffix; _nofit denotes NoFit; tail means are raw. T2 is in us^2, timestamp is Unix seconds, phase is rad, power is W.',
                 'sync_shot_index is included only where recorded identity can be established.', 'Units:']
        lines.extend(f'{key}: {label} ({unit}); {key}_nofit: NoFit ({unit})' for key, (label, unit) in METRICS.items())
        lines.extend(['', 'Warnings:'] + (warnings or ['None']))
        archive.writestr('README.txt', '\n'.join(lines).encode('utf-8'))
    return output.getvalue()


def labplot(rows, warnings, reference):
    worksheets = []
    summary_rows = summaries(rows)
    colors = [(0,114,178), (213,94,0), (0,158,115), (204,121,167), (230,159,0)]
    descriptors = dict(METRICS)
    descriptors.update({k + '_nofit': (v[0] + ' NoFit', v[1]) for k, v in METRICS.items()})
    descriptors.update({'tail_mean_up_raw': ('Tail mean UP', 'V'), 'tail_mean_dw_raw': ('Tail mean DOWN', 'V'),
                        'interferometer_phase': ('Phase', 'rad'), 'intf_alpha_applied': ('I alpha', ''),
                        'power_meter_power_w': ('Power', 'W')})
    for field, (label, unit) in descriptors.items():
        groups = defaultdict(list)
        for row in rows:
            x, y = _number(row.get('phase_noise_repeat')), _number(row.get(field))
            if row['reanalysis_status'] == 'ok' and x is not None and y is not None and (field != 'interferometer_phase' or row.get('interferometer_phase_valid') is True):
                groups[(f"{row['node_name']} [{row['node_id']}]", row.get('phase_noise_t2_us2'))].append((x, y))
        curves = []
        for index, ((node, t2), points) in enumerate(groups.items()):
            points.sort()
            curves.append(Curve(f'{node} T2={t2}', [p[0] for p in points], [p[1] for p in points], colors[index % len(colors)]))
        if curves:
            worksheets.append(Worksheet(label, [Plot(label, 'Science shot within T', f'{label} ({unit})' if unit else label, curves)]))
        mean_curves = []
        for index, item in enumerate(summary_rows):
            node, name, t2, metric, count, mean, std = item
            if metric == field and _number(t2) is not None and mean is not None:
                mean_curves.append(Curve(f'{name} [{node}] mean', [float(t2)], [mean], colors[index % len(colors)],
                                         line=False, y_error_plus=[std] if std is not None else None,
                                         y_error_minus=[std] if std is not None else None))
        if mean_curves:
            worksheets.append(Worksheet(label + ' statistics', [Plot(label + ' mean and sample standard deviation', 'T2 (us2)', f'{label} ({unit})' if unit else label, mean_curves)]))
    if not worksheets:
        raise ValueError('No valid recalculated values to plot')
    return build_project('MIGA Phase Noise ' + reference[-1], worksheets,
                         comment='Recalculated results. Missing and rejected shots are omitted from curves. ' + '; '.join(warnings))
