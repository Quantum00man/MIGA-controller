"""Bounded two-node fringe amplitude optimization using the Archive TF pipeline."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
from scipy.optimize import differential_evolution, minimize

from app.analysis.transfer_function import build_differential_transfer_function_summary, is_transfer_response_record, bragg_phase_modulation_rad
from app.core.data_loader import DataLoader


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=True).encode()).hexdigest()


class AmplitudeProblem:
    def __init__(self, loader, reference, slave, frequency, shot_min=None, shot_max=None):
        self.loader, self.reference = loader, reference
        self.slave, self.frequency = slave, float(frequency)
        if slave == 'master' or not slave or not math.isfinite(self.frequency):
            raise ValueError('Select a Slave and a finite target frequency')
        if shot_min is not None and shot_max is not None and shot_min > shot_max:
            raise ValueError('Shot range minimum must not exceed maximum')
        self.root = loader.get_run_dir(*reference)
        self.manifest = loader._load_sync_manifest_payload(self.root)
        if not self.manifest:
            raise ValueError('A SYNC archive is required')
        self.contexts = loader.archive_phase_reference_contexts(*reference)
        self.nodes = ('master', slave)
        self.configs, self.points, self.directories = {}, {}, {}
        for node in self.nodes:
            calibration = (self.contexts.get(node) or {}).get('effective_calibration')
            if not isinstance(calibration, dict):
                raise ValueError(f'No effective fringe calibration for {node}')
            amplitude = float((calibration.get('parameter_values') or {}).get('A', 0))
            if not math.isfinite(amplitude) or amplitude <= 0:
                raise ValueError(f'Invalid fringe amplitude for {node}')
            directory = loader._resolve_archive_node_dir(self.root, node)
            config = loader._load_config_data(directory)
            if str(config.get('mode') or '').lower() not in {'transfer_function', 'transfer_burst_time_scan'}:
                raise ValueError('Both nodes must be Transfer Function archives')
            self.directories[node] = directory
            self.configs[directory] = config
            self.points[directory] = loader._read_results_csv(directory, max_points=None)
        self.initial = np.array([self.contexts[node]['effective_calibration']['parameter_values']['A'] for node in self.nodes], dtype=float)
        self.scales = (loader.load_sync_transfer_normalization(*reference).get('slave_scales') or {})
        self.signature = digest({'contexts': {n: self.contexts[n]['effective_calibration'] for n in self.nodes},
            'configs': list(self.configs.values()), 'points': list(self.points.values()),
            'pairs': self.manifest.get('pairs'), 'scales': self.scales, 'slave': slave})
        # Cache disk inputs once; every candidate follows the production baseline
        # and phase conversion path, including each node's own legacy settings.
        self.engine = DataLoader()
        self.engine._resolve_archive_node_dir = lambda root, node: self.directories[node]
        self.engine._load_config_data = lambda directory: self.configs[directory]
        self.engine._read_results_csv = lambda directory, **kwargs: self.points[directory]
        selected = []
        for pair in self.manifest.get('pairs') or []:
            if str(pair.get('slave_node_id') or '') != slave:
                continue
            master, target = pair.get('master') or {}, pair.get('slave') or {}
            if not all(is_transfer_response_record(row) for row in (master, target)):
                continue
            try:
                frequencies = [float(row['transfer_frequency_hz']) for row in (master, target)]
                phases = [float(row['transfer_phase_deg']) for row in (master, target)]
                shot = int(master.get('sync_shot_index', master.get('step')))
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isclose(f, self.frequency, rel_tol=1e-9, abs_tol=1e-9) for f in frequencies):
                continue
            if phases[0] != phases[1] or phases[0] not in (0, 90):
                continue
            if master.get('sync_shot_index') is not None and target.get('sync_shot_index') is not None and master['sync_shot_index'] != target['sync_shot_index']:
                continue
            if shot_min is not None and shot < shot_min or shot_max is not None and shot > shot_max:
                continue
            selected.append(pair)
        self.target_manifest = {'pairs': selected}
        initial_pairs = self.convert(self.initial, self.target_manifest)
        self.fixed_indices = [index for index, pair in enumerate(initial_pairs) if self.valid(pair) and self.normalizable(pair)]
        self.target_manifest = {'pairs': [selected[i] for i in self.fixed_indices]}
        self.before = self.summary(self.convert(self.initial, self.target_manifest))
        if not self.before or not self.before.get('quadrature_complete'):
            raise ValueError('Both 0° and 90° quadratures need valid paired shots')
        if min(self.before['delta_s_0deg_count'], self.before['delta_s_90deg_count']) < 2:
            raise ValueError('At least two valid paired shots per quadrature are required')
        self.excluded_count = len(selected) - len(self.fixed_indices)

    @staticmethod
    def valid(pair):
        return all(row.get('interferometer_phase_valid') is True
            and row.get('interferometer_phase') is not None
            and math.isfinite(float(row['interferometer_phase']))
            for row in (pair.get('master') or {}, pair.get('slave') or {}))

    def normalizable(self, pair):
        try:
            scale = float(self.scales.get(self.slave, 1))
        except (TypeError, ValueError):
            return False
        return math.isfinite(scale) and scale > 0 and all(
            bragg_phase_modulation_rad(row.get('transfer_frequency_modulation_mhz'), row.get('transfer_atom_mirror_distance_m')) is not None
            for row in (pair['master'], pair['slave']))

    def calibrations(self, amplitudes):
        result = {}
        for node, amplitude in zip(self.nodes, amplitudes):
            calibration = deepcopy(self.contexts[node]['effective_calibration'])
            calibration['parameter_values']['A'] = float(amplitude)
            xs = calibration.get('fit_x') or []
            if xs:
                params = calibration['parameter_values']
                omega = calibration['bragg']['angular_frequency_rad_per_us2']
                calibration['fit_y'] = [params['C'] + amplitude * math.cos(omega * x + params['phi0']) for x in xs]
            result[node] = calibration
        return result

    def convert(self, amplitudes, manifest):
        contexts = {node: {'effective_calibration': cal, 'has_override': True}
            for node, cal in self.calibrations(amplitudes).items()}
        return self.engine._apply_sync_phase_reference_overrides(manifest, contexts, self.root).get('pairs') or []

    def summary(self, pairs):
        rows = build_differential_transfer_function_summary(pairs, self.scales)
        return next((row for row in rows if math.isclose(row['frequency_hz'], self.frequency, rel_tol=1e-9, abs_tol=1e-9)), None)

    def objective(self, fractions):
        pairs = self.convert(self.initial * np.asarray(fractions), self.target_manifest)
        if not all(self.valid(pair) for pair in pairs):
            return 1e12
        row = self.summary(pairs)
        if not row or not row.get('quadrature_complete'):
            return 1e12
        if (row['delta_s_0deg_count'], row['delta_s_90deg_count']) != (self.before['delta_s_0deg_count'], self.before['delta_s_90deg_count']):
            return 1e12
        return row['differential_s2'] / max(self.before['differential_s2'], 1e-20)

    def optimize(self, master_bound, slave_bound, shot_min=None, shot_max=None):
        for bound in (master_bound, slave_bound):
            if not math.isfinite(bound) or not 0 < bound <= .5:
                raise ValueError('Amplitude ranges must be greater than 0 and at most 50%')
        bounds = [(1 - b, 1 + b) for b in (master_bound, slave_bound)]
        candidate = differential_evolution(self.objective, bounds, seed=42, popsize=6, maxiter=35, tol=1e-7, polish=False)
        polished = minimize(self.objective, candidate.x, method='L-BFGS-B', bounds=bounds)
        choices = [np.ones(2), candidate.x, polished.x]
        best = min(choices, key=self.objective)
        amplitudes = self.initial * best
        after_pairs = self.convert(amplitudes, self.target_manifest)
        after = self.summary(after_pairs)
        full = self.convert(amplitudes, self.manifest)
        before_full = self.convert(self.initial, self.manifest)
        warnings = []
        for node, fraction, (lo, hi) in zip(self.nodes, best, bounds):
            if min(abs(fraction - lo), abs(fraction - hi)) < 1e-5:
                warnings.append(f'{node} amplitude reached its allowed boundary')
        if after['differential_s2'] >= self.before['differential_s2']:
            warnings.append('No lower target value was found within the allowed ranges.')
        before_valid = sum(self.valid(pair) for pair in before_full if str(pair.get('slave_node_id')) == self.slave)
        after_valid = sum(self.valid(pair) for pair in full if str(pair.get('slave_node_id')) == self.slave)
        if after_valid < before_valid:
            warnings.append(f'{before_valid - after_valid} fewer paired shots are valid across the full frequency range; inspect the preview.')
        warnings.append('Both amplitudes can grow together and reduce the response scale; inspect all frequencies before applying.')
        if self.excluded_count:
            warnings.append(f'{self.excluded_count} initially invalid or unnormalizable target pairs were excluded; the remaining sample set was held fixed.')
        return {'kind': 'transfer_amplitude', 'slave_node_id': self.slave, 'reference_node_id': 'master', 'target_node_id': self.slave,
            'transfer_frequency_hz': self.frequency, 'input_signature': self.signature,
            'parameters_before': dict(zip(self.nodes, self.initial.tolist())),
            'parameters_after': dict(zip(self.nodes, amplitudes.tolist())),
            'bounds': {node: [float(a * lo), float(a * hi)] for node, a, (lo, hi) in zip(self.nodes, self.initial, bounds)},
            'before': self.before, 'after': after, 'warnings': warnings,
            'pair_count': len(after_pairs), 'excluded_pair_count': self.excluded_count,
            'preview_rows': build_differential_transfer_function_summary([p for p in full if str(p.get('slave_node_id')) == self.slave], self.scales),
            'before_rows': build_differential_transfer_function_summary([p for p in before_full if str(p.get('slave_node_id')) == self.slave], self.scales),
            'settings': {'master_bound_fraction': master_bound, 'slave_bound_fraction': slave_bound,
                'shot_index_min': shot_min, 'shot_index_max': shot_max}}

    def apply(self, result):
        if result.get('input_signature') != self.signature:
            raise ValueError('Archive data, baseline, calibration or normalization changed; discard and recompute the preview')
        amplitudes = [float(result['parameters_after'][node]) for node in self.nodes]
        settings = result['settings']
        bounds = [float(settings['master_bound_fraction']), float(settings['slave_bound_fraction'])]
        if any(not math.isfinite(b) or not 0 < b <= .5 for b in bounds):
            raise ValueError('Invalid optimization bounds')
        if any(not math.isfinite(a) or abs(a / initial - 1) > bound + 1e-9 for a, initial, bound in zip(amplitudes, self.initial, bounds)):
            raise ValueError('Optimized amplitudes exceed their allowed ranges')
        if self.objective(np.array(amplitudes) / self.initial) > 1 + 1e-8:
            raise ValueError('Preview no longer improves the target or invalidates paired shots')
        calibrations = self.calibrations(amplitudes)
        payload = {'reference_node_id': 'master', 'target_node_id': self.slave,
            'optimized_reference_calibration': calibrations['master'],
            'optimized_target_calibration': calibrations[self.slave],
            'settings': {'transfer_frequency_hz': self.frequency, **settings}}
        return self.loader.apply_sync_phase_calibration_optimization(*self.reference, payload)
