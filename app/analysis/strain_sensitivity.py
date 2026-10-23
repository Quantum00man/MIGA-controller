"""Ideal long-wavelength strain calibration, shared by server-side exports.

Keep numerical definitions aligned with static/strain-sensitivity.js. Noise and
geometry are fixed inputs; only the path-response SEM is propagated.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def strain_sensitivity_rows(
    rows: Iterable[Dict[str, Any]], pairs: Iterable[Dict[str, Any]], settings: Dict[str, Any],
) -> list[Dict[str, Any]]:
    """Use recorded paired baselines, rejecting missing or inconsistent geometry."""
    baselines = {}
    for pair in pairs:
        master, slave = pair.get('master') or {}, pair.get('slave') or {}
        if any(record.get('transfer_zero_phase_baseline')
               or record.get('intf_alpha_calibration_block_id') is not None
               or record.get('power_meter_invalid_reason') == 'threshold_exceeded'
               or record.get('error') for record in (master, slave)):
            continue
        if not all(record.get('interferometer_phase_valid') is True
                   and _number(record.get('interferometer_phase')) is not None
                   for record in (master, slave)):
            continue
        phase = _number(master.get('transfer_phase_deg') if master.get('transfer_phase_deg') is not None else slave.get('transfer_phase_deg'))
        if phase not in (0, 90):
            continue
        frequency = _number(master.get('transfer_frequency_hz') if master.get('transfer_frequency_hz') is not None else slave.get('transfer_frequency_hz'))
        a, b = (_number(record.get('transfer_atom_mirror_distance_m')) for record in (master, slave))
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        key = (str(pair.get('slave_node_id') or slave.get('sync_node_id') or 'slave'), frequency)
        baseline = a - b
        if key not in baselines:
            baselines[key] = baseline
        elif abs(baselines[key] - baseline) > 1e-10:
            baselines[key] = float('nan')

    # Inputs are validated by StrainSensitivitySettings at the export boundary.
    noise = (settings['transfer_strain_sigma_mrad'] / 1000 * math.sqrt(2 * settings['transfer_strain_cycle_time_s'])
             if settings['transfer_strain_noise_model'] == 'white_phase'
             else settings['transfer_strain_phase_asd_rad_sqrt_hz'])
    k0 = 2 * math.pi / (settings['transfer_strain_wavelength_nm'] * 1e-9)
    result = []
    for source in rows:
        row = {**source, **settings, 'strain_response_rad': None,
               'strain_phase_asd_rad_sqrt_hz': noise, 'strain_asd_sqrt_hz': None,
               'strain_asd_sem_sqrt_hz': None}
        distance = _number(baselines.get((str(source.get('slave_node_id', 'slave')), _number(source.get('frequency_hz')))))
        row['strain_baseline_m'] = distance
        q, sem = _number(source.get('differential_path_s2')), _number(source.get('differential_path_s2_sem'))
        if distance is None or distance == 0:
            row['strain_status'] = 'missing_or_inconsistent_baseline'
        elif q is None or q < 0:
            row['strain_status'] = 'missing_response'
        elif q == 0:
            row['strain_status'] = 'response_null'
        else:
            response = k0 * abs(distance) * math.sqrt(q)
            asd = noise / response
            error = asd * (sem / q) / 2 if sem is not None and sem >= 0 else None
            row.update(strain_response_rad=response, strain_asd_sqrt_hz=asd,
                       strain_asd_sem_sqrt_hz=error,
                       strain_status='missing_uncertainty' if sem is None or sem < 0 else 'weak_response' if sem >= q else 'ok')
        result.append(row)
    return result
