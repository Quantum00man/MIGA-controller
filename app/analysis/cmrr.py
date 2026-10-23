"""Normalized residual CMRR: 10 log10(D / mean(single-node S2)).

Paired normalized quadrature means define the reference. Delta-method errors
retain within-shot Master/Slave covariance; quadrature acquisitions are independent.
"""
import math


def _number(value):
    if value is None or isinstance(value, bool) or value == '':
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def cmrr_rows(rows, pairs, slave_scales=None):
    groups = {}
    for pair in pairs:
        master, slave = pair.get('master') or {}, pair.get('slave') or {}
        if any(p.get('transfer_zero_phase_baseline') or p.get('intf_alpha_calibration_block_id') is not None
               or p.get('power_meter_invalid_reason') == 'threshold_exceeded' or p.get('error')
               or p.get('interferometer_phase_valid') is not True for p in (master, slave)):
            continue
        frequency = _number(master.get('transfer_frequency_hz') if master.get('transfer_frequency_hz') is not None else slave.get('transfer_frequency_hz'))
        phase = _number(master.get('transfer_phase_deg') if master.get('transfer_phase_deg') is not None else slave.get('transfer_phase_deg'))
        slave_id = str(pair.get('slave_node_id') or slave.get('sync_node_id') or 'slave')
        scale = _number((slave_scales or {}).get(slave_id, 1))
        if frequency is None or phase not in (0, 90) or scale is None or scale <= 0:
            continue
        values = []
        for record in (master, slave):
            value, fm, distance = (_number(record.get(key)) for key in
                                  ('interferometer_phase', 'transfer_frequency_modulation_mhz', 'transfer_atom_mirror_distance_m'))
            if value is None or fm is None or distance is None or fm <= 0 or distance <= 0:
                break
            values.append(value / (4 * math.pi * fm * 1e6 * distance / 299792458))
        if len(values) == 2:
            values[1] /= scale
            groups.setdefault((slave_id, frequency), {0: [], 90: []})[phase].append(values)

    result = []
    for source in rows:
        row = {**source, 'cmrr_master_s2': None, 'cmrr_slave_s2': None, 'cmrr_reference_s2': None,
               'cmrr_ratio': None, 'cmrr_db': None, 'cmrr_sem_db': None, 'cmrr_status': 'missing_quadratures'}
        group = groups.get((str(source.get('slave_node_id', 'slave')), _number(source.get('frequency_hz'))))
        if not group or not all(group[phase] for phase in (0, 90)):
            result.append(row)
            continue
        means = [(sum(v[0] for v in group[p]) / len(group[p]), sum(v[1] for v in group[p]) / len(group[p])) for p in (0, 90)]
        master_s2 = sum(m*m for m,s in means)
        slave_s2 = sum(s*s for m,s in means)
        reference = (master_s2 + slave_s2) / 2
        difference = _number(source.get('differential_s2'))
        row.update(cmrr_master_s2=master_s2, cmrr_slave_s2=slave_s2, cmrr_reference_s2=reference)
        if reference <= 0:
            row['cmrr_status'] = 'reference_null'
        elif difference is None or difference < 0:
            row['cmrr_status'] = 'missing_response'
        elif difference == 0:
            row.update(cmrr_ratio=0, cmrr_status='zero_residual_minus_infinity')
        else:
            row.update(cmrr_ratio=difference/reference,
                       cmrr_db=10 * (math.log10(difference) - math.log10(reference)), cmrr_status='missing_uncertainty')
            reconstructed = sum((m-s)**2 for m,s in means)
            if not math.isclose(difference, reconstructed, rel_tol=1e-7, abs_tol=1e-24):
                row['cmrr_status'] = 'response_mismatch'
            elif all(len(group[p]) >= 2 for p in (0, 90)):
                variance = 0
                for phase, (m,s) in zip((0,90),means):
                    gm = (10/math.log(10)) * (2*(m-s)/difference - m/reference)
                    gs = (10/math.log(10)) * (-2*(m-s)/difference - s/reference)
                    sample = group[phase]
                    # Equivalent to g^T Cov(mean_M, mean_S) g, without cancellation.
                    variance += sum((gm*(a-m)+gs*(b-s))**2 for a,b in sample) / (len(sample)*(len(sample)-1))
                error = math.sqrt(variance)
                row.update(cmrr_sem_db=error, cmrr_status='ok')
                sem = _number(source.get('differential_s2_sem'))
                if sem is not None and sem >= difference:
                    row['cmrr_status'] = 'weak_response'
        result.append(row)
    return result
