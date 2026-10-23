/* Ideal long-wavelength GW strain calibration from the paired path-normalized response.
 * No subtraction of the TF calibration noise floor; noise inputs describe the detector output.
 */
(function (root) {
    'use strict';
    const defaults = Object.freeze({
        transfer_strain_noise_model: 'white_phase',
        transfer_strain_sigma_mrad: 74,
        transfer_strain_cycle_time_s: 2.6,
        transfer_strain_phase_asd_rad_sqrt_hz: 0.074 * Math.sqrt(2 * 2.6),
        transfer_strain_wavelength_nm: 780,
    });
    const number = value => value === null || value === undefined || value === '' || typeof value === 'boolean'
        ? null : (Number.isFinite(Number(value)) ? Number(value) : null);
    function settings(input = {}) {
        return Object.fromEntries(Object.entries(defaults).map(([key, fallback]) => [key, input[key] ?? fallback]));
    }
    function calculate(row, baseline, input = {}) {
        const p = settings(input);
        const q = number(row.differential_path_s2);
        const semQ = number(row.differential_path_s2_sem);
        const wavelength = number(p.transfer_strain_wavelength_nm);
        const cycle = number(p.transfer_strain_cycle_time_s);
        const sigma = number(p.transfer_strain_sigma_mrad);
        const fixed = number(p.transfer_strain_phase_asd_rad_sqrt_hz);
        const distance = number(baseline);
        const result = {...row, ...p, strain_baseline_m: distance,
            strain_response_rad: null, strain_phase_asd_rad_sqrt_hz: null,
            strain_asd_sqrt_hz: null, strain_asd_sem_sqrt_hz: null, strain_status: 'invalid_parameters'};
        if (!(wavelength > 0) || !(cycle > 0)) return result;
        let noise;
        if (p.transfer_strain_noise_model === 'white_phase' && sigma !== null && sigma >= 0) {
            noise = sigma / 1000 * Math.sqrt(2 * cycle);
        } else if (p.transfer_strain_noise_model === 'fixed_asd' && fixed !== null && fixed >= 0) {
            noise = fixed;
        } else return result;
        result.strain_phase_asd_rad_sqrt_hz = noise;
        if (distance === null || distance === 0) return {...result, strain_status: 'missing_or_inconsistent_baseline'};
        if (q === null || q < 0) return {...result, strain_status: 'missing_response'};
        if (q === 0) return {...result, strain_status: 'response_null'};
        const response = 2 * Math.PI / (wavelength * 1e-9) * Math.abs(distance) * Math.sqrt(q);
        const asd = noise / response;
        if (!Number.isFinite(asd) || !Number.isFinite(response) || response <= 0) return result;
        const error = semQ !== null && semQ >= 0 ? asd * (semQ / q) / 2 : null;
        return {...result, strain_response_rad: response, strain_asd_sqrt_hz: asd,
            strain_asd_sem_sqrt_hz: Number.isFinite(error) ? error : null,
            strain_status: semQ === null || semQ < 0 ? 'missing_uncertainty' : (semQ >= q ? 'weak_response' : 'ok')};
    }
    function rows(responseRows, pairs, input) {
        // Recover actual per-frequency geometry from recorded pairs, including legacy runs.
        // Different baselines cannot share a single input calibration.
        const baselines = new Map();
        for (const pair of pairs || []) {
            const master = pair.master || {}, slave = pair.slave || {};
            if ([master, slave].some(record => record.transfer_zero_phase_baseline
                || record.intf_alpha_calibration_block_id != null
                || record.power_meter_invalid_reason === 'threshold_exceeded' || record.error)) continue;
            if (![master, slave].every(record => record.interferometer_phase_valid === true
                && number(record.interferometer_phase) !== null)) continue;
            if (![0, 90].includes(number(master.transfer_phase_deg ?? slave.transfer_phase_deg))) continue;
            const frequency = number(master.transfer_frequency_hz ?? slave.transfer_frequency_hz);
            const a = number(master.transfer_atom_mirror_distance_m), b = number(slave.transfer_atom_mirror_distance_m);
            const key = `${pair.slave_node_id || slave.sync_node_id || 'slave'}:${frequency}`;
            if (a === null || b === null || a <= 0 || b <= 0) continue;
            const baseline = a - b;
            if (!baselines.has(key)) baselines.set(key, baseline);
            else if (Math.abs(baselines.get(key) - baseline) > 1e-10) baselines.set(key, NaN);
        }
        return responseRows.map(row => calculate(row,
            baselines.get(`${row.slave_node_id}:${number(row.frequency_hz)}`), input));
    }
    function figure(values, input, scale = 'linear', xTitle = 'Frequency (Hz)') {
        const p = settings(input), log = scale === 'log';
        const valid = values.filter(row => row.strain_asd_sqrt_hz !== null && (!log || row.strain_asd_sqrt_hz > 0));
        const clipped = log && valid.some(row => row.strain_asd_sem_sqrt_hz >= row.strain_asd_sqrt_hz);
        const invalid = values.length - valid.length;
        const positiveBounds = valid.flatMap(row => {
            const lower = row.strain_asd_sqrt_hz - (row.strain_asd_sem_sqrt_hz ?? 0);
            return lower > 0 ? [row.strain_asd_sqrt_hz, lower] : [row.strain_asd_sqrt_hz];
        });
        const logFloor = positiveBounds.length ? Math.min(...positiveBounds) / 2 : 0;
        const cycle = number(p.transfer_strain_cycle_time_s);
        const annotations = [{text: `Ideal GW coupling · ${p.transfer_strain_noise_model === 'white_phase' ? 'white phase noise' : 'fixed phase ASD'} · readout Nyquist ${cycle > 0 ? (1 / (2 * cycle)).toPrecision(4) : '?'} Hz<br>Higher frequencies are model equivalents`,
            xref: 'paper', yref: 'paper', x: 0, y: -0.23, xanchor: 'left', showarrow: false, font: {size: 11}}];
        if (!valid.length) annotations.push({text: 'No finite strain sensitivity: check path response, paired distances and noise parameters', xref: 'paper', yref: 'paper', x: 0.5, y: 0.5, showarrow: false});
        if (invalid || clipped || valid.some(row => row.strain_status === 'weak_response')) annotations.push({
            text: `${invalid ? `${invalid} invalid/null point(s) omitted. ` : ''}${clipped ? 'Log lower error bounds crossing zero are clipped.<br>' : ''}${valid.some(row => row.strain_status === 'weak_response') ? 'Weak response: first-order errors are approximate.' : ''}`,
            xref: 'paper', yref: 'paper', x: 0, y: 1.07, xanchor: 'left', showarrow: false, font: {size: 11}});
        // Plotly expects absolute data errors even on log axes. Preserve missing data as gaps.
        const plotted = values.map(row => row.strain_asd_sqrt_hz !== null && (!log || row.strain_asd_sqrt_hz > 0) ? row : null);
        return {data: [{
            x: values.map(row => number(row.frequency_hz)), y: plotted.map(row => row?.strain_asd_sqrt_hz ?? null),
            error_y: {type: 'data', symmetric: false, visible: true,
                array: plotted.map(row => row?.strain_asd_sem_sqrt_hz ?? null),
                arrayminus: plotted.map(row => !row || row.strain_asd_sem_sqrt_hz === null ? null
                    : log ? Math.min(row.strain_asd_sem_sqrt_hz, row.strain_asd_sqrt_hz - logFloor) : row.strain_asd_sem_sqrt_hz)},
            customdata: values.map(row => [row.strain_asd_sem_sqrt_hz, row.strain_status]),
            type: 'scatter', mode: 'lines+markers', name: 'Strain ASD ± propagated SEM', connectgaps: false,
            line: {color: '#6f42c1', width: 2}, marker: {color: '#6f42c1', size: 7},
            hovertemplate: 'Frequency=%{x:.9g} Hz<br>Strain ASD=%{y:.6g} Hz⁻¹ᐟ²<br>SEM=%{customdata[0]:.6g}<br>%{customdata[1]}<extra></extra>',
        }], layout: {
            title: 'Strain Sensitivity Function', margin: {l: 90, r: 35, t: 80, b: 115},
            xaxis: {title: xTitle, showline: true, mirror: true, ticks: 'inside', gridcolor: '#dde5ee'},
            yaxis: {title: 'Strain ASD (Hz⁻¹ᐟ²)', type: scale, exponentformat: 'power', showexponent: 'all', ...(log ? {} : {rangemode: 'tozero'}), showline: true, mirror: true, ticks: 'inside', gridcolor: '#dde5ee'},
            annotations, showlegend: false, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', hovermode: 'closest',
        }};
    }
    root.StrainSensitivity = {defaults, settings, calculate, rows, figure};
    if (typeof module !== 'undefined' && module.exports) module.exports = root.StrainSensitivity;
})(typeof globalThis !== 'undefined' ? globalThis : window);
