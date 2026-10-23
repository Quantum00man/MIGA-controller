/* Negative-is-better normalized CMRR. Keep aligned with app/analysis/cmrr.py. */
(function(root) {
    'use strict';
    const number = v => v === null || v === undefined || v === '' || typeof v === 'boolean' ? null : Number.isFinite(Number(v)) ? Number(v) : null;
    function rows(sourceRows, pairs, scales = {}) {
        const groups = new Map();
        const key = (id, f) => JSON.stringify([String(id), number(f)]);
        for (const pair of pairs || []) {
            const master = pair.master || {}, slave = pair.slave || {};
            if ([master,slave].some(p => p.transfer_zero_phase_baseline || p.intf_alpha_calibration_block_id != null
                || p.power_meter_invalid_reason === 'threshold_exceeded' || p.error || p.interferometer_phase_valid !== true)) continue;
            const frequency = number(master.transfer_frequency_hz ?? slave.transfer_frequency_hz);
            const phase = number(master.transfer_phase_deg ?? slave.transfer_phase_deg);
            const id = String(pair.slave_node_id || slave.sync_node_id || 'slave');
            const scale = number(scales[id] ?? 1);
            if (frequency === null || ![0,90].includes(phase) || !(scale > 0)) continue;
            const values = [master,slave].map(p => {
                const value = number(p.interferometer_phase), fm = number(p.transfer_frequency_modulation_mhz), distance = number(p.transfer_atom_mirror_distance_m);
                return value !== null && fm > 0 && distance > 0 ? value / (4*Math.PI*fm*1e6*distance/299792458) : null;
            });
            if (values.some(v => v === null)) continue;
            values[1] /= scale;
            const k = key(id, frequency), group = groups.get(k) || {0:[],90:[]};
            group[phase].push(values); groups.set(k,group);
        }
        return sourceRows.map(source => {
            const row = {...source, cmrr_master_s2:null, cmrr_slave_s2:null, cmrr_reference_s2:null,
                cmrr_ratio:null, cmrr_db:null, cmrr_sem_db:null, cmrr_status:'missing_quadratures'};
            const group = groups.get(key(source.slave_node_id || 'slave',source.frequency_hz));
            if (!group || !group[0].length || !group[90].length) return row;
            const means = [0,90].map(p => [0,1].map(i => group[p].reduce((sum,v) => sum+v[i],0)/group[p].length));
            const master = means.reduce((sum,[m,s]) => sum+m*m,0), slave = means.reduce((sum,[m,s]) => sum+s*s,0);
            const reference = (master+slave)/2, difference = number(source.differential_s2);
            Object.assign(row,{cmrr_master_s2:master,cmrr_slave_s2:slave,cmrr_reference_s2:reference});
            if (reference <= 0) return {...row,cmrr_status:'reference_null'};
            if (difference === null || difference < 0) return {...row,cmrr_status:'missing_response'};
            if (difference === 0) return {...row,cmrr_ratio:0,cmrr_status:'zero_residual_minus_infinity'};
            Object.assign(row,{cmrr_ratio:difference/reference,cmrr_db:10*(Math.log10(difference)-Math.log10(reference)),cmrr_status:'missing_uncertainty'});
            const reconstructed = means.reduce((sum,[m,s]) => sum+(m-s)**2,0);
            if (Math.abs(difference-reconstructed) > Math.max(1e-24,1e-7*Math.max(difference,reconstructed))) return {...row,cmrr_status:'response_mismatch'};
            if ([0,90].every(p => group[p].length >= 2)) {
                const variance = [0,90].reduce((total,p,i) => {
                    const [m,s] = means[i], sample = group[p];
                    const gm = 10/Math.LN10 * (2*(m-s)/difference-m/reference);
                    const gs = 10/Math.LN10 * (-2*(m-s)/difference-s/reference);
                    return total+sample.reduce((sum,[a,b]) => sum+(gm*(a-m)+gs*(b-s))**2,0)/(sample.length*(sample.length-1));
                },0);
                Object.assign(row,{cmrr_sem_db:Math.sqrt(variance),cmrr_status:'ok'});
                if (number(source.differential_s2_sem) !== null && number(source.differential_s2_sem) >= difference) row.cmrr_status='weak_response';
            }
            return row;
        });
    }
    function figure(values, xTitle='Frequency (Hz)') {
        const omitted = values.filter(r => r.cmrr_db === null).length;
        return {data:[{x:values.map(r=>number(r.frequency_hz)),y:values.map(r=>r.cmrr_db),
            error_y:{type:'data',array:values.map(r=>r.cmrr_sem_db),visible:true},
            customdata:values.map(r=>[r.cmrr_ratio,r.cmrr_reference_s2,r.cmrr_status]),
            type:'scatter',mode:'lines+markers',name:'CMRR ± SEM',connectgaps:false,
            line:{color:'#0072b2',width:2},marker:{color:'#0072b2',size:7},
            hovertemplate:'x=%{x:.9g}<br>CMRR=%{y:.6g} dB<br>Residual/reference=%{customdata[0]:.6g}<br>Mean single-node S²=%{customdata[1]:.6g}<br>%{customdata[2]}<extra></extra>'}],
            layout:{title:'Common-Mode Rejection Ratio (CMRR)',margin:{l:85,r:35,t:65,b:110},
                xaxis:{title:xTitle,showline:true,mirror:true},
                yaxis:{title:'CMRR (dB) · more negative = better',type:'linear',showline:true,mirror:true},
                annotations:[{text:'10 log₁₀[differential S² / mean(single-node S²)]<br>Always a linear dB axis; no noise-floor subtraction.',xref:'paper',yref:'paper',x:0,y:-.22,xanchor:'left',showarrow:false,font:{size:11}},
                    ...(omitted ? [{text:`${omitted} undefined/zero-residual point(s) omitted; zero residual corresponds to −∞ dB.`,xref:'paper',yref:'paper',x:0,y:1.07,xanchor:'left',showarrow:false,font:{size:11}}] : [])],
                showlegend:false,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',hovermode:'closest'}};
    }
    root.CMRR={rows,figure};
    if(typeof module!=='undefined' && module.exports) module.exports=root.CMRR;
})(typeof globalThis!=='undefined'?globalThis:window);
