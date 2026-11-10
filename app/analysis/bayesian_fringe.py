"""Random-phase fringe inference and reproducible report generation.

All probabilities, amplitudes and noise widths are in percentage points.
"""
import csv
import hashlib
import io
import json
import math
import textwrap
import zipfile
from datetime import datetime, timezone

import numpy as np
from scipy.special import logsumexp

VERSION = "1.0"


def select_shots(loader, req, require_minimum=True):
    root = loader.get_run_dir(req.year, req.month, req.day, req.run_id)
    node, identity = loader._resolve_archive_node_identity(root, req.node_id or None)
    config = loader._load_config_data(node)
    mode = str(config.get("mode") or "").lower()
    if mode not in ("standard", "phase_noise", "phase_noise_analyze", "", "scan"):
        raise ValueError("Only Standard and Phase Noise Analyze archives are supported")
    phase = mode in ("phase_noise", "phase_noise_analyze")
    if phase and req.t2 is None:
        raise ValueError("Select a T value for Phase Noise Analyze")
    field = "intf_p%s%s" % (req.channel, "_nofit" if req.source == "nofit" else "")
    points = loader._read_results_csv(node, max_points=None)
    records = []
    for point in points:
        step = point.get("step")
        t2 = point.get("phase_noise_t2_us2")
        if t2 is None:
            t2 = point.get("interferometer_phase_reference_t2_us2")
        if t2 is None:
            t2 = point.get("parameter")
        if phase and (t2 is None or not math.isclose(float(t2), req.t2, rel_tol=1e-9, abs_tol=1e-9)):
            continue
        if req.shot_min is not None and (step is None or step < req.shot_min):
            continue
        if req.shot_max is not None and (step is None or step > req.shot_max):
            continue
        value = point.get(field)
        reason = ""
        if point.get("intf_alpha_calibration_block_id") is not None:
            reason = "calibration shot"
        elif phase and point.get("phase_noise_science_shot") == 0:
            reason = "non-science shot"
        elif value is None or not math.isfinite(float(value)):
            reason = "missing or non-finite probability"
        records.append({"step": step, "timestamp": point.get("timestamp"), "t2_us2": t2,
                        "probability_percent": float(value) if value is not None and math.isfinite(float(value)) else None,
                        "included": not bool(reason), "exclusion_reason": reason})
    if require_minimum and sum(r["included"] for r in records) < 20:
        raise ValueError("At least 20 valid shots are required; more may be needed for reliable inference")
    return records, identity, mode


def infer(values, bounds=None, grid=25, phase_points=96):
    x = np.asarray(values, dtype=float)
    span = max(float(np.ptp(x)), 0.1)
    if bounds is None:
        bounds = {"A": [0.001, max(span, 0.2)], "B": [float(x.min()-span*.25), float(x.max()+span*.25)],
                  "sigma": [max(span*.002, .001), max(span*.3, .05)]}
    for name in ("A", "B", "sigma"):
        pair = bounds.get(name)
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(math.isfinite(float(v)) for v in pair) or pair[0] >= pair[1]:
            raise ValueError("Invalid bounds for " + name)
    if bounds["A"][0] < 0 or bounds["sigma"][0] <= 0:
        raise ValueError("Amplitude must be non-negative and sigma must be positive")
    axes = [np.linspace(*bounds[k], grid) for k in ("A", "B", "sigma")]
    # Gauss-Legendre integration of the uniform phase, in log space.
    nodes, weights = np.polynomial.legendre.leggauss(phase_points)
    cosine = np.cos((nodes + 1)*np.pi/2)
    logweights = np.log(weights/2)
    likelihood = np.empty((grid, grid, grid))
    s = axes[2][None, :, None]
    for i, a in enumerate(axes[0]):
        for j, b in enumerate(axes[1]):
            total = np.zeros(grid)
            for chunk in np.array_split(x, max(1, math.ceil(len(x)/128))):
                residual = chunk[:, None, None] - b - a*cosine[None, None, :]
                logp = logsumexp(-.5*(residual/s)**2 + logweights, axis=2) - np.log(axes[2])[None, :] - .5*np.log(2*np.pi)
                total += logp.sum(axis=0)
            likelihood[i,j,:] = total
    posterior = np.exp(likelihood - logsumexp(likelihood))
    marginals = [posterior.sum(axis=tuple(j for j in range(3) if j != i)) for i in range(3)]
    result = {}
    warnings = []
    for name, axis, p in zip(("A", "B", "sigma"), axes, marginals):
        mean = float(axis @ p)
        sd = float(np.sqrt(((axis-mean)**2) @ p))
        result[name] = {"mean": mean, "sd": sd}
        if p[0] + p[-1] > .01:
            warnings.append(name + " posterior has more than 1% probability at grid boundaries; expand bounds and rerun.")
    result["contrast_2A"] = {"mean": 2*result["A"]["mean"], "sd": 2*result["A"]["sd"]}
    if any(float(np.max(p)) > .5 for p in marginals):
        warnings.append("A grid cell holds more than 50% posterior mass; refine the grid before trusting uncertainty estimates.")
    if result["sigma"]["mean"] < np.pi * result["A"]["mean"] / phase_points:
        warnings.append("The phase quadrature may be too coarse for this noise width; increase phase points and compare results.")
    warnings.append("Uniform phase coverage and independent stationary Gaussian output noise are assumptions, not established by this fit.")
    warnings.append("Check grid and phase integration convergence and phase coverage before interpreting uncertainties.")
    return {"parameters": result, "bounds": bounds, "grid": grid, "phase_points": phase_points,
            "axes": {k: a.tolist() for k,a in zip(("A","B","sigma"), axes)},
            "marginals": {k: p.tolist() for k,p in zip(("A","B","sigma"), marginals)}, "warnings": warnings}


def compute(loader, req):
    records, identity, mode = select_shots(loader, req)
    values = [r["probability_percent"] for r in records if r["included"]]
    result = infer(values, req.bounds, req.grid, req.phase_points)
    par = result["parameters"]
    a, b, noise = (par[k]["mean"] for k in ("A", "B", "sigma"))
    nodes, weights = np.polynomial.legendre.leggauss(req.phase_points)
    xx = np.linspace(min(values) - np.ptp(values)*.2, max(values) + np.ptp(values)*.2, 400)
    yy = np.exp(logsumexp(-.5*((xx[:,None]-b-a*np.cos((nodes+1)*np.pi/2))/noise)**2 + np.log(weights/2), axis=1))/(noise*np.sqrt(2*np.pi))
    result.update(records=records, count=len(values), excluded=len(records)-len(values), node=identity.get("resolved_node") or "local", mode=mode,
                  model_x=xx.tolist(), model_density=yy.tolist())
    return result


def package(req, result):
    # Matplotlib is imported only for export; the noninteractive canvas avoids GUI state.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.special import logsumexp as lse
    config = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    config.update(algorithm_version=VERSION, resolved_bounds=result["bounds"], resolved_node=result["node"],
                  probability_unit="percent; widths in percentage points", prior="uniform on finite parameter grid",
                  phase_model="uniform on [0, pi]", noise_model="independent Gaussian")
    shots = io.StringIO(newline="")
    writer = csv.DictWriter(shots, fieldnames=list(result["records"][0]))
    writer.writeheader(); writer.writerows(result["records"])
    data = shots.getvalue().encode("utf-8")
    config["shots_csv_sha256"] = hashlib.sha256(data).hexdigest()
    pdf = io.BytesIO()
    values = np.array([r["probability_percent"] for r in result["records"] if r["included"]])
    with PdfPages(pdf) as pages:
        fig = Figure(figsize=(8.27, 11.69), layout="constrained")
        gs = fig.add_gridspec(4, 1, height_ratios=[1.25, 1.0, 1.5, 1.6])
        ax = fig.add_subplot(gs[0]); ax.axis("off")
        ax.text(0,1,"Bayesian reconstruction of interferometer fringes", fontsize=16, weight="bold", va="top")
        ax.text(0,.76,"Random-phase distribution analysis  |  Gaussian output noise", fontsize=10, color="#425a70")
        ax.text(0,.5, f"Run: {req.year}-{req.month}-{req.day} / {req.run_id}   |   Node: {result['node']}\nChannel: P{req.channel} {req.source}   |   T²: {req.t2 if req.t2 is not None else 'not selected'} µs²\nShot range: {req.shot_min if req.shot_min is not None else 'first'} to {req.shot_max if req.shot_max is not None else 'last'}   |   Included: {result['count']}   |   Excluded: {result['excluded']}\nGenerated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", fontsize=9, va="top", linespacing=1.6)
        ax = fig.add_subplot(gs[1]); ax.axis("off")
        ax.text(0,1,"Results", fontsize=12, weight="bold", va="top")
        par=result["parameters"]
        for n,(key,label) in enumerate((("A","Amplitude A"),("contrast_2A","Contrast 2A"),("B","Offset B"),("sigma","Noise σ"))):
            ax.text(.02,.75-n*.19, f"{label:<20} {par[key]['mean']:.4g} ± {par[key]['sd']:.3g}", fontsize=10, family="monospace")
        ax.text(.68,.45,"Percent scale; ± posterior SD", fontsize=8, color="#425a70")
        ax=fig.add_subplot(gs[2]); ax.plot([r["step"] for r in result["records"] if r["included"]], values, ".-", lw=.6, ms=3)
        ax.set(xlabel="Shot index", ylabel="Interferometer probability (%)", title="Selected shot sequence")
        ax.grid(alpha=.2)
        ax=fig.add_subplot(gs[3]); ax.hist(values,bins="auto",density=True,color="#9cb7ca",edgecolor="white",label="Shots")
        xx=np.linspace(values.min()-np.ptp(values)*.2,values.max()+np.ptp(values)*.2,400)
        a,b,s=(par[k]["mean"] for k in ("A","B","sigma"))
        nodes,w=np.polynomial.legendre.leggauss(req.phase_points)
        y=np.exp(lse(-.5*((xx[:,None]-b-a*np.cos((nodes+1)*np.pi/2))/s)**2+np.log(w/2),axis=1))/(s*np.sqrt(2*np.pi))
        ax.plot(xx,y,color="#b33232",label="Model at posterior means")
        ax.set(xlabel="Interferometer probability (%)",ylabel="Probability density",title="Distribution and fitted model"); ax.legend(); ax.grid(alpha=.2)
        pages.savefig(fig)
        fig=Figure(figsize=(8.27,11.69),layout="constrained"); gs=fig.add_gridspec(5,1,height_ratios=[.6,1,1,1,1.7])
        ax=fig.add_subplot(gs[0]); ax.axis("off"); ax.text(0,1,"Posterior distributions and method",fontsize=16,weight="bold",va="top")
        for i,key in enumerate(("A","B","sigma"),1):
            ax=fig.add_subplot(gs[i]); ax.plot(result["axes"][key],result["marginals"][key],color="#24557a"); ax.set(xlabel=key+" (percentage points)",ylabel="Probability mass"); ax.grid(alpha=.2)
        ax=fig.add_subplot(gs[4]); ax.axis("off")
        text=("Model: P = B + A cos(phi) + noise; phi uniform on [0, pi]; independent Gaussian noise.\n"
              "Uniform priors on the finite grids below. Posterior SD is not a 95% credible interval.\n"
              f"Grid: {req.grid} points per parameter; phase quadrature: {req.phase_points} points.\n"
              f"Bounds: A {result['bounds']['A']}, B {result['bounds']['B']}, σ {result['bounds']['sigma']}.\n"
              "Phase uniformity and independence cannot be established from this fit alone.\n"
              "Inspect the shot sequence and repeat with wider/finer grids to test stability.\n"
              + "\n".join(textwrap.fill(w, 105) for w in result["warnings"]))
        ax.text(0,.98,text,fontsize=8,va="top",wrap=True,linespacing=1.4)
        pages.savefig(fig)
    output=io.BytesIO()
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.pdf",pdf.getvalue()); z.writestr("configuration.json",json.dumps(config,indent=2,ensure_ascii=False)); z.writestr("shots.csv",data)
    return output.getvalue()
