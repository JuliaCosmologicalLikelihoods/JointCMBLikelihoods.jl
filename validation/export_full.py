"""
Full fixture exporter for the frozen Hillik joint Planck+ACT+SPT likelihood.

Reference: mtristram/hillik @ d0e455cf6788eb2a1d38237a93153e696b9bc257
Paper: arXiv:2511.04733 (Tristram et al. 2026)

Exports per-experiment NPZ fixtures at multiple parameter points:
  - theory Dl arrays
  - per-component foreground spectra
  - intermediate model stages (sky model, instrument, binned)
  - residual vectors and chi2
"""
import os
os.environ["COBAYA_PACKAGES_PATH"] = "/tmp/hillik_ref_packages"
import numpy as np
from cobaya.yaml import yaml_load_file
from cobaya.component import get_component_class

DATA = "/tmp/hillik_data"
OUT = "/tmp/hillik_fixtures"
os.makedirs(OUT, exist_ok=True)

exec(open("/tmp/hillik_ref/export_fixtures.py").read().split("# ----------------------------------------------------------------------------\n# Instantiate")[0].split('"""')[2])  # reuse P, theory loading

# rebuild P and theory (exec'd above)
def theory_dict(lmax):
    return {
        "TT": dlTT[: lmax + 1],
        "TE": dlTE[: lmax + 1],
        "ET": dlTE[: lmax + 1],
        "EE": dlEE[: lmax + 1],
    }

def build(name, yml, extra=None):
    info = yaml_load_file(yml)
    info["path"] = DATA
    if extra:
        info.update(extra)
    cls = get_component_class(name)
    return cls(info)

plk = build("hillik_planck.TTTEEE", "/tmp/hillik_frozen/hillik_planck/TTTEEE.yaml")
spt = build("hillik_spt.TTTEEE", "/tmp/hillik_frozen/hillik_spt/TTTEEE.yaml",
            {"data_folder": "spt_candl_data/SPT3G_D1_TnE_v0"})
act = build("hillik_act.TTTEEE_PACT", "/tmp/hillik_frozen/hillik_act/TTTEEE_PACT.yaml")

# ============================================================================
# Per-experiment deep-dive exports at a given parameter point P
# ============================================================================
def export_planck(P, tag):
    dl = theory_dict(plk.lmax)
    out = {}
    # per-mode: data, weights, per-fg contributions, cal factors, residuals
    from itertools import combinations
    for mode in ("TT", "EE", "TE", "ET"):
        out[f"dldata_{mode}"] = plk._dldata[mode]
        out[f"dlweight_{mode}"] = plk._dlweight[mode]
        cal = np.array([plk._calibration(mode, m1, m2, P) for m1, m2 in combinations(plk._mapnames, 2)])
        out[f"cal_{mode}"] = cal
        dlmodel = np.zeros((plk._nxspec, plk.lmax + 1))
        for ifg, fg in enumerate(plk.fgs[mode]):
            contrib = fg.compute_dl(P)
            out[f"fg{ifg}_{fg.name}_{mode}"] = np.array(contrib)
            dlmodel += np.array(contrib)
        out[f"dlmodel_{mode}"] = dlmodel
    # residual vector assembly
    Xl = []
    if True:
        Rspec = plk._compute_residuals(P, dl, 'TT')
        out["Rspec_TT"] = Rspec
        Rl = plk._xspectra_to_xfreq(Rspec, plk._dlweight['TT'])
        out["Rl_TT"] = Rl
        Xl += plk._select_spectra(Rl, 'TT')
        Rspec = plk._compute_residuals(P, dl, 'EE')
        out["Rspec_EE"] = Rspec
        Rl = plk._xspectra_to_xfreq(Rspec, plk._dlweight['EE'])
        out["Rl_EE"] = Rl
        Xl += plk._select_spectra(Rl, 'EE')
        Rl = 0; Wl = 0
        Rspec = plk._compute_residuals(P, dl, 'TE')
        out["Rspec_TE"] = Rspec
        RlTE, WlTE = plk._xspectra_to_xfreq(Rspec, plk._dlweight['TE'], normed=False)
        Rspec = plk._compute_residuals(P, dl, 'ET')
        out["Rspec_ET"] = Rspec
        RlET, WlET = plk._xspectra_to_xfreq(Rspec, plk._dlweight['ET'], normed=False)
        Rl = RlTE + RlET; Wl = WlTE + WlET
        out["Rl_TE"] = Rl / Wl
        Xl += plk._select_spectra(Rl / Wl, 'TE')
    delta = np.asarray(Xl).astype('float32')
    out["delta_dl"] = delta
    chi2 = plk._fast_chi_squared(plk._invkll, delta)
    # frozen compute_chi2 rounding protection against the float32 cast
    alpha = 8. - np.ceil(np.log10(chi2))
    chi2 = np.float64(np.round(chi2 * 10**alpha)) * 10**(-alpha)
    out["chi2"] = chi2
    out["invkll"] = plk._invkll
    np.savez(f"{OUT}/planck_{tag}.npz", **out)
    return chi2

def export_act(P, tag):
    dl = theory_dict(act.lmax_bpw)
    out = {}
    dl_fg = {pol: act._compute_all_fg(fgs, P) for pol, fgs in act.fgs.items()}
    for pol, arr in dl_fg.items():
        out[f"dlfg_{pol}"] = arr
    delta = []
    for ispec, spec in enumerate(act.spectra):
        exp1, exp2 = spec["experiments"]
        for pol in spec["polarizations"]:
            bpw = spec[pol]["bpw"]
            X_model = bpw.weight.T @ (dl[pol][bpw.values] + dl_fg[pol][ispec, bpw.values])
            cal = act._calibration(P, pol, exp1, exp2)
            out[f"cal_{ispec}_{pol}"] = cal
            out[f"Xmodel_{ispec}_{pol}"] = X_model
            out[f"leff_{ispec}_{pol}"] = spec[pol]["leff"]
            delta += list(spec[pol]["dl"] - X_model / cal)
    delta = np.array(delta)
    out["delta_dl"] = delta
    chi2 = act._fast_chi_squared(act.inv_cov, delta)
    out["chi2"] = chi2
    out["inv_cov"] = act.inv_cov
    np.savez(f"{OUT}/act_{tag}.npz", **out)
    return chi2

def export_spt(P, tag):
    dl = theory_dict(spt.lmax)
    out = {}
    sky = spt.compute_sky_model(dl, **P)
    for spec, arr in sky.items():
        out[f"sky_{spec.replace(' ', '_')}"] = arr
    dl_model = spt.apply_spt_corrections(sky, **P)
    for spec, arr in dl_model.items():
        out[f"model_{spec.replace(' ', '_')}"] = arr
    db_model = {spec: spt.windows[spec] @ dl_model[spec] for spec in spt.spectra_to_fit}
    for spec, arr in db_model.items():
        out[f"binned_{spec.replace(' ', '_')}"] = arr
    delta = np.concatenate(
        [(spt.bandpowers[spec] - db_model[spec])[spt.spec_bin_min[i]-1:spt.spec_bin_max[i]]
         for i, spec in enumerate(spt.spectra_to_fit)])
    out["delta_dl"] = delta
    chi2 = delta @ spt._inv_bpcov @ delta
    out["chi2"] = chi2
    out["inv_bpcov"] = spt._inv_bpcov
    np.savez(f"{OUT}/spt_{tag}.npz", **out)
    return chi2

# ============================================================================
# Baseline
# ============================================================================
c_plk = export_planck(P, "baseline")
c_act = export_act(P, "baseline")
c_spt = export_spt(P, "baseline")
print(f"baseline: PLK={c_plk:.6f} ACT={c_act:.6f} SPT={c_spt:.6f} joint={c_plk+c_act+c_spt:.6f}")

# ============================================================================
# Perturbation points
# ============================================================================
import copy
points = {}

# one-at-a-time per sector
oat = {
    "common_fg": {"Atsz": 3.5, "Acib": 5.0, "Aksz": 2.0, "xi": -0.2, "beta_cib": 1.6, "beta_radio": -0.7},
    "plk_cal": {"PLK_cal_100A": 1.01, "PLK_pe_100A": 1.05},
    "plk_dust": {"PLK_Adust100TT": 30.0, "PLK_Adust143EE": 0.35},
    "plk_ps": {"PLK_radio_TT": 70.0, "PLK_cib_ps": 20.0},
    "act_cal": {"ACT_cal": 0.99, "ACT_pe_dr6_pa5_f090": 1.02},
    "act_band": {"ACT_band_shift_dr6_pa5_f090": 1.0, "ACT_band_shift_dr6_pa5_f150": -2.0},
    "act_fg": {"ACT_AdustTT": 8.5, "ACT_radio_TT": 5.0, "ACT_cib_ps": 9.0, "ACT_AdustEE": 0.25},
    "spt_cal": {"SPT3G_cal": 1.01, "SPT3G_cal_90": 1.001},
    "spt_beam": {"SPT3G_beta_1": 1.5, "SPT3G_beta_pol_90": 0.3},
    "spt_leak": {"SPT3G_T2P2_90": -0.008, "SPT3G_T2P2_150": -0.016},
    "spt_fg": {"SPT3G_AdustTT": 3.0, "SPT3G_radio_TT": 1.0, "SPT3G_cib_ps": 8.0},
    "spt_kappa": {"SPT3G_kappa": 0.0005},
}
for name, updates in oat.items():
    Q = copy.deepcopy(P); Q.update(updates)
    points[name] = Q

# simultaneous multipoint
points["multi1"] = {**copy.deepcopy(P),
    "Atsz": 3.3, "Acib": 5.5, "xi": -0.1, "PLK_Adust100TT": 28.0, "ACT_AdustTT": 8.2,
    "SPT3G_AdustEE": 0.07, "SPT3G_cal": 1.005, "ACT_cal": 0.995}
points["multi2"] = {**copy.deepcopy(P),
    "beta_cib": 1.55, "beta_radio": -0.95, "PLK_radio_TT": 45.0, "ACT_cib_ps": 5.0,
    "SPT3G_radio_TT": 0.4, "SPT3G_beta_2": 1.0, "ACT_band_shift_dr6_pa6_f150": -2.0}
points["multi3"] = {**copy.deepcopy(P),
    "PLK_pe_143A": 0.99, "PLK_pe_217B": 1.05, "ACT_pe_dr6_pa6_f090": 1.03,
    "SPT3G_pe_220": 1.01, "SPT3G_T2P2_220": -0.035, "Aksz": 2.5}
points["multi4"] = {**copy.deepcopy(P),
    "PLK_alpha_dustTT": -2.9, "PLK_Adust217TT": 12.0, "PLK_Adust217TE": 0.5,
    "SPT3G_AdustTE": 0.15, "ACT_AdustTE": 0.5, "xi": 0.3}
points["multi5"] = {**copy.deepcopy(P),
    "Atsz": 2.5, "Acib": 3.5, "Aksz": 0.5, "PLK_cib_ps": 15.0, "SPT3G_cib_ps": 6.0,
    "ACT_radio_TT": 1.5, "SPT3G_beta_pol_150": 0.4, "SPT3G_kappa": -0.0004}

# boundary points (sqrt amplitudes, bounded priors)
points["boundary_low"] = {**copy.deepcopy(P), "Atsz": 1e-6, "Acib": 1e-6, "Aksz": 1e-6, "xi": -1.0}
points["boundary_high"] = {**copy.deepcopy(P, ), "PLK_radio_TT": 150.0, "PLK_cib_ps": 100.0,
    "ACT_radio_TT": 20.0, "ACT_cib_ps": 20.0, "SPT3G_radio_TT": 10.0, "SPT3G_cib_ps": 10.0}

summary = {}
for name, Q in points.items():
    cp = export_planck(Q, name)
    ca = export_act(Q, name)
    cs = export_spt(Q, name)
    summary[name] = (cp, ca, cs, cp + ca + cs)
    print(f"{name}: PLK={cp:.6f} ACT={ca:.6f} SPT={cs:.6f} joint={cp+ca+cs:.6f}")

# parameter values export
import json
with open(f"{OUT}/params_baseline.json", "w") as f:
    json.dump({k: float(v) for k, v in P.items()}, f, indent=1)
with open(f"{OUT}/chi2_summary.json", "w") as f:
    json.dump({k: list(map(float, v)) for k, v in summary.items()}, f, indent=1)

# theory spectra
np.savez(f"{OUT}/theory.npz", dlTT=dlTT, dlTE=dlTE, dlEE=dlEE)
print("ALL FIXTURES EXPORTED")
