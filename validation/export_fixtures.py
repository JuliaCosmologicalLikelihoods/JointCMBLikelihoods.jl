"""
Fixture exporter for the frozen Hillik joint Planck+ACT+SPT likelihood.

Reference: mtristram/hillik @ d0e455cf6788eb2a1d38237a93153e696b9bc257
Paper: arXiv:2511.04733 (Tristram et al. 2026)

Instantiates the three paper-era likelihoods, evaluates them at the paper
posterior-mean nuisance values with a fixed theory spectrum, and dumps
per-experiment NPZ fixtures for the Julia reimplementation.
"""
import os
os.environ["COBAYA_PACKAGES_PATH"] = "/tmp/hillik_ref_packages"
import numpy as np
from cobaya.yaml import yaml_load_file
from cobaya.component import get_component_class

DATA = "/tmp/hillik_data"
OUT = "/tmp/hillik_fixtures"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Theory spectra: D_l in muK^2, ell = 0..9050 (ACT DR6 data-release theory)
# ----------------------------------------------------------------------------
theory_dir = "/home/marcobonici/Desktop/work/CosmologicalLikelihoods/ACTLikelihood.jl/data/ACT_DR6_TTTEEE_filtered"
dlTT = np.loadtxt(theory_dir + "/cmb_theory_tt.txt")
dlTE = np.loadtxt(theory_dir + "/cmb_theory_te.txt")
dlEE = np.loadtxt(theory_dir + "/cmb_theory_ee.txt")
LMAX_THEORY = min(len(dlTT), len(dlTE), len(dlEE)) - 1
print(f"Theory Dl loaded: lmax = {LMAX_THEORY}")

# ----------------------------------------------------------------------------
# Baseline nuisance values: paper posterior means (arXiv:2511.04733 Tables 4-5)
# ----------------------------------------------------------------------------
P = {}
# --- common foreground (10 sampled: 6 amplitudes/xi/beta_cib/beta_radio + 3 dust betas are per-survey)
P["xi"] = 0.044
P["Atsz"] = 2.98
P["Acib"] = 4.32
P["Aksz"] = 1.20
P["beta_cib"] = 1.79
P["beta_radio"] = -0.88
# fixed
P["alpha_tsz"] = 0.0
P["T_cib"] = 25.0
P["T_dust"] = 19.6
P["beta_dusty"] = P["beta_cib"]  # lambda alias
P["alpha_cib"] = 1.2  # unused with file templates

# --- Planck (12 instrument + 11 FG sampled)
P["A_planck"] = 1.001
for m, v in dict(PLK_cal_100A=1.0007, PLK_cal_100B=0.9961, PLK_cal_143B=1.0002,
                 PLK_cal_217A=1.0026, PLK_cal_217B=1.0001).items():
    P[m] = v
P["PLK_cal_143A"] = 1.0  # fixed reference
for m, v in dict(PLK_pe_100A=1.007, PLK_pe_100B=1.013, PLK_pe_143A=0.962,
                 PLK_pe_143B=1.022, PLK_pe_217A=1.035, PLK_pe_217B=1.021).items():
    P[m] = v
for f, v in dict(PLK_Adust100TT=24.2, PLK_Adust143TT=21.0, PLK_Adust217TT=9.71,
                 PLK_Adust100TE=1.477, PLK_Adust143TE=0.792, PLK_Adust217TE=0.434,
                 PLK_Adust100EE=0.628, PLK_Adust143EE=0.293, PLK_Adust217EE=0.174).items():
    P[f] = v
P["PLK_alpha_dustTT"] = -2.62
for f in (100, 143, 217):
    P[f"PLK_alpha_dust{f}TT"] = P["PLK_alpha_dustTT"]
P["PLK_alpha_dustTE"] = -2.4
P["PLK_alpha_dustEE"] = -2.4
for f in (100, 143, 217):
    P[f"PLK_alpha_dust{f}TE"] = -2.4
    P[f"PLK_alpha_dust{f}EE"] = -2.4
P["PLK_beta_dustTT"] = 1.510
P["PLK_beta_dustTE"] = 1.588
P["PLK_beta_dustEE"] = 1.612
P["PLK_beta_dustET"] = P["PLK_beta_dustTE"]
P["PLK_alpha_dustET"] = P["PLK_alpha_dustTE"]
for f in (100, 143, 217):
    P[f"PLK_alpha_dust{f}ET"] = P["PLK_alpha_dustET"]
    P[f"PLK_Adust{f}ET"] = P[f"PLK_Adust{f}TE"]
P["PLK_radio_TT"] = 58.06
P["PLK_cib_ps"] = 9.71
# fixed zeros
for f1, f2 in [(100,100),(100,143),(100,217),(143,143),(143,217),(217,217)]:
    P[f"PLK_ps_{f1}x{f2}"] = 0.0
    P[f"PLK_dustT_{f1}"] = 0.0
    P[f"PLK_dustE_{f1}"] = 0.0

# --- ACT (14 instrument + 5 FG likelihood params; ACT_radio_EE is dead/prior-only)
P["ACT_cal"] = 0.9868
for m, v in dict(ACT_cal_dr6_pa4_f220=0.9908, ACT_cal_dr6_pa5_f090=0.9981,
                 ACT_cal_dr6_pa6_f090=0.9990, ACT_cal_dr6_pa6_f150=1.0038).items():
    P[m] = v
P["ACT_cal_dr6_pa5_f150"] = 1.0  # fixed reference
P["ACT_pe_dr6_pa4_f220"] = 1.0  # fixed (PA4 pol unused)
for m, v in dict(ACT_pe_dr6_pa5_f090=1.0015, ACT_pe_dr6_pa6_f090=1.0109,
                 ACT_pe_dr6_pa5_f150=1.0067, ACT_pe_dr6_pa6_f150=1.0085).items():
    P[m] = v
for m, v in dict(ACT_band_shift_dr6_pa4_f220=0.06, ACT_band_shift_dr6_pa5_f090=0.46,
                 ACT_band_shift_dr6_pa6_f090=0.92, ACT_band_shift_dr6_pa5_f150=-1.64,
                 ACT_band_shift_dr6_pa6_f150=-1.00).items():
    P[m] = v
P["ACT_AdustTT"] = 7.81
P["ACT_AdustTE"] = 0.413
P["ACT_AdustEE"] = 0.185
P["ACT_beta_dustTT"] = 1.5
P["ACT_beta_dustTE"] = 1.5
P["ACT_beta_dustEE"] = 1.5
P["ACT_alpha_dustTT"] = -2.6
P["ACT_alpha_dustTE"] = -2.4
P["ACT_alpha_dustEE"] = -2.4
P["ACT_radio_TT"] = 3.01
P["ACT_cib_ps"] = 7.45
P["ACT_radio_EE"] = 0.2  # dead parameter (prior-only)

# --- SPT (22 instrument + 5 FG)
P["SPT3G_cal"] = 1.0013
P["SPT3G_cal_90"] = 1.0004
P["SPT3G_cal_150"] = 1.0  # fixed reference
P["SPT3G_cal_220"] = 1.0092
for m, v in dict(SPT3G_pe_90=1.0014, SPT3G_pe_150=1.0030, SPT3G_pe_220=0.9981).items():
    P[m] = v
P["SPT3G_kappa"] = -0.00008
for i, v in dict(SPT3G_beta_1=-1.15, SPT3G_beta_2=-0.50, SPT3G_beta_3=0.03,
                 SPT3G_beta_4=-1.85, SPT3G_beta_5=0.24, SPT3G_beta_6=-1.40,
                 SPT3G_beta_7=0.15, SPT3G_beta_8=-0.20, SPT3G_beta_9=0.42).items():
    P[i] = v
for f, v in dict(SPT3G_beta_pol_90=0.555, SPT3G_beta_pol_150=0.716,
                 SPT3G_beta_pol_220=0.681).items():
    P[f] = v
for f, v in dict(SPT3G_T2P2_90=-0.0073, SPT3G_T2P2_150=-0.0154,
                 SPT3G_T2P2_220=-0.0291).items():
    P[f] = v
P["SPT3G_AdustTT"] = 2.30
P["SPT3G_AdustTE"] = 0.105
P["SPT3G_AdustEE"] = 0.057
P["SPT3G_beta_dustTT"] = 1.48
P["SPT3G_beta_dustTE"] = 1.51
P["SPT3G_beta_dustEE"] = 1.51
P["SPT3G_alpha_dustTT"] = -2.53
P["SPT3G_alpha_dustTE"] = -2.42
P["SPT3G_alpha_dustEE"] = -2.42
P["SPT3G_radio_TT"] = 0.68
P["SPT3G_cib_ps"] = 7.33
P["SPT3G_radio_EE"] = 0.0  # fixed
P["SPT3G_radio_TE"] = 0.0  # fixed
for f in (90, 150, 220):
    P[f"SPT3G_ps_90x90" if f == 90 else ""] = P.get("", 0)  # placeholder no-op
# SPT ps and dustT/E amplitudes fixed to zero
for a, b in [("90","90"),("90","150"),("90","220"),("150","150"),("150","220"),("220","220")]:
    P[f"SPT3G_ps_{a}x{b}"] = 0.0
for f in (90, 150, 220):
    P[f"SPT3G_dustT_{f}"] = 0.0
    P[f"SPT3G_dustE_{f}"] = 0.0

# ----------------------------------------------------------------------------
# Instantiate the three likelihoods
# ----------------------------------------------------------------------------
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

print(f"dof: PLK={plk.dof()} ACT={act.dof()} SPT={spt.dof()}")
assert plk.dof() == 4872 and act.dof() == 1139 and spt.dof() == 1392

# ----------------------------------------------------------------------------
# Evaluate at baseline
# ----------------------------------------------------------------------------
def theory_dict(lmax):
    return {
        "TT": dlTT[: lmax + 1],
        "TE": dlTE[: lmax + 1],
        "ET": dlTE[: lmax + 1],
        "EE": dlEE[: lmax + 1],
    }

dl_plk = theory_dict(plk.lmax)
dl_act = theory_dict(act.lmax_bpw)
# SPT code indexes dl_cmb[mode][ell] with absolute ell: pass full 0..lmax arrays
dl_spt = theory_dict(spt.lmax)

chi2_plk = plk.compute_chi2(dl_plk, **P)
chi2_act = act.compute_chi2(dl_act, **P)
chi2_spt = spt.compute_chi2(dl_spt, **P)
print(f"chi2: PLK={chi2_plk:.6f}/{plk.dof()}  ACT={chi2_act:.6f}/{act.dof()}  SPT={chi2_spt:.6f}/{spt.dof()}")
print(f"joint chi2 = {chi2_plk + chi2_act + chi2_spt:.6f} / 7403")

np.savez(
    OUT + "/baseline.npz",
    chi2_plk=chi2_plk, chi2_act=chi2_act, chi2_spt=chi2_spt,
    delta_plk=plk.delta_dl, delta_act=np.array(act.delta_dl),
)
print("saved", OUT + "/baseline.npz")
