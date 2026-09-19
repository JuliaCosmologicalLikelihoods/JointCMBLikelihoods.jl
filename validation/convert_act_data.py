"""
Convert the ACT DR6 SACC release into the JointCMBLikelihoods ACT artifact,
applying the paper-era TTTEEE_PACT selection (frozen Hillik d0e455cf).

Selection: leff (bin-center) cuts TT>=2000, TE/ET>=1500, EE>=1000 for all
map pairs; PA4_f220 pairs are TT-only. Result: 1139-element vector.

Output layout:
  data_vector.npy   (1139,) bandpower Dl [muK^2]
  inv_cov.npy       (1139, 1139) inverse covariance
  windows/*.npy     one (n_ell, n_bin) bandpower window per selected (spec, pol)
  leff.npy          (1139,) effective multipoles
  spec_index.json   ordered (exp1, exp2, pol) list + bin counts
  bandpasses.npz    per-map nu + bandpass (for chromatic SEDs)
  beams/*.npy        per-map chromatic beam (n_ell, n_nu) normalized at ell=0
  metadata.json     provenance

Usage: python convert_act_data.py <dr6_data.fits> <output_dir>
"""
import json
import os
import sys

import numpy as np
import sacc

MAPS = ["dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150", "dr6_pa6_f090", "dr6_pa6_f150"]
POL_DICT = {"T": "0", "E": "e", "B": "b"}
EXT_DICT = {"T": "0", "E": "2", "B": "2"}
LMIN_GLOBAL, LMAX_GLOBAL = 2, 8501
LMAX_THEORY = 9000


def get_cl_name(pol, exp1, exp2):
    p1, p2 = pol
    tname1 = exp1 + "_s" + EXT_DICT[p1]
    tname2 = exp2 + "_s" + EXT_DICT[p2]
    if p2 == "T":
        dt = "cl_" + POL_DICT[p2] + POL_DICT[p1]
    else:
        dt = "cl_" + POL_DICT[p1] + POL_DICT[p2]
    return dt, tname1, tname2


def main(input_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "windows"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "beams"), exist_ok=True)

    data = sacc.Sacc.load_fits(input_file)

    # paper-era PACT selection: 15 map pairs; f220 pairs TT-only;
    # auto pairs (same map twice) use [TT, TE, EE]; cross pairs [TT, TE, ET, EE]
    from itertools import combinations_with_replacement
    spectra = []
    for e1, e2 in combinations_with_replacement(MAPS, 2):
        if "f220" in e1 or "f220" in e2:
            pols = ["TT"]
        elif e1 == e2:
            pols = ["TT", "TE", "EE"]
        else:
            pols = ["TT", "TE", "ET", "EE"]
        spectra.append({"experiments": [e1, e2], "polarizations": pols})

    select_ind = []
    spec_index = []
    leff = []
    for ispec, spec in enumerate(spectra):
        e1, e2 = spec["experiments"]
        for pol in spec["polarizations"]:
            # paper cuts on leff (bin centers)
            lmin = {"TT": 2000, "TE": 1500, "ET": 1500, "EE": 1000}[pol]
            lmax = LMAX_GLOBAL
            dt, t1, t2 = get_cl_name(pol, e1, e2)
            ls, dls = data.get_ell_cl(dt, t1, t2)
            sel = [i for i, (l, dl) in enumerate(zip(ls, dls)) if lmin <= l <= lmax]
            ind = data.indices(dt, (t1, t2), ell__gt=lmin, ell__lt=lmax)
            assert len(ind) == len(sel), (pol, e1, e2, len(ind), len(sel))
            bpw = data.get_bandpower_windows(ind)
            n_bin = len(ind)
            np.save(
                os.path.join(out_dir, "windows", f"{ispec:02d}_{pol}.npy"),
                bpw.weight,  # (n_ell, n_bin)
            )
            np.save(
                os.path.join(out_dir, "windows", f"{ispec:02d}_{pol}_values.npy"),
                bpw.values,
            )
            spec_index.append(
                {"ispec": ispec, "experiments": [e1, e2], "pol": pol, "n_bins": n_bin}
            )
            leff += list(ls[sel])
            select_ind += list(ind)

    data_vector = np.concatenate(
        [data.get_ell_cl(*_)[1] for _ in []]
    ) if False else np.array(
        [dl for i in select_ind for dl in [data.mean[i]]]
    )
    leff = np.array(leff)
    n = len(select_ind)
    print(f"selected {n} bins (expect 1139)")
    assert n == 1139, n

    covmat = data.covariance.covmat[np.ix_(select_ind, select_ind)]
    inv_cov = np.linalg.inv(covmat)

    np.save(os.path.join(out_dir, "data_vector.npy"), data_vector)
    np.save(os.path.join(out_dir, "inv_cov.npy"), inv_cov)
    np.save(os.path.join(out_dir, "leff.npy"), leff)

    # beams + bandpasses for chromatic SEDs (same info in T or E)
    # frozen hillik: lmax_bpw = max(bpw.values) over selected spectra = 8501
    lmax_bpw = max(
        int(np.load(os.path.join(out_dir, "windows", f"{s:02d}_{p}_values.npy")).max())
        for s in range(len(spectra)) for p in spectra[s]["polarizations"]
    )
    beam_arrays = {}
    for m in MAPS:
        bpl = data.tracers[m + "_s0"]
        beam = bpl.beam[: lmax_bpw + 1, :] / bpl.beam[0, :][np.newaxis, ...]
        np.save(os.path.join(out_dir, "beams", f"{m}.npy"), beam)
        beam_arrays[m] = (np.asarray(bpl.nu), np.asarray(bpl.bandpass))
    np.savez(
        os.path.join(out_dir, "bandpasses.npz"),
        **{f"{m}_nu": v[0] for m, v in beam_arrays.items()},
        **{f"{m}_bandpass": v[1] for m, v in beam_arrays.items()},
    )

    with open(os.path.join(out_dir, "spec_index.json"), "w") as f:
        json.dump({"spectra": spec_index, "lmax_bpw": lmax_bpw}, f, indent=1)

    meta = {
        "source": "ACT DR6 dr6_data.fits (ACTDR6MFLike v1.0)",
        "selection": "paper-era TTTEEE_PACT leff cuts: TT>=2000, TE/ET>=1500, EE>=1000; f220 pairs TT-only",
        "maps": MAPS,
        "data_vector_size": n,
        "convention": "Dl in muK^2; X_model = bpw.weight.T @ (dl_cmb + dl_fg) / cal",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("metadata written")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
