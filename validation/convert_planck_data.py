"""
Convert the HiLLiPoP v4.2 PR4 TT/TE/EE data into the JointCMBLikelihoods
Planck artifact.

Reproduces exactly the frozen Hillik (d0e455cf) Planck pipeline:
- 6 maps (100A, 100B, 143A, 143B, 217A, 217B), 15 cross-spectra
- lite binning: ell 30-250 unbinned + 251-2500 in bins of 10
- per-xfreq multipole ranges from binning_v4.2.fits (LMIN/LMAX per mode)
- 4872-element data vector (TT: 15 xfreq cut + EE + TE/ET combined)
- invfll_PR4_v4.2_TTTEEE_bin.fits inverse covariance (4872^2), /1e24, float32

Output layout (NPZ + NPY):
  data_vector.npy     (4872,)  binned data Dl [muK^2]
  inv_cov.npy         (4872, 4872) float32 inverse covariance
  dldata_tt/ee/te/et.npy    (15, 2501) unbinned cross-spectra [muK^2]
  dlweight_tt/ee/te/et.npy   (15, 2501) inverse-variance weights
  cal_factors are computed at runtime from parameters.
  lmins_tt/ee/te.npy  (6,) per-xfreq multipole ranges
  lmaxs_tt/ee/te.npy  (6,)
  binning.npz         lite binning (lmins, lmaxs)
  metadata.json       provenance

Usage: python convert_planck_data.py <hillik_data_root> <output_dir>
  <hillik_data_root> contains planck_2020/hillipop/{data,foregrounds}
"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from itertools import combinations

MAPNAMES = ["100A", "100B", "143A", "143B", "217A", "217B"]
FREQS = [100, 100, 143, 143, 217, 217]
NFREQ = 3

# lite binning (hillipop.py lite_lmins/lite_lmaxs)
LITE_LMINS = list(np.arange(30, 251, 1)) + list(np.arange(251, 2500, 10))
LITE_LMAXS = list(np.arange(30, 251, 1)) + list(np.arange(251, 2500, 10) + 9)
LMAX = 2500


def xspec2xfreq():
    # (f1, f2) xfreq index per 15 xspecs, matching hillipop._xspec2xfreq
    list_fqs = [(f1, f2) for f1 in range(NFREQ) for f2 in range(f1, NFREQ)]
    freqs = sorted(set(FREQS))
    out = []
    for m1 in range(6):
        for m2 in range(m1 + 1, 6):
            f1 = freqs.index(FREQS[m1])
            f2 = freqs.index(FREQS[m2])
            out.append(list_fqs.index((f1, f2)))
    return out


def read_dl_xspectra(basename, hdu=1):
    """Dl (TT, EE, TE, ET) in muK^2, shape (mode, nxspec=15, lmax+1)."""
    with fits.open(f"{basename}_{MAPNAMES[0]}x{MAPNAMES[1]}.fits") as hdus:
        nhdu = len(hdus)
    if nhdu == 1:
        hdu = 0
    dldata = []
    for m1, m2 in combinations(MAPNAMES, 2):
        data = fits.getdata(f"{basename}_{m1}x{m2}.fits", hdu) * 1e12
        tmpcl = list(data[[0, 1, 3], : LMAX + 1])
        data = fits.getdata(f"{basename}_{m2}x{m1}.fits", hdu) * 1e12
        tmpcl.append(data[3, : LMAX + 1])
        dldata.append(tmpcl)
    dldata = np.transpose(np.array(dldata), (1, 0, 2))
    return dict(zip(["TT", "EE", "TE", "ET"], dldata))


def read_multipole_ranges(filename):
    lmins, lmaxs = {}, {}
    with fits.open(filename) as hdus:
        for hdu in hdus[1:]:
            tag = hdu.header["spec"]
            lmins[tag] = hdu.data.LMIN
            lmaxs[tag] = hdu.data.LMAX
    lmins["ET"] = lmins["TE"]
    lmaxs["ET"] = lmaxs["TE"]
    return lmins, lmaxs


class Bins:
    """Port of hillik_planck/bins.py Bins (only what the converter needs)."""

    def __init__(self, lmins, lmaxs):
        lmins = np.asarray(lmins)
        lmaxs = np.asarray(lmaxs)
        cut = np.logical_and(lmaxs >= 2, lmins >= 2)
        self.lmins = lmins[cut]
        self.lmaxs = lmaxs[cut]
        self._derive()

    def _derive(self):
        self.lmin = min(self.lmins)
        self.lmax = max(self.lmaxs)
        self.nbins = len(self.lmins)
        self.lbin = (self.lmins + self.lmaxs) / 2.0
        self.dl = self.lmaxs - self.lmins + 1

    def cut_binning(self, lmin, lmax):
        sel = np.where((self.lmins >= lmin) & (self.lmaxs <= lmax))[0]
        self.lmins = self.lmins[sel]
        self.lmaxs = self.lmaxs[sel]
        self._derive()

    def bin_operators(self):
        # hillipop._select_spectra bins with Dl=False: pure average of Dl
        p = np.zeros((self.nbins, self.lmax + 1))
        for b, (a, z) in enumerate(zip(self.lmins, self.lmaxs)):
            dl = z - a + 1
            p[b, a : z + 1] = 1.0 / dl
        return p


def select_spectra(wf, lmins, lmaxs, xspec2xfreq, cl, nxfreq=6):
    """Flatten per-xfreq cut+binned spectra, matching _select_spectra."""
    xl = []
    for xf in range(nxfreq):
        idx = [i for i, v in enumerate(xspec2xfreq) if v == xf]
        # hillipop: lmin = self._lmins[mode][self._xspec2xfreq.index(xf)]
        # index() returns the FIRST xspec with this xfreq
        first = xspec2xfreq.index(xf)
        lmin = lmins[first]
        lmax = lmaxs[first]
        mywf = Bins(LITE_LMINS, LITE_LMAXS)
        mywf.cut_binning(lmin, lmax)
        p = mywf.bin_operators()
        # cl indexed by xfreq
        xl += list(cl[xf, : mywf.lmax + 1] @ p.T)
    return xl


def main(data_root, out_dir):
    data_folder = os.path.join(data_root, "planck_2020", "hillipop", "data")
    os.makedirs(out_dir, exist_ok=True)

    basename = os.path.join(data_folder, "dl_PR4_v4.2")
    dldata = read_dl_xspectra(basename)
    dlsig = read_dl_xspectra(basename, hdu=2)
    dlweight = {}
    for m, w8 in dlsig.items():
        w8 = np.array(w8)
        w8[w8 == 0] = np.inf
        dlweight[m] = 1 / w8**2

    lmins, lmaxs = read_multipole_ranges(
        os.path.join(data_folder, "binning_v4.2.fits")
    )

    # inverse covariance
    fname = os.path.join(data_folder, "invfll_PR4_v4.2_TTTEEE_bin.fits")
    data = fits.getdata(fname)
    nel = int(np.sqrt(data.size))
    invkll = data.reshape((nel, nel)) / 1e24
    print(f"invkll: {nel} (expect 4872)")
    # The frozen Hillik chi2 uses scipy dsymv, which reads only the LOWER
    # triangle of the (float32, slightly asymmetric) matrix. Symmetrize from
    # the lower triangle so a plain product reproduces the frozen value.
    L = np.tril(invkll)
    invkll = L + L.T - np.diag(np.diag(L))

    # assemble data vector: TT then EE then TE(ET combined)
    x2f = xspec2xfreq()

    # xfreq-averaged data (normed) for TT and EE; TE/ET combined with weights
    def xspectra_to_xfreq(cl, weight, normed=True):
        xcl = np.zeros((6, LMAX + 1))
        xw8 = np.zeros((6, LMAX + 1))
        for xs in range(15):
            xcl[x2f[xs]] += weight[xs] * cl[xs]
            xw8[x2f[xs]] += weight[xs]
        xw8[xw8 == 0] = np.inf
        return xcl / xw8 if normed else (xcl, xw8)

    Xl = []
    Xl += select_spectra(None, lmins["TT"], lmaxs["TT"], x2f,
                         xspectra_to_xfreq(dldata["TT"], dlweight["TT"]))
    n_tt = len(Xl)
    Xl += select_spectra(None, lmins["EE"], lmaxs["EE"], x2f,
                         xspectra_to_xfreq(dldata["EE"], dlweight["EE"]))
    n_ee = len(Xl) - n_tt
    RlTE, WlTE = xspectra_to_xfreq(dldata["TE"], dlweight["TE"], normed=False)
    RlET, WlET = xspectra_to_xfreq(dldata["ET"], dlweight["ET"], normed=False)
    Xl += select_spectra(None, lmins["TE"], lmaxs["TE"], x2f,
                         (RlTE + RlET) / (WlTE + WlET))
    n_te = len(Xl) - n_tt - n_ee
    data_vector = np.array(Xl)
    print(f"data vector: TT={n_tt} EE={n_ee} TE={n_te} total={len(data_vector)}")
    assert len(data_vector) == 4872, len(data_vector)
    assert nel == 4872

    # save
    np.save(os.path.join(out_dir, "data_vector.npy"), data_vector)
    np.save(os.path.join(out_dir, "inv_cov.npy"), invkll.astype(np.float32))
    for mode in ("TT", "EE", "TE", "ET"):
        np.save(os.path.join(out_dir, f"dldata_{mode.lower()}.npy"), dldata[mode])
        np.save(os.path.join(out_dir, f"dlweight_{mode.lower()}.npy"), dlweight[mode])
    for mode in ("TT", "EE", "TE"):
        np.save(os.path.join(out_dir, f"lmins_{mode.lower()}.npy"), lmins[mode])
        np.save(os.path.join(out_dir, f"lmaxs_{mode.lower()}.npy"), lmaxs[mode])
    np.savez(
        os.path.join(out_dir, "binning.npz"),
        lmins=np.asarray(LITE_LMINS), lmaxs=np.asarray(LITE_LMAXS),
    )
    np.save(os.path.join(out_dir, "xspec2xfreq.npy"), np.asarray(x2f))

    meta = {
        "source": "planck_2020_hillipop_TTTEEE_v4.2",
        "maps": MAPNAMES,
        "frequencies": FREQS,
        "xspec2xfreq": x2f,
        "lmax": LMAX,
        "binning": "30-250 unbinned, 251-2500 in bins of 10",
        "cuts": {"TT": n_tt, "EE": n_ee, "TE": n_te},
        "data_vector_size": len(data_vector),
        "inv_cov_dtype": "float32",
        "convention": "Dl in muK^2; invkll = invfll/1e24; TE+ET weight-combined",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("metadata written")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
