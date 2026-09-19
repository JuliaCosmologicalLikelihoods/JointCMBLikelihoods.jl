"""
Convert the four shared Hillik foreground templates into the
JointCMBLikelihoods shared-templates artifact.

Templates (Dl in muK^2, ell=2..13500 contiguous):
  dl_tsz_planck_13500.dat       tSZ template
  dl_ksz_planck_2_13500.dat     kSZ template
  cib_extra.dat                  clustered CIB template
  sz_x_cib_template_13500.dat    tSZ x CIB template

All are normalized at lnorm=3000 at load time by the likelihood code
(convention: template / template[3000]); the artifact stores raw values.

Usage: python convert_shared_templates.py <hillik_foregrounds_dir> <output_dir>
"""
import json
import os
import sys

import numpy as np

TEMPLATES = {
    "tsz": "dl_tsz_planck_13500.dat",
    "ksz": "dl_ksz_planck_2_13500.dat",
    "cib": "cib_extra.dat",
    "szxcib": "sz_x_cib_template_13500.dat",
}


def main(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = {}
    for name, fname in TEMPLATES.items():
        l, d = np.loadtxt(os.path.join(src_dir, fname), unpack=True)
        assert np.all(np.diff(l) == 1) and int(l[0]) == 2
        np.save(os.path.join(out_dir, f"{name}_ell.npy"), l.astype(np.int64))
        np.save(os.path.join(out_dir, f"{name}_dl.npy"), d)
        meta[name] = {
            "source_file": fname,
            "ell_min": int(l.min()),
            "ell_max": int(l.max()),
            "n": len(l),
            "unit": "Dl muK^2 (raw, unnormalized)",
        }
    meta["normalization"] = "likelihood divides by template[lnorm]; lnorm=3000 (fgmodel default)"
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("shared templates written:", list(TEMPLATES))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
