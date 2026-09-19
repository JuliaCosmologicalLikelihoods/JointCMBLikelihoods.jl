"""Instantiate the three frozen Hillik likelihoods from the paper-era joint parfile."""
import os
os.environ["COBAYA_PACKAGES_PATH"] = "/tmp/hillik_ref_packages"
from cobaya.yaml import yaml_load_file
from cobaya.component import get_component_class

DATA = "/tmp/hillik_data"
configs = {
    "hillik_planck.TTTEEE": "/tmp/hillik_frozen/hillik_planck/TTTEEE.yaml",
    "hillik_spt.TTTEEE":    "/tmp/hillik_frozen/hillik_spt/TTTEEE.yaml",
    "hillik_act.TTTEEE_PACT": "/tmp/hillik_frozen/hillik_act/TTTEEE_PACT.yaml",
}

for name, yml in configs.items():
    info = yaml_load_file(yml)
    info["path"] = DATA
    cls = get_component_class(name)
    inst = cls(info)
    inst.set_logger()
    inst.initialize()
    print(f"{name}: dof = {inst.dof()}")
