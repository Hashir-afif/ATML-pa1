"""Task 4 methods. ``build_method(cfg)`` maps a resolved config to a method object."""
from task4.methods.gcsc import GCSC
from task4.methods.proser import PROSER
from task4.methods.vanilla import Vanilla


def build_method(cfg: dict):
    m = cfg["method"]
    if m == "vanilla":
        return Vanilla()
    if m == "gcsc":
        return GCSC()
    if m == "proser":
        p = cfg["proser"]
        return PROSER(cfg["model"]["num_classes"], p["dummy_classifiers"], p["beta"], p["gamma"], p["mixup_beta_alpha"])
    raise ValueError(f"unknown method {m!r}")
